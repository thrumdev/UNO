import comfy
import comfy.model_management as mm
from einops import rearrange
import folder_paths
import node_helpers
import torch
import torchvision.transforms.functional as TVF

from .uno.flux import util as uno_util
from .uno.flux.model import Flux as FluxModel

from PIL import Image

def print_sd_weightnames(sd, name):
    print(f"finding weight names for {name}")
    for k in sd:
        if "double_block" in k:
            print(f"Double block key: {k}")
            break

    for k in sd:
        if "single_block" in k:
            print(f"single block key: {k}")
            break

    for k in sd:
        if "img_in" in k:
            print(f"img_in key: {k}")
            break

    for k in sd:
        if "vector_in" in k:
            print(f"vector_in key: {k}")
            break

# returns a function that, when called, returns the given model
def make_fake_model_builder(model: FluxModel):
    def return_model(image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        # expected in the adapter.
        model.patch_size = 2
        print(f"setting model dtype={dtype}")
        model.dtype = dtype
        return model.to(device)

    return return_model

class UnoComfyAdapter(comfy.model_base.Flux):
    def __init__(self, model_config, model: FluxModel, device=None):
        super().__init__(model_config, device=device, unet_model=make_fake_model_builder(model))

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        ref_img = kwargs.get("ref_img", None)
        if ref_img is not None:
            # kind of a hack but hopefully works.
            out["ref_img"] = comfy.conds.CONDConstant(ref_img)
        return out

class UnoFluxModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Flux Checkpoint or LoRA"}),
                "config_name": (["flux-dev", "flux-dev-fp8", "flux-schnell"], {"default": "flux-dev"}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the UNO LoRA file."}),
                "lora_rank": ("INT", {"default": 512, "min": 16, "max": 512, "tooltip": "The number of ranks to apply the UNO LoRa atop the Flux weights"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "uno"
    DESCRIPTION = "Load and apply the UNO LoRa on top of a loaded Flux model."

    def loadmodel(self, model, config_name, lora_name, lora_rank):
        # extract model state dict. this should apply LoRA patches as well.
        mm.load_models_gpu([model], force_patch_weights=True)
        sd = model.model.state_dict_for_saving()

        # load uno lora safetensors
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        uno_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # strip out prefix
        key_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        sd = comfy.utils.state_dict_prefix_replace(sd, {key_prefix: ""}, filter_keys=True)
        unet_config = comfy.model_detection.detect_unet_config(sd, "")

        print_sd_weightnames(sd, "fluxmodel")
        print_sd_weightnames(uno_sd, "uno")

        assert unet_config is not None

        model_config = comfy.supported_models.Flux(unet_config)
        print("Created Flux:", type(model_config), hasattr(model_config, "unet_config"))
        print(f"  unet config len={len(model_config.unet_config)}")

        # instantiate model class, update using lora
        with torch.device("meta"):
            model = FluxModel(uno_util.configs[config_name].params)
        model = uno_util.set_lora(model, lora_rank, device="meta")

        # ensure device and type are consistent across both state dicts. strip out prefix
        if sd:
            dtype = next(iter(sd.values())).dtype
            device = next(iter(sd.values())).device

            model_config.unet_config['dtype'] = dtype
            uno_sd = {k: v.to(dtype=dtype, device=device) for k, v in uno_sd.items()}

        # merge state dicts and load
        sd.update(uno_sd)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print(f"Loaded UNO LoRa. missing_count={len(missing)} unexpected_count={len(unexpected)}")
        
        # DEBUG: see whether LoRa parameters make sense.
        for name, param in model.named_parameters():
            if "lora" in name:
                print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")

        # instantiate adapter.
        model = UnoComfyAdapter(model_config, model)

        # return model patcher
        offload_device = mm.unet_offload_device()
        load_device = mm.get_torch_device()
        model = model.to(offload_device)
        model = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (model,)

class UnoConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "vae": ("VAE", { "tooltip": "Flux VAE" })
            },
            "optional": {
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "append"
    CATEGORY = "uno"
    DESCRIPTION = "Provide 1-4 reference images for UNO to be VAE encoded and attached to the conditioning"

    def append(self, conditioning, vae, ref_image_1 = None, ref_image_2 = None, ref_image_3 = None, ref_image_4 = None):
        ref_img = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        ref_img = [r for r in ref_img if r is not None]

        for r in ref_img:
            print(f"ref image shape={r.shape} dtype={r.dtype}")
            assert r.shape[0] == 1

        # just copied from inference.py
        long_size = 512 if len(ref_img) <= 1 else 320

        def preprocess(x):
            device = x.device
            # assume x is a tensor of shape [B, H, W, 3]
            # convert to image, resize
            if x.dtype == torch.float32 and x.max() <= 1.0:
                x = (x * 255).clamp(0, 255).to(torch.uint8)

            x = Image.fromarray((x.cpu().numpy().astype("uint8")))
            print(f"image shape: {x.size}")
            x = preprocess_ref(x, long_size=long_size)
            print(f"after preprocess shape: {x.size}")

            x = TVF.to_tensor(x)
            # x = x * 2.0 - 1.0
            print(f"after tensor conversion shape: {x.shape}")
            print(f"dtype={x.dtype}, min={x.min()}, max={x.max()}")


            x = x.unsqueeze(0).to(device=device, dtype=torch.float32)
            x = rearrange(x, "b c h w -> b h w c")
            print(f"after unsqueeze: {x.shape}")

            return x

            x = vae.encode(x)
            print("post-VAE ref: ", x.shape, x.mean().item(), x.std().item())
            return x

        ref_img = [preprocess(r[0]) for r in ref_img]

        return ref_img[0]

        # set the conditioning map.
        if len(ref_img) > 0:
            c = node_helpers.conditioning_set_values(conditioning, {"ref_img": ref_img})
        else:
            c = conditioning
        return (c, )

#copied from pipeline.py
def preprocess_ref(raw_image: Image.Image, long_size: int = 512):
    # 获取原始图像的宽度和高度
    image_w, image_h = raw_image.size

    # 计算长边和短边
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # 按新的宽高进行等比例缩放
    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    # 计算裁剪的起始坐标以实现中心裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    print(f"after resize: {raw_image.size}")

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    print(f"after crop: {raw_image.size}")

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image


NODE_CLASS_MAPPINGS = {
    "UnoFluxModelLoader": UnoFluxModelLoader,
    "UnoConditioning": UnoConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnoFluxModelLoader": "UNO Model Loader",
    "UnoConditioning": "Conditioning for UNO sampling",
}
