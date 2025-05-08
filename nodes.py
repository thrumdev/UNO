import comfy
import comfy.model_management as mm
import folder_paths
import node_helpers
import torch

from .uno.flux import util as uno_util
from .uno.flux.model import Flux as FluxModel


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

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "uno"
    DESCRIPTION = "Provide 1-4 reference images for UNO to be VAE encoded and attached to the conditioning"

    def append(self, conditioning, vae, ref_image_1 = None, ref_image_2 = None, ref_image_3 = None, ref_image_4 = None):
        ref_img = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        ref_img = [r for r in ref_img if r is not None]

        # this line is copied more or less verbatim from VaeEncode
        ref_img = [vae.encode(pixels[:,:,:,:3]) for pixels in ref_img]

        # set the conditioning map.
        c = node_helpers.conditioning_set_values(conditioning, {"ref_img": ref_img})
        return (c, )

NODE_CLASS_MAPPINGS = {
    "UnoFluxModelLoader": UnoFluxModelLoader,
    "UnoConditioning": UnoConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnoFluxModelLoader": "UNO Model Loader",
    "UnoConditioning": "Conditioning for UNO sampling",
}
