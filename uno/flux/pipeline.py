import os

import torch
from einops import rearrange
from PIL import ExifTags, Image
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode

from uno.flux.modules.layers import (
    DoubleStreamBlockLoraProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    SingleStreamBlockProcessor,
)
from uno.flux.sampling import denoise, get_noise, get_schedule, prepare, prepare_multi_ip, unpack
from uno.flux.util import (
    get_lora_rank,
    load_ae,
    load_checkpoint,
    load_clip,
    load_flow_model,
    load_flow_model_only_lora,
    load_flow_model_quintized,
    load_t5,
)


def find_nearest_scale(image_h, image_w, predefined_scales):
    """
    根据图片的高度和宽度，找到最近的预定义尺度。

    :param image_h: 图片的高度
    :param image_w: 图片的宽度
    :param predefined_scales: 预定义尺度列表 [(h1, w1), (h2, w2), ...]
    :return: 最近的预定义尺度 (h, w)
    """
    # 计算输入图片的长宽比
    image_ratio = image_h / image_w
    
    # 初始化变量以存储最小差异和最近的尺度
    min_diff = float('inf')
    nearest_scale = None

    # 遍历所有预定义尺度，找到与输入图片长宽比最接近的尺度
    for scale_h, scale_w in predefined_scales:
        predefined_ratio = scale_h / scale_w
        diff = abs(predefined_ratio - image_ratio)
        
        if diff < min_diff:
            min_diff = diff
            nearest_scale = (scale_h, scale_w)
    
    return nearest_scale

def preprocess_ref(raw_image, predefined_scales):
    if predefined_scales is not None:
        image_w, image_h = raw_image.size
        # 为了varlen inference，把图片按bucket进行resize
        bucket_h, bucket_w = find_nearest_scale(image_h, image_w, predefined_scales)

        aspect_ratio = bucket_w / bucket_h
        resize = RandomResizedCrop(
                                    size=(bucket_h, bucket_w),  # Crop to target width height
                                    scale=(1, 1),  # Do not scale.
                                    ratio=(aspect_ratio, aspect_ratio),  # Keep target aspect ratio.
                                    interpolation=InterpolationMode.LANCZOS  # Use LANCZO for downsample.
                                )
        crop_top_coord, crop_left_coord, _, _ = resize.get_params(raw_image, scale=(1, 1), ratio=(
                            aspect_ratio, aspect_ratio))
        crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
        raw_image = resize(raw_image)
    raw_image = raw_image.convert("RGB") 
    return raw_image

def preprocess_ref(raw_image, long_size):
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

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image

class UNOPipeline:
    def __init__(
        self,
        model_type: str,
        device: torch.device,
        offload: bool = False,
        only_lora: bool = False,
        lora_rank: int = 16
    ):
        self.device = device
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device) 
        elif only_lora:
            self.model = load_flow_model_only_lora(
                model_type, device="cpu" if offload else self.device, lora_rank=lora_rank
            )
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)  


    def load_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            from safetensors.torch import load_file as load_sft
            print("Loading checkpoint to replace old keys")
            # load_sft doesn't support torch.device
            if ckpt_path.endswith('safetensors'):
                sd = load_sft(ckpt_path, device='cpu')
                missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
            else:
                dit_state = torch.load(ckpt_path, map_location='cpu')
                sd = {}
                for k in dit_state.keys():
                    sd[k.replace('module.','')] = dit_state[k]
                missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
                self.model.to(str(self.device))
            print(f"missing keys: {missing}\n\n\n\n\nunexpected keys: {unexpected}")

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)


    def __call__(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        true_gs: float = 3,
        neg_prompt: str = '',
        neg_image_prompt: Image = None,
        timestep_to_start_cfg: int = 0,
        **kwargs
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            neg_prompt=neg_prompt,
            **kwargs
        )

    @torch.inference_mode()
    def gradio_generate(self, prompt, width, height, guidance,
                        num_steps, seed,ref_width, image_prompt1,
                        image_prompt2,image_prompt3,image_prompt4,**kargs):
        ref_imgs = [image_prompt1,image_prompt2,image_prompt3,image_prompt4]
        ref_imgs = [img for img in ref_imgs if isinstance(img, Image.Image)]
        ref_imgs = [preprocess_ref(img, ref_width) for img in ref_imgs]
        transform = Compose([
                    ToTensor(),
                    Normalize([0.5], [0.5]),
                ])    
        ref_imgs = [transform(each) for each in ref_imgs]            
        kargs['multi_ip'] = True    

        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt=prompt, width=width, height=height, guidance=guidance,
                   num_steps=num_steps, seed=seed, ref_imgs=ref_imgs,**kargs)

        filename = f"output/gradio/{seed}_{prompt[:20]}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "UNO"
        exif_data[ExifTags.Base.Model] = self.model_type
        info = f"{prompt=}, {seed=}, {width=}, {height=}, {guidance=}, {num_steps=}"
        exif_data[ExifTags.Base.ImageDescription] = info
        img.save(filename, format="png", exif=exif_data)
        return img, filename

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        timestep_to_start_cfg = 1e5,
        true_gs = 3.5,
        neg_prompt = "",
        ref_imgs = None,
        batch = None,
        **kargs
    ):
        single_ip = kargs.get("single_ip", False)
        multi_ip = kargs.get("multi_ip", False)
        pe = kargs.get("pe", "d")
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            self.model, self.ae = self.model.to(self.device),  self.ae.to(self.device)
            if not single_ip and not multi_ip:
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt, pe=pe)
                neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt, pe=pe)
            elif single_ip: 
                ref_img = ref_imgs
                prompt = prompt
                x_1_ref = self.ae.encode(ref_img.unsqueeze(0).to(self.device).to(torch.float32))
                inp_cond = prepare(
                    t5=self.t5, clip=self.clip,
                    img=x, prompt=prompt, ref_img=x_1_ref.to(torch.bfloat16), pe=pe
                )   
                neg_inp_cond = prepare(
                    t5=self.t5, clip=self.clip,
                    img=x, prompt=neg_prompt, ref_img=x_1_ref.to(torch.bfloat16), pe=pe
                )
            elif multi_ip:
                batch = kargs.get("batch", None)
                try:
                    ref_img1 = batch["ref_img1"]
                    ref_img2 = batch["ref_img2"]
                    prompt = batch["txt"]
                    x_1_ref = self.ae.encode(ref_img1.to(self.device).to(torch.float32))
                    x_2_ref = self.ae.encode(ref_img2.to(self.device).to(torch.float32))
                    inp_cond = prepare_multi_ip(
                        t5=self.t5, clip=self.clip,
                        img=x,
                        prompt=prompt, ref_imgs=(x_1_ref.to(torch.bfloat16), x_2_ref.to(torch.bfloat16)), pe=pe
                    )
                    neg_inp_cond = prepare_multi_ip(
                        t5=self.t5, clip=self.clip,
                        img=x,
                        prompt=neg_prompt, ref_imgs=(x_1_ref.to(torch.bfloat16), x_2_ref.to(torch.bfloat16)), pe=pe
                    )            
                except:
                    print('start gradio inference')         
                    x_1_refs = [
                        self.ae.encode(ref_img.unsqueeze(0).to(self.device).to(torch.float32))
                        for ref_img in ref_imgs
                    ]
                    x_1_refs = [each.to(torch.bfloat16) for each in x_1_refs]
                    inp_cond = prepare_multi_ip(
                        t5=self.t5, clip=self.clip,
                        img=x,
                        prompt=prompt, ref_imgs=x_1_refs, pe=pe
                    )
                    neg_inp_cond = prepare_multi_ip(
                        t5=self.t5, clip=self.clip,
                        img=x,
                        prompt=neg_prompt, ref_imgs=x_1_refs, pe=pe
                    )
            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                **kargs
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()
