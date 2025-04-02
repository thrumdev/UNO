import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode


def preprocess_image(image, target_height_ref1, target_width_ref1):
    h, w = image.shape[-2:]
    if h < target_height_ref1 or w < target_width_ref1:
        # 计算长宽比
        aspect_ratio = w / h
        if h < target_height_ref1:
            new_h = target_height_ref1
            new_w = new_h * aspect_ratio
            if new_w < target_width_ref1:
                new_w = target_width_ref1
                new_h = new_w / aspect_ratio
        else:
            new_w = target_width_ref1
            new_h = new_w / aspect_ratio
            if new_h < target_height_ref1:
                new_h = target_height_ref1
                new_w = new_h * aspect_ratio
    else:
        aspect_ratio = w / h
        tgt_aspect_ratio = target_width_ref1 / target_height_ref1
        if aspect_ratio > tgt_aspect_ratio:
            new_h = target_height_ref1
            new_w = new_h * aspect_ratio
        else:
            new_w = target_width_ref1
            new_h = new_w / aspect_ratio
    resize_transform = transforms.Resize((math.ceil(new_h), math.ceil(new_w)))
    image = resize_transform(image)
    # 这里可以继续添加后续的裁剪或其他处理操作，这里用随机裁剪
    crop = transforms.RandomCrop(size=(target_height_ref1, target_width_ref1))
    image = crop(image)
    return image

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


class FLUXPairedDataset(Dataset):

    def __init__(self, json_file, image_root_path, size, is_eval=False, args=None, samples=None):
        super().__init__()

        self.size = size
        assert self.size in [512,1024]
        self.image_root_path = image_root_path
        self.is_eval = is_eval
        self.args = args

        with open(json_file, 'r') as f:
            data = json.load(f)
        if not self.is_eval:            
            for k, v in data.items():
                v['dict_key'] = k
            data = sorted(data.items(), key=lambda x: x[0].split('/')[0])
            # shuffle
            data = self._shuffle_in_groups(data, 64)

            self.data = list(map(lambda x: x[1], data))
        else:
            self.data = data
        if samples is not None:
            self.data = samples

        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

    def _shuffle_in_groups(self, lst, group_size):
        groups = [lst[i: i + group_size] for i in range(0, len(lst), group_size)]
        random.shuffle(groups)
        
        shuffled_lst = []
        for group in groups:
            shuffled_lst.extend(group)
        
        return shuffled_lst

    def _process_image_varlen(self, image, predefined_scales=None, bbox=None, mask_rle=None, cropped_ratio=None):
        if mask_rle is not None:
            raw_image = cv2.imread(os.path.join(self.image_root_path, image))
            mask_decode = coco_mask.decode(mask_rle)
            mask = cv2.cvtColor(mask_decode, cv2.COLOR_GRAY2BGR)
            raw_image = raw_image * mask

            # 创建一个纯白色背景图片
            mask_reverse = (mask==0)*255
            raw_image = raw_image + mask_reverse

            raw_image = Image.fromarray(cv2.cvtColor(raw_image.astype(np.uint8),cv2.COLOR_BGR2RGB)) 
        else:
            raw_image = Image.open(os.path.join(self.image_root_path, image))
        if cropped_ratio is not None:
            # 防止白边出现，先中心crop一下
            w, h = raw_image.size
            crop_w, crop_h = (1-cropped_ratio)*w, (1-cropped_ratio)*h
            crop = CenterCrop((int(crop_h), int(crop_w)))
            raw_image = crop(raw_image)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            img_w, img_h = raw_image.size
            if mask_rle is not None:
                # 中心不变，四周按比例膨胀框大小
                xc, yc = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1, y2-y1
                x1, y1 = xc-w/2 * 1.2, yc-h/2 * 1.2
                x2, y2 = xc+w/2 * 1.2, yc+h/2 * 1.2
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
            raw_image = raw_image.crop((x1, y1, x2, y2))
        crop_coords_top_left = None
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
        
        return raw_image, crop_coords_top_left

    def _process_eval_data(self, idx):
        item = self.data[idx] 
        # read
        dict_key = item["index"]
        img1 = os.path.join(self.image_root_path, item["image_path"])
        txt1 = item["text"]

        predefined_scales = [(384, 512), (512, 384), [512, 512]]  # 预定义尺度列表        
        # read and process image
        raw_image1, ref_crop_coords_top_left1 = self._process_image_varlen(img1, predefined_scales)

        image = self.transform(raw_image1)

        save_ref = os.path.join(self.args.save_path, 'ref_imgs')
        if self.args.rank == 1 and not os.path.exists(os.path.join(save_ref, f"{dict_key}.jpg")):
            os.makedirs(save_ref, exist_ok=True)
            # 创建一个可以在图像上绘图的对象
            draw = ImageDraw.Draw(raw_image1) 
            # 选择字体和大小
            font = ImageFont.truetype("DejaVuSans.ttf", 10)

            # 在图像上绘制文本
            draw.text((8,8), txt1, font=font, fill=(255, 0, 0))
            raw_image1.save(os.path.join(save_ref, f"{dict_key}.jpg"))
            ## 存图片 ##  
        
        return {
            "index": dict_key,
            # "dict_key": dict_key,
            "img": image,            
            "ref_img": image,
            "txt": txt1
        }

    def __getitem__(self, idx):
        if self.is_eval:
            return self._process_eval_data(idx)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

def FLUXPaired_collate(batch):
    base_size = batch[0]['img'].shape[1:]
    for sample in batch[1:]:
        size = sample['img'].shape[1:]
        if base_size != size:
            print(f"Found diff resolution in one batch size, key: {sample['img1_path']}.")
            sample['img'] = TVF.resize(sample['img'], size=base_size, interpolation=TVF.InterpolationMode.BILINEAR)
            sample['ref_img'] = TVF.resize(
                sample['ref_img'],
                size=base_size,
                interpolation=TVF.InterpolationMode.BILINEAR
            )
    return default_collate(batch)


def get_flux_paired_eval_loader(args, samples=None):
    # Initialize dataset.
    eval_dataset = FLUXPairedDataset(
        json_file = args.eval_data_json,
        image_root_path = args.eval_img_root,
        size = args.data_resolution,
        is_eval=True,
        args = args,
        samples=samples
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        collate_fn=FLUXPaired_collate)

    eval_dataloader_iter = iter(eval_dataloader)

    # print(f"[RANK:{args.rank}] Dataset loaded.")

    return eval_dataloader, eval_dataloader_iter


class FLUXPairedMultiIPDataset(Dataset):

    def __init__(self, json_file, image_root_path, size, is_eval=False, args=None):
        super().__init__()

        self.size = size
        assert self.size in [512,1024]
        self.image_root_path = image_root_path
        self.is_eval = is_eval

        if self.is_eval:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
        else:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                v['dict_key'] = k
            data = sorted(data.items(), key=lambda x: x[0].split('/')[-3])
            # 根据分辨率，以组进行shuffle
            data = self._shuffle_in_groups(data, 64)

            self.data = list(map(lambda x: x[1], data))    

        self.args = args


        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

    def _shuffle_in_groups(self, lst, group_size):
        # 将 list 切分成相邻的组
        groups = [lst[i: i + group_size] for i in range(0, len(lst), group_size)]
        
        # 对组进行 shuffle
        random.shuffle(groups)
        
        # 将 shuffle 后的组重新组合成一个新的列表
        shuffled_lst = []
        for group in groups:
            shuffled_lst.extend(group)
        
        return shuffled_lst
    
    def _horizontal_concat(self, images):
        widths, heights = zip(*(img.size for img in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in images:
            new_im.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        return new_im

    def process_bbox(self, raw_image, bbox, rorate_angle=0, mask_rle=None, expand_ratio=1.2):
        """
        rorate_angle: rorate bbox
        mask_rle: if is not None, expand bbox
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            img_w, img_h = raw_image.size

            # 步骤 1: 随机旋转图像
            angle = random.uniform(-rorate_angle, rorate_angle)
            # 先裁剪出 bbox 内的图像
            cropped_image = raw_image.crop((x1, y1, x2, y2))
            # 对裁剪后的图像进行旋转
            rotated_image = cropped_image.rotate(angle, expand=True, fillcolor=(255, 255, 255))

            # 步骤 2: 计算旋转后图像的新边界框
            # 计算旋转后图像的中心
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            # 计算旋转后图像的新宽高
            rotated_w, rotated_h = rotated_image.size
            # 重新计算旋转后图像的边界框
            x1_rotated = xc - rotated_w / 2
            y1_rotated = yc - rotated_h / 2
            x2_rotated = xc + rotated_w / 2
            y2_rotated = yc + rotated_h / 2

            if mask_rle is not None:
                # 步骤 3: 中心不变，四周按比例膨胀框大小
                xc, yc = (x1_rotated + x2_rotated) / 2, (y1_rotated + y2_rotated) / 2
                w, h = x2_rotated - x1_rotated, y2_rotated - y1_rotated
                x1_new = xc - w / 2 * 1.2
                y1_new = yc - h / 2 * 1.2
                x2_new = xc + w / 2 * 1.2
                y2_new = yc + h / 2 * 1.2
                x1_new = max(0, x1_new)
                y1_new = max(0, y1_new)
                x2_new = min(img_w, x2_new)
                y2_new = min(img_h, y2_new)

                # 步骤 4: 创建一个白色背景图像并将旋转后的图像粘贴到合适位置
                expanded_w = int(x2_new - x1_new)
                expanded_h = int(y2_new - y1_new)
                expanded_image = Image.new('RGB', (expanded_w, expanded_h), color=(255, 255, 255))
                paste_x = int((expanded_w - rotated_w) / 2)
                paste_y = int((expanded_h - rotated_h) / 2)
                expanded_image.paste(rotated_image, (paste_x, paste_y))

                return expanded_image
            return rotated_image
        return raw_image


    def _process_image_varlen(
        self,
        image: str,
        predefined_scales: list[tuple[int, int]] | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        mask_rle: dict | None = None,
        cropped_ratio: float | None = None,
        rorate_angle: float = 0.,
        expand_ratio: float = 1.2,
    ):
        if mask_rle is not None:
            raw_image = cv2.imread(os.path.join(self.image_root_path, image))
            mask_decode = coco_mask.decode(mask_rle)
            mask = cv2.cvtColor(mask_decode, cv2.COLOR_GRAY2BGR)
            raw_image = raw_image * mask

            # 创建一个纯白色背景图片
            mask_reverse = (mask==0)*255
            raw_image = raw_image + mask_reverse

            raw_image = Image.fromarray(cv2.cvtColor(raw_image.astype(np.uint8),cv2.COLOR_BGR2RGB)) 
        else:
            raw_image = Image.open(os.path.join(self.image_root_path, image))
        if cropped_ratio is not None:
            # 防止白边出现，先中心crop一下
            w, h = raw_image.size
            crop_w, crop_h = (1 - cropped_ratio) * w, (1 - cropped_ratio) * h
            crop = CenterCrop((int(crop_h), int(crop_w)))
            raw_image = crop(raw_image)
        if bbox is not None:
            raw_image = self.process_bbox(
                raw_image, bbox,
                rorate_angle=rorate_angle, mask_rle=mask_rle, expand_ratio=expand_ratio
            )
        crop_coords_top_left = None
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
        
        return raw_image, crop_coords_top_left

    def _process_eval_data(self, item):
        img1 = os.path.join(self.image_root_path, item["image_path"][0])
        img2 = os.path.join(self.image_root_path, item["image_path"][1])
        txt1 = item["text"]
        index = item["index"]
        predefined_scales=[
            # w    h
            (320, 256),
            (384, 256),        
            (320, 320),
            (256, 320),        
            (256, 384),
        ]
        raw_image1, ref_crop_coords_top_left1 = self._process_image_varlen(img1, predefined_scales)
        raw_image2, ref_crop_coords_top_left2 = self._process_image_varlen(img2, predefined_scales)
        # 将原始图像转换为 PyTorch 张量
        ref_img1 = self.transform(raw_image1)
        ref_img2 = self.transform(raw_image2)

        if self.args.rank == 1 and not os.path.exists(os.path.join(self.args.output_dir, f"ref_{index}.png")):
            ## 存图片 ##
            save_concat_img = self._horizontal_concat([raw_image1, raw_image2])
            # 创建一个可以在图像上绘图的对象
            draw = ImageDraw.Draw(save_concat_img) 
            # 选择字体和大小
            font = ImageFont.truetype("DejaVuSans.ttf", 10)

            # 在图像上绘制文本
            draw.text((8,8), txt1, font=font, fill=(255, 0, 0))
            os.makedirs(self.args.output_dir, exist_ok=True)
            save_concat_img.save(os.path.join(self.args.output_dir, f"ref_{index}.png"))
            ## 存图片 ##    
        
        return {
            "img1_path": img1,
            "txt": txt1,
            "ref_img1": ref_img1,
            "ref_img2": ref_img2,
            "index": torch.tensor(index)
        }


    def __getitem__(self, idx):
        if self.is_eval:
            return self._process_eval_data(self.data[idx])
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.data)


def get_flux_paired_multiip_eval_loader(args):
    # Initialize dataset.
    eval_dataset = FLUXPairedMultiIPDataset(
        json_file = args.eval_data_json,
        image_root_path = args.eval_img_root,
        size = args.data_resolution,
        is_eval=True,
        args = args
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True)

    eval_dataloader_iter = iter(eval_dataloader)

    # print(f"[RANK:{args.rank}] Dataset loaded.")

    return eval_dataloader, eval_dataloader_iter