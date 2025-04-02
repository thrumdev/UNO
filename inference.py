import argparse
import os

from accelerate import Accelerator
from PIL import Image
import json
import itertools

from uno.flux.pipeline import UNOPipeline, preprocess_ref


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, help="The input text prompt")
    parser.add_argument("--image_paths", type=str, help="The input image path", nargs='*')
    parser.add_argument("--eval_json_path", type=str, help="The json path for evaluation dataset")
    parser.add_argument("--offload", action='store_true', help="Offload model to CPU when not in use")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="The number of images for per prompt")
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )

    parser.add_argument("--width", type=int, default=512, help="The width for generated image")
    parser.add_argument("--height", type=int, default=512, help="The height for generated image")
    parser.add_argument("--ref_size", type=int, default=512, help="The longest side size for ref image")
    parser.add_argument("--num_steps", type=int, default=25, help="The num_steps for diffusion process")
    parser.add_argument("--guidance", type=float, default=4, help="The guidance for diffusion process")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference")
    parser.add_argument("--save_path", type=str, default='output/inference', help="Path to save")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--only_lora", action='store_true', help="Load lora model", default=False
    )  # TODO 看看这个和下面那个参数
    parser.add_argument("--concat_refs", action='store_true', help="Concat ref in the result", default=False)
    parser.add_argument("--lora_rank", type=int, default=512)
    parser.add_argument("--data_resolution", type=int, default=512)
    parser.add_argument("--pe", type=str, default='d')
    return parser

def main(args):
    accelerator = Accelerator()

    pipeline = UNOPipeline(
        args.model_type,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank
    )

    assert args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
    
    if args.eval_json_path is not None:
        with open(args.eval_json_path, "rt") as f:
            data_dicts = json.load(f)
        data_root = os.path.dirname(args.eval_json_path)
    else:
        data_root = "./"
        data_dicts = [{"prompt": args.prompt, "image_paths": args.image_paths}]

    for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
        if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
            continue

        ref_imgs = [
            Image.open(os.path.join(data_root, img_path)).convert("RGB")
            for img_path in data_dict["image_paths"]
        ]
        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]

        image_gen = pipeline(
            prompt=data_dict["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed + j,
            ref_imgs=ref_imgs,
            pe=args.pe,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        os.makedirs(args.save_path, exist_ok=True)
        image_gen.save(os.path.join(args.save_path, f"{i}_{j}.png"))

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
