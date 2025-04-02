import argparse
import datetime
import os
import time

import torch
from PIL import Image
from tqdm import tqdm

from image_datasets.dataset import get_flux_paired_eval_loader, get_flux_paired_multiip_eval_loader
from image_datasets.dreambench import get_dreambench_multiip_dataloader
from uno.flux.pipeline import UNOPipeline


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

    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    parser.add_argument("--eval_data_json", type=str, required=True)
    parser.add_argument("--eval_img_root", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--only_lora", action='store_true', help="Load lora model", default=False
    )
    parser.add_argument(
        "--single_ip", action='store_true', help="", default=False
    )
    parser.add_argument(
        "--reference_generation", action='store_true', help="", default=False
    )    
    parser.add_argument(
        "--multi_ip", action='store_true', help="", default=False
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--data_resolution", type=int, default=512)
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument("--node_num", type=int, default=1)  
    parser.add_argument("--concept101", action='store_true', help="", default=False)  
    parser.add_argument("--dreambench_multi", action='store_true', help="", default=False)         
    parser.add_argument(
        "--split_to_per_rank", action='store_true', help="whether split eval dataset to each rank", default=False
    )
    parser.add_argument("--pe", type=str, default='d')      
    return parser


def main(args):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)

    args.device = "cuda"
    args.rank = rank

    if args.single_ip:
        if not args.reference_generation:
            if args.split_to_per_rank:  
                import json
                with open(args.eval_data_json, 'r') as f:
                    data = json.load(f)
                (
                    eval_dataloader,
                    eval_dataloader_iter
                ) = get_flux_paired_eval_loader(args, samples=data[rank::world_size])
            else:
                eval_dataloader, eval_dataloader_iter = get_flux_paired_eval_loader(args)                 
        else:
            import json

            from image_datasets.generation_dataset import FLUXPairedGeneration_eval_loader
            with open(args.eval_data_json, 'r') as f:
                data = json.load(f)

            print('checking...................')
            for k, v in data.items():
                v['dict_key'] = k     
            data = list(map(lambda x: x[1], data.items()))[args.node_id::args.node_num]      
            print(f'node id/node num: {args.node_id}/{args.node_num}, data length: {len(data)}')  

            new_data = []
            for i in tqdm(range(len(data))): 
                for bbox_idx in range(len(data[i]["florence"]["subject2_result"][1:])):
                    # index = os.path.basename(data[i]['dict_key']) + "_bbox" + str(bbox_idx)
                    index = '_'.join([
                        data[i]['dict_key'].split('/')[-4],
                        data[i]['dict_key'].split('/')[-3],
                        data[i]['dict_key'].split('/')[-1]
                    ]) + f"_bbox{bbox_idx}"
                    if not os.path.exists(os.path.join(args.save_path, f"{index}_{rank}.png")):
                        new_data.append(data[i])
                        break
            print(f'before {len(data)}, after {len(new_data)}')
            data = new_data
            eval_dataloader, eval_dataloader_iter = FLUXPairedGeneration_eval_loader(data, args)
    elif args.multi_ip:
        if args.concept101:
            if args.split_to_per_rank:  
                import json
                with open('./eval/MIP-Adapter/all_evaluate_data_prompt.json', 'r') as f:
                    samples = json.load(f)     
                print(f'total org img nums: {len(samples)}')       
                samples = samples[rank::world_size]
            eval_dataloader = Concept101MultiIP_Dataloader(args, samples=samples)
            eval_dataloader_iter = iter(eval_dataloader)
        elif args.dreambench_multi:
            if args.split_to_per_rank:  
                import json
                with open('./eval/dreambooth_dataset/total_multiip_750.json', 'r') as f:
                    samples = json.load(f)     
                print(f'total org img nums: {len(samples)}')       
                samples = samples[rank::world_size]
                eval_dataloader = get_dreambench_multiip_dataloader(args, samples=samples)
            else:
                eval_dataloader = get_dreambench_multiip_dataloader(args)
            eval_dataloader_iter = iter(eval_dataloader)            
        else:
            eval_dataloader, eval_dataloader_iter = get_flux_paired_multiip_eval_loader(args)
    # # for debug
    # for btch_idx, data in enumerate(eval_dataloader):
    #     continue

    pipeline = UNOPipeline(
        args.model_type,
        args.device,
        args.offload,
        only_lora=args.only_lora, 
        lora_rank=args.lora_rank
    )
    # load open-source model
    pipeline.load_ckpt(args.dit_ckpt)
    if args.use_lora:
        print('load lora:', args.lora_local_path, args.lora_repo_id, args.lora_name)
        pipeline.set_lora(args.lora_local_path, args.lora_repo_id, args.lora_name, args.lora_weight)

    if os.path.exists(args.prompt):
        if args.prompt.endswith('.txt'):
            with open(args.prompt, 'r') as f:
                prompts = f.readlines()
                prompts = [p.strip() for p in prompts]
        elif args.prompt.endswith('.json'):
            import json
            with open(args.prompt, "rt") as f:
                data_dicts = json.load(f)
            prompts = [data_dict["text"] for data_dict in data_dicts]
    else:
        prompts = [args.prompt]

    # for idx, prompt in tqdm(enumerate(prompts)):
    start_time = time.time()
    idx = -1
    while True:
        idx += 1
        result_imgs = []
        batch = next(eval_dataloader_iter)
        print(f'{idx}/{len(eval_dataloader)}, {datetime.timedelta(seconds = (time.time() - start_time))}')    
        if world_size > 1:
            print('num_images_per_prompt is invalid, each rank process one seed')
            args.num_images_per_prompt = 1
        for j in range(args.num_images_per_prompt):
            result = pipeline(
                prompt=prompts[0], # fake prompt
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed if args.split_to_per_rank else args.seed + rank,
                true_gs=args.true_gs,
                neg_prompt=args.neg_prompt,
                timestep_to_start_cfg=args.timestep_to_start_cfg,
                image_prompt=image_prompt,
                batch=batch,
                single_ip=args.single_ip,
                multi_ip=args.multi_ip,
                pe=args.pe
            )
            os.makedirs(args.save_path, exist_ok=True)
            # ind = len(os.listdir(args.save_path))
            if args.num_images_per_prompt==1:
                if args.split_to_per_rank:
                    result.save(os.path.join(args.save_path, f"{batch['index'][0]}_0.png"))
                    save_path = os.path.join(args.save_path, f"{batch['index'][0]}_0.png")
                    print(f"save in {save_path}")   
                else:                 
                    result.save(os.path.join(args.save_path, f"{batch['index'][0]}_{rank}.png"))
                    save_path = os.path.join(args.save_path, f"{batch['index'][0]}_{rank}.png")
                    print(f"save in {save_path}")
            else:
                result.save(os.path.join(args.save_path, f"{batch['index'][0]}_{j}.png"))
                args.seed = args.seed + 1
                result_imgs.append(result)
                concat_imgs = horizontal_concat(result_imgs)
                concat_imgs.save(os.path.join(args.save_path, f"{batch['index'][0]}_concat.png"))

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
