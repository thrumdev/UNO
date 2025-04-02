import torch
import gradio as gr

from uno.flux.pipeline import UNOPipeline


def create_demo(
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False,
        ckpt_path: str = "",
    ):

    pipeline = UNOPipeline(model_type, device, offload, only_lora=True, lora_rank=512)
    pipeline.load_ckpt(None)

    with gr.Blocks() as demo:
        gr.Markdown(f"# UNO by UNO team")
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="handsome woman in the city")
                    with gr.Row():
                        image_prompt1 = gr.Image(label="ref img1", visible=True, interactive=True, type="pil")
                        image_prompt2 = gr.Image(label="ref img2", visible=True, interactive=True, type="pil")
                        image_prompt3 = gr.Image(label="ref img3", visible=True, interactive=True, type="pil")
                        image_prompt4 = gr.Image(label="ref img4", visible=True, interactive=True, type="pil")

                    with gr.Row():
                        with gr.Column():
                            ref_width = gr.Slider(128, 512, 512, step=16, label="Long side of Ref Images")
                        with gr.Column():
                            gr.Markdown(
                                    "📌 The recommended ref scale is related to the ref img number.\n"
                                )
                            gr.Markdown(
                                    "   1->512 / 2->320 / 3...n->256"
                                )

                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(512, 2048, 512, step=16, label="Gneration Width")
                            height = gr.Slider(512, 2048, 512, step=16, label="Gneration Height")
                        with gr.Column():
                            gr.Markdown(
                                    "📌 The model trained on 512 buckets.\n"
                                )
                            gr.Markdown(
                                    "The size closer to 512 is more stable, and the higher size gives a better visual effect but is less stable"
                                )
                            gr.Markdown(
                                    "You can use our recommended size 704"
                                )

                    with gr.Accordion("Generation Options", open=False):
                        with gr.Row():
                            num_steps = gr.Slider(1, 50, 25, step=1, label="Number of steps")
                            guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="Guidance", interactive=True)
                        seed = gr.Textbox(-1, label="Seed (-1 for random)")

                    generate_btn = gr.Button("Generate")

                with gr.Column():
                    output_image = gr.Image(label="Generated Image")
                    download_btn = gr.File(label="Download full-resolution")


            inputs = [
                prompt, width, height, guidance, num_steps,
                seed, ref_width, image_prompt1, image_prompt2, image_prompt3, image_prompt4
            ]
            generate_btn.click(
                fn=pipeline.gradio_generate,
                inputs=inputs,
                outputs=[output_image, download_btn],
            )


    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", help="Model name")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Folder with checkpoints in safetensors format")
    parser.add_argument("--port", type=int, default=7860, help="Port to use for the Gradio app")
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload, args.ckpt_dir)
    demo.launch(server_name="0.0.0.0", server_port=args.port)
