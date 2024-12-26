import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)



def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)

def preprocess_image(image: Image.Image) -> Image.Image:
    processed_image = pipeline.preprocess_image(image)
    return processed_image

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images

def generate_image_from_text(prompt: str) -> Image.Image:
    try:
        generated_image = text_to_image_pipe(prompt=prompt).images[0]
        return generated_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None
    
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    

def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh

def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def image_to_3d(
    image: Image.Image,
    prompt: str,
    is_prompt: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    req: gr.Request,
) -> Tuple[dict, str, Optional[Image.Image]]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    
    generated_image = None
    if is_prompt:
        image = generate_image_from_text(prompt)
        generated_image = image
        if image is None:
            return {}, "Error generating image from text.", None

    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )

    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)

    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()

    return state, video_path, generated_image


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.

    Returns:
        str: The path to the extracted Gaussian file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split an image into multiple views.
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]

def determine_state(text_prompt: str) -> bool:
    """
    Determine whether the text prompt is provided.
    Returns True if text prompt is non-empty, otherwise False.
    """
    return bool(text_prompt.strip()) if text_prompt else False

with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Text/Image to 3D Asset Generator
    * Enter a text prompt or upload an image to create a 3D asset. Click "Generate" to proceed.
    """)
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            with gr.Tabs() as input_tabs:
                # Tab for Text Input
                with gr.Tab(label="Text Input", id=0) as text_input_tab:
                    text_prompt = gr.Textbox(label="Text Prompt", placeholder="Enter a description")
                    generate_image_btn = gr.Button("Generate Image from Text", variant="secondary")
                
                # Tab for Image Input
                with gr.Tab(label="Image Input", id=1) as image_input_tab:
                    image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)

                # Tab for Multiple Images
                with gr.Tab(label="Multiple Images", id=2) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images. 
                        
                        *NOTE: this is an experimental algorithm without training a specialized model. It may not produce the best results for all images, especially those having different poses or inconsistent details.*
                    """)
            with gr.Row():
                generated_image_output = gr.Image(label="Generated Image", format="png", type="pil")

            with gr.Row():
                download_image = gr.DownloadButton(label="Download Generated Image", interactive=False, visible=False)

            # Generation Settings
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("**Stage 1: Sparse Structure Generation**")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("**Stage 2: Structured Latent Generation**")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic")
                
            # Extraction Settings Accordion
            

            # Main Control Buttons
            with gr.Row():
                
                generate_btn = gr.Button("Generate 3D Asset", variant="primary")
            gr.Markdown("Click 'Generate' to start the asset creation process.")
        
        # Right Column: Outputs
        with gr.Column():
            # Display Outputs
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)

            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)

            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
                
            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)
            

            
            # Download Buttons for GLB and Gaussian
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)
            
            # Extraction Buttons (Initially Disabled)
            

    # Hidden States for managing component visibility and logic
    is_multiimage = gr.State(False)
    output_buf = gr.State()

    with gr.Row(visible=False) as single_image_example:
        examples = gr.Examples(
            examples=[f'assets/example_image/{image}' for image in os.listdir("assets/example_image")],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=64,
        )
    with gr.Row(visible=False) as multiimage_example:
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            fn=split_image,
            outputs=[multiimage_prompt],
            run_on_click=True,
            examples_per_page=8,
        )

    # Initialize session (no need to pass req as input)
    demo.load(start_session)  # Gradio will handle passing `req` here
    demo.unload(end_session)

    # Input tab selection logic
    text_input_tab.select(
        lambda: tuple([False, gr.Row.update(visible=False), gr.Row.update(visible=False), gr.Row.update(visible=True)]),
        outputs=[is_multiimage, single_image_example, multiimage_example, generated_image_output]
    )

    generate_image_btn.click(
    generate_image_from_text,
    inputs=[text_prompt],  # Input: the text prompt
    outputs=[generated_image_output],  # Output: display the generated image
)


    image_input_tab.select(
        lambda: tuple([False, gr.Row.update(visible=True), gr.Row.update(visible=False), gr.Row.update(visible=False)]),
        outputs=[is_multiimage, single_image_example, multiimage_example, generated_image_output]
    )
    multiimage_input_tab.select(
        lambda: tuple([True, gr.Row.update(visible=False), gr.Row.update(visible=True),gr.Row.update(visible=False)]),
        outputs=[is_multiimage, single_image_example, multiimage_example, generated_image_output]
    )

    # Example image processing logic
    image_prompt.upload(preprocess_image, inputs=[image_prompt], outputs=[image_prompt])
    multiimage_prompt.upload(preprocess_images, inputs=[multiimage_prompt], outputs=[multiimage_prompt])

    
    is_prompt = gr.State(False)

    

    # Button handlers and logic for generation
    generate_btn.click(
    get_seed,
    inputs=[randomize_seed, seed],
    outputs=[seed],
).then(
    lambda text_prompt: determine_state(text_prompt),
    inputs=[text_prompt],
    outputs=[is_prompt],
).then(
    image_to_3d,
    inputs=[
        image_prompt,  # Use the generated image for 3D asset creation
        text_prompt,
        is_prompt,
        seed,
        ss_guidance_strength,
        ss_sampling_steps,
        slat_guidance_strength,
        slat_sampling_steps
    ],
    outputs=[output_buf, video_output, generated_image_output],  # Ensure the correct outputs are shown
).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # Extract GLB handler
    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )

    # Extract Gaussian handler
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    

    # Clear model output when GLB is downloaded
    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )

if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("D:\\Harish\\img3D-Trellis\\TRELLIS-image-large")
    pipeline.cuda()
    text_to_image_pipe = StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-1b", torch_dtype=torch.float16)
    text_to_image_pipe = text_to_image_pipe.to("cuda")
    demo.launch()