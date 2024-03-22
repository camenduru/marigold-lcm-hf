import functools
import os
import shutil
import zipfile
from io import BytesIO

import gradio as gr
import imageio as imageio
import numpy as np
import torch as torch
from PIL import Image
from diffusers import UNet2DConditionModel, LCMScheduler
from gradio_imageslider import ImageSlider
from huggingface_hub import login
from tqdm import tqdm

from extrude import extrude_depth_3d
from marigold_depth_estimation_lcm import MarigoldDepthConsistencyPipeline

default_seed = 2024

default_image_denoise_steps = 4
default_image_ensemble_size = 1
default_image_processing_res = 768
default_image_reproducuble = True

default_video_depth_latent_init_strength = 0.1
default_video_denoise_steps = 1
default_video_ensemble_size = 1
default_video_processing_res = 768
default_video_out_fps = 15
default_video_out_max_frames = 100

default_bas_plane_near = 0.0
default_bas_plane_far = 1.0
default_bas_embossing = 20
default_bas_denoise_steps = 4
default_bas_ensemble_size = 1
default_bas_processing_res = 768
default_bas_size_longest_px = 512
default_bas_size_longest_cm = 10
default_bas_filter_size = 3
default_bas_frame_thickness = 5
default_bas_frame_near = 1
default_bas_frame_far = 1


def process_image(
    pipe,
    path_input,
    denoise_steps=default_image_denoise_steps,
    ensemble_size=default_image_ensemble_size,
    processing_res=default_image_processing_res,
    reproducible=default_image_reproducuble,
):
    input_image = Image.open(path_input)

    pipe_out = pipe(
        input_image,
        denoising_steps=denoise_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        batch_size=1 if processing_res == 0 else 0,
        seed=default_seed if reproducible else None,
        show_progress_bar=False,
    )

    depth_pred = pipe_out.depth_np
    depth_colored = pipe_out.depth_colored
    depth_16bit = (depth_pred * 65535.0).astype(np.uint16)

    path_output_dir = os.path.splitext(path_input)[0] + "_output"
    os.makedirs(path_output_dir, exist_ok=True)

    name_base = os.path.splitext(os.path.basename(path_input))[0]
    path_out_fp32 = os.path.join(path_output_dir, f"{name_base}_depth_fp32.npy")
    path_out_16bit = os.path.join(path_output_dir, f"{name_base}_depth_16bit.png")
    path_out_vis = os.path.join(path_output_dir, f"{name_base}_depth_colored.png")

    np.save(path_out_fp32, depth_pred)
    Image.fromarray(depth_16bit).save(path_out_16bit, mode="I;16")
    depth_colored.save(path_out_vis)

    return (
        [path_out_16bit, path_out_vis],
        [path_out_16bit, path_out_fp32, path_out_vis],
    )


def process_video(
    pipe,
    path_input,
    depth_latent_init_strength=default_video_depth_latent_init_strength,
    denoise_steps=default_video_denoise_steps,
    ensemble_size=default_video_ensemble_size,
    processing_res=default_video_processing_res,
    out_fps=default_video_out_fps,
    out_max_frames=default_video_out_max_frames,
    progress=gr.Progress(),
):
    path_output_dir = os.path.splitext(path_input)[0] + "_output"
    os.makedirs(path_output_dir, exist_ok=True)

    name_base = os.path.splitext(os.path.basename(path_input))[0]
    path_out_vis = os.path.join(path_output_dir, f"{name_base}_depth_colored.mp4")
    path_out_16bit = os.path.join(path_output_dir, f"{name_base}_depth_16bit.zip")

    reader = imageio.get_reader(path_input)

    meta_data = reader.get_meta_data()
    fps = meta_data["fps"]
    size = meta_data["size"]
    duration_sec = meta_data["duration"]

    if fps <= out_fps:
        frame_interval, out_fps = 1, fps
    else:
        frame_interval = round(fps / out_fps)
        out_fps = fps / frame_interval

    out_duration_sec = out_max_frames / out_fps
    if duration_sec > out_duration_sec:
        gr.Warning(
            f"Only the first ~{int(out_duration_sec)} seconds will be processed; "
            f"use alternative setups for full processing"
        )

    writer = imageio.get_writer(path_out_vis, fps=out_fps)
    zipf = zipfile.ZipFile(path_out_16bit, "w", zipfile.ZIP_DEFLATED)
    prev_depth_latent = None

    pbar = tqdm(desc="Processing Video", total=out_max_frames)

    out_frame_id = 0
    for frame_id, frame in enumerate(reader):
        if not (frame_id % frame_interval == 0):
            continue
        out_frame_id += 1
        pbar.update(1)
        if out_frame_id > out_max_frames:
            break

        frame_pil = Image.fromarray(frame)

        pipe_out = pipe(
            frame_pil,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=False,
            batch_size=0,
            depth_latent_init=prev_depth_latent,
            depth_latent_init_strength=depth_latent_init_strength,
            seed=default_seed,
            show_progress_bar=False,
        )

        prev_depth_latent = pipe_out.depth_latent

        processed_frame = pipe_out.depth_colored
        processed_frame = imageio.core.util.Array(np.array(processed_frame))
        writer.append_data(processed_frame)

        processed_frame = (65535 * np.clip(pipe_out.depth_np, 0.0, 1.0)).astype(
            np.uint16
        )
        processed_frame = Image.fromarray(processed_frame, mode="I;16")

        archive_path = os.path.join(
            f"{name_base}_depth_16bit", f"{out_frame_id:05d}.png"
        )
        img_byte_arr = BytesIO()
        processed_frame.save(img_byte_arr, format="png")
        img_byte_arr.seek(0)
        zipf.writestr(archive_path, img_byte_arr.read())

    reader.close()
    writer.close()
    zipf.close()

    return (
        path_out_vis,
        [path_out_vis, path_out_16bit],
    )


def process_bas(
    pipe,
    path_input,
    plane_near=default_bas_plane_near,
    plane_far=default_bas_plane_far,
    embossing=default_bas_embossing,
    denoise_steps=default_bas_denoise_steps,
    ensemble_size=default_bas_ensemble_size,
    processing_res=default_bas_processing_res,
    size_longest_px=default_bas_size_longest_px,
    size_longest_cm=default_bas_size_longest_cm,
    filter_size=default_bas_filter_size,
    frame_thickness=default_bas_frame_thickness,
    frame_near=default_bas_frame_near,
    frame_far=default_bas_frame_far,
):
    if plane_near >= plane_far:
        raise gr.Error("NEAR plane must have a value smaller than the FAR plane")

    path_output_dir = os.path.splitext(path_input)[0] + "_output"
    os.makedirs(path_output_dir, exist_ok=True)

    name_base, name_ext = os.path.splitext(os.path.basename(path_input))

    input_image = Image.open(path_input)

    pipe_out = pipe(
        input_image,
        denoising_steps=denoise_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        seed=default_seed,
        show_progress_bar=False,
    )

    depth_pred = pipe_out.depth_np * 65535

    def _process_3d(
        size_longest_px,
        filter_size,
        vertex_colors,
        scene_lights,
        output_model_scale=None,
        prepare_for_3d_printing=False,
    ):
        image_rgb_w, image_rgb_h = input_image.width, input_image.height
        image_rgb_d = max(image_rgb_w, image_rgb_h)
        image_new_w = size_longest_px * image_rgb_w // image_rgb_d
        image_new_h = size_longest_px * image_rgb_h // image_rgb_d

        image_rgb_new = os.path.join(
            path_output_dir, f"{name_base}_rgb_{size_longest_px}{name_ext}"
        )
        image_depth_new = os.path.join(
            path_output_dir, f"{name_base}_depth_{size_longest_px}.png"
        )
        input_image.resize((image_new_w, image_new_h), Image.LANCZOS).save(
            image_rgb_new
        )
        Image.fromarray(depth_pred).convert(mode="F").resize(
            (image_new_w, image_new_h), Image.BILINEAR
        ).convert("I").save(image_depth_new)

        path_glb, path_stl = extrude_depth_3d(
            image_rgb_new,
            image_depth_new,
            output_model_scale=size_longest_cm * 10
            if output_model_scale is None
            else output_model_scale,
            filter_size=filter_size,
            coef_near=plane_near,
            coef_far=plane_far,
            emboss=embossing / 100,
            f_thic=frame_thickness / 100,
            f_near=frame_near / 100,
            f_back=frame_far / 100,
            vertex_colors=vertex_colors,
            scene_lights=scene_lights,
            prepare_for_3d_printing=prepare_for_3d_printing,
        )

        return path_glb, path_stl

    path_viewer_glb, _ = _process_3d(
        256, filter_size, vertex_colors=False, scene_lights=True, output_model_scale=1
    )
    path_files_glb, path_files_stl = _process_3d(
        size_longest_px, filter_size, vertex_colors=True, scene_lights=False, prepare_for_3d_printing=True
    )

    return path_viewer_glb, [path_files_glb, path_files_stl]


def run_demo_server(pipe):
    process_pipe_image = functools.partial(process_image, pipe)
    process_pipe_video = functools.partial(process_video, pipe)
    process_pipe_bas = functools.partial(process_bas, pipe)
    os.environ["GRADIO_ALLOW_FLAGGING"] = "never"

    gradio_theme = gr.themes.Default()
    # gradio_theme.set(
    #     section_header_text_size="20px",
    #     section_header_text_weight="bold",
    # )

    with gr.Blocks(
        theme=gradio_theme,
        title="Marigold-LCM Depth Estimation",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        gr.Markdown(
            """
            <h1 align="center">Marigold-LCM Depth Estimation</h1>
            <p align="center">
            <a title="Website" href="https://marigoldmonodepth.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
            </a>
            <a title="arXiv" href="https://arxiv.org/abs/2312.02145" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
            </a>
            <a title="Github" href="https://github.com/prs-eth/marigold" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/prs-eth/marigold?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
            </a>
            <a title="Social" href="https://twitter.com/antonobukhov1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
            </a>
            </p>
            <p align="justify">
                Marigold-LCM is the fast version of Marigold, the state-of-the-art depth estimator for images in the wild.
                It combines the power of the original Marigold 10-step estimator and the Latent Consistency Models, delivering high-quality results in as little as <b>one step</b>.
                We provide three functions in this demo: Image, Video, and Bas-relief 3D processing — <b>see the tabs below</b>. 
                Upload your content into the <b>left</b> side, or click any of the <b>examples</b> below.
                Wait a second (for images and 3D) or a minute (for videos), and interact with the result in the <b>right</b> side.
                To avoid queuing, fork the demo into your profile.
            </p>
        """
        )

        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Input Image",
                            type="filepath",
                        )
                        with gr.Row():
                            image_submit_btn = gr.Button(
                                value="Compute Depth", variant="primary"
                            )
                            image_reset_btn = gr.Button(value="Reset")
                        with gr.Accordion("Advanced options", open=False):
                            image_denoise_steps = gr.Slider(
                                label="Number of denoising steps",
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=default_image_denoise_steps,
                            )
                            image_ensemble_size = gr.Slider(
                                label="Ensemble size",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=default_image_ensemble_size,
                            )
                            image_processing_res = gr.Radio(
                                [
                                    ("Native", 0),
                                    ("Recommended", 768),
                                ],
                                label="Processing resolution",
                                value=default_image_processing_res,
                            )
                    with gr.Column():
                        image_output_slider = ImageSlider(
                            label="Predicted depth (red-near, blue-far)",
                            type="filepath",
                            show_download_button=True,
                            show_share_button=True,
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )
                        image_output_files = gr.Files(
                            label="Depth outputs",
                            elem_id="download",
                            interactive=False,
                        )
                gr.Examples(
                    fn=process_pipe_image,
                    examples=[
                        os.path.join("files", "image", name)
                        for name in [
                            "arc.jpeg",
                            "berries.jpeg",
                            "butterfly.jpeg",
                            "cat.jpg",
                            "concert.jpeg",
                            "dog.jpeg",
                            "doughnuts.jpeg",
                            "einstein.jpg",
                            "food.jpeg",
                            "glasses.jpeg",
                            "house.jpg",
                            "lake.jpeg",
                            "marigold.jpeg",
                            "portrait_1.jpeg",
                            "portrait_2.jpeg",
                            "pumpkins.jpg",
                            "puzzle.jpeg",
                            "road.jpg",
                            "scientists.jpg",
                            "surfboards.jpeg",
                            "surfer.jpeg",
                            "swings.jpg",
                            "switzerland.jpeg",
                            "teamwork.jpeg",
                            "wave.jpeg",
                        ]
                    ],
                    inputs=[image_input],
                    outputs=[image_output_slider, image_output_files],
                    cache_examples=True,
                )

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Input Video",
                            sources=["upload"],
                        )
                        with gr.Row():
                            video_submit_btn = gr.Button(
                                value="Compute Depth", variant="primary"
                            )
                            video_reset_btn = gr.Button(value="Reset")
                    with gr.Column():
                        video_output_video = gr.Video(
                            label="Output video depth (red-near, blue-far)",
                            interactive=False,
                        )
                        video_output_files = gr.Files(
                            label="Depth outputs",
                            elem_id="download",
                            interactive=False,
                        )
                gr.Examples(
                    fn=process_pipe_video,
                    examples=[
                        os.path.join("files", "video", name)
                        for name in [
                            "cab.mp4",
                            "elephant.mp4",
                            "obama.mp4",
                        ]
                    ],
                    inputs=[video_input],
                    outputs=[video_output_video, video_output_files],
                    cache_examples=True,
                )

            with gr.Tab("Bas-relief (3D)"):
                gr.Markdown(
                    """
                    <p align="justify">
                        This part of the demo uses Marigold-LCM to create a bas-relief model. 
                        The models are watertight, with correct normals, and exported in the STL format, which makes them <b>3D-printable</b>.
                        Start by uploading the image and click "Create" with the default parameters. 
                        To improve the result, click "Clear", adjust the geometry sliders below, and click "Create" again.
                    </p>
                    """,
                )
                with gr.Row():
                    with gr.Column():
                        bas_input = gr.Image(
                            label="Input Image",
                            type="filepath",
                        )
                        with gr.Row():
                            bas_submit_btn = gr.Button(value="Create 3D", variant="primary")
                            bas_clear_btn = gr.Button(value="Clear")
                            bas_reset_btn = gr.Button(value="Reset")
                        with gr.Accordion("3D printing demo: Main options", open=True):
                            bas_plane_near = gr.Slider(
                                label="Relative position of the near plane (between 0 and 1)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.001,
                                value=default_bas_plane_near,
                            )
                            bas_plane_far = gr.Slider(
                                label="Relative position of the far plane (between near and 1)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.001,
                                value=default_bas_plane_far,
                            )
                            bas_embossing = gr.Slider(
                                label="Embossing level",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=default_bas_embossing,
                            )
                        with gr.Accordion("3D printing demo: Advanced options", open=False):
                            bas_denoise_steps = gr.Slider(
                                label="Number of denoising steps",
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=default_bas_denoise_steps,
                            )
                            bas_ensemble_size = gr.Slider(
                                label="Ensemble size",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=default_bas_ensemble_size,
                            )
                            bas_processing_res = gr.Radio(
                                [
                                    ("Native", 0),
                                    ("Recommended", 768),
                                ],
                                label="Processing resolution",
                                value=default_bas_processing_res,
                            )
                            bas_size_longest_px = gr.Slider(
                                label="Size (px) of the longest side",
                                minimum=256,
                                maximum=1024,
                                step=256,
                                value=default_bas_size_longest_px,
                            )
                            bas_size_longest_cm = gr.Slider(
                                label="Size (cm) of the longest side",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=default_bas_size_longest_cm,
                            )
                            bas_filter_size = gr.Slider(
                                label="Size (px) of the smoothing filter",
                                minimum=1,
                                maximum=5,
                                step=2,
                                value=default_bas_filter_size,
                            )
                            bas_frame_thickness = gr.Slider(
                                label="Frame thickness",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=default_bas_frame_thickness,
                            )
                            bas_frame_near = gr.Slider(
                                label="Frame's near plane offset",
                                minimum=-100,
                                maximum=100,
                                step=1,
                                value=default_bas_frame_near,
                            )
                            bas_frame_far = gr.Slider(
                                label="Frame's far plane offset",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=default_bas_frame_far,
                            )
                    with gr.Column():
                        bas_output_viewer = gr.Model3D(
                            camera_position=(75.0, 90.0, 1.25),
                            elem_classes="viewport",
                            label="3D preview (low-res, relief highlight)",
                            interactive=False,
                        )
                        bas_output_files = gr.Files(
                            label="3D model outputs (high-res)",
                            elem_id="download",
                            interactive=False,
                        )
                gr.Examples(
                    fn=process_pipe_bas,
                    examples=[
                        [
                            "files/basrelief/coin.jpg",  # input
                            0.0,  # plane_near
                            0.66,  # plane_far
                            15,  # embossing
                            4,  # denoise_steps
                            4,  # ensemble_size
                            768,  # processing_res
                            512,  # size_longest_px
                            10,  # size_longest_cm
                            3,  # filter_size
                            5,  # frame_thickness
                            0,  # frame_near
                            1,  # frame_far
                        ],
                        [
                            "files/basrelief/einstein.jpg",  # input
                            0.0,  # plane_near
                            0.5,  # plane_far
                            50,  # embossing
                            2,  # denoise_steps
                            1,  # ensemble_size
                            768,  # processing_res
                            512,  # size_longest_px
                            10,  # size_longest_cm
                            3,  # filter_size
                            5,  # frame_thickness
                            -15,  # frame_near
                            1,  # frame_far
                        ],
                        [
                            "files/basrelief/food.jpeg",  # input
                            0.0,  # plane_near
                            1.0,  # plane_far
                            20,  # embossing
                            2,  # denoise_steps
                            4,  # ensemble_size
                            768,  # processing_res
                            512,  # size_longest_px
                            10,  # size_longest_cm
                            3,  # filter_size
                            5,  # frame_thickness
                            -5,  # frame_near
                            1,  # frame_far
                        ],
                    ],
                    inputs=[
                        bas_input,
                        bas_plane_near,
                        bas_plane_far,
                        bas_embossing,
                        bas_denoise_steps,
                        bas_ensemble_size,
                        bas_processing_res,
                        bas_size_longest_px,
                        bas_size_longest_cm,
                        bas_filter_size,
                        bas_frame_thickness,
                        bas_frame_near,
                        bas_frame_far,
                    ],
                    outputs=[bas_output_viewer, bas_output_files],
                    cache_examples=True,
                )

        image_submit_btn.click(
            fn=process_pipe_image,
            inputs=[
                image_input,
                image_denoise_steps,
                image_ensemble_size,
                image_processing_res,
            ],
            outputs=[image_output_slider, image_output_files],
            concurrency_limit=1,
        )

        image_reset_btn.click(
            fn=lambda: (
                None,
                None,
                None,
                default_image_ensemble_size,
                default_image_denoise_steps,
                default_image_processing_res,
            ),
            inputs=[],
            outputs=[
                image_input,
                image_output_slider,
                image_output_files,
                image_ensemble_size,
                image_denoise_steps,
                image_processing_res,
            ],
            concurrency_limit=1,
        )

        video_submit_btn.click(
            fn=process_pipe_video,
            inputs=[video_input],
            outputs=[video_output_video, video_output_files],
            concurrency_limit=1,
        )

        video_reset_btn.click(
            fn=lambda: (None, None, None),
            inputs=[],
            outputs=[video_input, video_output_video, video_output_files],
            concurrency_limit=1,
        )

        def wrapper_process_pipe_bas(*args, **kwargs):
            out = list(process_pipe_bas(*args, **kwargs))
            out = [gr.Button(interactive=False), gr.Image(interactive=False)] + out
            return out

        bas_submit_btn.click(
            fn=wrapper_process_pipe_bas,
            inputs=[
                bas_input,
                bas_plane_near,
                bas_plane_far,
                bas_embossing,
                bas_denoise_steps,
                bas_ensemble_size,
                bas_processing_res,
                bas_size_longest_px,
                bas_size_longest_cm,
                bas_filter_size,
                bas_frame_thickness,
                bas_frame_near,
                bas_frame_far,
            ],
            outputs=[bas_submit_btn, bas_input, bas_output_viewer, bas_output_files],
            concurrency_limit=1,
        )

        bas_clear_btn.click(
            fn=lambda: (gr.Button(interactive=True), None, None),
            inputs=[],
            outputs=[
                bas_submit_btn,
                bas_output_viewer,
                bas_output_files,
            ],
            concurrency_limit=1,
        )

        bas_reset_btn.click(
            fn=lambda: (
                gr.Button(interactive=True),
                None,
                None,
                None,
                default_bas_plane_near,
                default_bas_plane_far,
                default_bas_embossing,
                default_bas_denoise_steps,
                default_bas_ensemble_size,
                default_bas_processing_res,
                default_bas_size_longest_px,
                default_bas_size_longest_cm,
                default_bas_filter_size,
                default_bas_frame_thickness,
                default_bas_frame_near,
                default_bas_frame_far,
            ),
            inputs=[],
            outputs=[
                bas_submit_btn,
                bas_input,
                bas_output_viewer,
                bas_output_files,
                bas_plane_near,
                bas_plane_far,
                bas_embossing,
                bas_denoise_steps,
                bas_ensemble_size,
                bas_processing_res,
                bas_size_longest_px,
                bas_size_longest_cm,
                bas_filter_size,
                bas_frame_thickness,
                bas_frame_near,
                bas_frame_far,
            ],
            concurrency_limit=1,
        )

        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )


def prefetch_hf_cache(pipe):
    process_image(pipe, "files/image/bee.jpg", 1, 1, 64)
    shutil.rmtree("files/image/bee_output")


def main():
    CHECKPOINT = "prs-eth/marigold-v1-0"
    CHECKPOINT_UNET_LCM = "prs-eth/marigold-lcm-v1-0"

    login(token=os.environ["HF_TOKEN_COLAB_RO"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = MarigoldDepthConsistencyPipeline.from_pretrained(
        CHECKPOINT,
        unet=UNet2DConditionModel.from_pretrained(
            CHECKPOINT_UNET_LCM, subfolder="unet", use_auth_token=True
        ),
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    try:
        import xformers

        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)
    prefetch_hf_cache(pipe)
    run_demo_server(pipe)


if __name__ == "__main__":
    main()
