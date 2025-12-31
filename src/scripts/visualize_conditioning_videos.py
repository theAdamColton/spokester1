from pathlib import Path
import time

from einops import repeat
import torch
import torchcodec
import torchvision
import jsonargparse

from src.net.supplemental_net import OpticalFlowExtractor, DepthExtractor
from src.net.helpers import _augment_condition
from src.game.auto_play import auto_playthrough


@torch.inference_mode()
def main(
    path: Path = Path("data") / "real_videos" / "hotline_toni_2k_24fps.mp4",
    output_path: Path = Path("runs") / "conditionings",
    height: int = 256,
    width: int = 256,
    num_frames: int = 32,
    device_str: str = "cuda",
    dtype_str: str = "bfloat16",
    depth_vis_shift: float = 0.6,
    depth_vis_scale: float = 0.4,
    flow_vis_shift: float = 0.5,
    flow_vis_scale: float = 1.0,
    should_augment: bool = True,
):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)

    output_path.mkdir(exist_ok=True, parents=True)

    flow_extractor = OpticalFlowExtractor().to(device, dtype)

    decoder = torchcodec.decoders.VideoDecoder(path, dimension_order="NHWC")
    pixel_values = decoder.get_frames_in_range(0, num_frames).data
    fps = decoder.metadata.average_fps

    _, og_h, og_w, _ = pixel_values.shape

    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(min(og_h, og_w)),
            torchvision.transforms.Resize((height, width)),
        ]
    )

    pixel_values = transform_fn(pixel_values.movedim(-1, 1)).movedim(1, -1)

    vis_path = output_path / "original.mp4"
    torchvision.io.write_video(
        str(vis_path), pixel_values, fps=fps, options={"crf": "18"}
    )
    print("saved", vis_path)

    n, h, w, c = pixel_values.shape

    pixel_values = pixel_values.to(device, dtype).div(255).mul(2).sub(1)

    pixel_values = pixel_values.unsqueeze(0)

    # Extract real flow
    st = time.time()
    flow = flow_extractor(pixel_values)[0].cpu()
    elapsed = time.time() - st
    print(
        "elapsed flow",
        elapsed,
        "seconds",
        " time per frame",
        1000 * elapsed / (n - 2),
        "ms",
    )

    # Extract real depth
    depth_extractor = DepthExtractor().to(device, dtype)
    st = time.time()
    depth = depth_extractor(pixel_values)[0].cpu()
    elapsed = time.time() - st
    print(
        "elapsed flow",
        elapsed,
        "seconds",
        " time per frame",
        1000 * elapsed / (n - 2),
        "ms",
    )

    # Print stats
    print("real flow mean", flow.mean((0, 1, 2)))
    print("real flow std", flow.std((0, 1, 2)))

    print("real depth mean", depth.mean())
    print("real depth median", depth.float().quantile(0.5))
    print("real depth std", depth.std())

    # Normalize
    flow = flow / flow.std()
    depth = (depth - depth.mean()) / depth.std()

    # Augment real
    depth = depth.unsqueeze(0)
    flow = flow.unsqueeze(0)
    torch_rng = torch.Generator().manual_seed(42)
    if should_augment:
        depth, flow = _augment_condition(depth, flow, torch_rng=torch_rng)
    depth = depth.squeeze(0)
    flow = flow.squeeze(0)

    # Prepare for visualization
    flow = flow.mul(flow_vis_scale).add(flow_vis_shift)
    flow = flow.clip(0, 1).mul(255).round().to(torch.uint8)
    flow = torch.cat((flow, torch.zeros_like(flow[..., -1:])), -1)
    depth = depth.mul(depth_vis_scale).add(depth_vis_shift)
    depth = depth.clip(0, 1).mul(255).round().to(torch.uint8)
    depth = repeat(depth, "n h w -> n h w c", c=3)

    # Save combined
    combined = torch.cat((depth, flow), -3)
    vis_path = output_path / "real.mp4"
    torchvision.io.write_video(str(vis_path), combined, fps=fps, options={"crf": "18"})
    print("saved", vis_path)

    # Obtain game observation
    depth, segs, flow = auto_playthrough(
        seed=42, height=height, width=width, max_length=num_frames, fps=fps
    )
    depth = torch.from_numpy(depth)
    segs = torch.from_numpy(segs)
    flow = torch.from_numpy(flow)

    # Save segmentation - traditionally rendered game
    vis_path = output_path / "segs.mp4"
    torchvision.io.write_video(str(vis_path), segs, fps=fps, options={"crf": "18"})
    print("saved", vis_path)

    # Print stats
    print("synth depth mean", depth.mean())
    print("synth depth std", depth.std())
    print("synth depth median", depth.quantile(0.5))
    print("synth flow mean", flow.mean((0, 1, 2))[:2])
    print("synth flow std", flow.std((0, 1, 2))[:2])

    # Normalize
    depth = (depth - depth.mean()) / depth.std()
    flow = flow[..., :2]
    flow = flow / flow.std()

    # Augment synth
    depth = depth.unsqueeze(0)
    flow = flow.unsqueeze(0)
    torch_rng = torch.Generator().manual_seed(42)
    if should_augment:
        depth, flow = _augment_condition(depth, flow, torch_rng=torch_rng)
    depth = depth.squeeze(0)
    flow = flow.squeeze(0)

    # Prepare for visualization
    depth = depth.mul(depth_vis_scale).add(depth_vis_shift)
    depth = depth.clip(0, 1).mul(255).round().to(torch.uint8)
    depth = repeat(depth, "... -> ... c", c=3)
    flow = flow.mul(flow_vis_scale).add(flow_vis_shift)
    flow = flow.clip(0, 1).mul(255).round().to(torch.uint8)
    flow = torch.cat((flow, torch.zeros_like(flow[..., -1:])), -1)

    vis = torch.cat((depth, flow), -3)
    vis_path = output_path / "synth.mp4"
    torchvision.io.write_video(str(vis_path), vis, fps=fps, options={"crf": "18"})
    print("saved", vis_path)


if __name__ == "__main__":
    jsonargparse.CLI(main)
