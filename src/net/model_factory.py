from pathlib import Path
import json

import torch
from huggingface_hub import snapshot_download

from src.net.configuring_net import ViTDenoiserConfig
from src.net.net import ViTDenoiser


def make_model(
    model_path_or_url: str = "adams-story/spokester1-vit-base",
    device=torch.device("cpu"),
    dtype=torch.float32,
    use_ema: bool = True,
):
    model_path = Path(model_path_or_url)
    if not model_path.exists():
        model_path = snapshot_download(model_path_or_url, repo_type="model")
        model_path = Path(model_path)

    checkpoint_paths = list(model_path.glob("*.pt"))
    checkpoint_paths.sort()
    latest_checkpoint_path = checkpoint_paths[-1]

    run_config_path = model_path / "run_config.json"

    with open(run_config_path) as f:
        run_config_dict = json.load(f)

    model_conf = ViTDenoiserConfig.from_dict(run_config_dict["pixel_denoiser"])

    # Load in float32 and then cast type
    model = ViTDenoiser(model_conf).to(device)
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)

    model_name = "ema_pixel_denoiser" if use_ema else "pixel_denoiser"
    model.load_state_dict(checkpoint[model_name])
    model = model.to(device, dtype).eval().requires_grad_(False)

    return model, run_config_dict
