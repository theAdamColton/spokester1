from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import json
from typing import Literal

import wandb
from einops import repeat
import jsonargparse
from tqdm import tqdm
import torchvision
import torch
import torch.nn.functional as F

from src.net.configuring_net import ViTDenoiserConfig
from src.net.configuring_video_data import SimpleVideoDataConfig
from src.net.net import ViTDenoiser
from src.net.flow_matching import FlowMatchingHelper
from src.net.optim import init_optimizer_and_scaler
from src.net.supplemental_net import (
    DepthExtractor,
    DinoEncoder,
    REPAProjector,
    OpticalFlowExtractor,
)
from src.net.helpers import (
    compute_loss,
    extract_real_patch_condition,
    extract_synth_patch_condition,
    generate_autoregressive,
)
from src.net.video_data import get_simple_video_dataloader
from src.net.torch_utils import clear_cuda_cache
from src.game.auto_play import auto_playthrough


def _sanitize(x):
    if isinstance(x, (int, str, float)):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, list):
        return [_sanitize(y) for y in x]
    if isinstance(x, dict):
        assert all(isinstance(k, str) for k in x.keys())
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_sanitize(x) for x in x)
    if isinstance(x, Path):
        return str(x)
    if x is None:
        return x
    raise ValueError(f"{x} type({type(x)}) can't be sanitized")


def sanitize_dict(d: dict) -> dict:
    return {k: _sanitize(v) for k, v in d.items()}


def lerp(a, b, t=0.5):
    return a + t * (b - a)


@dataclass
class MainConfig:
    # Datasets
    train_video_real: SimpleVideoDataConfig = field(
        default_factory=lambda: SimpleVideoDataConfig()
    )
    val_video: SimpleVideoDataConfig = field(
        default_factory=lambda: SimpleVideoDataConfig()
    )

    # Models
    # Dinov3 model used for REPA alignment
    dino_path_or_url: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    pixel_denoiser: ViTDenoiserConfig = field(
        default_factory=lambda: ViTDenoiserConfig()
    )

    # Measured in patchlets
    pixel_denoiser_temporal_window_size: int = 8

    patch_duration: int = 2
    patch_side_length: int = 16

    # Flow matching
    # This value should be the number of floats in each frame
    # 256 * 256 * 2 * 3
    flow_matching_time_shift_dim_pixels: float = 16**2 * 16**2 * 2 * 3

    # HParams
    pixel_denoiser_repa_weight: float = 0.5
    pixel_denoiser_repa_depth: int = 6

    # TREAD token dropping for training
    pixel_denoiser_token_drop_rate: float = 0.0
    pixel_denoiser_drop_at_layer: int = 2
    pixel_denoiser_undrop_at_layer: int = -2

    # Optimization
    ema_beta: float = 0.9995
    adamw_lr: float = 7e-5
    muon_lr: float = 1e-3
    muon_momentum: float = 0.95
    optim_mode: Literal["adamw", "muon"] = "muon"
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_weight_decay: float = 0.0
    num_lr_warmup_steps: int = 1000
    num_lr_steady_steps: int = 100_000
    num_lr_cooldown_steps: int = 50_000
    grad_clip_value: float = 1.0

    # Resumption
    resume_checkpoint_path: Path | None = None
    force_resume_new_wandb_run: bool = True

    # Logging
    validate_every_num_steps: int = 100
    wandb_log_every_num_steps: int = 50
    save_checkpoint_every_num_steps: int = 500
    max_num_checkpoints: int = 2

    # Device
    device_str: str = "cpu"
    dtype_str: str = "float32"

    should_compile: bool = False

    @property
    def device(self):
        return torch.device(self.device_str)

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)


def main(conf: MainConfig = MainConfig()):
    device = torch.device(conf.device_str)
    dtype = getattr(torch, conf.dtype_str)

    pixel_flow_matching_helper = FlowMatchingHelper(
        conf.flow_matching_time_shift_dim_pixels
    )
    pixel_denoiser = ViTDenoiser(conf.pixel_denoiser).to(device)
    num_params = sum(
        p.nelement() for p in pixel_denoiser.parameters() if p.requires_grad
    )
    print("Num parameters", num_params)
    ema_pixel_denoiser = ViTDenoiser(conf.pixel_denoiser).to(device)
    ema_pixel_denoiser.load_state_dict(pixel_denoiser.state_dict())
    pixel_denoiser_projector = REPAProjector(
        conf.pixel_denoiser.transformer.hidden_size, 768
    ).to(device)
    dino_encoder = DinoEncoder(conf.dino_path_or_url).to(device, dtype)
    depth_extractor = DepthExtractor().to(device, dtype)
    flow_extractor = OpticalFlowExtractor().to(device, dtype)

    if conf.should_compile:
        # you may have to tweak this
        torch._dynamo.config.cache_size_limit = 96
        dino_encoder = torch.compile(dino_encoder, fullgraph=True, dynamic=False)
        pixel_denoiser.forward = torch.compile(
            pixel_denoiser.forward, fullgraph=True, dynamic=False
        )
        ema_pixel_denoiser.forward = torch.compile(
            ema_pixel_denoiser.forward, fullgraph=True, dynamic=False
        )
        pixel_denoiser_projector.forward = torch.compile(
            pixel_denoiser_projector.forward, fullgraph=True, dynamic=False
        )
        depth_extractor = torch.compile(depth_extractor, fullgraph=True, dynamic=False)
        flow_extractor.model = torch.compile(
            flow_extractor.model, fullgraph=True, dynamic=False
        )

    # Optimizer
    trainable_params, optimizers, scaler = init_optimizer_and_scaler(
        pixel_denoiser,
        pixel_denoiser_projector,
        device=device,
        dtype=dtype,
        optim_mode=conf.optim_mode,
    )

    global_step = 0

    # Potentially resume run
    previous_wandb_run_id = None
    if conf.resume_checkpoint_path is not None:
        # Load the checkpoint
        # and try to load the previous wandb run id

        def _load_checkpoint():
            d = torch.load(
                conf.resume_checkpoint_path,
                map_location=conf.device,
                weights_only=False,
            )
            pixel_denoiser.load_state_dict(d["pixel_denoiser"])
            ema_pixel_denoiser.load_state_dict(d["ema_pixel_denoiser"])

            pixel_denoiser_projector.load_state_dict(d["pixel_denoiser_projector"])
            for o, o_checkpoint in zip(optimizers, d["optimizers"]):
                o.load_state_dict(o_checkpoint)
            nonlocal global_step
            global_step = d["global_step"] + 1

        _load_checkpoint()

        if not conf.force_resume_new_wandb_run:
            prev_run_config_path = (
                conf.resume_checkpoint_path.parent / "run_config.json"
            )
            with open(prev_run_config_path, "r") as f:
                prev_run_dict = json.load(f)
            previous_wandb_run_id = prev_run_dict.get("wandb_run_id", None)
            if previous_wandb_run_id:
                print(f"Resuming wandb run {previous_wandb_run_id}")

        print(f"Resumed from checkpoint {conf.resume_checkpoint_path}")

    num_total_train_steps = (
        conf.num_lr_warmup_steps + conf.num_lr_steady_steps + conf.num_lr_cooldown_steps
    )

    real_dataloader = get_simple_video_dataloader(
        conf.train_video_real, device=device, dtype=dtype
    )

    def _get_val_batch():
        synth_depth, synth_seg, synth_flow = auto_playthrough(
            seed=42,
            height=conf.val_video.height,
            width=conf.val_video.width,
            max_length=conf.val_video.duration,
            fps=24.0,
        )
        synth_batch = (
            torch.from_numpy(synth_depth).unsqueeze(0),
            torch.from_numpy(synth_flow).unsqueeze(0),
        )

        val_real_dataloader = get_simple_video_dataloader(conf.val_video)
        real_batch = next(iter(val_real_dataloader))

        return synth_batch, real_batch

    val_synth_batch, val_real_batch = _get_val_batch()

    # Logging Setup
    wandb_run = wandb.init(
        project="spokester-net-2", config=asdict(conf), id=previous_wandb_run_id
    )

    # Write the global run config
    run_path = Path("runs")
    run_path.mkdir(parents=True, exist_ok=True)
    run_path = run_path / f"run-{len(list(run_path.iterdir())):06}"
    run_path.mkdir(parents=True, exist_ok=False)
    log_file_path = run_path / "logs.jsonl"
    with open(run_path / "run_config.json", "w") as f:
        run_dict = sanitize_dict(asdict(conf))
        run_dict["wandb_run_id"] = wandb_run.id
        f.write(json.dumps(run_dict))

    def append_to_logs(row):
        with open(log_file_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _save_checkpoint():
        checkpoint_path = run_path / f"{global_step:08}.pt"
        torch.save(
            {
                "pixel_denoiser": pixel_denoiser.state_dict(),
                "ema_pixel_denoiser": ema_pixel_denoiser.state_dict(),
                "pixel_denoiser_projector": pixel_denoiser_projector.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers],
                "scaler": scaler.state_dict() if scaler is not None else None,
                "global_step": global_step,
            },
            checkpoint_path,
        )

        existing_checkpoints = list(run_path.glob("*.pt"))
        existing_checkpoints.sort()
        for p in existing_checkpoints[: -conf.max_num_checkpoints]:
            p.unlink()

    @clear_cuda_cache
    @torch.inference_mode()
    def validate(
        *,
        batch_synth,
        batch_real: dict,
        save_path: Path,
        save_name_prefix: str,
    ):
        batch_synth = tuple(x.to(device, dtype) for x in batch_synth)
        batch_real = {
            k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v
            for k, v in batch_real.items()
        }

        real_pixel_values = batch_real["pixel_values"]

        b, n, h, w, c = real_pixel_values.shape
        npn = n // conf.patch_duration
        nph = h // conf.patch_side_length
        npw = w // conf.patch_side_length

        log_dict = {}
        wandb_extra_dict = {}

        real_pixel_values_vis = (
            real_pixel_values.add(1).div(2).mul(255).round().to(torch.uint8).cpu()
        )

        synth_depth, synth_flow = batch_synth

        synth_pixel_values_vis = (
            synth_depth.div(synth_depth.max())
            .clip(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        synth_pixel_values_vis = repeat(synth_pixel_values_vis, "... -> ... c", c=3)

        def _generate_autoregressive_and_save(
            name: str,
            bottom_vis_video: torch.Tensor | None,
            patch_condition,
            # generation kwargs
            num_denoising_steps=20,
            cache_at_step=None,
            temporal_window_size=conf.pixel_denoiser_temporal_window_size,
            **kwargs,
        ):
            gen_torch_rng = torch.Generator(device).manual_seed(42)
            generated_pixel_values, *_ = generate_autoregressive(
                sample_shape=(b, n, h, w, c),
                patch_condition=patch_condition,
                pixel_denoiser=ema_pixel_denoiser,
                pixel_flow_matching_helper=pixel_flow_matching_helper,
                patch_duration=conf.patch_duration,
                patch_side_length=conf.patch_side_length,
                torch_rng=gen_torch_rng,
                num_denoising_steps=num_denoising_steps,
                temporal_window_size=temporal_window_size,
                cache_at_step=cache_at_step,
                dtype=conf.dtype,
                **kwargs,
            )

            generated_pixel_values_vis = (
                generated_pixel_values.clip(-1, 1)
                .add(1)
                .div(2)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
            )

            if bottom_vis_video is not None:
                vis_video = torch.cat(
                    (generated_pixel_values_vis, bottom_vis_video), -3
                )

            vis_save_path = save_path / f"{save_name_prefix}_{name}.mp4"
            vis_save_path.parent.mkdir(exist_ok=True)
            torchvision.io.write_video(
                str(vis_save_path),
                vis_video[0],
                24.0,
                options={"crf": "0"},
            )
            wandb_extra_dict[name] = wandb.Video(vis_save_path, format="mp4")

            return generated_pixel_values

        num_denoising_steps = 20
        for aug_level in [False, True]:
            for cache_at_step in [int(num_denoising_steps * 0.5), None]:
                aug_torch_rng = torch.Generator(device).manual_seed(42)
                synth_patch_condition = extract_synth_patch_condition(
                    synth_depth,
                    synth_flow,
                    patch_duration=conf.patch_duration,
                    patch_side_length=conf.patch_side_length,
                    should_augment=aug_level,
                    torch_rng=aug_torch_rng,
                )
                _generate_autoregressive_and_save(
                    name=f"autoregressive_synth_noise{aug_level}_cache_at_step{cache_at_step}_num_steps{num_denoising_steps}",
                    bottom_vis_video=synth_pixel_values_vis,
                    patch_condition=synth_patch_condition,
                    cache_at_step=cache_at_step,
                    num_denoising_steps=num_denoising_steps,
                )

                aug_torch_rng = torch.Generator(device).manual_seed(42)
                real_patch_condition = extract_real_patch_condition(
                    real_pixel_values,
                    depth_extractor=depth_extractor,
                    flow_extractor=flow_extractor,
                    patch_duration=conf.patch_duration,
                    patch_side_length=conf.patch_side_length,
                    should_augment=aug_level,
                    torch_rng=aug_torch_rng,
                )
                generated_pixel_values = _generate_autoregressive_and_save(
                    name=f"autoregressive_real_noise{aug_level}_cache_at_step{cache_at_step}_num_steps{num_denoising_steps}",
                    bottom_vis_video=real_pixel_values_vis,
                    patch_condition=real_patch_condition,
                    cache_at_step=cache_at_step,
                    num_denoising_steps=num_denoising_steps,
                )

                log_dict[
                    f"rec_loss_ar_noise{aug_level}_cache_at_step{cache_at_step}_num_steps{num_denoising_steps}"
                ] = F.l1_loss(generated_pixel_values, real_pixel_values)

        return log_dict, wandb_extra_dict

    # Training Loop
    real_dataloader_iter = iter(real_dataloader)
    prog_bar = tqdm(initial=global_step, total=num_total_train_steps)
    for global_step in range(global_step, num_total_train_steps):

        def _accumulate_denoising_grads():
            real_pixel_values = next(real_dataloader_iter)["pixel_values"]
            log_dict, _ = compute_loss(
                pixel_flow_matching_helper=pixel_flow_matching_helper,
                pixel_denoiser=pixel_denoiser,
                pixel_denoiser_projector=pixel_denoiser_projector,
                dino_encoder=dino_encoder,
                depth_extractor=depth_extractor,
                flow_extractor=flow_extractor,
                pixel_values=real_pixel_values,
                patch_duration=conf.patch_duration,
                patch_side_length=conf.patch_side_length,
                repa_alignment_depth=conf.pixel_denoiser_repa_depth,
                repa_weight=conf.pixel_denoiser_repa_weight,
                pixel_denoiser_token_drop_rate=conf.pixel_denoiser_token_drop_rate,
                pixel_denoiser_drop_at_layer=conf.pixel_denoiser_drop_at_layer,
                pixel_denoiser_undrop_at_layer=conf.pixel_denoiser_undrop_at_layer,
            )

            total_loss = log_dict["total_loss"]
            total_loss.backward()

            return sanitize_dict(log_dict)

        def _train_step():
            log_dict = {}

            # Accumulate denoising grads

            log_dict.update(_accumulate_denoising_grads())

            # LR Scheduling
            lr_mult = lerp(0.05, 1.0, global_step / conf.num_lr_warmup_steps)
            if global_step > (conf.num_lr_warmup_steps + conf.num_lr_steady_steps):
                lr_mult = lerp(
                    1.0,
                    0.05,
                    (global_step - conf.num_lr_warmup_steps - conf.num_lr_steady_steps)
                    / conf.num_lr_cooldown_steps,
                )
            lr_mult = min(1.0, lr_mult)
            lr_mult = max(1e-3, lr_mult)
            step_adamw_lr = conf.adamw_lr * lr_mult
            step_muon_lr = conf.muon_lr * lr_mult

            log_dict["learning_rate"] = step_adamw_lr
            for o in optimizers:
                for g in o.param_groups:
                    g["lr"] = step_muon_lr if g["use_muon"] else step_adamw_lr
                    g["weight_decay"] = (
                        conf.adamw_weight_decay if g["use_weight_decay"] else 0.0
                    )
                    if g["use_muon"]:
                        g["momentum"] = conf.muon_momentum
                    g["betas"] = conf.adamw_betas

            # Optimization
            if scaler is not None:
                for o in optimizers:
                    scaler.unscale_(o)

            torch.nn.utils.clip_grad_norm_(trainable_params, conf.grad_clip_value)

            if scaler is not None:
                for o in optimizers:
                    scaler.step(o)
                scaler.update()
            else:
                for o in optimizers:
                    o.step()
            for o in optimizers:
                o.zero_grad(set_to_none=True)

            # EMA update
            # ema warmup
            ema_beta = lerp(
                0.1, conf.ema_beta, min(global_step / conf.num_lr_warmup_steps, 1.0)
            )
            log_dict["ema_beta"] = ema_beta
            with torch.no_grad():
                for p, ema_p in zip(
                    pixel_denoiser.parameters(),
                    ema_pixel_denoiser.parameters(),
                ):
                    if not p.is_floating_point():
                        continue
                    if p.requires_grad:
                        ema_p.lerp_(p, 1 - ema_beta)
                    else:
                        ema_p.copy_(p)

            return log_dict

        log_dict = _train_step()
        wandb_extra_dict = {}

        # Validation
        should_validate = global_step % conf.validate_every_num_steps == 0
        if should_validate:
            save_path = run_path / "artifacts"
            save_name_prefix = f"{global_step:06}"
            val_dict, val_wandb_extra_dict = validate(
                batch_synth=val_synth_batch,
                batch_real=val_real_batch,
                save_path=save_path,
                save_name_prefix=save_name_prefix,
            )
            val_dict = sanitize_dict(val_dict)
            log_dict.update(val_dict)
            wandb_extra_dict.update(val_wandb_extra_dict)

        log_dict["global_step"] = global_step
        append_to_logs(log_dict)

        should_log_to_wandb = should_validate or (
            global_step % conf.wandb_log_every_num_steps == 0
        )
        if should_log_to_wandb:
            wandb_log_dict = copy.deepcopy(log_dict)
            wandb_log_dict.update(wandb_extra_dict)
            wandb.log(wandb_log_dict, step=global_step)

        if global_step > 0 and global_step % conf.save_checkpoint_every_num_steps == 0:
            _save_checkpoint()

        global_step += 1
        prog_bar.update(1)

    prog_bar.close()

    _save_checkpoint()


if __name__ == "__main__":
    jsonargparse.CLI(main)
