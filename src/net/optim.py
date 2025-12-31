import warnings

import torch
from torch import nn


def override_parameter_needs_weight_decay_(
    p: nn.Parameter | torch.Tensor, should_weight_decay: bool = True
):
    setattr(p, "_needs_weight_decay", should_weight_decay)
    return p


def init_optimizer_and_scaler(
    *models: nn.Module,
    device=torch.device("cpu"),
    dtype=torch.bfloat16,
    optim_mode: str = "muon",
):
    """
    requires that each model
    has their transformer blocks named 'blocks'
    """
    hidden_weights = []
    gains_biases = []
    nonhidden_weights = []

    def _categorize_params(model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_in_blocks = "blocks." in name
            is_multidim = param.ndim >= 2
            weight_decay_override = getattr(param, "_needs_weight_decay", None)
            is_weight_decay_overridden = weight_decay_override is not None

            # Specifically NO weight decay
            # No weight decay or Muon for these
            if is_weight_decay_overridden and not weight_decay_override:
                gains_biases.append(param)
                continue

            # Generic multidim block weights
            if is_in_blocks and is_multidim:
                hidden_weights.append(param)
                continue

            # Generic multidim non-block weights
            # (input projections, output projections, ...)
            if is_multidim:
                nonhidden_weights.append(param)
                continue

            # Specifically SHOULD weight decay
            if weight_decay_override:
                nonhidden_weights.append(param)
                continue

            # No weight decay or Muon for these
            gains_biases.append(param)

    use_muon = optim_mode == "muon"

    for model in models:
        parameter_names = [n for n, _ in model.named_parameters()]
        has_blocks = any("blocks." in n for n in parameter_names)
        if not has_blocks:
            warnings.warn(
                f"model {type(model)} doesn't have a modulelist named 'blocks' where the transformer blocks live!"
            )

        _categorize_params(model)

    param_groups = [
        {
            "params": hidden_weights,
            "lr": 0.0,
            "use_muon": use_muon,
            "use_weight_decay": True,
        },
        {
            "params": gains_biases,
            "lr": 0.0,
            "use_muon": False,
            "use_weight_decay": False,
        },
        {
            "params": nonhidden_weights,
            "lr": 0.0,
            "use_muon": False,
            "use_weight_decay": True,
        },
    ]

    trainable_parameters = hidden_weights + gains_biases + nonhidden_weights

    if optim_mode == "muon":
        optimizers = [
            torch.optim.Muon(
                (p for p in param_groups if p["use_muon"]),
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW((p for p in param_groups if not p["use_muon"])),
        ]
    elif optim_mode == "adamw":
        optimizers = [torch.optim.AdamW(param_groups)]
    elif optim_mode == "adafactor":
        optimizers = [torch.optim.Adafactor(param_groups)]
    else:
        raise ValueError(optim_mode)

    # Only use grad scaler for float16
    scaler = None
    if dtype == torch.float16:
        scaler = torch.GradScaler(device=device.type)

    return (
        trainable_parameters,
        optimizers,
        scaler,
    )
