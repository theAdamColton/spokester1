import gc
import functools

import torch
import matplotlib as mpl
from einops import rearrange, reduce, repeat

DEFAULT_DURATION_STEP_SIZE = 1.0


def clear_cuda_cache(func):
    """
    A decorator that performs garbage collection and clears the CUDA cache
    before and after the decorated function is executed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        result = func(*args, **kwargs)

        gc.collect()
        torch.cuda.empty_cache()

        return result

    return wrapper


def random_uniform(
    shape, a=0.0, b=1.0, device=torch.device("cpu"), dtype=torch.float32, torch_rng=None
):
    assert a < b
    u = torch.rand(shape, generator=torch_rng, device=device, dtype=dtype)
    u = u * (b - a) + a
    return u


def unsqueeze_leading(x, y):
    while x.ndim < y.ndim:
        x = x.unsqueeze(0)
    return x


def unsqueeze_trailing(y, x):
    while y.ndim < x.ndim:
        y = y.unsqueeze(-1)
    return y


def compute_smooth_rank(x, eps=1e-5):
    """
    x: Batch of representations, shape: (B Z)

    This is a metric studied in the 2023 paper:
    RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank

    Higher smooth rank
    """
    x = x.float()

    s = torch.linalg.svdvals(x)
    s_norm = s.norm(1)
    p = s / s_norm
    log_p = torch.log(p + eps)
    entropy = torch.exp(-(p * log_p).sum())
    return entropy


def hsl2rgb(hsl: torch.Tensor) -> torch.Tensor:
    # hsl: Channels-last image
    hsl_h, hsl_s, hsl_l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    _c = (-torch.abs(hsl_l * 2.0 - 1.0) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsl_l - _c / 2.0
    idx = (hsl_h * 6.0).type(torch.uint8)
    idx = idx % 6
    idx = repeat(idx, "... -> ... three", three=3)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.stack([_c, _x, _o], dim=-1)[idx == 0]
    rgb[idx == 1] = torch.stack([_x, _c, _o], dim=-1)[idx == 1]
    rgb[idx == 2] = torch.stack([_o, _c, _x], dim=-1)[idx == 2]
    rgb[idx == 3] = torch.stack([_o, _x, _c], dim=-1)[idx == 3]
    rgb[idx == 4] = torch.stack([_x, _o, _c], dim=-1)[idx == 4]
    rgb[idx == 5] = torch.stack([_c, _o, _x], dim=-1)[idx == 5]
    rgb += _m.unsqueeze(-1)
    return rgb


def convert_monochrome_to_hot_colormap(x: torch.Tensor):
    """
    x: (...,)
    returns uint8 tensor of shape
        (...,3)
    """
    hot_cmap = mpl.colormaps.get_cmap("hot")
    colors = hot_cmap(torch.arange(256))
    colors = torch.from_numpy(colors)
    colors = colors[:, :3]
    colors = colors.to(x.device)
    colors = colors.clamp(0, 1).mul(255).to(torch.uint8)
    if x.dtype.is_floating_point:
        raise ValueError(x.dtype)
    x = colors[x.long()]
    return x


def minmaxscale(x, eps=1e-6):
    x_min = reduce(x, "b ... d -> b d", "min")
    x_max = reduce(x, "b ... d -> b d", "max")

    for _ in range(x.ndim - 2):
        x_min = x_min.unsqueeze(1)
        x_max = x_max.unsqueeze(1)

    x_centered = x - x_min
    x_range = x_max - x_min
    x_range.clip_(eps)
    x = x_centered / x_range
    x = x.clip(0, 1)
    return x


def features_to_rgb(x, use_hsl: bool = True):
    og_shape = x.shape
    x = rearrange(x, "b ... d -> b (...) d")

    with torch.autocast(x.device.type, enabled=False):
        # Compute batched PCA lowrank
        # ... d -> ... 3
        x, *_ = torch.pca_lowrank(x.float(), 3, niter=20)

    x = minmaxscale(x)

    if use_hsl:
        # Treat top 3 PCA components as Lightness, Saturation, and Hue
        x = x.flip(-1)
        x = hsl2rgb(x)

    x = x.mul(255).round().to(torch.uint8)

    x = x.reshape(*og_shape[:-1], 3)

    return x
