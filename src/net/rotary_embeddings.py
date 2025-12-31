import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, einsum, repeat

from src.net.torch_utils import unsqueeze_leading


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    x: multihead features, shape: b h ... d

    freqs_cos,freqs_sin
        frequencies obtained from position coordinates
        shape: (b) (h) ... d//2
        where the batch and head dimensions are optional
    """

    og_dtype = x.dtype
    x = x.float()

    x1, x2 = x.chunk(2, dim=-1)

    # unsqueeze batch dim
    freqs_cos = unsqueeze_leading(freqs_cos, x)
    freqs_sin = unsqueeze_leading(freqs_sin, x)

    x = torch.cat(
        (x1 * freqs_cos - x2 * freqs_sin, x2 * freqs_cos + x1 * freqs_sin), -1
    )

    x = x.to(og_dtype)

    return x


def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x


def make_directions(n: int, d: int) -> torch.Tensor:
    g = _phi(d)
    alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return directions.float()


class Rope2DPositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, rope_theta: float = 100.0, head_dim: int = 64):
        super().__init__()

        inv_freq = 1 / rope_theta ** torch.arange(
            0, 1, 4 / head_dim, dtype=torch.float32
        )  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        patch_coords: torch.Tensor,
        dtype=torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.inv_freq.device
        with torch.autocast(device_type=device.type, enabled=False):  # Force float32
            # (b, height * width, 2, head_dim / 4) -> (b, height * width, head_dim / 2) -> (b, height * width, head_dim/2)
            angles = (
                2
                * math.pi
                * patch_coords[..., None]
                * unsqueeze_leading(self.inv_freq, patch_coords)
            )
            angles = rearrange(angles, "... nd df -> ... (nd df)")

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        # unsqueeze head dim
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class GoldenGateRoPEnD(nn.Module):
    """
    adapted from https://jerryxio.ng/posts/nd-rope/
    """

    def __init__(
        self,
        head_dim: int = 64,
        num_attention_heads: int = 1,
        num_zero_freqs: list[int] = [8, 8],
        min_freqs: list[float] = [0.2, 0.2],
        max_freqs: list[float] = [20.0, 20.0],
    ):
        """
        Dimension key:
            N: batch size
            L: number of tokens per sample
            P: pos_dim
            h: n_heads
            d: head_dim
            F: num_freqs == head_dim // 2
        """
        super().__init__()

        assert all(len(x) == len(num_zero_freqs) for x in (min_freqs, max_freqs))

        num_position_dimensions = len(num_zero_freqs)
        num_freqs = head_dim // 2

        omega_PF = []
        for i in range(num_position_dimensions):
            min_freq, max_freq = min_freqs[i], max_freqs[i]
            dim_num_zero_freqs = num_zero_freqs[i]
            dim_num_nonzero_freqs = num_freqs - dim_num_zero_freqs

            zero_freqs = torch.zeros(dim_num_zero_freqs, dtype=torch.float64)

            freqs = min_freq * (max_freq / min_freq) ** torch.linspace(
                0, 1, dim_num_nonzero_freqs, dtype=torch.float64
            )

            omega_F = torch.cat((zero_freqs, freqs))

            omega_PF.append(omega_F)

        omega_PF = torch.stack(omega_PF, 0)

        directions_hFP = make_directions(
            num_attention_heads * num_freqs, num_position_dimensions
        ).reshape(num_attention_heads, num_freqs, num_position_dimensions)

        # h F P, P F -> h F P
        freqs_hFP = directions_hFP * omega_PF.movedim(0, 1).unsqueeze(0)

        self.register_buffer("freqs_hFP", freqs_hFP, persistent=False)

    def forward(
        self, pos_NLP: torch.Tensor, mask_NL: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        *leading_dims, L, P = pos_NLP.shape
        pos_NLP = rearrange(pos_NLP, "... L P -> (...) L P")

        with torch.autocast(pos_NLP.device.type, enabled=False):
            pos_NLP = pos_NLP.float()
            freqs_hFP = self.freqs_hFP.float()
            theta_NhLF = einsum(freqs_hFP, pos_NLP, "h F P, N L P -> N h L F")

            if mask_NL is not None:
                theta_NhLF = theta_NhLF * mask_NL[:, None, :, None]

            cos_NhLF = torch.cos(theta_NhLF)
            sin_NhLF = torch.sin(theta_NhLF)

        _, *hLF = cos_NhLF.shape
        cos_NhLF = cos_NhLF.reshape(*leading_dims, *hLF)
        sin_NhLF = sin_NhLF.reshape(*leading_dims, *hLF)

        return cos_NhLF, sin_NhLF
