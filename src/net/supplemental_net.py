from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange

from memfof import MEMFOF
from memfof.corr import CorrBlock
from memfof.utils.utils import coords_grid, InputPadder


def coords_feature(fmap, b, x, y):
    H, W = fmap.shape[2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    b = b.long()
    x = torch.clamp(x, 0, W - 1).long()
    y = torch.clamp(y, 0, H - 1).long()
    res = fmap[b, :, y, x] * mask.float().unsqueeze(1)
    return res


class OpticalFlowExtractor(nn.Module):
    def __init__(
        self, path_or_url: str = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
    ):
        super().__init__()
        self.model = (
            MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
            .eval()
            .requires_grad_(False)
        )

    def _forward_flow(
        self,
        pixel_values: torch.Tensor,
        fmap_cache: Sequence[torch.Tensor | None],
        iters: int = 0,
    ):
        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(pixel_values.shape)
        pixel_values = padder.pad(pixel_values)
        B, _, _, H, W = pixel_values.shape
        dilation = torch.ones(B, 1, H // 16, W // 16, device=pixel_values.device)

        # run the context network
        cnet = self.model.cnet(
            torch.cat(
                [pixel_values[:, 0], pixel_values[:, 1], pixel_values[:, 2]], dim=1
            )
        )
        cnet = self.model.init_conv(cnet)
        net, context = torch.split(cnet, [self.model.dim, self.model.dim], dim=1)
        attention = self.model.att(context)

        # init flow
        flow_update = self.model.flow_head(net)
        weight_update = 0.25 * self.model.upsample_weight(net)

        flow_16x_21 = flow_update[:, 0:2]
        info_16x_21 = flow_update[:, 2:6]

        flow_16x_23 = flow_update[:, 6:8]
        info_16x_23 = flow_update[:, 8:12]

        if self.training or iters == 0:
            flow_up_21, info_up_21 = self.model._upsample_data(
                flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
            )
            flow_up_23, info_up_23 = self.model._upsample_data(
                flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
            )
            flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
            info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        if iters > 0:
            # run the feature network
            fmap1_16x = (
                self.model.fnet(pixel_values[:, 0])
                if fmap_cache[0] is None
                else fmap_cache[0].clone().to(cnet)
            )
            fmap2_16x = (
                self.model.fnet(pixel_values[:, 1])
                if fmap_cache[1] is None
                else fmap_cache[1].clone().to(cnet)
            )
            fmap3_16x = (
                self.model.fnet(pixel_values[:, 2])
                if fmap_cache[2] is None
                else fmap_cache[2].clone().to(cnet)
            )
            corr_fn_21 = CorrBlock(
                fmap2_16x, fmap1_16x, self.model.corr_levels, self.model.corr_radius
            )
            corr_fn_23 = CorrBlock(
                fmap2_16x, fmap3_16x, self.model.corr_levels, self.model.corr_radius
            )

        for itr in range(iters):
            B, _, H, W = flow_16x_21.shape
            flow_16x_21 = flow_16x_21.detach()
            flow_16x_23 = flow_16x_23.detach()

            coords21 = (
                coords_grid(B, H, W, device=pixel_values.device) + flow_16x_21
            ).detach()
            coords23 = (
                coords_grid(B, H, W, device=pixel_values.device) + flow_16x_23
            ).detach()

            corr_21 = corr_fn_21(coords21, dilation=dilation)
            corr_23 = corr_fn_23(coords23, dilation=dilation)

            corr = torch.cat([corr_21, corr_23], dim=1)
            flow_16x = torch.cat([flow_16x_21, flow_16x_23], dim=1)

            net = self.model.update_block(net, context, corr, flow_16x, attention)

            flow_update = self.model.flow_head(net)
            weight_update = 0.25 * self.model.upsample_weight(net)

            flow_16x_21 = flow_16x_21 + flow_update[:, 0:2]
            info_16x_21 = flow_update[:, 2:6]

            flow_16x_23 = flow_16x_23 + flow_update[:, 6:8]
            info_16x_23 = flow_update[:, 8:12]

            if self.training or itr == iters - 1:
                flow_up_21, info_up_21 = self.model._upsample_data(
                    flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
                )
                flow_up_23, info_up_23 = self.model._upsample_data(
                    flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
                )
                flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
                info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        new_fmap_cache = [None, None, None]
        if iters > 0:
            new_fmap_cache = [
                fmap1_16x.clone(),
                fmap2_16x.clone(),
                fmap3_16x.clone(),
            ]

        return {
            "flow": flow_predictions,
            "info": info_predictions,
            "nf": None,
            "fmap_cache": new_fmap_cache,
        }

    def forward(self, pixel_values: torch.Tensor, iters: int = 0):
        # Expects [b, n, h, w, c] in [-1,1]
        pixel_values = rearrange(pixel_values, "b n h w c -> b n c h w")

        b, n, c, h, w = pixel_values.shape

        if n < 3:
            raise ValueError(f"num frames {n} must be greater or equal to 3")

        fmap_cache = [None] * 3

        flow = torch.empty(
            b, n, 2, h, w, device=pixel_values.device, dtype=pixel_values.dtype
        )

        for i in range(1, n - 1):
            frames = pixel_values[:, i - 1 : i + 2]

            with torch.autocast(
                pixel_values.device.type,
                pixel_values.dtype,
                enabled=pixel_values.dtype != torch.float32,
            ):
                output = self._forward_flow(frames, fmap_cache=fmap_cache, iters=iters)

            frame_flow = output["flow"][-1][0, 1]  # FW [2, H, W]
            flow[:, i] = frame_flow

            fmap_cache = output["fmap_cache"]
            fmap_cache = [fmap_cache[1], fmap_cache[2], None]

        flow[:, 0] = flow[:, 1]
        flow[:, -1] = flow[:, -2]

        flow = rearrange(flow, "b n uv h w -> b n h w uv")

        return flow


class DinoEncoder(nn.Module):
    def __init__(
        self, dino_path_or_url: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    ):
        super().__init__()
        self.dino = (
            transformers.DINOv3ViTModel.from_pretrained(dino_path_or_url)
            .eval()
            .requires_grad_(False)
        )
        self.hidden_size = self.dino.config.hidden_size
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor):
        # Expects [..., h, w, c] in [-1, 1]
        *leading, h, w, c = pixel_values.shape
        pixel_values = rearrange(pixel_values, "... h w c -> (...) c h w")

        pixel_values = (pixel_values + 1) / 2
        pixel_values = (pixel_values - self.mean) / self.std

        features = self.dino(pixel_values).last_hidden_state

        # DINOv3 implementation detail: skipping register tokens if present
        # Assuming format [CLS, PATCHES...]
        cls_feature = features[:, 0]
        patch_features = features[:, 5:]  # skip sidechannel features

        # Layer norms
        patch_features = F.layer_norm(patch_features, (patch_features.shape[-1],))
        cls_feature = F.layer_norm(cls_feature, (cls_feature.shape[-1],))

        patch_features = patch_features.reshape(
            *leading, patch_features.shape[1], patch_features.shape[2]
        )
        cls_feature = cls_feature.reshape(*leading, cls_feature.shape[-1])

        return cls_feature, patch_features


class DepthExtractor(nn.Module):
    def __init__(self, path_or_url: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        self.model = (
            transformers.AutoModelForDepthEstimation.from_pretrained(path_or_url)
            .eval()
            .requires_grad_(False)
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor):
        # Expects [..., h, w, c] in [-1, 1]
        *leading, h, w, c = pixel_values.shape
        pixel_values = rearrange(pixel_values, "... h w c -> (...) c h w")

        pixel_values = (pixel_values + 1) / 2
        pixel_values = (pixel_values - self.mean) / self.std

        depth = self.model(pixel_values).predicted_depth

        # Interpolate to original resolution
        depth = F.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze(1)

        return depth.reshape(*leading, h, w)


class REPAProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        # Expects [B, D, T, H, W]
        return self.net(x)
