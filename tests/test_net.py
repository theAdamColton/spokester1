import unittest

import torch

from src.net.configuring_net import TransformerConfig, ViTDenoiserConfig
from src.net.flow_matching import FlowMatchingHelper
from src.net.helpers import generate, generate_autoregressive
from src.net.net import ViTDenoiser


class TestDenoiser(unittest.TestCase):
    @torch.inference_mode()
    def test_kv_cache_parallel_matches_autoregressive(self):
        b, l, d_patch = 3, 256, 16

        transformer_conf = TransformerConfig(
            hidden_size=64, head_dim=16, use_attention_gating=False
        )
        model_conf = ViTDenoiserConfig(
            input_size=d_patch,
            transformer=transformer_conf,
            num_blocks=2,
        )
        model = ViTDenoiser(model_conf)

        torch.nn.init.kaiming_normal_(model.proj_out.weight)

        x = torch.randn(b, l, d_patch)
        patch_coords = torch.randn(b, l, 2)
        timesteps = torch.rand(1, 1)

        q_ids = torch.arange(l).unsqueeze(-1)
        kv_ids = torch.arange(l).unsqueeze(0)
        causal_mask = q_ids >= kv_ids
        # b l l
        causal_mask = causal_mask.unsqueeze(0)

        output_gt = model(
            patches=x,
            patch_coords=patch_coords,
            timesteps=timesteps,
            attention_mask=causal_mask,
        ).prediction

        kv_cache = torch.empty(
            model_conf.num_blocks,
            2,
            b,
            transformer_conf.num_attention_heads,
            l,
            transformer_conf.head_dim,
        )

        outputs_ar = []

        for i in range(l):
            x_i = x[:, i : i + 1]
            patch_coords_i = patch_coords[:, i : i + 1]
            timesteps_i = timesteps

            q_ids = torch.full(
                (1,),
                i,
            ).unsqueeze(-1)
            causal_mask_kv_cache = q_ids >= kv_ids
            # b l s
            causal_mask_kv_cache = causal_mask_kv_cache.unsqueeze(0)

            output_ar = model(
                patches=x_i,
                patch_coords=patch_coords_i,
                timesteps=timesteps_i,
                attention_mask=causal_mask_kv_cache,
                kv_cache=kv_cache,
                kv_cache_length=i,
            ).prediction

            outputs_ar.append(output_ar)

            self.assertTrue(
                torch.allclose(
                    output_ar, output_gt[:, i : i + 1], rtol=1e-4, atol=1e-4
                ),
                f"difference at token {i}",
            )

        outputs_ar = torch.cat(outputs_ar, 1)
        self.assertTrue(torch.allclose(outputs_ar, output_gt, rtol=1e-4, atol=1e-4))

    @torch.inference_mode()
    def test_generate_autoregressive_matches_parallel(self):
        b, n, h, w, c = 3, 256, 64, 64, 3
        pn = 2
        ph = pw = 16
        d_patch = pn * ph * pw * c
        d_cond = 32
        npn = n // pn
        nph = h // ph
        npw = w // pw

        torch_rng = torch.Generator().manual_seed(42)
        fm_helper = FlowMatchingHelper()
        transformer_conf = TransformerConfig(
            hidden_size=64, head_dim=16, use_attention_gating=False
        )
        model_conf = ViTDenoiserConfig(
            input_size=d_patch,
            transformer=transformer_conf,
            condition_input_size=d_cond,
            num_blocks=2,
        )
        model = ViTDenoiser(model_conf).reset_weights_(torch_rng=torch_rng)
        torch.nn.init.kaiming_normal_(model.proj_out.weight)

        patch_condition = torch.randn(b, npn, nph * npw, d_cond, generator=torch_rng)
        noise = torch.randn(b, npn, nph * npw, d_patch, generator=torch_rng)

        temporal_window_size = int(
            torch.randint(2, npn, (1,), generator=torch_rng).item()
        )

        sample_gt, kv_cache_gt, kv_length_gt = generate(
            sample_shape=(b, n, h, w, c),
            pixel_flow_matching_helper=fm_helper,
            pixel_denoiser=model,
            noise=noise,
            patch_condition=patch_condition,
            patch_duration=pn,
            patch_side_length=ph,
            temporal_window_size=temporal_window_size,
            num_denoising_steps=1,
            reshape_out=False,
        )

        # Create a list of random query lengths
        # Instead of using one query token at a time
        # use a random number of query tokens at a time
        query_lengths = []
        while True:
            query_length = torch.randint(1, 8, (1,), generator=torch_rng).item()
            if sum(query_lengths) + query_length > npn:
                query_lengths.append(npn - sum(query_lengths))
                break
            query_lengths.append(query_length)

        kv_cache = None
        kv_cache_length = 0
        i = 0
        for npn_q in query_lengths:
            noise_q = noise[:, i : i + npn_q]
            patch_condition_q = patch_condition[:, i : i + npn_q]

            sample_ar, kv_cache, kv_cache_length = generate(
                sample_shape=(b, npn_q * pn, h, w, c),
                pixel_flow_matching_helper=fm_helper,
                pixel_denoiser=model,
                noise=noise_q,
                patch_condition=patch_condition_q,
                patch_duration=pn,
                patch_side_length=ph,
                temporal_window_size=temporal_window_size,
                num_denoising_steps=1,
                kv_cache=kv_cache,
                kv_cache_length=kv_cache_length,
                reshape_out=False,
            )

            self.assertTrue(
                torch.allclose(
                    sample_ar, sample_gt[:, i : i + npn_q], rtol=1e-4, atol=1e-4
                )
            )

            i += npn_q

        sample_ar, kv_cache_ar, _ = generate_autoregressive(
            sample_shape=(b, n, h, w, c),
            pixel_flow_matching_helper=fm_helper,
            pixel_denoiser=model,
            noise=noise,
            patch_condition=patch_condition,
            patch_duration=pn,
            patch_side_length=ph,
            temporal_window_size=temporal_window_size,
            num_denoising_steps=1,
            reshape_out=False,
        )

        self.assertTrue(torch.allclose(sample_ar, sample_gt, rtol=1e-4, atol=1e-4))
