from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimpleVideoDataConfig:
    path: Path = None
    height: int = 224
    width: int = 224
    duration: int = 8
    seed: int | None = None
    should_file_sampler_weight_by_duration: bool = False
    clip_decoder_should_random_start_seek: bool = False
    clip_decoder_frame_batch_size: int = 256
    clip_decoder_frame_skip_range: tuple[int, int] = (1, 1)
    clip_decoder_num_samples_to_decode_range: tuple[int, int] = (1, 1)
    num_workers: int = 0
    shuffle_size: int = 0
    batch_size: int = 1
