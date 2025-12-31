from pathlib import Path
import random
from typing import Callable

from tqdm import tqdm

import torch
import torchvision.transforms.v2 as transforms
from torchcodec.decoders import VideoDecoder

from src.net.configuring_video_data import SimpleVideoDataConfig
from src.net.data_pipeline import DataPipeline, RngMixin


def snap_to_multiple(value, multiple) -> int:
    return int(round(value / multiple) * multiple)


class VideoFileSampler(RngMixin):
    def __init__(
        self,
        path: Path | str,
        video_extensions=(".mp4",),
        should_weight_by_duration: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        if not isinstance(path, Path):
            path = Path(path).absolute()

        if path.is_file():
            self.video_files = [path]
        else:
            self.video_files = [
                p for p in path.rglob("*") if p.suffix.lower() in video_extensions
            ]

        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {path}")

        self.video_files.sort()

        self.should_weight_by_duration = should_weight_by_duration
        self.video_durations = []
        if should_weight_by_duration:
            for video_file in tqdm(self.video_files, "calculating video durations..."):
                decoder = VideoDecoder(video_file, seek_mode="approximate")
                duration = decoder.metadata.duration_seconds
                if duration is None:
                    raise ValueError(
                        f"no accessible duration metadata for video {video_file}"
                    )
                self.video_durations.append(duration)

        self.set_seed(seed)

    def sample_video(self):
        if not self.should_weight_by_duration:
            return self.rng.choice(self.video_files)

        return self.rng.choices(self.video_files, self.video_durations, k=1)[0]

    def __iter__(self):
        while True:
            video_path = self.sample_video()
            yield {"video_path": video_path}


class ClipDecoder(RngMixin):
    def __init__(
        self,
        should_random_start_seek: bool = False,
        batch_size: int = 32,
        min_duration: int = 8,
        max_duration: int = 8,
        frame_skip_range: tuple[int, int] = (1, 1),
        num_samples_to_decode_range: tuple[int, int] = (1, 1),
        duration_multiple_of: int = 1,
        seed: int | None = None,
    ):
        self.should_random_start_seek = should_random_start_seek
        self.batch_size = batch_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.frame_skip_range = frame_skip_range
        self.num_samples_to_decode_range = num_samples_to_decode_range
        self.duration_multiple_of = duration_multiple_of
        self.seed = seed
        self.set_seed(seed)

    def _yield_frames(self, source_row):
        video_path = source_row["video_path"]
        # Approximate mode only works for some types of videos - it should always work for MP4s
        decoder = VideoDecoder(
            str(video_path),
            dimension_order="NHWC",
            seek_mode="approximate",
        )

        num_total_frames = decoder.metadata.num_frames
        assert isinstance(num_total_frames, int)

        # Pick a random number of samples to decode from this video
        num_samples_to_decode = self.rng.randint(*self.num_samples_to_decode_range)
        # Pick the durations of the samples,
        # but don't exceed the num_total_frames
        sample_durations = []
        for _ in range(num_samples_to_decode):
            sample_duration = self.rng.randint(self.min_duration, self.max_duration)
            sample_duration = snap_to_multiple(
                sample_duration, self.duration_multiple_of
            )
            if sum(sample_durations) + sample_duration > num_total_frames:
                break
            sample_durations.append(sample_duration)

        num_frames_to_decode = sum(sample_durations)

        if num_frames_to_decode == 0:
            print(
                f"{video_path} not enough frames to decode a single sample! ({num_total_frames})"
            )
            return

        # Pick the sample rate
        frame_skip_amount = self.rng.randint(*self.frame_skip_range)

        # Pick the start index for decoding
        start_idx = 0
        if self.should_random_start_seek:
            start_idx = self.rng.randint(
                0, num_total_frames - num_frames_to_decode * frame_skip_amount
            )

        decode_idx = start_idx
        frame_buffer = []
        while sample_durations:
            num_remaining_frames = num_frames_to_decode - (decode_idx - start_idx)
            batch_size = min(self.batch_size, num_remaining_frames)
            try:
                frames = decoder.get_frames_in_range(
                    decode_idx,
                    decode_idx + batch_size * frame_skip_amount,
                    frame_skip_amount,
                ).data
            except Exception as e:
                print("Video clip sampler exception!", e)
                return

            decode_idx += batch_size * frame_skip_amount
            frame_buffer.append(frames)

            # print(
            #     f"decoded {decode_idx - start_idx} / {num_frames_to_decode} frames from {video_path}"
            # )

            # yield zero or more samples
            while (
                sample_durations
                and sum(x.shape[0] for x in frame_buffer) >= sample_durations[0]
            ):
                # Keep a list of batched frames to be concattenated to form the sample
                frames = []
                while sum(x.shape[0] for x in frames) < sample_durations[0]:
                    remainder = sample_durations[0] - sum(x.shape[0] for x in frames)

                    # If remainder is smaller than the item in the frame buffer,
                    # we only take part of it
                    if remainder < frame_buffer[0].shape[0]:
                        append, frame_buffer[0] = (
                            frame_buffer[0][:remainder],
                            frame_buffer[0][remainder:],
                        )
                        frames.append(append)
                    # Otherwise we pop the whole frame batch
                    else:
                        frames.append(frame_buffer.pop(0))

                frames = torch.cat(frames, 0)
                assert frames.shape[0] == sample_durations[0]

                row = dict(**source_row)
                row["pixel_values"] = frames
                row["video_metadata"] = decoder.metadata
                yield row

                # We are done with this sample!
                sample_durations.pop(0)

    def __call__(self, source):
        for row in source:
            for row in self._yield_frames(row):
                yield row


class CenterCropVideo:
    def __init__(self, size: tuple[int, int] = (256, 256)):
        self.size = size

    def __call__(self, row):
        pixel_values = row.pop("pixel_values")
        # n h w c -> n c h w
        pixel_values = pixel_values.permute(0, 3, 1, 2)

        *_, h, w = pixel_values.shape

        # TODO use resize mode
        transform_fn = transforms.Compose(
            [transforms.CenterCrop(min(h, w)), transforms.Resize(self.size)]
        )

        pixel_values = transform_fn(pixel_values)

        # n c h w -> n h w c
        pixel_values = pixel_values.permute(0, 2, 3, 1)

        row["pixel_values"] = pixel_values
        return row


class PrepareModelInputs:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, row):
        device, dtype = self.device, self.dtype

        # Try to use packed batch, but default to row
        batch = row.pop("packed_batch", row)

        patches = batch.pop("patches", None)
        pixel_values = batch.pop("pixel_values", None)
        # Send tensors to device, nonblocking
        if patches is not None:
            if patches.dtype != torch.uint8:
                raise ValueError(f"Expected uint8 patch data but got {patches.dtype}")
            patches = patches.to(device=device, dtype=dtype, non_blocking=True)

        if pixel_values is not None:
            if pixel_values.dtype != torch.uint8:
                raise ValueError(
                    f"Expected uint8 pixel data but got {pixel_values.dtype}"
                )
            pixel_values = pixel_values.to(
                device=device, dtype=dtype, non_blocking=True
            )

        batch = {
            k: v.to(device=device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

        # Scale from [0,255] to [-1,1]
        if patches is not None:
            patches = patches.div_(255).mul_(2).sub_(1)
        if pixel_values is not None:
            pixel_values = pixel_values.div_(255).mul_(2).sub_(1)

        if patches is not None:
            row["patches"] = patches
        if pixel_values is not None:
            row["pixel_values"] = pixel_values

        row.update(batch)

        return row


def get_simple_video_dataloader(
    config: SimpleVideoDataConfig = SimpleVideoDataConfig(),
    transform_fn: Callable[[dict], dict] | None = None,
    device=torch.device("cpu"),
    dtype=torch.float32,
):
    rng = None
    if config.seed is not None:
        rng = random.Random(config.seed)

    def get_new_seed():
        if rng is not None:
            return rng.randint(0, 999999999999999)

    video_file_sampler = VideoFileSampler(
        config.path,
        seed=get_new_seed(),
        should_weight_by_duration=config.should_file_sampler_weight_by_duration,
    )

    # * Each worker picks a random video file from the video files
    # * Each worker yields sequential clips of fixed duration from the file
    # * Each clip is resized
    # * Workers shuffle clips
    # * The main process shuffles clips from different workers
    # * The main process collates batches
    dataset = DataPipeline(video_file_sampler).compose(
        ClipDecoder(
            should_random_start_seek=config.clip_decoder_should_random_start_seek,
            batch_size=config.clip_decoder_frame_batch_size,
            min_duration=config.duration,
            max_duration=config.duration,
            frame_skip_range=config.clip_decoder_frame_skip_range,
            num_samples_to_decode_range=config.clip_decoder_num_samples_to_decode_range,
            seed=get_new_seed(),
        )
    )

    if transform_fn is not None:
        dataset = dataset.map(transform_fn)

    dataset = (
        dataset.map(CenterCropVideo((config.height, config.width)))
        .load_parallel(num_workers=config.num_workers, prefetch_factor=1)
        .shuffle(size=config.shuffle_size, seed=get_new_seed())
        .batched(batch_size=config.batch_size)
        .map(PrepareModelInputs(device, dtype))
    )

    return dataset
