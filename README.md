# Spokester1

Try to pedal your way through the congested boulevards of
New York City at ever increasing speeds. The twist is
the graphics pipeline is rendered using a diffusion transformer.


![spokester-sample](https://github.com/user-attachments/assets/c6a28603-26be-41d0-af2b-b2e5a861b7f9)

Using this codebase you can train your own neural graphics engine from scratch. You
can also download my pretrained diffusion model and play the game.

### Requirements

- Modern FFMPEG (n8.0)
- uv
- opengl
- yt-dlp (To download reference video for training)
- 24GB GPU memory and 16GB host memory (For training)
- 8GB GPU memory and 8GB host memory (For inference)

This project was tested on Linux and may need modification to run on
Windows.

### Playing Spokester1

run this command:

```
uv run python -m src.scripts.play_game_with_neural_graphics
```

### Training

Training a DiT-Base with 100 million parameters for ~300k steps
takes about 3 days using a single 3090 GPU.

First, download Hotline Toni from YouTube

```
mkdir -p data/real_videos
cd data/real_videos
yt-dlp "https://www.youtube.com/watch?v=FrOFEiZzazA"
```

Reencode the video using ffmpeg.

I recommend encoding into a low resolution video to be used during low res pretraining
and a higher resolution video to be used during subsequent high res pretraining.

These commands center crop and resize the video, slicing off the 23 second intro graphic and the
outro graphic.

Replace the Hotline Toni path with the particular one that you just downloaded

```
ffmpeg -i Hotline\ Toni\ ｜\ Three\ Borough\ Loop\ \[FrOFEiZzazA\].webm \
 -c:v libx264 -crf 18 -preset veryfast -tune fastdecode -g 12 -bf 0 \
 -filter:v "libplacebo=fps=24,crop='min(iw,ih):min(iw,ih)',libplacebo=downscaler=ewa_lanczos:w=448:h=448" \
 -an -y -ss 00:00:23 -to 00:34:12 hotline_toni_448p_24fps.mp4

ffmpeg -i Hotline\ Toni\ ｜\ Three\ Borough\ Loop\ \[FrOFEiZzazA\].webm \
 -c:v libx264 -crf 18 -preset veryfast -tune fastdecode -g 12 -bf 0 \
 -filter:v "libplacebo=fps=24,crop='min(iw,ih):min(iw,ih)',libplacebo=downscaler=ewa_lanczos:w=224:h=224" \
 -an -y -ss 00:00:23 -to 00:34:12 hotline_toni_224p_24fps.mp4
```

cd back to the root of this project

```
cd ../..
```

Modify the `conf/base.yaml` file with your particular device/dtype

Launch the low res pretraining. This script trains for 200k steps and saves checkpoints and generated samples to a new folder in `./runs/`

```
uv run python -m src.scripts.train_net --conf conf/base.yaml --conf conf/pretrain_lowres.yaml
```

After low res pretraining you can do some higher resolution finetuning.

Replace ./runs/run-000020/00131000.pt with your checkpoint to resume from

```
uv run python -m src.scripts.train_net --conf conf/base.yaml --conf conf/pretrain_highres.yaml \
 --conf.resume_checkpoint_path ./runs/run-000020/00131000.pt
```
