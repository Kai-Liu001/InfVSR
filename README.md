# InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution

[Ziqing Zhang](https://github.com/sjtuzzq), [Kai Liu](https://kai-liu001.github.io/), [Zheng Chen](https://zhengchen1999.github.io), [Xi Li](), [Yucong Chen](https://scholar.google.com/citations?user=BTEsUk8AAAAJ&hl), [Bingnan Duan](https://github.com/Bingnan), [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), and [Yulun Zhang](http://yulunzhang.com/), "InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution", ICML, 2026

<div>
  <a href="https://arxiv.org/abs/2510.00948"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
  <a href="https://huggingface.co/zzqingz/InfVSR"><img src="https://img.shields.io/badge/Hugging%20Face-InfVSR-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="InfVSR pretrained model"></a>
  <a href="https://huggingface.co/datasets/zzqingz/MovieLQ"><img src="https://img.shields.io/badge/Hugging%20Face-MovieLQ-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="MovieLQ dataset"></a>
  <a href="https://github.com/Kai-Liu001/InfVSR"><img src="https://visitor-badge.laobi.icu/badge?page_id=Kai-Liu001/InfVSR" alt="Visitors"></a>
  <a href="https://github.com/Kai-Liu001/InfVSR/stargazers"><img src="https://img.shields.io/github/stars/Kai-Liu001/InfVSR?style=social" alt="GitHub stars"></a>
</div>



#### 🔥🔥🔥 News

- **2026-10-02:** Inference code, pretrained weights, and the MovieLQ test set are released. Thanks for your patience and support! 🙏🙏🙏
- **2026-05-01:** InfVSR is accepted by ICML 26! 🎉🎉🎉
- **2025-09-25:** This repo is released. ⭐️⭐️⭐️

---
![](figs/cover.png)



> **Abstract:** Real-world videos often extend over thousands of frames. Existing generative video super-resolution (VSR) approaches, however, face two persistent challenges when processing long sequences: (1) inefficiency due to the heavy cost of multi-step denoising for full-length sequences; and (2) poor consistency hindered by temporal decomposition that causes artifacts and discontinuities. To break these limits, we propose InfVSR, which reformulates VSR as an autoregressive-one-step-diffusion paradigm, and enables streaming inference with video diffusion priors. First, we adapt the pretrained DiT into a causal structure, maintaining both local and global coherence via rolling KV-cache and joint visual guidance. Second, we distill the diffusion process into a single step efficiently, with patch-wise pixel supervision and cross-chunk distribution matching. To fill the gap in long-form video evaluation, we build a new benchmark tailored for extended sequences and further introduce semantic-level metrics to comprehensively assess temporal consistency. Our method pushes the frontier of long-form VSR, achieves state-of-the-art quality with enhanced semantic consistency, and delivers up to 58× speed-up over existing methods such as MGLD-VSR. Our code and models are available at https://github.com/Kai-Liu001/InfVSR.


## ⚒️ TODO

- [x] Release paper and supplementary material.
- [x] Release pretrained model and inference code.
- [x] Release MovieLQ dataset.


## 🔗 Contents


- [InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution](#infvsr-toward-consistency-driven-streaming-generative-video-super-resolution)
  - [🔥🔥🔥 News](#-news)
  - [Contents](#-contents)
  - [Method](#method)
  - [Inference](#inference)
  - [Dataset](#dataset)
  - [Results](#results)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## <a name="method"></a>💡 Method

<p align="center">
  <img width="900" src="figs/method.png">
</p>

## <a name="inference"></a> 🚀 Inference

### 1. Environment

Use Python 3.10 or later with CUDA-enabled PyTorch and a matching torchvision version. The model runs in BF16.

```bash
pip install -r requirements.txt
pip install huggingface_hub
```

FlashAttention is optional; PyTorch SDPA is available as a fallback.

### 2. Download the model

Download [**InfVSR**](https://huggingface.co/zzqingz/InfVSR) checkpoint (approximately 2.63 GiB), into `models/`:

```bash
hf download zzqingz/InfVSR InfVSR.ckpt --local-dir models
```

Also download the three external dependencies from their upstream repositories:

| File | Download |
| --- | --- |
| `Wan2.1_VAE.pth` | [Wan2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/37ec512624d61f7aa208f7ea8140a131f93afc9a/Wan2.1_VAE.pth) |
| `ram_swin_large_14m.pth` | [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/7c76cec1377de0d46df8091a032c4b667aea4459/ram_swin_large_14m.pth) |
| `DAPE.pth` | [DAPE](https://huggingface.co/alexnasa/SEESR/resolve/4059ae0246bf52f5ac5017821bb6e59a6849a758/DAPE.pth) |

Arrange the files as follows:

```text
models/
├── InfVSR.ckpt
├── Wan2.1_VAE.pth
├── ram_swin_large_14m.pth
└── DAPE.pth
```

### 3. Run inference

Single video:

```bash
python inference.py --input inputs/example.mp4 --output outputs/example --scale 4
```

Video folder:

```bash
python inference.py --input inputs/videos --output outputs/videos --scale 4
```

The default scale is **4×**; change `--scale` for another upscaling factor. Use `--model_dir` for a different weight directory and `--device cuda:1` to select a GPU. Weights are loaded locally.

Outputs are H.264 MP4 videos with the requested dimensions, selected input frame count, and original FPS preserved. The default pixel format is yuv420p; odd dimensions use yuv444p without resizing. Audio is not copied. Use `--fps` to override the frame rate or `--yuv444p` for lossless H.264/yuv444p. Choose a new output directory for each run.

Inference processes the entire video by default. Use `--max_frames 1000` to process only the first 1000 frames.

Scene detection runs online using PySceneDetect. Use `--no_scene_detection` to keep a single reference and continuous caches throughout the video.

## <a name="dataset"></a> 🤗 Dataset

[**MovieLQ**](https://huggingface.co/datasets/zzqingz/MovieLQ) is introduced in our paper to evaluate super-resolution performance on long real-world videos. It contains **10 real-world degraded videos, each 1,000 frames long**.

Download MovieLQ from Hugging Face:

```bash
hf download zzqingz/MovieLQ --repo-type dataset --local-dir inputs/MovieLQ
```

## <a name="results"></a> 🔎 Results

We achieved state-of-the-art performance. Detailed results can be found in the paper.

<details>
<summary>Click to expand</summary>

- quantitative comparisons in Table 1 (main paper)

<p align="center">
  <img width="900" src="figs/main-comparison.png">
</p>



- visual comparison in Figure 3 (main paper)

<p align="center">
  <img width="900" src="figs/visual-comp1.png">
</p>



- visual comparison in Figure 5 (main paper)

<p align="center">
  <img width="900" src="figs/visual-comp2.png">
</p>




- visual comparison in Figure 9 (supplemental material)

<p align="center">
  <img width="900" src="figs/visual-comp3.png">
</p>

- visual comparison in Figure 10 (supplemental material)

<p align="center">
  <img width="900" src="figs/visual-comp4.png">
</p>

</details>

## <a name="citation"></a>📎 Citation
If you find the code helpful in your research or work, please cite the following paper(s).

```
@inproceedings{zhang2026infvsr,
  title={InfVSR: Toward Consistency-Driven Streaming Generative Video Super-Resolution},
  author={Zhang, Ziqing and Liu, Kai and Chen, Zheng and Li, Xi and Chen, Yucong and Duan, Bingnan and Kong, Linghe and Zhang, Yulun},
  booktitle={ICML},
  year={2026}
}
```

## <a name="acknowledgements"></a> 💖 Acknowledgements

This project builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and [SeeSR (DAPE)](https://github.com/cswry/SeeSR). We thank the authors for sharing their code and pretrained models.
