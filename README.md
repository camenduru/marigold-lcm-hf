---
title: Marigold-LCM Depth Estimation
emoji: 🏵️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.22.0
app_file: app.py
pinned: true
license: cc-by-sa-4.0
models:
- prs-eth/marigold-v1-0
- prs-eth/marigold-lcm-v1-0
---

This is a demo of Marigold-LCM, the state-of-the-art depth estimator for images in the wild.
It combines the power of the original Marigold 10-step estimator and the Latent Consistency Models, delivering high-quality results in as little as one step.
Find out more in our paper titled ["Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation"](https://arxiv.org/abs/2312.02145)

```
@misc{ke2023repurposing,
      title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation}, 
      author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
      year={2023},
      eprint={2312.02145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
