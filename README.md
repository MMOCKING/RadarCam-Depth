# RadarCam-Depth

## Introduction

<img src="assets/cover.png" alt="cover" style="zoom:50%;" />

**Abstract:** We present a novel approach for metric dense depth estimation based on the fusion of a single-view image and a sparse, noisy Radar point cloud. The direct fusion of heterogeneous Radar and image data, or their encodings, tends to yield dense depth maps with significant artifacts, blurred boundaries, and suboptimal accuracy. To circumvent this issue, we learn to augment versatile and robust monocular depth prediction with the dense metric scale induced from sparse and noisy Radar data. We propose a Radar-Camera framework for highly accurate and fine-detailed dense depth estimation with four stages, including monocular depth prediction, global scale alignment of monocular depth with sparse Radar points, quasi-dense scale estimation through learning the association between Radar points and image patches, and local scale refinement of dense depth using a scale map learner. Our proposed method significantly outperforms the state-of-the-art Radar-Camera depth estimation methods by reducing the mean absolute error (MAE) of depth estimation by 25.6% and 40.2% on the challenging nuScenes dataset and our self-collected ZJU-4DRadarCam dataset, respectively.



![pipeline](assets/pipeline.png)

Our proposed RadarCam-Depth is comprised with four stages: **monocular depth prediction**, **global alignment of mono-depth** with sparse Radar depth, learned **quasi-dense scale estimation**, and **scale map learner** for refining local scale. $\mathbf{d}$ and $\mathbf{s}$ denotes the depth and scale, while $\mathbf{z}=1/\mathbf{d}$ is the inverse depth.

## Citation

```
@article{li2024radarcam,
  title={RadarCam-Depth: Radar-Camera Fusion for Depth Estimation with Learned Metric Scale},
  author={Li, Han and Ma, Yukai and Gu, Yaqing and Hu, Kewei and Liu, Yong and Zuo, Xingxing},
  journal={arXiv preprint arXiv:2401.04325},
  year={2024}
}
```

