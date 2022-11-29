# CenterPoint

3D Object Detection and Tracking using center points in the bird-eye view.

This forked repo were used for the lecture Advanced Deep Learning for Computer Vision. Our poster and contributions:
![Team2_cvpr_poster](https://user-images.githubusercontent.com/20690194/203155410-92ceca31-5006-4ae0-a3a9-6a3d89c0f1aa.png)

## Reference papers \ Repos:
1) Center Point: https://arxiv.org/abs/2006.11275
2) BiFPN horizon Robotics: https://arxiv.org/abs/2006.15505
3) BiFPN GitHub Repo: https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch
4) Metrics GitHub repo: https://github.com/rafaelpadilla/Object-Detection-Metrics

GCP VM Settings:
- Intel N1-highmem-4 (4vCPU, 26 GB Memory)
- Nvidia Tesla T4
- 500 GB SSD

Useful Links:
1) GCP GPU Zones: https://cloud.google.com/compute/docs/gpus/gpu-regions-zones

## Original Repo README.md:

3D Object Detection and Tracking using center points in the bird-eye view.

<p align="center"> <img src='docs/teaser.png' align="center" height="230px"> </p>

> [**Center-based 3D Object Detection and Tracking**](https://arxiv.org/abs/2006.11275),            
> Tianwei Yin, Xingyi Zhou, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 2006.11275](https://arxiv.org/abs/2006.11275))*  


    @article{yin2021center,
      title={Center-based 3D Object Detection and Tracking},
      author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
      journal={CVPR},
      year={2021},
    }

**CenterPoint is also implemented in the official [OpenPCDet repo](https://github.com/open-mmlab/OpenPCDet) and can reproduce results on nuScenes and Waymo**. 

This repo is an reimplementation of CenterPoint on the KITTI dataset. For nuScenes and Waymo, please refer to the [original repo](https://github.com/tianweiy/CenterPoint). Please refer to [INSTALL.md](docs/INSTALL.md) for installation. We provide two configs, [centerpoint.yaml](tools/cfgs/kitti_models/centerpoint.yaml) for the vanilla centerpoint model and [centerpoint_rcnn.yaml](tools/cfgs/kitti_models/centerpoint_rcnn.yaml) which combines centerpoint with PVRCNN. 


## Acknowledgement

Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Some util files are copied from [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Thanks OpenMMLab Development Team for their awesome codebases.



