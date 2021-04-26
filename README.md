# CenterPoint 

This repo is an reimplementation of CenterPoint on the KITTI dataset. For nuScenes and Waymo, please refer to the [original repo](https://github.com/tianweiy/CenterPoint). We provide two configs, [centerpoint.yaml](tools/cfgs/kitti_models/centerpoint.yaml) for the vanilla centerpoint model and [centerpoint_rcnn.yaml](tools/cfgs/kitti_models/centerpoint_rcnn.yaml) which combines centerpoint with PVRCNN. 

## Acknowledgement

Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Some util files are copied from [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Thanks OpenMMLab Development Team for their awesome codebases.