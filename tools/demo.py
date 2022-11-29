import argparse
import glob
from operator import gt
from pathlib import Path
import pickle
import mayavi.mlab as mlab
import numpy as np
import torch
import re

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', **kwargs):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.args = kwargs
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--sample_folder', type=str, default='test', help='Take demo sample from {test, train}')
    parser.add_argument('--bifpn', type=int, nargs='*', default=[], help='<Required> Set number of bifpn blocks')
    parser.add_argument('--bifpn_skip', dest='bifpn_skip', action='store_true', help='Use skip connections with BiFPN blocks')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    kitti_infos = include_kitti_data(cfg, args.sample_folder)
    gt_instance = list(filter(lambda x :x["image"]["image_idx"] == re.findall(f"[0-9]+", args.data_path)[-1], kitti_infos))
    gt_instance = gt_instance[0]["annos"]["gt_boxes_lidar"] if len(gt_instance) else None
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of CenterPoint----------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, bifpn=args.bifpn, bifpn_skip=args.bifpn_skip
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=gt_instance, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    logger.info('Demo done.')

def include_kitti_data(dataset_cfg, mode):
    """
    Extracts the scenes and images info from the .pkl files inti a dictionary that holds the gt_boxes, predicted_boxes, etc.
    """
    kitti_infos = []
    for info_path in dataset_cfg.DATA_CONFIG["INFO_PATH"][mode]:
        info_path =  Path(dataset_cfg.DATA_CONFIG.DATA_PATH) / info_path
        if not info_path.exists():
            continue
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_infos.extend(infos)

    return kitti_infos

if __name__ == '__main__':
    
    main()




