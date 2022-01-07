from urllib.parse import parse_qsl
import mayavi.mlab as mlab
import torch
from visual_utils import visualize_utils as V

class Visualizator:

    def __init__(self) -> None:
        pass
    
    def show():
        with torch.no_grad():
                for idx, data_dict in enumerate(demo_dataset):
                    logger.info(f'Visualized sample index: \t{idx + 1}')
                    data_dict = demo_dataset.collate_batch([data_dict])
                    load_data_to_gpu(data_dict)
                    pred_dicts, _ = model.forward(data_dict)
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], gt_boxes = gt_instance["annos"]["gt_boxes_lidar"], ref_boxes=pred_dicts[0]['pred_boxes'],
                        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                    )
                    mlab.show(stop=True)

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
        
        
    


