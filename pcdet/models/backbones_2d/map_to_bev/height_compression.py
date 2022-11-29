import torch.nn as nn
from ..BiFPN import BiFPN, BiFPN_Network
from torch import cat


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.counter = 0
        self.bifpn_sizes = kwargs.get('bifpn', [])
    
    def pseudo_image(self, x):
        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)
        return x 

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        
        batch_dict['spatial_features'] = self.pseudo_image(batch_dict["encoded_spconv_tensor"])
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        self.counter += 1
        return batch_dict
