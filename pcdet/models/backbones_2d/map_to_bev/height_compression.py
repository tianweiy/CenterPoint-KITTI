import torch.nn as nn
from ..BiFPN import BiFPN, BiFPN_Network
from torch import cat


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.counter = 0
        self.bifpn_sizes = kwargs['bifpn']

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        def pseudo_image(x):
            x = x.dense()
            N, C, D, H, W = x.shape
            x = x.view(N, C * D, H, W)
            return x
        layer_5 = pseudo_image(batch_dict["encoded_spconv_tensor"])
        if len(self.bifpn_sizes) > 0:
            layer_4 = pseudo_image(batch_dict["multi_scale_3d_features"]["x_conv4"])
            layer_3 = pseudo_image(batch_dict["multi_scale_3d_features"]["x_conv3"])

            if self.counter == 0:
                # TODO: Put BiFPN details in CFG file.
                self.bifpn = BiFPN_Network([layer_5.shape[1], layer_4.shape[1], layer_3.shape[1]], out_channels=self.bifpn_sizes)
                self.bifpn.to(self.bifpn.device)

            feature_maps = self.bifpn.forward(layer_5, layer_4, layer_3)
            pseudo = cat(feature_maps, axis=1)
            if self.counter == 0:
                self.last_block = nn.Sequential(
                    nn.Conv2d(pseudo.shape[1], self.model_cfg.NUM_BEV_FEATURES, kernel_size=1),
                    nn.BatchNorm2d(self.model_cfg.NUM_BEV_FEATURES),
                    nn.ReLU()
                ).to(self.bifpn.device)
            pseudo = self.last_block(pseudo)
        else:
            pseudo = layer_5

        

        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # spatial_features = encoded_spconv_tensor.dense()
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = pseudo
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        self.counter += 1
        return batch_dict
