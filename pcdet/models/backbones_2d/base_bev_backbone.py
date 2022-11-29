import numpy as np
import torch
import torch.nn as nn
from .BiFPN import BiFPN, BiFPN_Network, BiFPN_Network_SkipConnections
from torch import cat


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        """
        Processing the pseudo image with efficient 2D convolutions.
        """
        
        super().__init__()
        self.model_cfg = model_cfg
        self.bifpn_sizes = kwargs['bifpn']
        self.bifpn_skip = kwargs["bifpn_skip"]

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        
        # Creating the original network, consisting of repeating 2D convolutional blocks.
        for idx in range(num_levels):
            cur_layers = [nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )]
                
            for k in range(layer_nums[idx]):
                cur_layers.append(nn.Sequential(
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))
     
            self.blocks.append(nn.Sequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    
         # BiFPN should be inserted to the last nn.Sequential
        if  len(self.bifpn_sizes) > 0:
            bifpn = BiFPN_Network_SkipConnections if self.bifpn_skip else BiFPN_Network
            self.bifpn = bifpn([num_filters[idx]] * 3, out_channels=self.bifpn_sizes)
            self.bifpn.to(self.bifpn.device)     
            output_channels = num_filters[idx] * 3 if self.bifpn_skip else self.bifpn_sizes[-1] * 3 # Last_blockdeals the concatanted output of the network.
            self.last_block = nn.Sequential(
                nn.Conv2d(output_channels, *self.model_cfg.NUM_FILTERS, kernel_size=1),
                nn.BatchNorm2d(*self.model_cfg.NUM_FILTERS),
                nn.ReLU()
            ).to(self.bifpn.device)

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        spatial_features = data_dict['spatial_features']
        ups = []
        
        x = spatial_features
        for i in range(len(self.blocks)):
            
            # With BiFPN blocks, first run all the sequentials in the 2D-backbone networks, until the last one.
            # There, features are taken from different scales: First, middle and last conv blocks. They serve as input to the 
            # BiFPN subnetwork.
            if len(self.bifpn_sizes) > 0 and i + 1 == len(self.blocks):
                
                # Save the relevant scales in orig_layers, forward the last sequential, but save pointers to the blocks.
                orig_layers = []
                for j, layer in enumerate(self.blocks[i]):
                    x = layer(x)
                    if i + 1 == len(self.blocks):
                        orig_layers.append(x)
                
                # Run the BiFPN network on the chosen inputs.
                N = len(orig_layers)
                feature_maps = self.bifpn.forward(orig_layers[-1], orig_layers[-(int(N / 2) + 1)], orig_layers[-N])
                
                # The BiFPN outputs processed tesnors for all 3 inputs. Concatante them to one X.
                x = cat(feature_maps[:3], axis=1)
                # last block uses convolutions to adjust the number of channels, to the requested one by the deblocks upsampling.
                x = self.last_block(x) 
            else:
                x = self.blocks[i](x)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
                    
            if len(ups) > 1:
                x = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x = ups[0]

        
            if len(self.deblocks) > len(self.blocks):
                x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
