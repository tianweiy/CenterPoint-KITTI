from functools import partial

import spconv
import torch.nn as nn
import torch


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out

class BiFPN(nn.Module):
    """
    This is influenced by: https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch/blob/master/BiFPN.py
    And the paper: https://arxiv.org/abs/1911.09070

    """
    def __init__(self,  fpn_sizes):
        super().__init__()

        P3_channels, P4_channels, P5_channels = fpn_sizes

        self.out_chn = 64
        self.eps = 1e-4

        # TODO: notice that batchnorm comes here AFTER Relu on the BiFPN github, in contrast to the convention.
        block = post_act_block
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # 5
        self.p5_inp_conv = spconv.SparseSequential( 
            block(P5_channels, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p5_out_conv = spconv.SparseSequential( 
            block(self.out_chn, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p5_out_w1  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_upsample  = nn.Upsample([5, 200, 176], mode='nearest')

        # 4
        self.p4_inp_conv = spconv.SparseSequential( 
            block(P4_channels, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p4_td_conv = spconv.SparseSequential( 
            block(self.out_chn, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p4_out_conv = spconv.SparseSequential( 
            block(self.out_chn, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_upsample = nn.Upsample([11, 400, 352], mode='nearest')
        self.p4_downsample = nn.MaxPool3d(kernel_size=(2,1,1))
        
        # 3
        self.p3_inp_conv = spconv.SparseSequential( 
            block(P3_channels, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p3_out_conv = spconv.SparseSequential( 
            block(self.out_chn, self.out_chn, kernel_size=3, stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.p3_out_w1  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_downsample= nn.MaxPool3d(kernel_size=2)
        
    def forward(self, in_5, in_4, in_3):
        # Input convs, we keep the dimensions
        conv_5_in = self.p5_inp_conv(in_5)  # [2, 200, 176]
        conv_4_in = self.p4_inp_conv(in_4)  # [5, 200, 176]
        conv_3_in = self.p3_inp_conv(in_3)  # [11, 400, 352]

        # intermediate convs for 3 & 4 (top --> down path)
        conv_5_in_resized = self.p5_upsample(conv_5_in)
        t_in = (self.p4_td_w1 * conv_4_in + self.p4_td_w2 * conv_5_in_resized ) / (self.p4_td_w1 + self.p4_td_w2 + self.eps)
        td_4 = self.p4_td_conv(t_in)

        td_4_resized = self.p4_upsample(td_4)
        t_in = (self.p3_out_w1 * conv_3_in + self.p3_out_w2 * td_4_resized) / (self.p3_out_w1 + self.p3_out_w2 + self.eps)
        out_3 = self.p3_out_conv(t_in)

        # out convs 4 & 5 (down -- > top path)
        out_3_resized = self.p3_downsample(out_3)
        t_in = (self.p4_out_w1 * conv_4_in + self.p4_out_w2 * td_4 + self.p4_out_w3 * out_3_resized) / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + self.eps)
        out_4 = self.p4_out_conv(t_in)

        out_4_resized = self.p4_downsample(out_4)
        t_in = (self.p5_out_w1 * conv_5_in + self.p5_out_w2 * out_4_resized) / (self.p5_out_w1 + self.p5_out_w2 + self.eps)
        out_5 = self.p5_out_conv(t_in)

        return out_5, out_4, out_3
        
class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.grid_size = grid_size
        self.kwargs = kwargs
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

        
    def __repr__(self):
        #super().__repr__()
        return ('{name}({model_cfg}, in_ch={in_ch}, grid_size={grid_size},'
                'kwargs={kwargs})'
                .format(name=self.__class__.__name__, model_cfg=self.model_cfg, in_ch=self.input_channels, grid_size=self.sparse_shape, kwargs=self.kwargs))

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # print(f"xconv_3 = {x_conv3.spatial_shape}")
        x_conv4 = self.conv4(x_conv3)
        # print(f"xconv_4 = {x_conv4.spatial_shape}")
    

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        # print(f"out = {out.spatial_shape}")

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict
