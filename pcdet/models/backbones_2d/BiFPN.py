import torch.nn as nn
import torch

def conv_block(in_channels, out_channels, kernel_size=3, padding=1, eps=1e-3, momentum=0.01,
 stride=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
         stride=stride, bias=bias),
        nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
        nn.ReLU()
    ).cuda()


class BiFPN(nn.Module):
    """
    This is influenced by: https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch/blob/master/BiFPN.py
    And the paper: https://arxiv.org/abs/1911.09070 
    
    Those BiFPN layers are meant to processed multiscaled features, in order to refine the input data. It uses a weighted fusion,
    to learn the optimal relation between the different scales.
    In contrast to the original implementation, the batchnorm comes here AFTER Relu on the BiFPN github, in contrast to the convention.

    """
    def __init__(self,  fpn_sizes, out_channels=256, eps=1e-4, block_num=1):
        super().__init__()

        # The input channels of all 3 different inputs.
        P5_channels, P4_channels, P3_channels = fpn_sizes

        self.block_num = block_num
        self.out_chn = out_channels
        self.eps = eps

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 5
        self.p5_inp_conv = conv_block(P5_channels, self.out_chn)
        self.p5_out_conv = conv_block(self.out_chn, self.out_chn)
        
        self.p5_out_w1  = torch.tensor(1, dtype=torch.float, requires_grad=True, device=self.device)
        self.p5_out_w2  = torch.tensor(1, dtype=torch.float, requires_grad=True, device=self.device)

        # 4
        self.p4_inp_conv = conv_block(P4_channels, self.out_chn)
        self.p4_td_conv = conv_block(self.out_chn, self.out_chn)
        self.p4_out_conv = conv_block(self.out_chn, self.out_chn)

        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # 3
        self.p3_inp_conv = conv_block(P3_channels, self.out_chn)
        self.p3_out_conv = conv_block(self.out_chn, self.out_chn)

        self.p3_out_w1  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        

        if self.block_num == 1:
            self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p3_downsample= nn.MaxPool2d(kernel_size=2)
        
        self.cuda()
        
    def forward(self, in_5, in_4, in_3):

        conv_5_in = self.p5_inp_conv(in_5)  # [4, 256, 200, 176] 
        conv_4_in = self.p4_inp_conv(in_4)  # [4, 320, 200, 176]
        conv_3_in = self.p3_inp_conv(in_3)  # [4, 704, 400, 352]

        # intermediate convs for 3 & 4 (top --> down path)
        dev_fact = 1 / (self.p4_td_w1 + self.p4_td_w2 + self.eps)  # Operation optamization.
        t_in = (self.p4_td_w1 * conv_4_in + self.p4_td_w2 * conv_5_in ) * dev_fact 
        td_4 = self.p4_td_conv(t_in)

        if self.block_num == 1 and not (conv_3_in.shape == td_4.shape):
            resized_td_4 = self.p4_upsample(td_4) 
        else:
            resized_td_4 = td_4
        
        dev_fact = 1 / (self.p3_out_w1 + self.p3_out_w2 + self.eps)
        t_in = (self.p3_out_w1 * conv_3_in + self.p3_out_w2 * resized_td_4) * dev_fact
        out_3 = self.p3_out_conv(t_in)
        if self.block_num == 1 and not out_3.shape == td_4.shape:
            out_3 = self.p3_downsample(out_3)

        # out convs 4 & 5 (down -- > top path)
        dev_fact = 1 / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + self.eps)
        t_in = (self.p4_out_w1 * conv_4_in + self.p4_out_w2 * td_4 + self.p4_out_w3 * out_3) * dev_fact
        out_4 = self.p4_out_conv(t_in)

        dev_fact = 1 / (self.p5_out_w1 + self.p5_out_w2 + self.eps)
        t_in = (self.p5_out_w1 * conv_5_in + self.p5_out_w2 * out_4) * dev_fact
        out_5 = self.p5_out_conv(t_in)

        return out_5, out_4, out_3

class BiFPN_Network(nn.Module):
    """
    A sub-netowrk of BiFPN blocks, to use within a greater architechture.
    """
    def __init__(self, fpn_sizes, out_channels: list = [256], eps=1e-4):
        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_blocks = len(out_channels)
        self.layers = []

        for i in range(self.num_blocks):
            # From the second block on, all inputs would have the same number of channels, as specified in out_channels.
            if i > 0:
                fpn_sizes = [out_channels[i - 1]] * 3
            current = BiFPN(fpn_sizes, out_channels[i], eps, block_num=(i + 1))
            self.layers.append(current)
        
        # It is very important to use nn.ModuleList and not a regular one, since the latter causes an unknown behaviour.
        self.layers = nn.ModuleList(self.layers)
        self.cuda()
        
        
    def forward(self, in_5, in_4, in_3):

        for i in range(self.num_blocks):
            in_5, in_4, in_3 = self.layers[i](in_5, in_4, in_3)

        return in_5, in_4, in_3
    
    def num_of_params(self):
        layers_num_p = sum([x.numel() for p in self.layers for x in p.parameters()])
        return layers_num_p 


class BiFPN_Network_SkipConnections(nn.Module):
    """
    A BiFPN with skip connections. The skip connectios are the highroad from the 3 inputs to a BiFPN layers, to their corresponding 3 outputs.
    """
    def __init__(self, fpn_sizes, out_channels: list = [256], eps=1e-4):
        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_blocks = len(out_channels)
        self.layers = []
        
        # Thie final_convs layers are meant to make sure the outputs of the BiFPN layer match the size of the inputs.
        self.final_convs = nn.ModuleList([nn.ModuleList([conv_block(out_channels[i], fpn_sizes[j], kernel_size=1, padding=0).to(device=self.device) for j in range(len(fpn_sizes))]) for i in range(len(out_channels))])
        for i in range(self.num_blocks):
         
            current = BiFPN(fpn_sizes, out_channels[i], eps, block_num=(i + 1)).to(device=self.device)
            self.layers.append(current)

        self.layers = nn.ModuleList(self.layers)
        self.cuda()
        
    def forward(self, in_5, in_4, in_3):
        for i in range(self.num_blocks):
            inputs = [in_5, in_4, in_3]
            outs = list(self.layers[i](in_5, in_4, in_3))
            for j in range(len(outs)):
                outs[j] = self.final_convs[i][j](outs[j]) + inputs[j].to(device=self.device)
            in_5, in_4, in_3 = outs
        return in_5, in_4, in_3


    def is_same_spatial(self, x1, x2):
        H1, W1 = x1.shape[-2], x1.shape[-1]
        H2, W2 = x2.shape[-2], x2.shape[-1]
        return H1 == H2 and W1 == W2
    
    def is_same_channels(self, x1, x2):
        return x1.shape[-3]  == x2.shape[-3]
    
    def num_of_params(self):
        layers_num_p = sum([x.numel() for p in self.layers for x in p.parameters()])
        skip_convs_p = sum([s.numel() for p in self.final_convs for x in p for s in x.parameters()])
        return layers_num_p + skip_convs_p








