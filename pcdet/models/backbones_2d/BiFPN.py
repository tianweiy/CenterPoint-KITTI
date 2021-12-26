import torch.nn as nn
import torch

def conv_block(in_channels, out_channels, kernel_size=3, padding=1, eps=1e-3, momentum=0.01,
 stride=1, bias=False):
    # TODO: think if we need different kernel sizes, and if so - we can use the 1x1 convs.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
         stride=stride, bias=bias),
        nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
        nn.ReLU()
    )


class BiFPN(nn.Module):
    """
    This is influenced by: https://github.com/ViswanathaReddyGajjala/EfficientDet-Pytorch/blob/master/BiFPN.py
    And the paper: https://arxiv.org/abs/1911.09070

    """
    def __init__(self,  fpn_sizes, out_channels=256, eps=1e-4, block_num=1):
        super().__init__()

        P5_channels, P4_channels, P3_channels = fpn_sizes

        self.block_num = block_num
        self.out_chn = out_channels
        self.eps = eps

        # TODO: notice that batchnorm comes here AFTER Relu on the BiFPN github, in contrast to the convention.
        block = conv_block

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 5
        self.p5_inp_conv = block(P5_channels, self.out_chn)
        self.p5_out_conv = block(self.out_chn, self.out_chn)
        
        self.p5_out_w1  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2  = torch.tensor(1, dtype=torch.float, requires_grad=True)

        # 4
        self.p4_inp_conv = block(P4_channels, self.out_chn)
        self.p4_td_conv = block(self.out_chn, self.out_chn)
        self.p4_out_conv = block(self.out_chn, self.out_chn)

        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        
        # 3
        self.p3_inp_conv = block(P3_channels, self.out_chn)
        self.p3_out_conv = block(self.out_chn, self.out_chn)

        self.p3_out_w1  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2  = torch.tensor(1, dtype=torch.float, requires_grad=True)
        

        if self.block_num == 1:
            self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p3_downsample= nn.MaxPool2d(kernel_size=2)
        
    def forward(self, in_5, in_4, in_3):

        # Input convs, we keep the dimensions
        conv_5_in = self.p5_inp_conv(in_5)  # [4, 256, 200, 176] , 4 is batch_size
        conv_4_in = self.p4_inp_conv(in_4)  # [4, 320, 200, 176]
        conv_3_in = self.p3_inp_conv(in_3)  # [4, 704, 400, 352]

        # intermediate convs for 3 & 4 (top --> down path)
        dev_fact = 1 / (self.p4_td_w1 + self.p4_td_w2 + self.eps)       
        t_in = (self.p4_td_w1 * conv_4_in + self.p4_td_w2 * conv_5_in ) * dev_fact
        td_4 = self.p4_td_conv(t_in)

        resized_td_4 = self.p4_upsample(td_4) if self.block_num == 1 else td_4
        dev_fact = 1 / (self.p3_out_w1 + self.p3_out_w2 + self.eps)
        t_in = (self.p3_out_w1 * conv_3_in + self.p3_out_w2 * resized_td_4) * dev_fact
        out_3 = self.p3_out_conv(t_in)
        if self.block_num == 1:
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
    def __init__(self, fpn_sizes, out_channels: list = [256], eps=1e-4):
        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_blocks = len(out_channels)
        self.layers = []

        for i in range(self.num_blocks):
            if i > 0:
                fpn_sizes = [out_channels[i - 1]] * 3

            current = BiFPN(fpn_sizes, out_channels[i], eps, block_num=(i + 1)).to(device=self.device)
            self.layers.append(current)
            
        

    def forward(self, in_5, in_4, in_3):
        

        # TODO: Get rid of loop by packing inputs into tuple
        for i in range(self.num_blocks):
            in_5, in_4, in_3 = self.layers[i](in_5, in_4, in_3)

        return in_5, in_4, in_3







