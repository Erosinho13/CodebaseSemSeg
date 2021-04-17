import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import get_activation_layer


class ConvBlock(nn.Module):

    """
    use_bn     : bool                    | Whether to use BatchNorm layer.
    bn_eps     : float                   | Small float added to variance in Batch norm.
    activation : function or str or None | Activation function or name of activation function.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 groups=1, bias=False, use_bn=True, bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
    
        super().__init__()
        
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
            
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        
        if self.use_pad:
            x = self.pad(x)
        
        x = self.conv(x)
        
        if self.use_bn:
            x = self.bn(x)
        
        if self.activate:
            x = self.activ(x)
        
        return x

    
def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                     stride=stride, groups=groups, bias=bias)


def conv1x1_block(in_channels, out_channels, stride=1, padding=0, groups=1, bias=False,
                  use_bn=True, bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):    
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                     stride=stride, padding=padding, groups=groups, bias=bias,
                     use_bn=use_bn, bn_eps=bn_eps, activation=activation)


def conv3x3_block(in_channels, out_channels, stride=1, padding=1, dilation=1,
                  groups=1, bias=False, use_bn=True, bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=bias, use_bn=use_bn, bn_eps=bn_eps, activation=activation)

class InterpolationBlock(nn.Module):

    """
    Interpolation upsampling block.
    ----------
    scale_factor  : int            | Multiplier for spatial size.
    out_size      : tuple of 2 int | Spatial size of the output tensor for the bilinear
                                     interpolation operation.
    mode          : str            | Algorithm used for upsampling.
    align_corners : bool           | Whether to align the corner pixels of the input and
                                     output tensors.
    up            : bool           | Whether to upsample or downsample.
    """

    def __init__(self, scale_factor, out_size=None, mode="bilinear", align_corners=True, up=True):

        super().__init__()

        self.scale_factor = scale_factor
        self.out_size = out_size
        self.mode = mode
        self.align_corners = align_corners
        self.up = up

    def calc_out_size(self, x):
        if self.out_size is not None:
            return self.out_size
        if self.up:
            return tuple(s * self.scale_factor for s in x.shape[2:])
        else:
            return tuple(s // self.scale_factor for s in x.shape[2:])

    def forward(self, x, size=None):
        if (self.mode == "bilinear") or (size is not None):
            out_size = self.calc_out_size(x) if size is None else size
            return F.interpolate(input=x, size=out_size, mode=self.mode,
                                 align_corners=self.align_corners)
        return F.interpolate(input=x, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)

    def __repr__(self):
        s = '{name}(scale_factor={scale_factor}, out_size={out_size},' + \
            'mode={mode}, align_corners={align_corners}, up={up})'  # noqa
        return s.format(name=self.__class__.__name__, scale_factor=self.scale_factor,
                        out_size=self.out_size, mode=self.mode,
                        align_corners=self.align_corners, up=self.up)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        if self.mode == "bilinear":
            num_flops = 9 * x.numel()
        else:
            num_flops = 4 * x.numel()
        num_macs = 0
        return num_flops, num_macs


class PyramidPoolingZeroBranch(nn.Module):

    """
    Pyramid pooling zero branch.
    ----------
    in_size      : tuple of 2 int | Spatial size of output image for the upsampling operation.
    """

    def __init__(self, in_channels, out_channels, in_size):

        super().__init__()

        self.in_size = in_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = conv1x1_block(
            in_channels=in_channels, out_channels=out_channels)
        self.up = InterpolationBlock(
            scale_factor=None, mode="nearest", align_corners=None)

    def forward(self, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.up(x, size=in_size)
        return x


class AttentionRefinementBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = conv3x3_block(
            in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = conv1x1_block(in_channels=out_channels, out_channels=out_channels,
                                   activation=(lambda: nn.Sigmoid()))

    def forward(self, x):
        x = self.conv1(x)
        w = self.pool(x)
        w = self.conv2(w)
        x = x * w
        return x


class PyramidPoolingMainBranch(nn.Module):

    """
    Pyramid pooling main branch
    ----------
    scale_factor : float |Multiplier for spatial size.
    """

    def __init__(self, in_channels, out_channels, scale_factor):

        super().__init__()

        self.att = AttentionRefinementBlock(
            in_channels=in_channels, out_channels=out_channels)
        self.up = InterpolationBlock(
            scale_factor=scale_factor, mode="nearest", align_corners=None)
        self.conv = conv3x3_block(
            in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, y):
        x = self.att(x)
        x = x + y
        x = self.up(x)
        x = self.conv(x)
        return x


class FeatureFusion(nn.Module):

    """
    Feature fusion block
    ----------
    reduction    : int  | Squeeze reduction value.
    """

    def __init__(self, in_channels, out_channels, reduction=4):

        super().__init__()

        mid_channels = out_channels // reduction

        self.conv_merge = conv1x1_block(
            in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv1x1(in_channels=out_channels,
                             out_channels=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(in_channels=mid_channels,
                             out_channels=out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.conv_merge(x)
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x_att = x * w
        x = x + x_att
        return x


class PyramidPooling(nn.Module):

    """
    Pyramid Pooling module
    ----------
    x16_in_channels : int            | Number of input channels for x16.
    x32_in_channels : int            | Number of input channels for x32.
    y_out_channels  : int            | Number of output channels for y-outputs.
    y32_out_size    : tuple of 2 int | Spatial size of the y32 tensor.
    """

    def __init__(self, x16_in_channels, x32_in_channels, y_out_channels, y32_out_size):

        super().__init__()

        z_out_channels = 2 * y_out_channels

        self.pool32 = PyramidPoolingZeroBranch(in_channels=x32_in_channels,
                                               out_channels=y_out_channels, in_size=y32_out_size)
        self.pool16 = PyramidPoolingMainBranch(in_channels=x32_in_channels,
                                               out_channels=y_out_channels, scale_factor=2)
        self.pool8 = PyramidPoolingMainBranch(in_channels=x16_in_channels,
                                              out_channels=y_out_channels, scale_factor=2)
        self.fusion = FeatureFusion(
            in_channels=z_out_channels, out_channels=z_out_channels)

    def forward(self, x8, x16, x32):
        y32 = self.pool32(x32)
        y16 = self.pool16(x32, y32)
        y8 = self.pool8(x16, y16)
        z8 = self.fusion(x8, y8)
        return z8, y8, y16


class BiSeHead(nn.Module):

    """
    BiSeNet head (final) block
    """

    def __init__(self, in_channels, mid_channels, out_channels):

        super().__init__()

        self.conv1 = conv3x3_block(
            in_channels=in_channels, out_channels=mid_channels)
        self.conv2 = conv1x1(in_channels=mid_channels,
                             out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    
class MultiOutputSequential(nn.Sequential):

    """
    A sequential container with multiple outputs
    Modules will be executed in the order they are added
    ----------
    multi_output : bool | Whether to return multiple output
    dual_output  : bool | Whether to return dual output
    return_last  : bool | Whether to forcibly return last value
    """

    def __init__(self, multi_output=True, dual_output=False, return_last=True):
        
        super().__init__()
        
        self.multi_output = multi_output
        self.dual_output = dual_output
        self.return_last = return_last

    def forward(self, x):

        outs = []

        for module in self._modules.values():

            x = module(x)

            if hasattr(module, "do_output") and module.do_output:
                outs.append(x)
            elif hasattr(module, "do_output2") and module.do_output2:
                assert (type(x) == tuple)
                outs.extend(x[1])
                x = x[0]

        if self.multi_output:
            return [x] + outs if self.return_last else outs
        if self.dual_output:
            return x, outs
        return x


class BiSeNetV1(nn.Module):
    """
    backbone    : func -> nn.Sequential | Feature extractor
    aux         : bool                  | Whether to output an auxiliary results
    fixed_size  : bool                  | Whether to expect fixed spatial size of input image
    in_channels : int                   | Number of input channels.
    in_size     : tuple of two ints     | Spatial size of the expected input image
    num_classes : int                   | Number of classification classes
    """

    def __init__(self, backbone, aux=True, fixed_size=True, in_channels=3,
                 in_size=(640, 480), num_classes=19):

        super().__init__()

        assert (in_channels == 3)

        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux
        self.fixed_size = fixed_size

        
        backbone_out_channels = backbone.out_channels
        self.backbone = backbone

        y_out_channels = backbone_out_channels[0]
        z_out_channels = 2 * y_out_channels
        y32_out_size = (
            self.in_size[0] // 32, self.in_size[1] // 32) if fixed_size else None
        self.pool = PyramidPooling(x16_in_channels=backbone_out_channels[1],
                                   x32_in_channels=backbone_out_channels[2],
                                   y_out_channels=y_out_channels, y32_out_size=y32_out_size)
        self.head_z8 = BiSeHead(in_channels=z_out_channels, mid_channels=z_out_channels,
                                out_channels=num_classes)
        self.up8 = InterpolationBlock(scale_factor=(8 if fixed_size else None))

        if self.aux:
            mid_channels = y_out_channels // 2
            self.head_y8 = BiSeHead(in_channels=y_out_channels, mid_channels=mid_channels,
                                    out_channels=num_classes)
            self.head_y16 = BiSeHead(in_channels=y_out_channels, mid_channels=mid_channels,
                                     out_channels=num_classes)
            self.up16 = InterpolationBlock(scale_factor=(16 if fixed_size else None))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        
        assert (x.shape[2] % 32 == 0) and (x.shape[3] % 32 == 0)

        x8, x16, x32 = self.backbone(x)
        z8, y8, y16 = self.pool(x8, x16, x32)

        z8 = self.head_z8(z8)
        z8 = self.up8(z8)

        if self.aux:
            y8 = self.head_y8(y8)
            y16 = self.head_y16(y16)
            y8 = self.up8(y8)
            y16 = self.up16(y16)
            return z8, y8, y16
        else:
            return z8