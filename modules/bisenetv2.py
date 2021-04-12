import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DetailBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.S1 = nn.Sequential(
            BasicBlock(3, 64, kernel_size=3, stride=2),
            BasicBlock(64, 64, kernel_size=3, stride=1),
        )
        self.S2 = nn.Sequential(
            BasicBlock(64, 64, kernel_size=3, stride=2),
            BasicBlock(64, 64, kernel_size=3, stride=1),
            BasicBlock(64, 64, kernel_size=3, stride=1),
        )
        self.S3 = nn.Sequential(
            BasicBlock(64, 128, kernel_size=3, stride=2),
            BasicBlock(128, 128, kernel_size=3, stride=1),
            BasicBlock(128, 128, kernel_size=3, stride=1),
        )

    def forward(self, x):

        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)

        return x


class StemBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = BasicBlock(3, 16, kernel_size=3, stride=2)

        self.left = nn.Sequential(
            BasicBlock(16, 8, kernel_size=1, stride=1, padding=0),
            BasicBlock(8, 16, kernel_size=3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.fuse = BasicBlock(32, 16, kernel_size=3, stride=1)

    def forward(self, x):

        x = self.conv(x)

        x_left = self.left(x)
        x_right = self.right(x)

        x = torch.cat([x_left, x_right], dim=1)
        x = self.fuse(x)

        return x


class GAPooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)


class CEBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.gap_bn = nn.Sequential(
            GAPooling(),
            nn.BatchNorm2d(128)
        )

        self.conv_gap = BasicBlock(
            128, 128, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        feat = self.gap_bn(x)
        feat = self.conv_gap(feat)

        feat = feat + x
        feat = self.conv_last(feat)

        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_channels, out_channels, exp_ratio=6):

        super().__init__()

        mid_channels = in_channels*exp_ratio

        self.conv1 = BasicBlock(in_channels, in_channels,
                                kernel_size=3, stride=1)

        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            #             nn.ReLU(inplace=True), # not shown in paper
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2[1].last_bn = True

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)

        feat = feat + x

        feat = self.relu(feat)

        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_channels, out_channels, exp_ratio=6):

        super().__init__()

        mid_channels = in_channels*exp_ratio

        self.conv1 = BasicBlock(in_channels, in_channels, 3, stride=1)

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            #             nn.ReLU(inplace=True), # not shown in paper
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2[1].last_bn = True

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)

        shortcut = self.shortcut(x)

        feat = feat + shortcut
        feat = self.relu(feat)

        return feat


class SegmentBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.S1S2 = StemBlock()

        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )

        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )

        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        
        self.S5_5 = CEBlock()

    def forward(self, x):

        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)

        return feat2, feat3, feat4, feat5_4, feat5_5
    
    
class BGALayer(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

    def forward(self, x_det, x_sem):
        
        left1 = self.left1(x_det)
        left2 = self.left2(x_det)
        right1 = self.right1(x_sem)
        right2 = self.right2(x_sem)
        
        left = left1*right1
        right = self.up(left2*right2)
        out = self.conv(left+right)
        
        return out


class SegmentHead(nn.Module):

    def __init__(self, in_channels, mid_channels, n_classes, up_factor=8, aux=True):
        
        super().__init__()
        
        self.conv = BasicBlock(in_channels, mid_channels, kernel_size=3, stride=1)
#         self.drop = nn.Dropout(0.1)

        out_channels = n_classes*(up_factor**2)
        
#         if aux:
#             self.conv_out = nn.Sequential(
#                 BasicBlock(mid_channels, up_factor**2, kernel_size=3, stride=1),
#                 nn.Conv2d(up_factor**2, out_channels, kernel_size=1, stride=1, padding=0),
#                 nn.PixelShuffle(up_factor)
#             )
#         else:
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(up_factor)
        )

    def forward(self, x):
        feat = self.conv(x)
#         feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat
    
    
class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, output_aux=True, output_stride=2):
        
        super().__init__()
        
        self.output_aux = output_aux
        
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan?
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        
        if self.output_aux:
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        
        self.test_up = nn.Upsample(scale_factor=output_stride, mode='bilinear', align_corners=True)
        
        self.__init_weights()

    def __init_weights(self):
        
        for _, module in self.named_modules():
            
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                
                if not module.bias is None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                    
                nn.init.zeros_(module.bias)

    def forward(self, x, test=False):
        
        feat_det = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_seg = self.segment(x)
        feat_head = self.bga(feat_det, feat_seg)
        logits = self.head(feat_head)
        
        if self.output_aux and not test:
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        
        if test:
            return self.test_up(logits)
        
        return logits