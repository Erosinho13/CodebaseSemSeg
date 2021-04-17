import torch.nn as nn
from torchvision.models import resnet18, resnet101

class ResNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i>=5:
                results.append(x)
        return results
    
    
class ResNet18(ResNet):
    
    def __init__(self):
        super().__init__()
        self.feat_list = list(resnet18(pretrained=True).children())[:-2]
        self.features = nn.ModuleList(self.feat_list).eval()
        self.out_channels = [128, 256, 512]
        
        
class ResNet101(ResNet):
    
    def __init__(self):
        super().__init__()
        self.feat_list = list(resnet101(pretrained=True).children())[:-2]
        self.features = nn.ModuleList(self.feat_list).eval()
        self.out_channels = [512, 1024, 2048]