import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, vgg16


# ResNet 
class ResNet(nn.Module):
    def __init__(self, model, spatial_feature = False, pooling = "average"):
        super(ResNet, self).__init__()
        # pretrained model
        if model == "resnet50":
            resnet = resnet50(pretrained=True)
        elif model == "resnet101":
            resnet = resnet101(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # spatial_feature
        self.spatial_feature = spatial_feature

        # pooling
        if pooling == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1,1))
        elif pooling == "average":
            self.pooling =  nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):
        x=self.cnn(x)
        if self.spatial_feature:
            return x
        else:
            x = self.pooling(x)       
            x = x.view(x.size(0) , -1)    
            return x


# ResNet 
class VGG(nn.Module):
    def __init__(self, model, spatial_feature = False):
        super(VGG, self).__init__()
        # pretrained model
        if model == "vgg16":
            vgg = vgg16(pretrained=True)
        self.cnn = nn.Sequential(*list(vgg.children())[:-2])

        # spatial_feature
        self.spatial_feature = spatial_feature

        #pooling
        self.pooling = list(vgg.children())[-2]
        
    def forward(self, x):
        x=self.cnn(x)
        if self.spatial_feature:
            return x
        else:
            x = self.pooling(x)       
            x = x.view(x.size(0) , -1)    
            return x




#debug
if __name__ == '__main__':
    # batch,sequence,3,224,224
    tensor = torch.randn((8,3,224,224))

    net1 = ResNet("resnet101", spatial_feature = True, pooling = "max")
    result1 = net1(tensor)
    print(result1.size())

    net2 = ResNet("vgg16", spatial_feature = True)
    result2 = net2(tensor)
    print(result2.size())
    