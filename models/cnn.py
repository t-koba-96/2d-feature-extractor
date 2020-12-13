import torch
import torch.nn as nn
import torchvision.models 
import torch.nn.functional as F

def l2normalize(ten):
    norm = torch.norm(ten, dim=1, keepdim=True)
    return ten / norm


# image feature extractor
class ImageEncoder(nn.Module):
    def __init__(self, cnn_type="resnet50", pretrained=True):
        super(ImageEncoder, self).__init__()
        # get pretrained model
        self.cnn = getattr(torchvision.models, cnn_type)(pretrained)
        # replace final fc layer to output size
        if cnn_type.startswith('vgg'):
            # self.fc = nn.Linear(self.model.classifier._modules['6'].in_features,
            #                    out_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            #self.fc = nn.Linear(self.cnn.fc.in_features, out_size)
            self.cnn.fc = nn.Sequential()

    def forward(self, x):
        out = self.cnn(x)
        #out = self.fc(resout)
        normed_out = l2normalize(out).transpose(1,0)
        return normed_out


# image devided feature extractor
class Resnet_BlockEncoder(nn.Module):
    def __init__(self, cnn_type="resnet50", pretrained=True):
        super(Resnet_BlockEncoder, self).__init__()
        # get pretrained model
        resnet = getattr(torchvision.models, cnn_type)(pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        out = self.cnn(x)
        normed_out = l2normalize(out).view(out.size(0),-1).transpose(1,0)
        return normed_out



#debug
if __name__ == '__main__':
    # batch,sequence,3,224,224
    tensor = torch.randn((8,3,224,224))
    net = ImageEncoder()

    result = net(tensor)
    print(result.shape)
    