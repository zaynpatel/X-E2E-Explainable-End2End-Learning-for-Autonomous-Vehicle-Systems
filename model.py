import torch
import torchvision
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class CommaNetwork(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        """
        @param:
            in_channels (int): specify the number of input channels for the image
            out_channels (int): specify the number of output channels to be released from the U-Net model
        """

        super(CommaNetwork, self).__init__()
        self.down1 = double_conv(in_channels,64)
        self.down2 = max_down(64,128)
        self.down3 = max_down(128,256)
        self.down4 = max_down(256,512)
        self.down5 = max_down(512,512)
        self.up1 = Upsample(1024,256)
        self.up2 = Upsample(512,128)
        self.up3 = Upsample(256,64)
        self.up4 = Upsample(128,64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        # self.out_conv = nn.Conv2d(out_channels, out_channels, 1)	

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        # x = torch.relu(self.conv1(x))
        out = self.out_conv(x)
        return out

if __name__=="__main__":
    x = torch.randn(1,3,512,1024)
    c = CommaNetwork(3,6)
    print(c(x).shape)
    # summary(c, input_size=(1, 3, 512, 1024))

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet101", 
                     encoder_weights="imagenet",
                    #  activation="softmax", 
                     in_channels=3, 
                     classes=numClasses(),)
    out = model(x)
    out = torch.max(out, dim=1)[1]
    print(out.shape)
