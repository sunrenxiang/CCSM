from torch import nn
from models.densenet import densenet161
from models.attention import AttentionBlock
from models.uncertain_model import ConfBlock
import torch.nn.functional as F
from models.layers import SaveFeatures, UnetBlock_, UnetBlock, UnetBlock3d_, UnetBlock3d
import torch


def ComputePara(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k += l
    print("network paramenters:"+str(k))

def dim_tran(x):      
    x = x.permute(1,2,3,0)
    x = x.unsqueeze(0)
    return x
    
class DenseUnet_2d(nn.Module):

    def __init__(self):
        super().__init__()

        base_model = densenet161

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        self.up1 = UnetBlock_(2208,2112,768)
        self.up2 = UnetBlock(768,384,768)
        self.up3 = UnetBlock(384,96,384)
        self.up4 = UnetBlock(96,96,96)

        self.att1 = AttentionBlock(768)
        self.att2 = AttentionBlock(384)
        self.att3 = AttentionBlock(96)
        self.att4 = AttentionBlock(96)

        self.bn1 = nn.BatchNorm2d(64)      
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        self.conf = ConfBlock()

        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
            
        x = F.relu(self.rn(x))  # torch.Size([5, 2208, 7, 7])  torch.Size([1, 1024, 7, 7])
        x = self.up1(x, self.sfs[3].features)  # torch.Size([5, 768, 14, 14])
        x = self.att1(x)
        x = self.up2(x, self.sfs[2].features)  # torch.Size([5, 384, 28, 28])
        x = self.att2(x)
        x = self.up3(x, self.sfs[1].features)  # torch.Size([5, 96, 56, 56])
        x = self.att3(x)
        x = self.up4(x, self.sfs[0].features)  # torch.Size([5, 96, 112, 112])
        x = self.att4(x)

        x_fea = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_fea = self.conv1(x_fea)
        if dropout:
            x_fea = F.dropout2d(x_fea, p=0.3)
        x_fea = F.relu(self.bn1(x_fea))
        
        conf_map = self.conf(x_fea)
        x_out = self.conv2(x_fea)

        return x_out, conf_map

    def close(self):
        for sf in self.sfs: sf.remove()  