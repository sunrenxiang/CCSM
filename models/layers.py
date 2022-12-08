
from torch import nn
import torch
import torch.nn.functional as F

class UnetBlock(nn.Module):
    def __init__(self, up_in1, up_out, size):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(up_out)

        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        cat_p = torch.add(up_p, x_p)

        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))
                
        return cat_p   

class UnetBlock3d(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv3d(up_in1+up_in2, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm3d(up_out)


    def forward(self, up_p, x_p):

        n,c,rows,cols,deps = x_p.shape
        
        up_p = F.upsample(up_p, size=(rows,cols,deps), mode='trilinear')
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))
                
        return cat_p   

class UnetBlock_(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv2d(up_in2, up_in1, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(up_out)

        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.xavier_normal_(self.x_conv_.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):

        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)

        x_p = self.x_conv_(x_p)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p

class UnetBlock3d_(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv3d(up_in1*2, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv3d(up_in2, up_in1, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm3d(up_out)


    def forward(self, up_p, x_p):

        n,c,rows,cols,deps = x_p.shape
        
        up_p = F.upsample(up_p, size=(rows,cols,deps), mode='trilinear')
        x_p = self.x_conv_(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))
                
        return cat_p 

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()
    
