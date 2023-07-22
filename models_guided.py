import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import numpy as np


class CA(nn.Module):
    def __init__(self, channel):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PA(nn.Module):
    def __init__(self, channel):
        super(PA, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class Soft(nn.Module):
    def __init__(self, features):
        super(Soft, self).__init__()
        self.S1 = nn.Parameter(torch.FloatTensor(features, features))
        self.S2 = nn.Parameter(torch.FloatTensor(features, features))
        init.xavier_uniform_(self.S1)
        init.xavier_uniform_(self.S2)

    def forward(self, x):
        out = torch.matmul(torch.matmul(self.S1, x).permute(0, 2, 1), torch.matmul(self.S2, x))
        out = F.softmax(out, dim=2)
        return out


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GConv(nn.Module):
    def __init__(self, num_state, agg):
        super(GConv, self).__init__()
        self.num_state = num_state
        self.weight = nn.Parameter(
            torch.FloatTensor(num_state, num_state))
        self.agg = agg()
        init.xavier_uniform_(self.weight)

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.num_state)
        agg_feats = self.agg(features, A)
        out = torch.einsum('bnd,df->bnf', (agg_feats, self.weight))
        out = F.relu(out)
        return out


class gcn(nn.Module):
    def __init__(self, num_state):
        super(gcn, self).__init__()
        self.conv = GConv(num_state, MeanAggregator)

    def forward(self, x, A, train=True):
        x = self.conv(x, A)
        return x


class DGR(nn.Module):
    def __init__(self, in_channel=256, state_channel=256, node_num=128, normalize=True):
        super(DGR, self).__init__()
        self.normalize = normalize
        self.state = state_channel
        self.node_num = node_num
        self.conv1 = nn.Conv2d(in_channel, self.state, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel, self.state, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(int(self.node_num ** 0.5))
        self.conv3 = Soft(self.state)
        self.gcn = gcn(self.state)
        self.conv4 = nn.Conv2d(self.state, in_channel, kernel_size=1, bias=False)
        self.BN = nn.BatchNorm2d(in_channel, eps=1e-04)

    def forward(self, x):
        Batch = x.size(0)
        K = self.conv1(x).view(Batch, self.state, -1)
        proj = self.conv2(x)
        C = self.pool(proj).reshape(Batch, self.state, -1)
        P = C.permute(0, 2, 1)
        T = proj.reshape(Batch, self.state, -1)
        B = torch.matmul(P, T)
        B = F.softmax(B, dim=1)
        D = B
        V = torch.matmul(K, B.permute(0, 2, 1))
        if self.normalize:
            V = V * (1. / K.size(2))
        adj = self.conv3(V)
        V_rel = self.gcn(V.permute(0, 2, 1), adj)
        Y = torch.matmul(V_rel.permute(0, 2, 1), D)
        Y = Y.view(Batch, self.state, *x.size()[2:])
        out = x + self.BN(self.conv4(Y))
        return out


class RGB_to_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_to_HSV, self).__init__()
        self.eps = eps

    def forward(self, im):
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)
        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6
        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6
        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0
        value = img.max(1)[0]
        hsv = torch.cat([hue, saturation, value], dim=0)
        hsv = hsv.unsqueeze(0)
        return hsv


class RGB_to_Lab(nn.Module):
    def __init__(self):
        super(RGB_to_Lab, self).__init__()

    def forward(self, x):
        x1 = x.view(-1, 3, x.size(-2), x.size(-1))
        lab = torch.cat([F.relu(5.0 * (x1[:, 0] + 0.0193) / 1.0588),
                         F.relu(5.0 * (x1[:, 1] + 0.0146) / 1.0588),
                         F.relu(5.0 * (x1[:, 2] + 0.0173) / 1.0588)], dim=-1)
        lab = lab.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return lab


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True)]
                        #nn.ReflectionPad2d(1),
                        #nn.Conv2d(in_features, in_features, 3),
                        #nn.InstanceNorm2d(in_features)  ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)





class Generator_S2F(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator_S2F, self).__init__()
        # Initial convolution block
        self.shadow_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.HSV_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.Lab_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.RGB_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.down5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.down6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.down7 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down8 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        
        self.SP = DGR(in_channel=256, state_channel=256, node_num=128)
        self.ca = CA(256)
        self.pa = PA(256)
        self.rgb2hsv = RGB_to_HSV()
        self.rgb2lab = RGB_to_Lab()

        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(256)
        self.resblock3 = ResidualBlock(256)
        self.resblock4 = ResidualBlock(128)
        self.resblock5 = ResidualBlock(128)
        self.resblock6 = ResidualBlock(128)
        self.resblock7 = ResidualBlock(128)
        self.resblock8 = ResidualBlock(256)
        self.resblock9 = ResidualBlock(256)
        self.resblock10 = ResidualBlock(256)
        self.resblock11 = ResidualBlock(128)
        self.resblock12 = ResidualBlock(128)
        self.resblock13 = ResidualBlock(128)
        self.conv = nn.Conv2d(6, 3, kernel_size = 3, padding=1)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        
        # Output layer
        self.shadow_final_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7)
            )
        self.color_final_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7)
            )
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        ########################################################################
        s1 = self.shadow_initial_convolution_block(x)          #[1,64,400,400]
        s2 = self.down1(s1)                                    #[1,128,200,200]
        s3 = self.resblock1(s2)                                #[1,128,200,200]
        s4 = self.down4(s3)                                    #[1,256,100,100]
        s5 = self.resblock2(s4)                                #[1,256,100,100]
        s6 = self.SP(s5)                                       #[1,256,100,100]
        s7 = self.resblock3(s6)                                #[1,256,100,100]
        s8 = self.up1(s7)                                      #[1,128,200,200]
        s9 = self.resblock4(s8)                                #[1,128,200,200]
        s10 = self.up4(s9)                                     #[1,64,400,400]
        s_final = self.shadow_final_convolution_block(s10)     #[1,3,400,400]
        ########################################################################
        HSV1 = self.rgb2hsv(x)                           #[1,3,400,400]
        HSV2 = self.HSV_initial_convolution_block(HSV1)        #[1,64,400,400]
        Lab1 = self.rgb2lab(x)                           #[1,3,400,400]
        Lab2 = self.Lab_initial_convolution_block(Lab1)        #[1,64,400,400]
        RGB1 = self.RGB_initial_convolution_block(x)     #[1,64,400,400]
        RGB2 = HSV2 + Lab2 + RGB1                              #[1,64,400,400]

        HSV3 = self.down2(HSV2)                                #[1,128,200,200]
        Lab3 = self.down3(Lab2)                                #[1,128,200,200]
        RGB3 = self.down7(RGB2)                                #[1,128,200,200]
        RGB4 = HSV3 + Lab3 + RGB3                              #[1,128,200,200]

        HSV4 = self.down5(HSV3)                                #[1,256,100,100]
        Lab4 = self.down6(Lab3)                                #[1,256,100,100]
        RGB5 = self.down8(RGB4)                                #[1,256,100,100]
        RGB6 = HSV4 + Lab4 + RGB5                              #[1,256,100,100]
        RGB6 = self.ca(RGB6)
        RGB6 = self.pa(RGB6)

        HSV5 = self.up1(HSV4)                                  #[1,128,200,200]
        Lab5 = self.up2(Lab4)                                  #[1,128,200,200]
        RGB7 = self.up3(RGB6)                                  #[1,128,200,200]
        RGB8 = HSV5 + Lab5 + RGB7                              #[1,128,200,200]

        HSV6 = self.up4(HSV5)                                  #[1,64,400,400]
        Lab6 = self.up5(Lab5)                                  #[1,64,400,400]
        RGB9 = self.up6(RGB8)                                  #[1,64,400,400]
        RGB10 = HSV6 + Lab6 + RGB9                             #[1,64,400,400]
        c_final = self.color_final_convolution_block(RGB10)    #[1,3,400,400]
        ########################################################################
        final = torch.cat([s_final, c_final], dim=1)           #[1,6,400,400]
        final = self.conv(final)                               #[1,3,400,400]
        final_weight = self.sigmoid(final)
        c_final = final_weight * c_final
        out = c_final + s_final
        return (out + x).tanh() #(min=-1, max=1) #just learn a residual


class Generator_F2S(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator_F2S, self).__init__()
        # Initial convolution block
        self.shadow_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc+1, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.HSV_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.Lab_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.RGB_initial_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.down5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.down6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.down7 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.down8 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        
        self.SP = DGR(in_channel=256, state_channel=256, node_num=128)
        self.ca = CA(256)
        self.pa = PA(256)
        self.rgb2hsv = RGB_to_HSV()
        self.rgb2lab = RGB_to_Lab()

        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(256)
        self.resblock3 = ResidualBlock(256)
        self.resblock4 = ResidualBlock(128)
        self.resblock5 = ResidualBlock(128)
        self.resblock6 = ResidualBlock(128)
        self.resblock7 = ResidualBlock(128)
        self.resblock8 = ResidualBlock(256)
        self.resblock9 = ResidualBlock(256)
        self.resblock10 = ResidualBlock(256)
        self.resblock11 = ResidualBlock(128)
        self.resblock12 = ResidualBlock(128)
        self.resblock13 = ResidualBlock(128)
        self.conv = nn.Conv2d(6, 3, kernel_size = 3, padding=1)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
            )
        
        # Output layer
        self.shadow_final_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7)
            )
        self.color_final_convolution_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7)
            )
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, mask):
        ########################################################################
        x1 = torch.cat([x, mask], dim=1)                       #[1,4,400,400]
        s1 = self.shadow_initial_convolution_block(x1)          #[1,64,400,400]
        s2 = self.down1(s1)                                    #[1,128,200,200]
        s3 = self.resblock1(s2)                                #[1,128,200,200]
        s4 = self.down4(s3)                                    #[1,256,100,100]
        s5 = self.resblock2(s4)                                #[1,256,100,100]
        s6 = self.SP(s5)                                       #[1,256,100,100]
        s7 = self.resblock3(s6)                                #[1,256,100,100]
        s8 = self.up1(s7)                                      #[1,128,200,200]
        s9 = self.resblock4(s8)                                #[1,128,200,200]
        s10 = self.up4(s9)                                     #[1,64,400,400]
        s_final = self.shadow_final_convolution_block(s10)     #[1,3,400,400]
        ########################################################################
        HSV1 = self.rgb2hsv(x)                           #[1,3,400,400]
        HSV2 = self.HSV_initial_convolution_block(HSV1)        #[1,64,400,400]
        Lab1 = self.rgb2lab(x)                           #[1,3,400,400]
        Lab2 = self.Lab_initial_convolution_block(Lab1)        #[1,64,400,400]
        RGB1 = self.RGB_initial_convolution_block(x)     #[1,64,400,400]
        RGB2 = HSV2 + Lab2 + RGB1                              #[1,64,400,400]

        HSV3 = self.down2(HSV2)                                #[1,128,200,200]
        Lab3 = self.down3(Lab2)                                #[1,128,200,200]
        RGB3 = self.down7(RGB2)                                #[1,128,200,200]
        RGB4 = HSV3 + Lab3 + RGB3                              #[1,128,200,200]

        HSV4 = self.down5(HSV3)                                #[1,256,100,100]
        Lab4 = self.down6(Lab3)                                #[1,256,100,100]
        RGB5 = self.down8(RGB4)                                #[1,256,100,100]
        RGB6 = HSV4 + Lab4 + RGB5                              #[1,256,100,100]
        RGB6 = self.ca(RGB6)
        RGB6 = self.pa(RGB6)

        HSV5 = self.up1(HSV4)                                  #[1,128,200,200]
        Lab5 = self.up2(Lab4)                                  #[1,128,200,200]
        RGB7 = self.up3(RGB6)                                  #[1,128,200,200]
        RGB8 = HSV5 + Lab5 + RGB7                              #[1,128,200,200]

        HSV6 = self.up4(HSV5)                                  #[1,64,400,400]
        Lab6 = self.up5(Lab5)                                  #[1,64,400,400]
        RGB9 = self.up6(RGB8)                                  #[1,64,400,400]
        RGB10 = HSV6 + Lab6 + RGB9                             #[1,64,400,400]
        c_final = self.color_final_convolution_block(RGB10)    #[1,3,400,400]
        ########################################################################
        final = torch.cat([s_final, c_final], dim=1)           #[1,6,400,400]
        final = self.conv(final)                               #[1,3,400,400]
        final_weight = self.sigmoid(final)
        c_final = final_weight * c_final
        out = c_final + s_final
        return (out + x).tanh() #(min=-1, max=1) #just learn a residual


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]
        
        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)