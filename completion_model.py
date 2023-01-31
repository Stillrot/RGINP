from munch import Munch
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from models.forwardAttentionLayer import ForwardAttention
from models.reverseAttentionLayer import ReverseAttention, ReverseMaskConv

def build_model(args):
    lBAM_generator = LBAMModel(args.img_size, 3, 3, args.n_attrs)
    discriminator = Discriminator(args)
    classifier = aux_Classifier(args)
    label_predict = VGG_Net(args.n_attrs)
    nets = Munch(LBAM_generator=lBAM_generator, discriminator=discriminator, classifier=classifier, label_predict=label_predict)
    return nets

class LBAMModel(nn.Module):
    """
    https://github.com/Vious/LBAM_Pytorch
    """
    def __init__(self, img_size, inputChannels, outputChannels, n_attrs, max_conv_dim=512):
        super(LBAMModel, self).__init__()
        self.n_attrs = n_attrs
        self.img_size = img_size
        repeat_num = int(np.log2(img_size)) - 1
        self.ec1 = ForwardAttention(inputChannels, 64, bn = False)
        self.ec2 = ForwardAttention(64, 128)
        self.ec3 = ForwardAttention(128, 256)
        self.ec4 = ForwardAttention(256, 512)

        for i in range(5, repeat_num + 1):
            name = 'ec{:d}'.format(i)
            setattr(self, name, ForwardAttention(512, 512))
        
        dim_in = 64
        self.reverseConv1 = ReverseMaskConv(3, dim_in)

        for i in range(2, repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            name = 'reverseConv{:d}'.format(i)
            setattr(self, name, ReverseMaskConv(dim_in, dim_out))
            dim_in = dim_out

        self.dc1 = ReverseAttention(dim_in + n_attrs, dim_in, bnChannels = 1024)
        for i in range(2, repeat_num - 3):
            name = 'dc{:d}'.format(i)
            setattr(self, name, ReverseAttention(dim_in * 2, dim_in, bnChannels=1024))
        
        name = 'dc{:d}'.format(repeat_num - 3)
        setattr(self, name, ReverseAttention(dim_in * 2, 256, bnChannels = 512))

        name = 'dc{:d}'.format(repeat_num - 2)
        setattr(self, name, ReverseAttention(256 * 2, 128, bnChannels = 256))

        name = 'dc{:d}'.format(repeat_num - 1)
        setattr(self, name, ReverseAttention(128 * 2, 64, bnChannels = 128))

        name = 'dc{:d}'.format(repeat_num)
        setattr(self, name, nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False))

    def enc(self, inputImgs, masks):
        ef = []
        mu = []
        skipConnect = []
        forwardMap = []
        reverseMap = []
        revMu = []

        ef1, mu1, skipConnect1, forwardMap1 = self.ec1(inputImgs, masks)
        ef.append(ef1), mu.append(mu1), skipConnect.append(skipConnect1), forwardMap.append(forwardMap1)

        ef2, mu2, skipConnect2, forwardMap2 = self.ec2(ef1, mu1)
        ef.append(ef2), mu.append(mu2), skipConnect.append(skipConnect2), forwardMap.append(forwardMap2)

        ef3, mu3, skipConnect3, forwardMap3 = self.ec3(ef2, mu2)
        ef.append(ef3), mu.append(mu3), skipConnect.append(skipConnect3), forwardMap.append(forwardMap3)

        ef4, mu4, skipConnect4, forwardMap4 = self.ec4(ef3, mu3)
        ef.append(ef4), mu.append(mu4), skipConnect.append(skipConnect4), forwardMap.append(forwardMap4)

        ef5, mu5, skipConnect5, forwardMap5 = self.ec5(ef4, mu4)
        ef.append(ef5), mu.append(mu5), skipConnect.append(skipConnect5), forwardMap.append(forwardMap5)

        ef6, mu6, skipConnect6, forwardMap6 = self.ec6(ef5, mu5)
        ef.append(ef6), mu.append(mu6), skipConnect.append(skipConnect6), forwardMap.append(forwardMap6)

        ef7, _, _, _ = self.ec7(ef6, mu6)
        ef.append(ef7)


        reverseMap1, revMu1 = self.reverseConv1(1 - masks)
        reverseMap.append(reverseMap1), revMu.append(revMu1)

        reverseMap2, revMu2 = self.reverseConv2(revMu1)
        reverseMap.append(reverseMap2), revMu.append(revMu2)

        reverseMap3, revMu3 = self.reverseConv3(revMu2)
        reverseMap.append(reverseMap3), revMu.append(revMu3)

        reverseMap4, revMu4 = self.reverseConv4(revMu3)
        reverseMap.append(reverseMap4), revMu.append(revMu4)

        reverseMap5, revMu5 = self.reverseConv5(revMu4)
        reverseMap.append(reverseMap5), revMu.append(revMu5)

        reverseMap6, _ = self.reverseConv6(revMu5)
        reverseMap.append(reverseMap6)

        return Munch(ef = ef, mu = mu, skipConnect = skipConnect, forwardMap = forwardMap,
                reverseMap = reverseMap, revMu = revMu)
    
    def dec(self, inputImgs, masks, label, enc_feature):
        ef7 = enc_feature.ef[6]
        label_tile = label.view(label.size(0), -1, 1, 1).repeat(1, 1, ef7.size(2), ef7.size(2))

        forwardMap6 = enc_feature.forwardMap[5]
        reverseMap6 = enc_feature.reverseMap[5]
        skipConnect6 = enc_feature.skipConnect[5]
        ef7 = torch.cat([ef7, label_tile.cuda()], dim=1)
        concatMap6 = torch.cat((forwardMap6, reverseMap6), 1)
        dcFeatures1 = self.dc1(skipConnect6, ef7, concatMap6)

        forwardMap5 = enc_feature.forwardMap[4]
        reverseMap5 = enc_feature.reverseMap[4]
        skipConnect5 = enc_feature.skipConnect[4]
        concatMap5 = torch.cat((forwardMap5, reverseMap5), 1)
        dcFeatures2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)

        forwardMap4 = enc_feature.forwardMap[3]
        reverseMap4 = enc_feature.reverseMap[3]
        skipConnect4 = enc_feature.skipConnect[3]
        concatMap4 = torch.cat((forwardMap4, reverseMap4), 1)
        dcFeatures3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)

        forwardMap3 = enc_feature.forwardMap[2]
        reverseMap3 = enc_feature.reverseMap[2]
        skipConnect3 = enc_feature.skipConnect[2]
        concatMap3 = torch.cat((forwardMap3, reverseMap3), 1)
        dcFeatures4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)

        forwardMap2 = enc_feature.forwardMap[1]
        reverseMap2 = enc_feature.reverseMap[1]
        skipConnect2 = enc_feature.skipConnect[1]
        concatMap2 = torch.cat((forwardMap2, reverseMap2), 1)
        dcFeatures5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)

        forwardMap1 = enc_feature.forwardMap[0]
        reverseMap1 = enc_feature.reverseMap[0]
        skipConnect1 = enc_feature.skipConnect[0]
        concatMap1 = torch.cat((forwardMap1, reverseMap1), 1)
        dcFeatures6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)

        dcFeatures7 = self.dc7(dcFeatures6)

        return dcFeatures7

    def forward(self, inputImgs, masks, label, enc_feature, mode='enc_dec'):
        if mode == 'enc':
            enc_feature = self.enc(inputImgs, masks)
            return enc_feature

        if mode == 'dec':
            return self.dec(inputImgs, masks, label, enc_feature)

        if mode == 'enc_dec':
            enc_feature = self.enc(inputImgs, masks)
            return self.dec(inputImgs, masks, label, enc_feature)

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        inputChannels = 3
        self.args = args
        use_spectral_norm = True
        self.globalConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2 , inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_finalLayer = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fusionLayer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, batches):
        globalFt = self.globalConv(batches)
        globalFt = self.global_finalLayer(globalFt)

        return self.fusionLayer(globalFt).view(batches.size()[0], -1)

class aux_Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        inputChannels = 3
        self.args = args
        use_spectral_norm = True
        self.globalConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2 , inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_finalLayer = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_cls = nn.Sequential(
            nn.Conv2d(512, args.n_attrs, kernel_size=4)
        )

    def forward(self, batches):
        globalFt = self.globalConv(batches)
        globalFt = self.global_finalLayer(globalFt)

        return self.fc_cls(globalFt).view(batches.size()[0], -1)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class VGG_Net(nn.Module):
    def __init__(self, labels_num):
        super().__init__()
        
        self.vgg_ = models.vgg16(pretrained=False)
        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.Linear(1000, 512),
            nn.Linear(512, 256),
            nn.Linear(256, labels_num),
            nn.Sigmoid()
        )

    def forward(self, batches):
        x_ = self.vgg_(batches)
        label = self.fc_layers(x_)
        return label
