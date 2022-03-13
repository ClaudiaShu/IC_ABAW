import numpy as np
import torch.nn

from models import *
from utils import *

import pickle
import torchvision

class BaselineHRNet(nn.Module):
    def __init__(self, num_classes):
        super(BaselineHRNet, self).__init__()
        self.trunk1 = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        self.trunk2 = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.landmark = get_model_by_name('COFW', device='cuda')

        self.fc1 = nn.Linear(in_features=29*28*28, out_features=1000)
        self.BN1 = nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()

        self.BN2 = nn.BatchNorm1d(2000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=2000, out_features=num_classes)

    def forward(self, x):
        out1 = self.backbone(x)
        lm = self.landmark(x)
        bz, nj, hm_h, hm_w = lm.shape # batchsize, num_joints
        lm = lm.reshape(bz, nj*hm_h*hm_w)
        # out2 = self.relu1(self.BN1(self.fc1(lm)))
        out2 = self.fc1(lm)
        out3 = torch.cat([out1, out2], dim=1)
        out = self.fc2(self.relu2(self.BN2(out3)))
        return out


class BaseLineRNN(nn.Module):
    def __init__(self, num_classes=8, module='R3D', n_features=512,
                 hidden_size=512, num_layers=2, drop_gru=0.4,
                 embed_dim=512, num_heads=2, drop_att=0.7):
        super(BaseLineRNN, self).__init__()
        '''
        r3d_18 & mc3_18: 
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400

        N → number of sequences (mini batch)
        Cin → number of channels (3 for rgb)
        D → Number of images in a sequence
        H → Height of one image in the sequence
        W → Width of one image in the sequence
        '''
        self.baseline = torchvision.models.resnet18(pretrained=True).cuda()
        self.baseline.fc = nn.Identity()
        # pretrained on DEFW
        if module == "R3D":
            pretrainedr3d = torch.load(
                "/data/users/ys221/data/pretrain/resnet/r3d_18/r3d_18_fold3_epo144_UAR45.69_WAR56.92.pth")
            self.backbone = torchvision.models.video.resnet.r3d_18(pretrained=pretrainedr3d).cuda()
        elif module == "MC3":
            pretrainedmc3 = torch.load(
                "/data/users/ys221/data/pretrain/resnet/mc3_18/mc3_18_fold3_epo038_UAR46.85_WAR58.93.pth")
            self.backbone = torchvision.models.video.resnet.mc3_18(pretrained=pretrainedmc3).cuda()
        else:
            self.backbone = torchvision.models.video.resnet.r2plus1d_18(pretrained=True).cuda()
        # self.modelr3d.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.backbone.fc = nn.Identity()  # 512

        self.act = torch.nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

        # for i, p in enumerate(self.backbone.layer1.parameters()):
        #     p.requires_grad = False

        # landmark
        self.landmark = get_model_by_name('COFW', device='cuda')
        for i, p in enumerate(self.landmark.parameters()):
            p.requires_grad = False

        # attention
        self.att = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_att)

        # GRU
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, bidirectional=False, batch_first=True,
                          dropout=drop_gru, num_layers=num_layers)

        self.layer1 = torch.nn.Sequential(
            nn.Linear(in_features=29 * 28 * 28, out_features=1024),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512)
        )

        self.layer2 = torch.nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        '''
        e input clip of size 3 × L × H × W, where L
        is the number of frames in the clip
        L, 3, H, W

        :param x: bz, seq, c, h, w
        :return: output
        '''
        x_rnn = x.permute(0, 2, 1, 3, 4)
        out_rnn = self.act(self.backbone(x_rnn))

        b, l, c, h, w = x.shape
        x_att = x.reshape(b * l, c, h, w)
        out_res = self.baseline(x_att)
        out_res = out_res.reshape(b, l, -1)
        out_att, _ = self.att(out_res, out_res, out_res)
        out_gru, _ = self.gru(out_att)

        out_att = self.act(out_att[:, -1])
        out_gru = self.act(out_gru[:, -1])

        out_lm = self.landmark(x[:, int(l / 2), :, :, :])
        bz, lm, ih, iw = out_lm.shape
        out_lm = out_lm.reshape(bz, lm * ih * iw)
        out_lm = self.act(self.layer1(out_lm))

        out = torch.cat([out_rnn, out_att, out_gru, out_lm], dim=1)
        out = self.layer2(out)

        return out