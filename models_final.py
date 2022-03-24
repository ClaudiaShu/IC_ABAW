import numpy as np
import pynndescent.pynndescent_
import torch.nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from models import *
from utils import *
import pickle
import torchvision



class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False, drop_out=0.7):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if drop_out != 0:
                layer_list.append(nn.Dropout(drop_out))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.densenet169(pretrained=True).cuda()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out

# Baseline
class BaselineRES(nn.Module):
    def __init__(self, num_classes=8, emb_size=512):
        super(BaselineRES, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

        self.emb = MLP([in_features, emb_size], final_relu=True).cuda()  # emb
        self.out = MLP([in_features, 1024, num_classes], drop_out=0.5)  # class

    def forward(self, x):

        fea = self.backbone(x)
        emb = self.emb(fea)
        out = self.out(fea)

        return out


class BaselineINC(nn.Module):
    def __init__(self, num_classes=8, emb_size=512):
        super(BaselineINC, self).__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        in_features = self.backbone.logits.in_features
        self.backbone.logits = nn.Identity()
        # self.backbone.logits = nn.Linear(in_features=in_features, out_features=num_classes)

        self.emb = MLP([in_features, emb_size], final_relu=True).cuda()  # emb
        self.out = MLP([in_features, 256, num_classes], drop_out=0.5)  # class

    def forward(self, x):

        fea = self.backbone(x)
        emb = self.emb(fea)
        out = self.out(fea)

        return out


class BaselineHRN(nn.Module):
    def __init__(self, num_classes=8, emb_size=128):
        super(BaselineHRN, self).__init__()
        self.backbone = get_model_by_name('COFW', device='cuda')

        self.mlp = nn.Sequential(nn.Linear(29 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).cuda()

        self.emb = MLP([128, emb_size], final_relu=True).cuda()  # emb
        self.out = MLP([128, 64, num_classes], drop_out=0.5)  # class

    def forward(self, x):
        lm = self.backbone(x).detach()
        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        fea_lm = lm.reshape(bz * nj, hm_h * hm_w)
        (max_c, max_indices) = torch.max(fea_lm, dim=1)
        max_h = (max_indices / hm_w).to(torch.int).to(torch.float32)
        max_w = max_indices % hm_w
        lm = torch.stack((max_h / hm_h, max_w / hm_w, max_c), -1).reshape(bz, nj * 3)

        fea = self.mlp(lm)
        emb = self.emb(fea)
        out = self.out(fea)

        return out


# landmark as channel
# The landmark should freeze
class BaselineRES_C(nn.Module):
    def __init__(self, num_classes=8, emb_size=512):
        super(BaselineRES_C, self).__init__()
        self.landmark = get_model_by_name('COFW', device='cuda')
        self.landmark = self.landmark.eval()

        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()  # feature 1
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.backbone.fc.in_features  # resnet50 2048
        self.backbone.fc = nn.Identity()

        self.mlp1 = MLP([in_features, emb_size], final_relu=True).cuda()  # emb
        self.mlp2 = MLP([in_features, 1024, num_classes]).cuda()

    def forward(self, x):
        '''
        backbone: resnet50
        landmark as channel

        :param x: bz, c, h, w
        :return: output
        '''

        with torch.no_grad():
            self.landmark.eval()
            lm = self.landmark(x).detach()

        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        mask_lm = F.interpolate(lm, size=[112, 112])
        mask_lm = torch.mean(mask_lm, dim=1, keepdim=True)
        mask_lm = mask_lm.expand(bz, 1, 112, 112)

        x = torch.cat([x, mask_lm], dim=1)

        bk = self.backbone(x)  # feature

        emb = self.mlp1(bk)  # emb
        out = self.mlp2(bk)  # class

        return out


class BaselineINC_C(nn.Module):
    def __init__(self, num_classes=8, emb_size=512):
        super(BaselineINC_C, self).__init__()

        self.landmark = get_model_by_name('COFW', device='cuda')
        self.landmark = self.landmark.eval()

        self.backbone = InceptionResnetV1(pretrained='vggface2')
        self.backbone.conv2d_1a.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        in_features = self.backbone.logits.in_features
        self.backbone.logits = nn.Identity()  # feature 1 512

        self.mlp1 = MLP([in_features, emb_size], final_relu=True).cuda()  # emb
        self.mlp2 = MLP([in_features, 256, num_classes]).cuda()
    def forward(self, x):
        '''
        backbone: InceptionResnetV1 pretrained vggface2
        landmark as channel

        :param x: bz, c, h, w
        :return: output
        '''

        with torch.no_grad():
            self.landmark.eval()
            lm = self.landmark(x).detach()

        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        mask_lm = F.interpolate(lm, size=[112, 112])
        mask_lm = torch.mean(mask_lm, dim=1, keepdim=True)
        mask_lm = mask_lm.expand(bz, 1, 112, 112)

        x = torch.cat([x, mask_lm], dim=1)

        bk = self.backbone(x)  # feature

        emb = self.mlp1(bk)  # emb
        out = self.mlp2(bk)  # class

        return out


# landmark as modality
class BaselineRES_M(nn.Module):
    def __init__(self, num_classes=8):
        super(BaselineRES_M, self).__init__()
        # branch 1
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()  # feature 1
        in_features = self.backbone.fc.in_features  # resnet50 2048
        self.backbone.fc = nn.Identity()

        # branch 2
        # self.landmark = get_model_by_name('COFW', device='cuda')  # 29
        self.landmark = get_model_by_name('WFLW', device='cuda')  #  98
        # self.landmark = self.landmark.eval()

        # self.mlp21 = MLP([29 * 3, 512, 256, 128], final_relu=True).cuda()  # feature 2
        self.mlp21 = nn.Sequential(nn.Linear(98 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).cuda()

        self.out = MLP([in_features + 128, 1024, num_classes], drop_out=0.4)  # class

    def forward(self, x):
        '''
        backbone: resnet50
        landmark as modality

        :param x: bz, c, h, w
        :return: output
        '''
        bk = self.backbone(x)  # feature 1

        # with torch.no_grad():
        #     self.landmark.eval()
        lm = self.landmark(x).detach()

        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        fea_lm = lm.reshape(bz * nj, hm_h * hm_w)
        (max_c, max_indices) = torch.max(fea_lm, dim=1)
        max_h = (max_indices / hm_w).to(torch.int).to(torch.float32)
        max_w = max_indices % hm_w
        lm = torch.stack((max_h / hm_h, max_w / hm_w, max_c), -1).reshape(bz, nj * 3)
        lm = self.mlp21(lm)  # feature 2

        cat = torch.cat([bk, lm], dim=1)
        out = self.out(cat)

        return out


class BaselineINC_M(nn.Module):
    def __init__(self, num_classes=8):
        super(BaselineINC_M, self).__init__()
        # branch 1
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        in_features = self.backbone.logits.in_features
        self.backbone.logits = nn.Identity()  # feature 1 512

        self.landmark = get_model_by_name('COFW', device='cuda')
        # self.landmark = self.landmark.eval()

        self.mlp21 = nn.Sequential(nn.Linear(29 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).cuda()

        self.out = MLP([in_features + 128, 1024, num_classes], drop_out=0.4)  # class

    def forward(self, x):
        '''
        backbone: InceptionResnetV1 pretrained vggface2
        landmark as modality

        :param x: bz, c, h, w
        :return: output
        '''
        bk = self.backbone(x)  # feature 1

        # with torch.no_grad():
        #     self.landmark.eval()
        lm = self.landmark(x).detach()

        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        fea_lm = lm.reshape(bz * nj, hm_h * hm_w)
        (max_c, max_indices) = torch.max(fea_lm, dim=1)
        max_h = (max_indices / hm_w).to(torch.int).to(torch.float32)
        max_w = max_indices % hm_w
        lm = torch.stack((max_h / hm_h, max_w / hm_w, max_c), -1).reshape(bz, nj * 3)
        lm = self.mlp21(lm)  # feature 2

        cat = torch.cat([bk, lm], dim=1)
        out = self.out(cat)

        return out


# Triplet structure
class BaselineRES_T(nn.Module):
    def __init__(self, num_classes=8):
        super(BaselineRES_T, self).__init__()
        # branch 1
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()  # feature 1
        in_features = self.backbone.fc.in_features  # resnet50 2048
        self.backbone.fc = nn.Identity()
        self.mlp11 = MLP([in_features, 512], final_relu=True).cuda()  # emb1
        self.mlp12 = MLP([in_features, 512, num_classes]).cuda()  # class1

        # branch 2
        self.landmark = get_model_by_name('COFW', device='cuda')
        # self.landmark = self.landmark.eval()
        # for i, p in enumerate(self.landmark.parameters()):
        #     p.requires_grad = False
        #     if isinstance(p, nn.modules.batchnorm._BatchNorm):
        #         p.eval()

        # self.mlp21 = MLP([29*3, 512, 256, 128], final_relu=True).cuda()  # feature 2
        self.mlp21 = nn.Sequential(nn.Linear(29 * 2, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).cuda()
        self.mlp22 = MLP([128, 128], final_relu=True).cuda()  # emb2
        self.mlp23 = MLP([128, num_classes]).cuda()  # class2

        self.out = MLP([2048+128, 1024, num_classes], drop_out=0.5)  # class3

    def forward(self, x):
        '''
        branch1: resnet50

        :param x: bz, c, h, w
        :return: output
        '''
        bk = self.backbone(x)  # feature 1
        emb1 = self.mlp11(bk)  # emb 1
        class1 = self.mlp12(bk)  # class 1

        # with torch.no_grad():
        #     self.landmark.eval()
        lm = self.landmark(x).detach()

        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        fea_lm = lm.reshape(bz*nj, hm_h*hm_w)
        (max_c, max_indices) = torch.max(fea_lm, dim=1)
        #max_c = torch.gather(fea_lm, dim=1, index=max_indices)
        max_h = (max_indices / hm_w).to(torch.int).to(torch.float32)
        max_w = max_indices % hm_w

        # lm = torch.stack((max_h/hm_h, max_w/hm_w, max_c), -1).reshape(bz, nj*3)
        lm = torch.stack((max_h / hm_h, max_w / hm_w), -1).reshape(bz, nj * 2)
        # fea_lm = F.softmax(fea_lm, dim=-1)

        lm = self.mlp21(lm)  # feature 2
        emb2 = self.mlp22(lm)  # emb 2
        class2 = self.mlp23(lm)  # class 2

        cat = torch.cat([bk, lm], dim=1)
        class3 = self.out(cat)  #class 3

        return class1, class2, class3, emb1, emb2


class BaselineINC_T(nn.Module):
    def __init__(self, num_classes=8):
        super(BaselineINC_T, self).__init__()
        # branch 1
        # branch 1
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        in_features = self.backbone.logits.in_features
        self.backbone.logits = nn.Identity()  # feature 1 512

        self.mlp11 = MLP([in_features, 512], final_relu=True).cuda()  # emb1
        self.mlp12 = MLP([in_features, 256, num_classes]).cuda()  # class1

        # branch 2
        self.landmark = get_model_by_name('COFW', device='cuda')
        # self.landmark = self.landmark.eval()
        # for i, p in enumerate(self.landmark.parameters()):
        #     p.requires_grad = False
        #     if isinstance(p, nn.modules.batchnorm._BatchNorm):
        #         p.eval()

        # self.mlp21 = MLP([29*3, 512, 256, 128], final_relu=True).cuda()  # feature 2
        self.mlp21 = nn.Sequential(nn.Linear(29 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).cuda()
        self.mlp22 = MLP([128, 128], final_relu=True).cuda()  # emb2
        self.mlp23 = MLP([128, num_classes]).cuda()  # class2

        self.out = MLP([in_features+128, 256, num_classes], drop_out=0.5)  # class3

    def forward(self, x):
        '''
        branch1: InceptionResnetV1 pretrained vggface2

        :param x: bz, c, h, w
        :return: output
        '''
        bk = self.backbone(x)  # feature 1
        emb1 = self.mlp11(bk)  # emb 1
        class1 = self.mlp12(bk)  # class 1

        # with torch.no_grad():
        #     self.landmark.eval()
        lm = self.landmark(x).detach()

        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        fea_lm = lm.reshape(bz*nj, hm_h*hm_w)
        (max_c, max_indices) = torch.max(fea_lm, dim=1)
        #max_c = torch.gather(fea_lm, dim=1, index=max_indices)
        max_h = (max_indices / hm_w).to(torch.int).to(torch.float32)
        max_w = max_indices % hm_w

        lm = torch.stack((max_h/hm_h, max_w/hm_w, max_c), -1).reshape(bz, nj*3)
        # fea_lm = F.softmax(fea_lm, dim=-1)

        lm = self.mlp21(lm)  # feature 2
        emb2 = self.mlp22(lm)  # emb 2
        class2 = self.mlp23(lm)  # class 2

        cat = torch.cat([bk, lm], dim=1)
        class3 = self.out(cat)  #class 3

        return class1, class2, class3, emb1, emb2


# Recurrent
class BaselineRNN(nn.Module):
    def __init__(self, num_classes=8, module='R3D', n_features=512,
                 hidden_size=512, num_layers=2, drop_gru=0.4,
                 embed_dim=512, num_heads=2, drop_att=0.7):
        super(BaselineRNN, self).__init__()
        '''
        r3d_18 & mc3_18: 
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400

        N → number of sequences (mini batch)
        Cin → number of channels (3 for rgb)
        D → Number of images in a sequence
        H → Height of one image in the sequence
        W → Width of one image in the sequence
        '''
        self.baseline = torchvision.models.resnet50(pretrained=True).cuda()
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

        self.backbone.fc = nn.Identity()  # 512

        # landmark
        self.landmark = get_model_by_name('COFW', device='cuda')
        for i, p in enumerate(self.landmark.parameters()):
            p.requires_grad = False

        # attention
        self.att = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_att)

        # GRU
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, bidirectional=False, batch_first=True,
                          dropout=drop_gru, num_layers=num_layers)

        self.mlp_lm = nn.Sequential(nn.Linear(29 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).cuda()

        self.out = MLP([512*3 + 128, 1024, num_classes], drop_out=0.5)  # class

    def forward(self, x):
        '''
        e input clip of size 3 × L × H × W, where L
        is the number of frames in the clip
        L, 3, H, W

        :param x: bz, seq, c, h, w
        :return: output
        '''
        x_rnn = x.permute(0, 2, 1, 3, 4)
        out_rnn = self.backbone(x_rnn)

        b, l, c, h, w = x.shape
        x_att = x.reshape(b * l, c, h, w)
        out_res = self.baseline(x_att)
        out_res = out_res.reshape(b, l, -1)
        out_att, _ = self.att(out_res, out_res, out_res)
        out_gru, _ = self.gru(out_att)

        out_att = out_att[:, -1]
        out_gru = out_gru[:, -1]

        lm = self.landmark(x[:, int(l / 2), :, :, :])
        bz, nj, hm_h, hm_w = lm.shape  # batchsize, num_joints
        fea_lm = lm.reshape(bz * nj, hm_h * hm_w)
        (max_c, max_indices) = torch.max(fea_lm, dim=1)
        max_h = (max_indices / hm_w).to(torch.int).to(torch.float32)
        max_w = max_indices % hm_w
        lm = torch.stack((max_h / hm_h, max_w / hm_w, max_c), -1).reshape(bz, nj * 3)
        out_lm = self.mlp_lm(lm)  # feature 2

        out = torch.cat([out_rnn, out_att, out_gru, out_lm], dim=1)
        out = self.layer2(out)

        return out


class BaselineRES_test(nn.Module):
    def __init__(self, num_classes=8, emb_size=512):
        super(BaselineRES_test, self).__init__()
        weight = '/data/users/ys221/data/pretrain/Resnet50/resnet50_ft_weight.pkl'
        self.backbone = resnet50()
        load_state_dict(self.backbone, weight)
        in_features = self.backbone.fc.in_features  # 2048
        # self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        # self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

        self.emb = MLP([in_features, emb_size], final_relu=True).cuda()  # emb
        self.out = MLP([in_features, 1024, num_classes], drop_out=0.5)  # class

    def forward(self, x):
        fea = self.backbone(x)
        # print(fea.shape)
        emb = self.emb(fea)
        out = self.out(fea)

        return out


