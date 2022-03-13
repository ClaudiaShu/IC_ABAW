import argparse

import torchvision
from models import *
from config.default_aflw import _C as config_aflw


# parser = argparse.ArgumentParser(description='Model')
# parser.add_argument('--pretrain', default='/data/users/ys221/data/pretrain/HRNet/WFLW/hrnetv2_w18_imagenet_pretrained.pth', type=str)
# parser.add_argument('--aflw', default='/data/users/ys221/data/pretrain/HRNet/AFLW/HR18-AFLW.pth', type=str)
# parser.add_argument('--cofw', default='/data/users/ys221/data/pretrain/HRNet/COFW/HR18-COFW.pth', type=str)
# parser.add_argument('--wlfw', default='/data/users/ys221/data/pretrain/HRNet/WFLW/HR18-WFLW.pth', type=str)
# # parser.add_argument('--model_file', default="/data/users/ys221/data/pretrain/HRNet/WFLW/HR18-WFLW.pth", type=str)
# args = parser.parse_args()


# Gen = tdgan.Gen(clsn_ER=8, Nz=256, GRAY=False, Nb=6)
#
# # # instantiate face discriminator
# # Dis_FR = models.tdgan.Dis(GRAY=FLAG_GEN_GRAYIMG, cls_num=FR_cls_num + 1)
# # instantiate expression discriminator
# Dis_ER = tdgan.Dis(GRAY=True, cls_num=8)
#
# # instantiate Expression Clssification Module (M_ER)
# Dis_ER_val = tdgan.Dis()
# Dis_ER_val.enc = Gen.enc_ER
# Dis_ER_val.fc = Gen.fc_ER
#
# par_Enc_FR_dir = '/data/users/ys221/data/pretrain/TDGAN/examples/Enc_FR_G.pkl'
# par_Enc_ER_dir = '/data/users/ys221/data/pretrain/TDGAN/examples/Enc_ER_G.pkl'
# par_Dis_ER_dir = '/data/users/ys221/data/pretrain/TDGAN/examples/Dis_ER.pkl'
# par_dec_dir = '/data/users/ys221/data/pretrain/TDGAN/examples/dec.pkl'
# par_fc_ER_dir = '/data/users/ys221/data/pretrain/TDGAN/examples/fc_ER_G.pkl'
#
# Gen.enc_FR.load_state_dict(del_extra_keys(par_Enc_FR_dir))
# Gen.enc_ER.load_state_dict(del_extra_keys(par_Enc_ER_dir))
# Gen.dec.load_state_dict(del_extra_keys(par_dec_dir))
# Gen.fc_ER.load_state_dict(del_extra_keys(par_fc_ER_dir))
#
# # validation using M_ER
# Dis_ER_val.fc = nn.Identity()
# Dis_ER_val.enc.load_state_dict(del_extra_keys(par_Enc_ER_dir))
# # Dis_ER_val.fc.load_state_dict(del_extra_keys(par_fc_ER_dir),strict=False)\
#
# # validation using discriminator
# Dis_ER.fc = nn.Identity()
# Dis_ER.load_state_dict(del_extra_keys(par_Dis_ER_dir), strict=False)

# idcodegen = FaceCycle.codegeneration().cuda()

# pretrained_data = '/data/users/ys221/data/pretrain/D_ID/checkpoint/AttrEncoder.h5'
#
# model = torchvision.models.inception_v3(pretrained=False).cuda()
# model.load_state_dict(pretrained_data, strict=False)

model1 = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
model2 = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
model3 = torchvision.models.resnet18(pretrained=True)
print(model1)