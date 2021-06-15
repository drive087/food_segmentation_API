import os
import sys
import segmentation_models_pytorch as smp
import config as cfg
import torch
import copy
from pytorch2keras.converter import pytorch_to_keras

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# sys.path.append("/Users/thusk/Documents/CP/Senior Project/food_segmentation_API/weights/Unet++_timm-efficientnet-b2.pt")

# BACKBONE = cfg.seg_model['BACKBONE']
# CLASSES = cfg.seg_model['classes']
# ACTIVATION = cfg.seg_model['activation']
# PRETRAINED = cfg.seg_model['pretrain-path']

# model = smp.UnetPlusPlus(BACKBONE, classes=CLASSES, activation=ACTIVATION, encoder_weights=None)
# model.load_state_dict(copy.deepcopy(torch.load(PRETRAINED, map_location=torch.device('cpu'))))
# # model = model.double()
# model = model.eval()

# x = torch.randn(1, 3, 320, 320, requires_grad=False)
# k_model = pytorch_to_keras(model, x, [(3, None, None,)], verbose=True)
# k_model.save('unetplusplus_efficientnetb2_keras.h5')