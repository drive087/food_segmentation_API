import preprocessing

seg_model = {
    "BACKBONE": "timm-efficientnet-b2",
    "classes": 7,
    "activation": "softmax",
    "pretrain-path": "./weights/Unet++_timm-efficientnet-b2.pt",
    "ENCODER": "timm-efficientnet-b2",
    "ENCODER_WEIGHTS": "imagenet"
}
use_anonymous = True