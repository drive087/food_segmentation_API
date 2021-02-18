import preprocessing

seg_model = {
    "BACKBONE": "efficientnetb1",
    "classes": 7,
    "activation": "softmax",
    "pretrain-path": "./weights/kuay_yai_mak_2_model.h5",
}
use_anonymous = True