import cv2
import albumentations as A
import segmentation_models_pytorch as smp
import config as cfg


ENCODER = cfg.seg_model['ENCODER']
ENCODER_WEIGHTS = cfg.seg_model['ENCODER_WEIGHTS']


def pre_img(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320,320))
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    sample = get_preprocessing(preprocessing_fn)(image=image)
    image = sample['image'] 
    return image

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float64')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor),
    ]
    return A.Compose(_transform)