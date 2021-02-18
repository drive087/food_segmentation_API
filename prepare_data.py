import cv2
import albumentations as A
import segmentation_models as sm
# import config as cfg
# import config as cfg




def pre_img(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256,256), cv2.INTER_AREA)
    preprocess_input = sm.get_preprocessing('efficientnetb1')

    sample = get_preprocessing(preprocess_input)(image=image)
    image = sample['image'] 
    return image

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
    ]
    return A.Compose(_transform)