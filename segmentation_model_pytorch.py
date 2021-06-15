import segmentation_models_pytorch as smp
import config as cfg
import numpy as np
import torch
import copy

BACKBONE = cfg.seg_model['BACKBONE']
CLASSES = cfg.seg_model['classes']
ACTIVATION = cfg.seg_model['activation']
PRETRAINED = cfg.seg_model['pretrain-path']

class Food_Segmentation_model_pytorch():

    def __init__(self):
        
        self.model = smp.UnetPlusPlus(BACKBONE, classes=CLASSES, activation=ACTIVATION, encoder_weights=None)
        self.model.load_state_dict(copy.deepcopy(torch.load(PRETRAINED, map_location=torch.device('cpu'))))
        self.model = self.model.double()

    def predict(self, img):
        
        x_tensor = torch.from_numpy(img)
        # x_tensor = torch.reshape(x_tensor, (3, 320, 320))
        # x_tensor = x_tensor.Double()
        x_tensor = x_tensor.to('cpu').unsqueeze(0)
        
        y_pred = self.model.predict(x_tensor)
        y_pred = y_pred.squeeze()
        y_pred = y_pred.argmax(axis=0)
  
        return y_pred

    def decode_segmap(self, image, nc=7):
  
        label_colors = np.array([(0, 0, 0),  # 0=background black
                    # 1=rice white,   2=meat brown, 3=egg yellow, 4=vegetable green,  5=beverage blue,6=etc purple
                    (255, 255, 255), (153, 76, 0), (255, 255, 0), (102, 255, 102), (0, 0, 255), (255, 102, 255)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)
        return rgb