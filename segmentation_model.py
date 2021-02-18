import segmentation_models as sm
import config as cfg
import numpy as np

class Food_Segmentation_model():

    def __init__(self):
        
        self.model = sm.Unet(cfg.seg_model['BACKBONE'], classes=cfg.seg_model['classes'], activation=cfg.seg_model['activation'], encoder_weights=None)
        self.model.load_weights(cfg.seg_model['pretrain-path'])

    def predict(self, img):
        y_pred = self.model.predict(np.reshape(img, (1,256,256,3)))
        y_pred = y_pred.squeeze()
        y_pred = y_pred.argmax(axis=2)
  
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