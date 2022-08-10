import torch
import os
import sys
import cv2

from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.autograd import Variable

try:
    # avoid import fault
    sys.path.append(os.path.dirname(__file__))
    from .u2net_models import U2NET, U2NETP
    from .u2net_transform import RescaleT, ToTensorLab
except:
    from u2net_models import U2NET, U2NETP
    from u2net_transform import RescaleT, ToTensorLab


from easycv.predictors.builder import build_predictor, PREDICTORS


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

@PREDICTORS.register_module()
class SODPredictor(object):
    """SODPredictor predict
        evtorch style predictor.predict, do salient object detection , borrow some code & 
        pretrain model from https://github.com/xuebinqin/U-2-Net 
        input list of RGB Image or RGB numpy array, output list of return dict. 
    """
    def __init__(self, model_name='u2net', model_path=None):
        """SODPredictor initialize with model_name
        Args:
            model_name (str): Required, u2net(147M) / u2netp(4.7M) supported, Default u2net
            model_path (str) : Optional, use input model_path to init weights, if none ,we download pretrain model from release/evtorch_thirdparty.
                we will do cache for this kind of load weights
        Return:
            None
        """ 

        def load_url_weights(name, url_index="http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/release/evtorch_thirdparty/u2net_sod/", map_location=None):
            os.makedirs('.easycv_cache', exist_ok=True)
            local_model = os.path.join('.easycv_cache', name+'.pth')
            if os.path.exists(local_model):
                weights = torch.load(local_model)
                if weights is not None:
                    print("load U2NET from  %s success!"%(local_model))
            else:
                url_model = os.path.join(url_index, name) + '.pth'
                try:
                    s = request.urlopen(url_model).read()
                    m = BytesIO(s)
                    if map_location is not None:
                        weights = torch.load(m, map_location=map_location)
                    else:
                        weights = torch.load(m)
                except:
                    print("Failed to load %s from %s, please ensure access to %s  or provide face detector model !"%(name, url_model, url_model))
                    weights = None

                with open(local_model, 'wb') as ofile:
                        torch.save(weights, ofile)
                if weights is not None:
                    print("load U2NET from  %s success!"%(url_model))
                
            return weights

        if(model_name=='u2net'):
            print("SODPredictor Build U2NET---173.6 MB")
            net = U2NET(3,1)
        elif(model_name=='u2netp'):
            print("SODPredictor Build U2NEP---4.7 MB")
            net = U2NETP(3,1)
        else:
            print("model_name %s doesn't supported now:"%model_name)

        if model_path is not None:
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(model_path))
                net.cuda()
            else:
                net.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('load U2NET from %s success!'%model_path)
        else:
            if torch.cuda.is_available():
                net.load_state_dict(load_url_weights(model_name))
                net.cuda()
            else:
                net.load_state_dict(load_url_weights(model_name, map_location='cpu'))

        net.eval()
        self.net = net
        self.model_name = model_name
        self.transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

        return

    def getitem(self, img):
        """ Wrapper of U2 SOD net's preprocess, input a Image(RGB)
        Args:
            img (str): Required, Image(RGB) or np.ndarry(RGB)
        """
        # we should notice U2 project use cv2 BGR input                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add dummy label & index, need optimize!
        imname = 'tmp'
        imidx = np.array([0])
        label = img
        sample =  {'imidx':imidx, 'image':img, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample)

        inputs_test = sample['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        # expand 0 dimension
        inputs_test = torch.unsqueeze(inputs_test, 0) 
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        
        return inputs_test

    def predict(self, img_list):
        """SODPredictor predict
            evtorch style predictor.predict, input list of RGB Image or RGB numpy array, output list of return dict
        Args:
            img_list (str): Required, List[Image(RGB)] to be infer
            require_box (bool) : Optional, generate boundingbox for mask by cv2, default to be False
        Return:
            return_res : [{ 
                "mask": np.ndarray,
                "bbox": list[list[int]]
            }]
        """
        return_res = []
        for img in img_list:
            if type(img) is not np.ndarray:
                img = np.array(img) 
            
            ow, oh = img.shape[1], img.shape[0]

            inputs_test = self.getitem(img)
            
            d1,d2,d3,d4,d5,d6,d7= self.net(inputs_test)
            
            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            del d2,d3,d4,d5,d6,d7
            
            # get cpu outout
            predict = pred
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()

            # convert 2 uint8
            img = (predict_np * 255).astype(np.uint8)
            img = cv2.resize(img, (ow, oh))

            # get contours and bbox
            cv2_major = cv2.__version__.split('.')[0]
            if cv2_major == '3':
                _, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # cv2.drawContours(img,contours,-1,(255,0,255),3)  
            bbox = []
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,255), 2)
                bbox.append([x,y,x+w-1, y+h-1])
            return_res.append({'mask':img, 'bbox': bbox,})

        return return_res



if  __name__ =="__main__":
    import sys
    input_path = sys.argv[1]
    sod = SODPredictor(model_name='u2netp')
    img = Image.open(input_path)
    res  = sod.predict([img])
    print(res[0]['bbox'])
    cv2.imwrite('test.jpg', res[0]['mask'])

