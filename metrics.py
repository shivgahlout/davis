import numpy as np
from sklearn.metrics import accuracy_score

def cal_iou(predictions, gt):
    eps=1e-7
    intersection=np.sum(gt.astype('uint8') & predictions.astype('uint8'))
    union = np.sum(gt.astype('uint8') | predictions.astype('uint8'))
    
    return (intersection+eps)/(union+eps)


