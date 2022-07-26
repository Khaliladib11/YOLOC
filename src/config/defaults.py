# https://github.com/rbgirshick/yacs
from yacs.config import CfgNode as CN

_C = CN()

# Dataset params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATASETNAME = 'Bdd100K'
_C.DATASET.IMAGESROOT = 'dataset/images/bdd100k/images/100k'
_C.DATASET.LABELROOT = 'dataset/det_annotations'
_C.DATASET.MASKROOT = 'dataset/da_seg_annotations'
_C.DATASET.LANEROOT = 'dataset/ll_seg_annotations'
_C.DATASET.TRAIN = 'train'
_C.DATASET.TEST = 'val'
_C.DATASET.IMAGEFORMAT = 'jpg'


cfg = _C
