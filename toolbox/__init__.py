# from .metrics import averageMeter, runningScore
from .log import get_logger
import torch.nn as nn
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .losses.loss import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LovaszSoftmax, LDAMLoss, MscCrossEntropyLoss

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, compute_speed_2,compute_speed_yuanzu,setup_seed, group_weight_decay,save_ckpt_resume


def get_dataset(cfg):
    # assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900',  'irseg_sam',
    #                           'irseg_mil', 'irseg_noaug',
    #                           ]

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'),  IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PST900
        return PST900(cfg, mode='train'), PST900(cfg, mode='test')
    if cfg['dataset'] == 'pst900_dan':
        from .datasets.pst900_danshuru import PST900
        return PST900(cfg, mode='train'), PST900(cfg, mode='test')



def get_model(cfg):
    if cfg['model_name'] == "models_rgbx_diy_x_2_x":
        from models_rgbx_diy_x_2_x.builder import EncoderDecoder as segmodel
        return segmodel(cfg_in=cfg)

