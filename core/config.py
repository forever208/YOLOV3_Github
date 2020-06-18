from easydict import EasyDict as edict

__C                           = edict()
# you can get config by: from config import cfg
cfg                           = __C    # cfg = {}

# YOLO options
__C.YOLO                      = edict()    # cfg = {YOLO:{}}
__C.YOLO.CLASSES              = "C:/PycharmProjects/YOLOV3_Github/data/classes/coco.names"
__C.YOLO.ANCHORS              = "C:/PycharmProjects/YOLOV3_Github/data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()     # cfg = {YOLO:{}, TRAIN:{}}
__C.TRAIN.ANNOT_PATH          = "C:/PycharmProjects/COCO/one_txt/train_labels.txt"
__C.TRAIN.BATCH_SIZE          = 16
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = False
__C.TRAIN.LR_INIT             = 1e-6
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 30


# TEST options
__C.TEST                      = edict()    #cfg = {YOLO:{}, TRAIN:{}, TEST:{}}
__C.TEST.ANNOT_PATH           = "C:/PycharmProjects/COCO/one_txt/val_labels.txt"
__C.TEST.BATCH_SIZE           = 1
__C.TEST.INPUT_SIZE           = [416]
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "C:/PycharmProjects/YOLOV3_Github/data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.3


