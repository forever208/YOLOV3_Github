
import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES         = np.array(cfg.YOLO.STRIDES)    # [8, 16, 32]
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH    # 0.5


"""网络直接输出 (tx, ty, tw, th)，tx ty 为相对于当前 cell 的偏移值，tw th 为相对于匹配的 anchor 宽高的比值"""
def YOLOv3(input_layer):
    """
    build the whole network of YOLOv3
    :param input_layer: [batch, 416, 416, 3]
    :return: 3 scale predictions [[b, 52, 52, anchors*(5+classes)], [b, 26, 26, anchors*(5+classes)], [b, 13, 13, anchors*(5+classes)]]
    """
    route_1, route_2, conv = backbone.darknet53(input_layer)    # call darknet to get the basic network, [52,52,256], [26,26,512], [13,13,1024]

    """large scale output [batch, 13, 13, 3*(5+c)]"""
    conv = common.convolutional(conv, (1, 1, 1024,  512), conv_trainable=False)
    conv = common.convolutional(conv, (3, 3,  512, 1024), conv_trainable=False)
    conv = common.convolutional(conv, (1, 1, 1024,  512), conv_trainable=False)
    conv = common.convolutional(conv, (3, 3,  512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024,  512))
    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS+5)), activate=False, bn=False)


    """the medium scale output, [batch, 26, 26, 3*(NUM_CLASS+5)]"""
    conv = common.convolutional(conv, (1, 1,  512,  256), conv_trainable=False)
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)    # concat in channel dimension

    conv = common.convolutional(conv, (1, 1, 768, 256), conv_trainable=False)
    conv = common.convolutional(conv, (3, 3, 256, 512), conv_trainable=False)
    conv = common.convolutional(conv, (1, 1, 512, 256), conv_trainable=False)
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS+5)), activate=False, bn=False)


    """the small scale output, [batch, 52, 52, 3*(NUM_CLASS+5)]"""
    conv = common.convolutional(conv, (1, 1, 256, 128), conv_trainable=False)
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)    # concat in channel dimension

    conv = common.convolutional(conv, (1, 1, 384, 128), conv_trainable=False)
    conv = common.convolutional(conv, (3, 3, 128, 256), conv_trainable=False)
    conv = common.convolutional(conv, (1, 1, 256, 128), conv_trainable=False)
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS+5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


"""decode 1 scale output of YOLOv3 [x,y,w,h,c], from offset into coordinates relating to 416*416"""
def decode(conv_output, i=0):
    """
    :param conv_output: one scale output: [batch, grid, grid, 3*(5+c)], grid is 13/26/52
    :param i: i=0,1,2 correspond to 3 scales, i.e. (conv_sbbox, conv_mbbox, conv_lbbox)
    :return: decoded output [batch, grid, grid, 3, 5+num_classes]
    """
    conv_shape       = tf.shape(conv_output)    # [batch, 13, 13, 3*(5+c)]
    batch_size       = conv_shape[0]    # batch
    output_size      = conv_shape[1]    # 13/26/52

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5+NUM_CLASS))    # reshape to [b,13,13,3,5+c]

    """YOLOv3 directly output (tx, ty, tw, th) where tx ty are the offsets regarding to its cell，tw th are the ratio of matched anchor"""
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]    # [batch, 13, 13, 3, xy]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]    # [batch, 13, 13, 3, wh]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]    # [batch, 13, 13, 3, confidence]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]    # [batch, 13, 13, 3, class_prob]

    # compute the grid, output_size = 13/26/52
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])    # (13,13): [[0,0,...,0], [1,1,...,1],...,[13,13,...,13]]
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])    # (13,13): [[0,1,...,13], [0,1,...,13],...,[0,1,...,13]]

    # compute the coordinate of grid cells
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)    # concat based on the last dimension, --> (13, 13, 2)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])    # (1,13,13,2,1) --> (batch, 13, 13, 3, 2)
    xy_grid = tf.cast(xy_grid, tf.float32)    # convert to float, (batch, 13, 13, 3, 2)

    # get the absolute coordinate [x, y, w, h] relating to 416*416
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)    # shrink confidence value to [0,1]
    pred_prob = tf.sigmoid(conv_raw_prob)    # shrink classes prob to [0,1]

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)    # decoded output into shape (batch, 13, 13, 3, (5+num_classes))


def bbox_iou(boxes1, boxes2):
    """
    compute the IOU between 2 boxes
    :param boxes1: 6D tensor with shape [batch, 13, 13, 3, 1, xywh]
    :param boxes2: 6D tensor with shape [batch, 13, 13, 3, 1, xywh]
    :return: IOU value
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]    # output [batch, 13, 13, 3, 1, w*h]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # get (x,y) coordinate of top left and bottom right for boxes
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,    # top left (x,y) = (x-0.5*w, y-0.5h)
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)    # bottom right (x,y) = (x+0.5*w, y+0.5h)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])     # top left coordinate(x,y) of intersection
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])    # bottom right coordinate(x,y) of intersection

    inter_section = tf.maximum(right_down - left_up, 0.0)    # [batch, 13, 13, 3, (w+h)]
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    """
    :param boxes1: predicted bbox [batch, 13, 13, 3, xywh], xywh are coordinates in 416*416]
    :param boxes2: label bbox     [batch, 13, 13, 3, xywh], xywh are coordinates in 416*416]
    :return: giou []      ,∈ (-1,1]
    """

    # [x,y,w,h] --> [xmin,ymin,xmax,ymax]
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,    # top left (x,y) = (x-0.5*w, y-0.5h)
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)    # bottom right (x,y) = (x+0.5*w, y+0.5h)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # make sure [xmin,ymin,xmax,ymax]
    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])    # (xmax-xmin)*(ymax-ymin)
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])    # top_left coordinate of intersection area
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])    # right_bottom coordinate of intersection area
    inter_section = tf.maximum(right_down - left_up, 0.0)    # w,h of intersection area
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-10

    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    return giou


"""compute loss for one scale"""
def compute_loss(pred, conv, label, bboxes, i=0):
    """
    :param pred: (batch, grid, grid, 3, 5+c), predicted output after decoding, coordinates are relating to 416
    :param conv: (batch, grid, grid, 3*(5+c)), direct output of YOLOv3
    :param label: (batch, grid, grid, 3, 5+c),                                 coordinates are relating to 416
    :param bboxes: (batch, [[x,y,w,h], [x,y,w,h]...]) 3D gt bboxes coordinates
    :param i:
    :return: giou_loss, confidence_loss, classes_loss of each image
    """
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size    # original image size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5+NUM_CLASS))    # reshape to [batch, grid, grid, 3, 5+c]

    # direct output of YOLOv3
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    # decoded output of YOLOv3
    pred_xywh     = pred[:, :, :, :, 0:4]    # [x,y,w,h] relating to 416
    pred_conf     = pred[:, :, :, :, 4:5]    # conf

    # gt labels
    label_xywh    = label[:, :, :, :, 0:4]    # [x,y,w,h] relating to 416
    respond_bbox  = label[:, :, :, :, 4:5]    # conf: 0 or 1
    label_prob    = label[:, :, :, :, 5:]    # class label [0,0,0,0,1,...,0,0]

    """loss of bounding boxes -- giou"""
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)    # shape: (1,)
    input_size = tf.cast(input_size, tf.float32)    # convert to float

    # bbox_loss_scale = 2-w*h/(416*416), small boxes get higher scale, large boxes get lower scale
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1-giou)    # only the position has gt bbox will get giou loss

    """loss of box confidence"""
    # compute the IOU between decoded output and gt labels
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])

    # get the predicted bbox that has the highest IOU with ground truth
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)    # (batch, 13, 13, 1, 1)

    # if the label=0 && max_iou＜0.5, the bbox will be regarded as background (no object)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32 )    # respond_bgd [batch, 13, 13, 3, 0/1]

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # compute the cross_entropy loss given labels and predicted logits
    conf_loss = conf_focal * (
            # when ground truth confidence = 1 (the box has objects in reality)
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))

    """loss of classes"""
    # only compute the box that has objects in reality
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    """average loss"""
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss





