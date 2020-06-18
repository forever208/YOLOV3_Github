
import cv2
import random
import colorsys
import numpy as np
from core.config import cfg

def load_weights(model, weights_file):
    wf = open(weights_file, 'rb')    # read weights file, rb: read binary

    # 权重文件为float32浮点数，权重文件的前160 bit存储5个int32值，它们构成文件的头部
    # 1.主要版本号， 2.次要版本号， 3.颠覆号码， 4, 5.网络看到的图像（训练期间）
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)    # 读取二进制文件， count: 读入元素个数

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)    # given layer name, get a corresponding layer
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            """get bn weights"""
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)    # load batch_norm weights: [beta, gamma, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]    # reshape to [gamma, beta, mean, variance]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1

        else:
            """get conv bias"""
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        """get conv weights"""
        conv_shape = (filters, in_dim, k_size, k_size)    # darknet shape (out_dim, in_dim, height, width)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])    # tf shape (height, width, in_dim, out_dim)

        """set weights"""
        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])    # when there is batch_norm, no bias needs to be loaded
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])    # when there is no barch_norm, bias must be loaded

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names    # names={0: 'person', 1: 'bicycle', 2: 'car',...}


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)    # 9 anchors array: (height, width)


def image_preporcess(image, target_size, gt_boxes=None):
    """
    resize a single image to target size and shrink to [0,1]
    :param image: [org_h, org_w, channels]
    :param target_size: [h, w]
    :param gt_boxes:
    :return: processed image [h, w, c]
    """

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))    # scale by ratio

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)    # construct target*target*3 image with pixels intensity 128
    dw, dh = (iw - nw) // 2, (ih-nh) // 2    # dw, dh is the area to be padded
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized    # fill up with valid image
    image_paded = image_paded / 255.    # shrink to [0,1]

    if gt_boxes is None:
        return image_paded
    else:    # if raw image has ground truth boxes, they will be resize as well
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    :param image:
    :param bboxes: [x_min, y_min, x_max, y_max, probability, class_id]
    :param classes: {0: 'person', 1: 'bicycle', 2: 'car',...}
    :param show_label: switch
    :return: image with boxes and text
    """

    """different classes will be shown in different colors"""
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]    # [(0.1,1,1), (0.2,1,1)...(1,1,1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))    # convert hsv to rgb [(0.1,1,1), (0.2,1,1)...(1,1,1)]
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))    # [(0,255,255), (25,255,255)...(255,255,255)]

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)    # coor [xmin, ymin, xmax, ymax]
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]    # different classes marked as different colors
        bbox_thick = int(0.6 * (image_h + image_w) / 600)    # thick of the box
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])    # c1: left top coordinate,  c2: bottom right coordinate
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)    # draw rectangle on images

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)    # add test of class name and score on images
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]    # compute the text size
            cv2.rectangle(image, c1, (c1[0]+t_size[0], c1[1]-t_size[1]-3), bbox_color, -1)  # draw text frame on images
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,    # draw text on images
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    remove redundant boxes that point at the same object
    :param bboxes: [xmin, ymin, xmax, ymax, score, class]
    :param iou_threshold:
    :param sigma:
    :param method:
    :return:
    """
    classes_in_img = list(set(bboxes[:, 5]))    # remove duplicates of class, return [1,2,3...n]
    best_bboxes = []

    for cls in classes_in_img:    # for each class, remove duplicate class bboxes
        cls_mask = (bboxes[:, 5] == cls)    # mask array technique
        cls_bboxes = bboxes[cls_mask]    # find all bboxes with the same class

        while len(cls_bboxes) > 0:    # when bboxes is not non
            max_ind = np.argmax(cls_bboxes[:, 4])    # get the index of the highest score box
            best_bbox = cls_bboxes[max_ind]    # get the bbox
            best_bboxes.append(best_bbox)    # save the highest IOU box
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])    # save the rest boxes
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])    # compute IOU between highest score bbox and rest bboxes
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':    # weight=0 if the IOU is greater than threshold
                iou_mask = iou > iou_threshold    # iou_mask = True/False
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]    # remove boxes whose score <= 0

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    """
    transform [xmin, ymin, xmax, ymax] into original coordinates, discard low score and out-of-bounds boxes
    :param pred_bbox: [-1, 5+num_classes]
    :param org_img_shape: practical image size
    :param input_size: 416
    :param score_threshold: 0.3/0.5
    :return:
    """

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    """1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)"""
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    """2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)"""
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    """3. reset boxes out of border as coordinate 0"""
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    """4. discard boxes that are out of border"""
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))    # compute the box area
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))    # locate boxes whose area > 0

    """5. discard some boxes with low scores"""
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)




