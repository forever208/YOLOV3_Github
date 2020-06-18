import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


class Dataset(object):
    """
    Generating training or testing dataset and its labels
    """
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH    # path of training set or test set
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE    # 4 batches for train, 2 batches for test
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE    # 416
        self.strides = np.array(cfg.YOLO.STRIDES)    # ndarray [8,16,32]
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)    # {0:'person', 1:'bicycle', 2:'car',...}
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))    # 9 anchors array: [height, width], shape=(3,3,2)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE    # 3
        self.max_bbox_per_scale = 150

        # get gt label "C/.../.jpg xmin, ymin, xmax, ymax, class_ind"
        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)    # number of images
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))     # np.ceil: round up
        self.batch_count = 0


    """given annotation txt file, return gt label string list: "image_path + bbox + class index"""""
    def load_annotations(self, dataset_type):
        """
        :param dataset_type: train or test
        :return: ["C:/.../000002.jpg xmin, ymin, xmax, ymax, class_ind xmin, ymin, xmax, ymax, class_ind", "C:/...", ...]
        """
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]    # line.strip() remove the \n
        np.random.shuffle(annotations)
        return annotations


    def __iter__(self):    # generate an iterator
        return self


    def __next__(self):    # return the next iterator, this function will run after the Dataset class has been instantiated,
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)      # still 416
            self.train_output_sizes = self.train_input_size // self.strides    # [52 26 13]

            """initialise image tensor and ground truth tensor"""
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)    # (b,416,416,3)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],    # (b,52,52,3,5+c)
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],    # (b,26,26,3,5+c)
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],    # (b,13,13,3,5+c)
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)    # (b,150,4)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)    # (b,150,4)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)    # (b,150,4)

            num = 0
            """for every batch, for every image, get image data and labels"""
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples

                    annotation = self.annotations[index]    # get one annotation "image_path + bbox coordinates + class index"

                    # parse one annotation to get a resized image and bboxes
                    image, bboxes = self.parse_annotation(annotation)    # bboxes: 2D [[xmin, ymin, xmax, ymax, class_id], [xmin, ymin, xmax, ymax, class_id]...]

                    """for every gt bbox, match it with one anchor, generate 4D label tensors (52,52,3,5+c), (26,26,3,5+c), (13,13,3,5+c) respectively, coordinates are relating to 416"""
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration


    def random_horizontal_flip(self, image, bboxes):    # data augmentation
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes


    def random_crop(self, image, bboxes):    # data augmentation
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes


    def random_translate(self, image, bboxes):    # data augmentation
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes


    def parse_annotation(self, annotation):
        """
        read an image, resize it to 416*416, resize bboxes as well
        :param annotation: 'C:/PycharmProjects/YOLOV3 Github/yymnist/Images/000002.jpg xmin, ymin, xmax, ymax, class_id xmin, ymin, xmax, ymax, class_ind'
        :return: bboxes: 2D tensor [[xmin, ymin, xmax, ymax, class_id], [xmin, ymin, xmax, ymax, class_id]...]
        """
        line = annotation.split()
        image_path = line[0]    # 'C:/PycharmProjects/YOLOV3_Github/yymnist/Images/000002.jpg'
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)    # read an image
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])    # [[xmin, ymin, xmax, ymax, class_id], [xmin, ymin, xmax, ymax, class_id]...]

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize an image to 416*416 size and shrink to [0,1], resize bboxes as well
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes


    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area


    """given an image, match every gt bbox with one anchor, generate gt label with 5D tensor ((52,52,3,5+c), (26,26,3,5+c), (13,13,3,5+c))"""
    def preprocess_true_boxes(self, bboxes):
        """
        :param bboxes: ground truth bboxes that contain n objects, [[xmin, ymin, xmax, ymax, class_id], [xmin, ymin, xmax, ymax, class_id]...]
        :return: label_sbbox: gt label in 52 scale, 4D tensor (52,52,3,5+c), xywh are relating to 416 size
                 label_mbbox: gt label in 26 scale, 4D tensor (52,52,3,5+c), xywh are relating to 416 size
                 label_lbbox: gt label in 13 scale, 4D tensor (52,52,3,5+c), xywh are relating to 416 size
                 sbboxes: gt bboxes in 52 scale, 2D tensor [[x,y,w,h], [x,y,w,h]...], relating to 416 size
                 mbboxes: gt bboxes in 26 scale, 2D tensor [[x,y,w,h], [x,y,w,h]...], relating to 416 size
                 lbboxes: gt bboxes in 13 scale, 2D tensor [[x,y,w,h], [x,y,w,h]...], relating to 416 size
        """

        # initialise gt label: 5D (52,52,3,5+c), (26,26,3,5+c), (13,13,3,5+c)
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5+self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]    # 3D (150,4), (150,4), (150,4)
        bbox_count = np.zeros((3,))

        """for every gt bbox, match it with an anchor, assign gt's [x,y,w,h] in the position of the anchor"""
        for bbox in bboxes:    # for every bbox [xmin, ymin, xmax, ymax, class_id]
            bbox_coor = bbox[:4]    # [xmin, ymin, xmax, ymax]
            bbox_class_ind = bbox[4]    # [class_id]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0      # [0,0,1,...0]
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot*(1-deta) + deta*uniform_distribution    # smooth label --> [0.1, 0,1, 0.7...0.1]

            # [xmin, ymin, xmax, ymax]  --> [x, y, w, h]
            bbox_xywh = np.concatenate([(bbox_coor[2:]+bbox_coor[:2])*0.5, bbox_coor[2:]-bbox_coor[:2]], axis=-1)

            # [x, y, w, h] --> [[x/8, y/8, w/8, h/8], [x/16, y/16, w/16, h/16], [x/32, y/32, w/32, h/32]
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]    # 2D with shape (3,4)

            iou = []
            exist_positive = False
            for i in range(3):    # for every scale
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))    # shape (3,4)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5    # set anchor's xy in the current grid center
                anchors_xywh[:, 2:4] = self.anchors[i]    # get anchor's wh

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)    # compute iou between gt and 3 anchors
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):    # if the gt bbox matches with 1 anchor, locate the box in the position of the anchor
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:    # if the gt box has no iou > 0.3, match the gt box with the highest iou anchor
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




