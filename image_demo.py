import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image
import os

# gpus = tf.config.experimental.list_physical_devices('GPU')    # return all GPUs
# tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)    # dynamically use GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # use CPU

input_size   = 416
image_path   = "./demo/kite.jpg"


"""build model and load weights"""
input_layer  = tf.keras.Input(shape=(input_size, input_size, 3))    # instantiate an input layer with tensor
feature_maps = YOLOv3(input_layer)    # chain input layer and hidden layers, get output [[b,52,52,3*(5+c)], [b,26,26,3*(5+c)], [b,13,13,3*(5+c)]]
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)    # decode into [x,y,w,h,c] corresponding to 416*416
    bbox_tensors.append(bbox_tensor)    # bbox_tensors [[b,52,52,3,5+c], [b,26,26,3,5+c], [b,13,13,3,5+c]]
model = tf.keras.Model(input_layer, bbox_tensors)    # generate a model object based on input layer and output layer
utils.load_weights(model, "./yolov3.weights")
model.summary()    # print model information


"""image proprecess"""
original_image      = cv2.imread(image_path)    # read test image
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)    # convert BGR to RGB mode
original_image_size = original_image.shape[:2]

image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])    # tranform original image to [416, 416 ,3]
image_data = image_data[np.newaxis, ...].astype(np.float32)    # add a dimension, [None, 416, 416, 3]


"""input image and make predictions"""
pred_bbox = model.predict(image_data)    # generates output predictions for the input sample
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]    # reshape to [[-1,5+c], [-1,5+c], [-1,5+c]]
pred_bbox = tf.concat(pred_bbox, axis=0)    # concat into [-1, 5+c]
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)    # discard low score and out-of-frame boxes
bboxes = utils.nms(bboxes, 0.45, method='nms')    # discard duplicate boxes which point at the same object


"""draw bboxes and show"""
image = utils.draw_bbox(original_image, bboxes)    # draw boxes on the image
image = Image.fromarray(image)
image.show()