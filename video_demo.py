
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode

gpus = tf.config.experimental.list_physical_devices('GPU')    # return all GPUs
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)    # dynamically use GPU memory

video_path      = "C:/PycharmProjects/YOLOV3_Github/docs/station.mp4"
num_classes     = 80
input_size      = 416


"""build model and load weights"""
input_layer  = tf.keras.Input(shape=(input_size, input_size, 3))    # instantiate an input layer with tensor
feature_maps = YOLOv3(input_layer)    # chain input layer and hidden layers, get output [[b,52,52,3*(5+c)], [b,26,26,3*(5+c)], [b,13,13,3*(5+c)]]
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)    # decode [x,y,w,h,c] corresponding to 416*416
    bbox_tensors.append(bbox_tensor)    # bbox_tensors [[b,52,52,3,5+c], [b,26,26,3,5+c], [b,13,13,3,5+c]]
model = tf.keras.Model(input_layer, bbox_tensors)    # instantiate a Keras model based on input layer and output layer
utils.load_weights(model, "C:/PycharmProjects/YOLOV3_Github/yolov3.weights")    # generate hidden layers and load weights
model.summary()    # print model information


"""read video and show predicted images"""
vid = cv2.VideoCapture(video_path)    # open the video
while True:
    return_value, frame = vid.read()    # read video frame by frame; return Ture/False to return_value; save frame data to frame
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # convert to RGB mode
    else:
        raise ValueError("No image!")    # raise an error manually
    frame_size = frame.shape[:2]    # [h, w]
    image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])    # resize image and shirnk pixels to [0,1]
    image_data = image_data[np.newaxis, ...].astype(np.float32)    # [None, h, w, c]

    prev_time = time.time()
    pred_bbox = model.predict_on_batch(image_data)    # make predictions, output [[b,52,52,3,5+c], [b,26,26,3,5+c], [b,13,13,3,5+c]]
    curr_time = time.time()
    exec_time = curr_time - prev_time

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]    # reshape to [[-1,5+c], [-1,5+c], [-1,5+c]]
    pred_bbox = tf.concat(pred_bbox, axis=0)    # concat into [-1, 5+c]
    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)    # discard low score and out-of-bound boxes
    bboxes = utils.nms(bboxes, 0.45, method='nms')    # discard duplicate boxes which point at the same object
    image = utils.draw_bbox(frame, bboxes)    # draw boxes on the image

    result = np.asarray(image)    # convert into numpy array
    info = "time: %.2f ms" %(1000*exec_time)
    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break