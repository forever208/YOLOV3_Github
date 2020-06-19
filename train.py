import os
import shutil    # high-level file operation
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm    # show progress bar
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

# gpus = tf.config.experimental.list_physical_devices('GPU')    # return all GPUs
# tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)    # dynamically use GPU memory
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""get training images and labels"""
trainset = Dataset('train')    # creat a Dataset object which will be used for generating batch_image, (batch_smaller_label, batch_medium_label, batch_larger_label)
logdir = "C:/PycharmProjects/YOLOV3_Github/data/log"    # file for saving logs
steps_per_epoch = len(trainset)    # 250
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch    # 2 epochs
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch    # 30 train epochs
tf.print('steps_per_epoch:%d' %steps_per_epoch)


"""build model and load weights"""
input_tensor = tf.keras.layers.Input([416, 416, 3])    # create an input object with tensor shape (416,416,3), no batch is needed
conv_tensors = YOLOv3(input_tensor)    # chain input layer and hidden layers, get output [[b,52,52,3*(5+c)], [b,26,26,3*(5+c)], [b,13,13,3*(5+c)]]
output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)    # output layer [batch, grid, grid, 3*(5+c)]
    output_tensors.append(pred_tensor)    # decoded output layer [batch, grid, grid, 3*(5+c)]
model = tf.keras.Model(input_tensor, output_tensors)    # generate a model object based on input layer and output layer
utils.load_weights(model, "./yolov3.weights")    # load weights
model.summary()


# """set up optimizer, tensorboard"""
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# if os.path.exists(logdir): shutil.rmtree(logdir)    # remove logdir folder and its files recursively
# writer = tf.summary.create_file_writer(logdir)    # creat a summary file which can be visualised by TensorBoard
#
#
# """compute loss, apply gradients"""
# for epoch in range(cfg.TRAIN.EPOCHS):    # for every epoch
#     for image_data, target in trainset:    # for every batch
#         with tf.GradientTape() as tape:
#             pred_result = model(image_data, training=True)
#             giou_loss = 0
#             conf_loss = 0
#             prob_loss = 0
#
#             # sum up loss over 3 scales
#             for i in range(3):
#                 conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]    # get output and decoded ouput(pred) simultanuously
#                 loss_items = compute_loss(pred, conv, *target[i], i)    # compare decoded output and label for loss computation
#                 giou_loss += loss_items[0]
#                 conf_loss += loss_items[1]
#                 prob_loss += loss_items[2]
#             total_loss = giou_loss + conf_loss + prob_loss  # compute loss
#
#             # compute gradient
#             gradients = tape.gradient(total_loss, model.trainable_variables)
#
#             # apply gradients to update weights
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#             # print loss
#             tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
#                      "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
#                                                                giou_loss, conf_loss, prob_loss, total_loss))
#             global_steps.assign_add(1)
#
#             # """alternatively, you could manually set up lr decay schedule"""
#             # if global_steps < warmup_steps:
#             #     lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
#             # else:
#             #     lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
#             #         (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
#             #
#             # # update learning rate
#             # optimizer.lr.assign(lr.numpy())
#
#             # write loss into the summary
#             with writer.as_default():
#                 tf.summary.scalar("lr", optimizer.lr, step=global_steps)
#                 tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
#                 tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
#                 tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
#                 tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
#             writer.flush()
#
#     tf.print('epoch:%d' %epoch)
#     model.save_weights("./model.h5")    # only save weights after each epoch



"""compute validation loss"""
testset = Dataset('test')
sum_giou_loss = 0
sum_conf_loss = 0
sum_prob_loss = 0
sum_total_loss = 0
n = 0
for image_data, target in testset:    # for every batch
    test_result = model(image_data, training=False)
    giou_loss = 0
    conf_loss = 0
    prob_loss = 0

    # sum up loss over 3 scales
    for i in range(3):
        conv, pred = test_result[i * 2], test_result[i * 2 + 1]    # get output and decoded ouput(pred) simultanuously
        loss_items = compute_loss(pred, conv, *target[i], i)    # compare decoded output and label for loss computation
        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
    total_loss = giou_loss + conf_loss + prob_loss

    sum_giou_loss += giou_loss
    sum_conf_loss += conf_loss
    sum_prob_loss += prob_loss
    sum_total_loss += total_loss
    n += 1

    tf.print("val_giou_loss: %4.2f   val_conf_loss: %4.2f   "
             "val_prob_loss: %4.2f   val_total_loss: %4.2f" % (giou_loss, conf_loss, prob_loss, total_loss))

# average loss among of all validation set
tf.print("average_giou_loss: %4.2f   average_conf_loss: %4.2f   "
         "average_prob_loss: %4.2f   average_total_loss: %4.2f" % (sum_giou_loss/n, sum_conf_loss/n, sum_prob_loss/n, sum_total_loss/n))