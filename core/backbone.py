import core.common as common


def darknet53(input_data):
    """
    build the darknet 53
    :param input_data: [416,416,3]
    :return: 3 feature maps [52,52,256], [26,26,512], [13,13,1024]
    """
    input_data = common.convolutional(input_data, (3, 3,  3,  32), conv_trainable=False)    # 32 filters [3,3,3], output [416,416,32]
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, conv_trainable=False)    # 64 filters [3,3,3], output [208, 208, 64]

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, conv_trainable=False)
    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True, conv_trainable=False)

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128, conv_trainable=False)
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, conv_trainable=False)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256, conv_trainable=False)
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, conv_trainable=False)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512, conv_trainable=False)
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, conv_trainable=False)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024, conv_trainable=False)

    return route_1, route_2, input_data    # [52,52,256], [26,26,512], [13,13,1024]


