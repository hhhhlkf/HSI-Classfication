from keras import backend as ker_b  # 用于处理张量
import tensorflow as tf  # 用于机器学习
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Layer, Conv2DTranspose, \
    GlobalMaxPooling2D  # 用于全局平均池化
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Conv3D, Dense, Dropout, Flatten, Input,
                          Lambda, add, concatenate)  # 用于构建神经网络层
from tensorflow.python.keras.initializers import Constant
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.models import Model  # 用于定义模型
from keras.utils import np_utils, plot_model
from keras import regularizers
"""
haar wavelet transform
"""


# 定义一个函数，用于对图像的y轴进行小波变换
def WaveletTransformAxisY(batch_img):
    # 将图像的奇数行和偶数行分别提取出来
    odd_img = batch_img[:, 0::2]
    even_img = batch_img[:, 1::2]
    # 计算图像的低频成分，即奇数行和偶数行的平均值
    L = (odd_img + even_img) / 2.0
    # 计算图像的高频成分，即奇数行和偶数行的差值的绝对值
    H = ker_b.abs(odd_img - even_img)
    # 返回低频成分和高频成分
    return L, H


# 定义一个函数，用于对图像的x轴进行小波变换
def WaveletTransformAxisX(batch_img):
    # 将图像沿着y轴翻转，然后转置，相当于对图像进行了顺时针旋转90度的操作
    tmp_batch = ker_b.permute_dimensions(batch_img, [0, 2, 1])[:, :, ::-1]
    # 对旋转后的图像进行y轴的小波变换
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # 将变换后的图像再次转置，然后沿着x轴翻转，相当于对图像进行了逆时针旋转90度的操作，恢复原来的方向
    dst_L = ker_b.permute_dimensions(_dst_L, [0, 2, 1])[:, ::-1, ...]
    dst_H = ker_b.permute_dimensions(_dst_H, [0, 2, 1])[:, ::-1, ...]
    # 返回低频成分和高频成分
    return dst_L, dst_H


# 定义一个函数，用于对图像进行小波变换
def WaveletTransform(batch_image1):
    #     print(f"batch_image1 = {batch_image1.shape}")
    # 将图像的维度顺序从[batch_size, height, width, channels]变为[batch_size, channels, height, width]
    batch_image = ker_b.permute_dimensions(batch_image1, [0, 3, 1, 2])
    # 将图像的每个颜色通道分别提取出来，形成一个列表
    color_channels = [batch_image[:, i:i + 1, :, :] for i in range(0, batch_image.shape[1], 1)]
    # 定义一个空列表，用于存储第一层小波变换的结果
    wavelet_data_level1 = []
    # 定义一个空列表，用于存储第二层小波变换的结果
    wavelet_data_level2 = []
    # 定义一个空列表，用于存储第三层小波变换的结果
    wavelet_data_level3 = []
    # 定义一个空列表，用于存储第四层小波变换的结果
    wavelet_data_level4 = []

    # 对第一层小波变换进行循环
    for channel1 in color_channels:
        # print(f"channel1 = {channel1.shape}")
        # 将颜色通道的维度顺序从[batch_size, 1, height, width]变为[batch_size, height, width]
        channel2 = tf.reshape(channel1, shape=(-1, channel1.shape[2], channel1.shape[3]))
        # print(f"channel2 = {channel2.shape}")
        # 对图像的y轴进行小波变换，得到低频和高频成分
        wavelet_L, wavelet_H = WaveletTransformAxisY(channel2)
        # 对低频成分的x轴进行小波变换，得到水平和垂直方向的低频细节
        channel_wavelet_LL, channel_wavelet_LH = WaveletTransformAxisX(wavelet_L)
        # 对高频成分的x轴进行小波变换，得到水平和垂直方向的高频细节
        channel_wavelet_HL, channel_wavelet_HH = WaveletTransformAxisX(wavelet_H)
        # 将四个子图像添加到第一层小波变换的结果列表中
        wavelet_data_level1.extend([channel_wavelet_LL])
        wavelet_data_level1.extend([channel_wavelet_LH])
        wavelet_data_level1.extend([channel_wavelet_HL])
        wavelet_data_level1.extend([channel_wavelet_HH])

        # level 2 decomposition
        wavelet_L2, wavelet_H2 = WaveletTransformAxisY(channel_wavelet_LL)
        channel_wavelet_LL2, channel_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
        channel_wavelet_HL2, channel_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)
        wavelet_data_level2.extend(
            [channel_wavelet_LL2])  # , channel_wavelet_LH2, channel_wavelet_HL2, channel_wavelet_HH2])
        wavelet_data_level2.extend([channel_wavelet_LH2])
        wavelet_data_level2.extend([channel_wavelet_HL2])
        wavelet_data_level2.extend([channel_wavelet_HH2])
        # level 3 decomposition
        wavelet_L3, wavelet_H3 = WaveletTransformAxisY(channel_wavelet_LL2)
        channel_wavelet_LL3, channel_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
        channel_wavelet_HL3, channel_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)
        wavelet_data_level3.extend(
            [channel_wavelet_LL3])  # , channel_wavelet_LH3, channel_wavelet_HL3, channel_wavelet_HH3])
        wavelet_data_level3.extend([channel_wavelet_LH3])
        wavelet_data_level3.extend([channel_wavelet_HL3])
        wavelet_data_level3.extend([channel_wavelet_HH3])
        # level 4 decomposition
        wavelet_L4, wavelet_H4 = WaveletTransformAxisY(channel_wavelet_LL3)
        channel_wavelet_LL4, channel_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
        channel_wavelet_HL4, channel_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)
        wavelet_data_level4.extend([
            channel_wavelet_LL4])  # , channel_wavelet_LH4, channel_wavelet_HL4, channel_wavelet_HH4])#print('shape
        # before')
        wavelet_data_level4.extend([channel_wavelet_LH4])
        wavelet_data_level4.extend([channel_wavelet_HL4])
        wavelet_data_level4.extend([channel_wavelet_HH4])

    # 将第一层小波变换的结果列表沿着第二个维度堆叠起来，形成一个[batch_size, 4*channels, height/2, width/2]的张量
    transform_batch = ker_b.stack(wavelet_data_level1, axis=1)
    # print(f"wavelet_data_level_1: {transform_batch.shape}")
    # 将第二层小波变换的结果列表沿着第二个维度堆叠起来，形成一个[batch_size, 4*channels, height/4, width/4]的张量
    transform_batch_l2 = ker_b.stack(wavelet_data_level2, axis=1)
    # print(f"wavelet_data_level_2: {transform_batch_l2.shape}")
    # 将第三层小波变换的结果列表沿着第二个维度堆叠起来，形成一个[batch_size, 4*channels, height/8, width/8]的张量
    transform_batch_l3 = ker_b.stack(wavelet_data_level3, axis=1)
    # print(f"wavelet_data_level_3: {transform_batch_l3.shape}")
    # 将第四层小波变换的结果列表沿着第二个维度堆叠起来，形成一个[batch_size, 4*channels, height/16, width/16]的张量
    transform_batch_l4 = ker_b.stack(wavelet_data_level4, axis=1)
    # print(f"wavelet_data_level_4: {transform_batch_l4.shape}")
    # 将每一层小波变换的张量的维度顺序从[batch_size, channels, height, width]变为[batch_size, height, width, channels]
    decorum_level_1 = ker_b.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decorum_level_2 = ker_b.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decorum_level_3 = ker_b.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decorum_level_4 = ker_b.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])

    return [decorum_level_1, decorum_level_2, decorum_level_3, decorum_level_4]


class ConvNeXtBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num, k_size=7, drop_path=0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dw_conv = Conv2D(dim, kernel_size=k_size, padding="same", groups=dim, name="ConvNeXt2d__" + num)
        self.norm = LayerNormalization(axis=-1, epsilon=1e-6, name="ConvNextLayer__" + num)
        self.pw_conv1 = Dense(4 * dim, name="ConvNextDense1__" + num)
        self.act = Activation("gelu", name="ConvNextGELU__" + num)
        self.pw_conv2 = Dense(dim, name="ConvNextDense2__" + num)
        self.gamma = self.add_weight(name="gamma" + num, shape=(dim,),
                                     initializer=Constant(layer_scale_init_value),
                                     trainable=True) if layer_scale_init_value > 0 else None
        self.drop_path = Dropout(drop_path, name="gamma" + num) if drop_path > 0. else Lambda(lambda x: x)

    def call(self, x):
        input = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x


def get_wavelet_cnn_model(window_size, K):
    input_shape = window_size, window_size, K
    input_ = Input(input_shape, name='the_input')

    wavelet1 = Lambda(WaveletTransform, name='wavelet1')

    input_l1, input_l2, input_l3, input_l4 = wavelet1(input_)

    # level one decomposition starts
    conv_1 = Conv2D(48, kernel_size=(5, 5), padding='same')(input_l1)
    #     norm_1 = BatchNormalization(name='norm_1')(conv_1)
    norm_1 = LayerNormalization(axis=-1, epsilon=1e-6)(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)

    conv_1_2 = Conv2D(48, kernel_size=(5, 5), strides=(4, 4), padding='same', name='conv_1_2')(relu_1)
    #     norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
    norm_1_2 = LayerNormalization(axis=-1, epsilon=1e-6)(conv_1_2)
    relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

    conv_1_3 = Conv2D(48, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_1_3')(relu_1_2)
    #     norm_1_3 = BatchNormalization(name='norm_1_3')(conv_1_3)
    norm_1_3 = LayerNormalization(axis=-1, epsilon=1e-6)(conv_1_3)
    relu_1_3 = Activation('relu', name='relu_1_3')(norm_1_3)

    conv_1_4 = Conv2D(48, kernel_size=(1, 1), padding='same', name='conv_1_4')(relu_1_3)
    #     norm_1_4 = BatchNormalization(name='norm_1_4')(conv_1_4)
    norm_1_4 = LayerNormalization(axis=-1, epsilon=1e-6)(conv_1_4)
    relu_1_4 = Activation('relu', name='relu_1_4')(norm_1_4)

    # level two decomposition starts
    conv_a = Conv2D(filters=48, kernel_size=(5, 5), padding='same', name='conv_a')(input_l2)
    norm_a = BatchNormalization(name='norm_a')(conv_a)
    relu_a = Activation('relu', name='relu_a')(norm_a)

    conv_a1 = Conv2D(48, kernel_size=(5, 5), strides=(4, 4), padding='same', name='conv_a1')(relu_a)
    norm_a1 = BatchNormalization(name='norm_a1')(conv_a1)
    relu_a1 = Activation('relu', name='relu_a1')(norm_a1)

    # conv_a2= Conv2D(32, kernel_size=(1, 1), padding='same', name='conv_a2')(relu_a1)
    # norm_a2= BatchNormalization(name='norm_a2')(conv_a2)
    # relu_a2= Activation('relu', name='relu_a2')(norm_a2)

    # level three decomposition starts
    #     conv_b = Conv2D(filters=32, kernel_size=(3, 3), padding='same', name='conv_b')(input_l3)
    conv_b = Conv2D(filters=48, kernel_size=(3, 3), padding='same', name='conv_b')(input_l3)
    norm_b = BatchNormalization(name='norm_b')(conv_b)
    relu_b = Activation('relu', name='relu_b')(norm_b)

    #     conv_b1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_b1')(relu_b)
    conv_b1 = Conv2D(48, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_b1')(relu_b)
    norm_b1 = BatchNormalization(name='norm_b1')(conv_b1)
    relu_b1 = Activation('relu', name='relu_b1')(norm_b1)

    # conv_b2 = Conv2D(32, kernel_size=(1, 1),padding='same', name='conv_b2')(relu_b1)
    # norm_b2 = BatchNormalization(name='norm_b2')(conv_b2)
    # relu_b2 = Activation('relu', name='relu_b2')(norm_b2)

    # level four decomposition start
    #     conv_c = Conv2D(32, kernel_size=(1, 1), padding='same', name='conv_c')(input_l4)
    conv_c = Conv2D(48, kernel_size=(1, 1), padding='same', name='conv_c')(input_l4)
    norm_c = BatchNormalization(name='norm_c')(conv_c)
    relu_c = Activation('relu', name='relu_c')(norm_c)

    concate_level = concatenate([relu_1_4, relu_a1, relu_b1, relu_c])

    # 通道注意力
    b, h, w, c = concate_level.get_shape()
    weight = GlobalAveragePooling2D(keepdims=True)(concate_level)
    #     dense1 = Dense(160 // 16, use_bias=False,name ='attention_dense1')(weight)
    dense1 = Dense(192 // 16, use_bias=False, name='attention_dense1')(weight)
    relu1 = Activation('relu', name='attention_relu1')(dense1)
    dense2 = Dense(192, use_bias=False, name='attention_dense2')(relu1)
    sigmod1 = Activation('sigmoid', name='attention_sigmod1')(dense2)
    attention = concate_level * tf.tile(sigmod1, (1, h, w, 1))

    sap = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True), name='sap')(attention)
    smp = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True), name='smp')(attention)

    concat_U = concatenate([sap, smp])
    #     print(f"s_U:{concat_U.shape}")
    convs_U = Conv2D(1, kernel_size=(3, 3), padding="same", name='convs_U')(concat_U)
    sigmoid_U = Activation('sigmoid', name='sigmoid_U2')(convs_U)
    #     print(f"sigmoid:{sigmoid_U.shape},concate_level:{concate_level.shape}")
    attentions_U = tf.multiply(sigmoid_U, concate_level)

    softmax_U = Activation('softmax', name='softmaxx_U')(attentions_U)

    conv_2 = Conv2D(192, kernel_size=(1, 1), padding='same', name='conv_2', kernel_regularizer=regularizers.l2(0.0002))(
        softmax_U)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)

    conv_3 = Conv2D(128, kernel_size=(1, 1), padding='same', name='conv_3')(relu_2)
    norm_3 = BatchNormalization(name='norm_3')(conv_3)
    relu_3 = Activation('relu', name='relu_3')(norm_3)

    conv_4 = Conv2D(64, kernel_size=(1, 1), padding='same', name='conv_4')(relu_3)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    relu_4 = Activation('relu', name='relu_4')(norm_4)

    # norm_4 = normalize(conv_4,gama,beta)
    # relu_4 = Activation('relu', name='relu_4')(norm_4)
    # pool = GlobalAveragePooling2D(keepdims=True)(norm_5)

    pool = GlobalAveragePooling2D(keepdims=True)(relu_4)
    #     flatten_layer = Flatten()(pool)

    conv_last = Conv2D(16, kernel_size=(1, 1), padding="same", name="conv_last")(pool)
    softmax_last = Activation('softmax', name='softmax_last')(conv_last)
    output_layers = Flatten()(softmax_last)

    #     dense_layer2 = Dense(units=128, activation='relu')(flatten_layer)
    #     dense_layer2 = Dropout(0.4)(dense_layer2)
    #     output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

    model = Model(inputs=input_, outputs=output_layers)

    model.summary()
    # plot_model(model, to_file='wavelet_cnn_0.5.png')

    return model
