import logging
from sklearn.decomposition import FactorAnalysis  # 用于降维
import numpy as np
import scipy.io as sio  # 用于读写mat文件
from sklearn.model_selection import train_test_split  # 用于划分数据集
import tensorflow as tf  # 用于机器学习
from typing import Any, Callable, Dict, List, Optional, Union

from tensorflow import truediv
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


# 定义一个函数，接受一个三维数组X和一个因子数目numComponents，默认为75
def applyFA(X, numComponents=75):
    # 将X的第一和第二维合并，得到一个二维数组newX
    newX = np.reshape(X, (-1, X.shape[2]))
    # 创建一个因子分析对象fa，指定因子数目和随机种子
    fa = FactorAnalysis(n_components=numComponents, random_state=0)
    # 对newX进行因子分析，得到一个新的二维数组newX，每一行是一个样本的因子值
    newX = fa.fit_transform(newX)
    # 将newX的第一维恢复为X的第一维，得到一个三维数组newX，每个元素是一个样本的因子值
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    # 返回新的数组newX和因子分析对象fa
    return newX, fa


"""
dataset management: KSC, IP, SA, PU
"""


def loadData(name):
    data, labels = None, None
    if name == 'KSC':
        data = sio.loadmat('input/ksc/KSC.mat')['KSC']
        labels = sio.loadmat('input/ksc/KSC_gt.mat')['KSC_gt']
    if name == 'IP':
        data = sio.loadmat('input/indian-pines/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('input/indian-pines/Indian_pines_gt.mat')['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat('input/salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('input/salinas/Salinas_gt.mat')['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat('input/paviau/paviaU.mat')['paviaU']
        labels = sio.loadmat('input/paviau/paviaU_gt.mat')['paviaU_gt']

    return data, labels


def splitTrainTestSet(X, y, testRatio, randomState=355):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, train_size=1 - testRatio,
                                                        random_state=randomState,
                                                        stratify=y)
    missing_labels = np.setdiff1d(y_test, y_train)  # 找出测试集中没有的标签
    test_y = np.unique(y_test)  # 测试集中的标签
    train_y = np.unique(y_train)  # 训练集中的标签
    print(f"missing label:{missing_labels}, test_y:{test_y}, train_y:{train_y}")
    # 将测试集中没有的标签加入到训练集中
    for i in missing_labels:
        index = np.where(y_test == i)[0][0]
        y_train = np.append(y_train, i)
        y_test = np.delete(y_test, index)
        c, w, h = X_test[index].shape
        X_train = np.append(X_train, X_test[index].reshape((1, c, w, h)), axis=0)
        X_test = np.delete(X_test, index, axis=0)
    return X_train, X_test, y_train, y_test


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=8, removeZeroLabels=True):
    margin = int(windowSize / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


# 用于对数据进行归一化的函数
def normalize(X, gamma, beta):
    eps = 1e-6  # 定义一个很小的常数，用于避免除零错误
    gx = tf.norm(X, axis=(1, 2), keepdims=True)  # 计算X在第1和第2维度上的范数，保持维度不变
    nx = gx / (tf.reduce_mean(gx, axis=-1, keepdims=True) + eps)  # 计算X在最后一个维度上的范数的均值，并用gx除以它，得到归一化后的nx
    # signed g,b
    normalized_X = gamma * (X * nx) + beta  # 用gamma和beta对X进行缩放和偏移，得到归一化后的X
    return normalized_X  # 返回归一化后的X


def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    """Counts and returns model FLOPs.
    Args:
      model: A model instance.
      inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
        shape specifications to getting corresponding concrete function.
      output_path: A file path to write the profiling results to.
    Returns:
      The model's FLOPs.
    """
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            logging.info(
                'Failed to count model FLOPs with error %s, because the build() '
                'methods in keras layers were not called. This is probably because '
                'the model was not feed any input, e.g., the max train step already '
                'reached before this run.', e)
            return None
    return None


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch
