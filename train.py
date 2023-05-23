from structure import *
from utils import *
from keras.optimizers import Adam, SGD  # 用于优化模型
from tensorflow.keras.optimizers.experimental import AdamW
from keras.callbacks import ModelCheckpoint  # 用于保存模型
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)  # 用于评估模型
import spectral  # 用于处理光谱数据


def train():
    # init parameters
    dataset = 'IP'
    test_ratio = 0.90
    windowSize = 24
    K = 12
    reduction = 4
    PATCH_SIZE = windowSize
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # load dataset
    data, labels = loadData(dataset)
    print(f"dataset:{dataset}, data.shape:{data.shape}, labels.shape:{labels.shape}")
    data, fa = applyFA(data, numComponents=K)
    print(f"after applyFA, data.shape:{data.shape}")
    data, labels = createImageCubes(data, labels, windowSize=windowSize)
    print(f"after createImageCubes, data.shape:{data.shape}, labels.shape:{labels.shape}")
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(data, labels, test_ratio)
    print(f"after splitTrainTestSet, Xtrain.shape:{Xtrain.shape}, Xtest.shape:{Xtest.shape}, "
          f"ytrain.shape:{ytrain.shape}, ytest.shape:{ytest.shape}")
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    print(f"after reshape, Xtrain.shape:{Xtrain.shape}")
    ytrain = np_utils.to_categorical(ytrain)
    print(f"after to_categorical, ytrain.shape:{ytrain.shape}")

    # build model
    model = get_wavelet_cnn_model(windowSize, K)

    # optimizer and loss
    adamW = AdamW(learning_rate=0.001, weight_decay=1e-06)
    # sgd = SGD(learning_rate=0.001, momentum=0.90, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=adamW, metrics=['accuracy'])

    # train model
    filepath = "best-model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    # fit model
    history = model.fit(x=Xtrain, y=ytrain, batch_size=32, epochs=150, callbacks=callbacks_list)

    # plot loss
    plt.figure(figsize=(7, 7))
    plt.grid()
    plt.plot(history.history['loss'])

    # test model
    Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    ytest = np_utils.to_categorical(ytest)

    # classification report
    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
    classification = classification_report(np.argmax(ytest, axis=1), y_pred_test, digits=4)
    print(classification)

    m = tf.keras.metrics.Accuracy()  # 准确率
    m.update_state(np.argmax(ytest, axis=1), y_pred_test)  # 传入预测值和真实值
    print(f"accuracy:{m.result().numpy()}")  # 打印准确率

    flops = try_count_flops(model)
    print(f"flops:{flops}")
    print(f"M flops:{flops}")
    reports(Xtest, ytest, dataset)


def reports(X_test, y_test, name, model, flag: bool = False):
    # start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # end = time.time()
    # print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    print(f"oa:{oa}")
    print(f"aa:{aa}")
    print(f"kappa:{kappa}")

    if flag:
        write_file(classification, confusion, Test_Loss, Test_accuracy, oa, each_acc, aa, kappa)


def write_file(classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa):
    classification = str(classification)
    confusion = str(confusion)
    file_name = "classification_report.txt"

    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))


def save(dataset, model, windowSize, K, ):
    X, y = loadData(dataset)
    height = y.shape[0]
    width = y.shape[1]
    PATCH_SIZE = windowSize
    K = 3
    X, fa = applyFA(X, numComponents=K)
    X = padWithZeros(X, PATCH_SIZE // 2)
    # calculate the predicted image
    outputs = np.zeros((height, width))
    for i in range(height):
        print('i:', i, 'height:', height)
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                image_patch = Patch(X, i, j, PATCH_SIZE)
                X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                   1).astype('float32')
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction + 1

    ground_truth = spectral.imshow(classes=y, figsize=(7, 7))
    predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))
    spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)


if __name__ == "__main__":
    train()
