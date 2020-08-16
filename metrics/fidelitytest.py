import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt


def fidelity(ori, gen, show=True):
    ori_idx = int(len(ori) * .8)
    gen_idx = int(len(gen) * .8)

    x_train = np.vstack((ori[:ori_idx],
                         gen[:gen_idx]))

    x_test = np.vstack((ori[ori_idx:],
                        gen[gen_idx:]))

    y_train = np.concatenate((np.ones(len(ori[:ori_idx])),
                              np.zeros(len(gen[:gen_idx]))))

    y_test = np.concatenate((np.ones(len(ori[ori_idx:])),
                             np.zeros(len(gen[gen_idx:]))))

    ts_classifier = Sequential([GRU(6, input_shape=(ori.shape[1], ori.shape[2]), name='GRU'),
                                Dense(1, activation='sigmoid', name='OUT')])
    ts_classifier.compile(loss='binary_crossentropy', optimizer='adam',
                          metrics=[AUC(name='AUC'), 'accuracy'])

    result = ts_classifier.fit(x_train, y_train,
                               validation_data=(x_test, y_test),
                               epochs=200, batch_size=None,
                               verbose=0)

    final_acc = ts_classifier.evaluate(x_test, y_test)

    acc = result.history['acc']
    auc = result.history['AUC']

    if show:
        plt.plot(acc)
        plt.show()

        plt.plot(auc)
        plt.show()
    # print(final_acc)
    metrics = {'acc': acc, 'auc': auc}
    return metrics, final_acc[2]
