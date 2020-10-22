import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('spam_train.csv', usecols=range(1, 59))
    df.fillna(0)
    array = np.array(df).astype(float)
    x_list, y_list = [], []
    for i in range(4000):
        mat = array[i, 0:57]
        label = array[i, 57]
        x_list.append(mat)
        y_list.append(label)
    x = np.array(x_list)
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    x[:, -3] /= np.mean(x[:, -3])
    y = np.array(y_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=40)
    w, b = train(x_train, y_train, 30)
    acc = validate(x_test, y_test, w, b)
    print('acc on val data is ' + str(acc))


def sigmoid(x):
    if x >= 0:
        return 1/(1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def train(x_train, y_train, epoch):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    b = 0
    w = np.ones(dim)
    lr = 1
    lr_b = 0
    lr_w = np.zeros(dim)
    reg_rate = 0.001

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        for j in range(num):
            y_pre = w.dot(x_train[j, :]) + b
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j]-sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * w[k]

        b_g /= num
        w_g /= num
        lr_b += b_g ** 2
        lr_w += w_g ** 2
        b -= lr / np.sqrt(lr_b) * b_g
        w -= lr / np.sqrt(lr_w) * w_g

        if i % 3 == 0:
            acc = 0
            result = np.zeros(num)
            for j in range(num):
                y_pre = w.dot(x_train[j, :]) + b
                sig = sigmoid(y_pre)
                if sig >= 0.5:
                    result[j] = 1
                if result[j] == y_train[j]:
                    acc += 1.0
            print('after ' + str(i) + ' acc is: ' + str(acc / num))
    return w, b


def validate(x_test, y_test, w, b):
    num = 400
    acc = 0
    result = np.zeros(num)
    for i in range(num):
        y_pre = w.dot(x_test[i, :]) + b
        sig = sigmoid(y_pre)
        if sig >= 0.5:
            result[i] = 1
        if result[i] == y_test[i]:
            acc += 1.0
    return acc / num


if __name__ == "__main__":
    main()
