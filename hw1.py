import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('hw1train.csv', usecols=range(3, 27))
    # 数据预处理
    x_train, x_test, y_train, y_test = dataprocess(df)
    w, b = train(x_train, y_train, 1600)
    # 在测试集看效果
    loss = validate(x_test, y_test, w, b)
    print('loss:'+str(loss))


def dataprocess(df):
    x_list, y_list = [], []
    df = df.replace(['NR'], [0.0])
    array = np.array(df).astype(float)
    for i in range(0, 4320, 18):
        for j in range(15):
            label = array[i + 9, j + 9]
            mat = array[i + 9, j:j + 9]
            mat = np.append(mat, array[i + 7, j + 9])
            mat = np.append(mat, array[i + 12, j + 9])
            mat = np.append(mat, array[i + 16, j + 9])
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    return x_train, x_test, y_train, y_test


def train(x_train, y_train, epoch):
    b = 0  # bias
    w = np.ones(12)  # weights
    lr = 1
    reg_rate = 0.01
    lr_b = 0  # 用于存放偏置值的梯度平方和
    lr_w = np.zeros(12)  # 用于存放权重的梯度平方和
    for i in range(epoch):
        b_grad = 0
        w_grad = np.zeros(12)
        for j in range(3240):
            b_grad -= 2.0 * (y_train[j] - b - w.dot(x_train[j, :]))
            for k in range(12):
                w_grad[k] -= 2.0 * (y_train[j] - b - w.dot(x_train[j, :])) * x_train[j, k]

        b_grad /= 3240
        w_grad /= 3240

        for m in range(12):
            w_grad[m] += reg_rate * w[m]

        lr_b += b_grad ** 2
        lr_w += w_grad ** 2

        b -= lr / np.sqrt(lr_b) * b_grad
        for m in range(12):
            w[m] -= lr / np.sqrt(lr_w[m]) * w_grad[m]

        if i % 200 == 0:
            loss = 0
            for j in range(3240):
                loss += (y_train[j] - b - w.dot(x_train[j, :])) ** 2
            loss /= 3240
            print(str(i) + ':' + str(loss))

    return w, b


def validate(x_test, y_test, w, b):
    loss = 0;
    for i in range(360):
        loss += (y_test[i] - b -w.dot(x_test[i, :])) ** 2
    loss /= 400
    return loss


if __name__ == "__main__":
    main()
