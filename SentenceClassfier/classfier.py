import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def getStopWords():
    path = "HGD_StopWords.txt"
    StopWords = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    return StopWords


def train(x_train, y_train, epochs, dim):
    num = len(x_train)
    b = 0
    w = np.ones(dim)
    lr = 1
    reg_rate = 0.1
    lr_b = 0
    lr_w = np.zeros(dim)
    for i in range(epochs):
        b_grad = 0
        w_grad = np.zeros(dim)
        for j in range(num):
            b_grad -= 2.0 * (y_train[j] - b - w.dot(x_train[j, :]))
            for k in range(dim):
                w_grad[k] -= 2.0 * (y_train[j] - b - w.dot(x_train[j, :])) * x_train[j, k]

        b_grad /= num
        w_grad /= num

        for m in range(dim):
            w_grad[m] += reg_rate * w[m]

        lr_b += b_grad ** 2
        lr_w += w_grad ** 2

        b -= lr / np.sqrt(lr_b) * b_grad
        for m in range(dim):
            w[m] -= lr / np.sqrt(lr_w[m]) * w_grad[m]
        print("epochs:" + str(i))

    return w, b


def validate(x_test, y_test, w, b):
    loss = 0
    cnt = 0
    for i in range(len(x_test)):
        y_pred = b + w.dot(x_test[i, :])
        loss += (y_test[i] - y_pred) ** 2
        if y_pred >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        if y_test[i] == y_pred:
            cnt += 1
    loss /= len(x_test)
    cnt /= len(x_test)
    return loss, cnt


def knn(x_train, y_train, x_test, y_test):
    kn = KNeighborsClassifier()
    kn.fit(x_train, y_train)
    y_pre = kn.predict(x_test)
    test_acc = accuracy_score(y_pre.astype('int'), y_test.astype('int'))
    print("knn分类器准确率" + str(test_acc))
    return test_acc


def randforest(x_train, y_train, x_test, y_test):
    Rf = RandomForestClassifier(n_estimators=20)
    Rf.fit(x_train, y_train)
    y_pre = Rf.predict(x_test)
    test_acc = accuracy_score(y_pre.astype('int'), y_test.astype('int'))
    print("随机森林准确率" + str(test_acc))
    return test_acc


def decisiontree(x_train, y_train, x_test, y_test):
    Dt = DecisionTreeClassifier()
    Dt.fit(x_train, y_train)
    y_pre = Dt.predict(x_test)
    test_acc = accuracy_score(y_pre.astype('int'), y_test.astype('int'))
    print("决策树准确率" + str(test_acc))
    return test_acc


def SVMc(x_train, y_train, x_test, y_test):
    SVM = SVC(kernel='rbf', C=1.0, gamma='auto')#径向基核函数在文本分类效果较好
    SVM.fit(x_train, y_train)
    y_pre = SVM.predict(x_test)
    test_acc = accuracy_score(y_pre.astype('int'), y_test.astype('int'))
    print("SVM准确率" + str(test_acc))
    return test_acc


def main():
    tf_idf_acc = []
    cnt_acc = []

    df = pd.read_csv("dataset.csv", index_col=0, encoding='GBK')
    df = df.fillna('')
    array = np.array(df)
    comment = array[:, 0]
    labels = array[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(comment, labels, train_size=0.9, random_state=42)
    stopwords = getStopWords()

    #TF-IDF
    TF_Vec = TfidfVectorizer(max_df=0.8, min_df=3, stop_words=frozenset(stopwords))
    x_train_tfidfvec = TF_Vec.fit_transform(x_train)
    x_test_tfidfvec = TF_Vec.transform(x_test)
    x_train_tfidfnparray = x_train_tfidfvec.A
    x_test_tfidfnparray = x_test_tfidfvec.A

    #CountVect
    Cnt_Vec = CountVectorizer(max_df=0.8, min_df=3, stop_words=frozenset(stopwords))
    x_train_cntvec = Cnt_Vec.fit_transform(x_train)
    x_test_cntvec = Cnt_Vec.transform(x_test)
    x_train_cntnparray = x_train_cntvec.A
    x_test_cntnparray = x_test_cntvec.A

    w, b = train(x_train_tfidfnparray, y_train, 10, len(x_train_tfidfnparray[0]))
    loss, acc = validate(x_test_tfidfnparray, y_test, w, b)
    print("total loss with tf-idf = " + str(loss))
    print("acc on test_set = " + str(acc))
    tf_idf_acc.append(acc)

    w, b = train(x_train_cntnparray, y_train, 10, len(x_train_cntnparray[0]))
    loss, acc = validate(x_test_cntnparray, y_test, w, b)
    print("total loss with cnt = " + str(loss))
    print("acc on test_set = " + str(acc))
    cnt_acc.append(acc)

    print("tf-idf")
    tf_idf_acc.append(knn(x_train_tfidfvec, y_train.astype('int'), x_test_tfidfvec, y_test))
    tf_idf_acc.append(randforest(x_train_tfidfvec, y_train.astype('int'), x_test_tfidfvec, y_test))
    tf_idf_acc.append(decisiontree(x_train_tfidfvec, y_train.astype('int'), x_test_tfidfvec, y_test))
    tf_idf_acc.append(SVMc(x_train_tfidfvec, y_train.astype('int'), x_test_tfidfvec, y_test))
    print("cnt")
    cnt_acc.append(knn(x_train_cntvec, y_train.astype('int'), x_test_cntvec, y_test))
    cnt_acc.append(randforest(x_train_cntvec, y_train.astype('int'), x_test_cntvec, y_test))
    cnt_acc.append(decisiontree(x_train_cntvec, y_train.astype('int'), x_test_cntvec, y_test))
    cnt_acc.append(SVMc(x_train_cntvec, y_train.astype('int'), x_test_cntvec, y_test))

    x_name = ['logistics', 'knn', 'randforest', 'decesionTree', 'SVM']
    x = np.arange(len(x_name))
    width = 0.3
    plt.bar(x-0.15, tf_idf_acc, width=width, label='tf_idf', color='red', tick_label=x_name)
    plt.bar(x+0.15, cnt_acc, width=width, label='cnt', color='orange', tick_label=x_name)

    # for a, b in zip(x, tf_idf_acc):
    #     plt.text(a-0.15, b+0.1, b, ha='center', va='bottom')
    # for a, b in zip(x, cnt_acc):
    #     plt.text(a+0.15, b+0.1, b, ha='center', va='bottom')

    plt.xticks(x, x_name, rotation=0, fontsize=10)
    plt.legend(loc="upper left")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.ylim(0.4, 1)
    plt.ylabel('准确率')
    plt.xlabel('分类方法')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (15.0, 8.0)
    plt.title("compare")
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    main()
