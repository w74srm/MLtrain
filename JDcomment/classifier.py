import warnings

warnings.filterwarnings('ignore')  # 过滤无用warning

import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


def loadDataSet():
    commentlist = []
    labellist = []
    label = pd.read_csv('label.csv')
    outlabellist = label['0'].tolist()
    comment = pd.read_csv('comments.csv', encoding='GBK')
    outcomment = comment['0'].tolist()
    print(len(outcomment))
    cnt = 0
    for item in outcomment:
        if type(item) is float:
            del outlabellist[cnt]
        else:
            item = item.split(' ')
            commentlist.append(item)
        cnt += 1
    for item in outlabellist:
        if item > 3:
            labellist.append(1)
        else:
            labellist.append(0)
    return commentlist, labellist


def evaluation(outputs, labels):
    # outputs => 预测值，概率（float）
    # labels => 真实值，标签（0或1）
    outputs[outputs >= 0.5] = 1  # 大于等于 0.5 为正面
    outputs[outputs < 0.5] = 0  # 小于 0.5 为负面
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy


def train_word2vec(commentlist):
    model = Word2Vec(commentlist, size=100, window=5, min_count=1, workers=4, iter=5, sg=0)
    return model


class Preprocess():
    def __init__(self, sen_len, w2v_path):
        self.w2v_path = w2v_path  # word2vec的存储路径
        self.sen_len = sen_len  # 句子的固定长度
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 读取之前训练好的 word2vec
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 这里的 word 只会是 "<PAD>" 或 "<UNK>"
        # 把一个随机生成的表征向量 vector 作为 "<PAD>" 或 "<UNK>" 的嵌入
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        # 它的 index 是 word2idx 这个词典的长度，即最后一个
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 获取训练好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 遍历嵌入后的单词
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i + 1), end='\r')
            # 新加入的 word 的 index 是 word2idx 这个词典的长度，即最后一个
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        # 把 embedding_matrix 变成 tensor
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 将 <PAD> 和 <UNK> 加入 embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度，即 sen_len 的长度
        if len(sentence) > self.sen_len:
            # 如果句子长度大于 sen_len 的长度，就截断
            sentence = sentence[:self.sen_len]
        else:
            # 如果句子长度小于 sen_len 的长度，就补上 <PAD> 符号，缺多少个单词就补多少个 <PAD>
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self, sentences):
        # 把句子里面的字变成相对应的 index
        sentence_list = []
        for i, sen in enumerate(sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    # 没有出现过的单词就用 <UNK> 表示
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)


class JDDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class Atten_BiLSTM(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(Atten_BiLSTM, self).__init__()
        # embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否将 embedding 固定住，如果 fix_embedding 为 False，在训练过程中，embedding 也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 64),
                                        nn.Dropout(dropout),
                                        nn.Linear(64, 32),
                                        nn.Dropout(dropout),
                                        nn.Linear(32, 16),
                                        nn.Dropout(dropout),
                                        nn.Linear(16, 1),
                                        nn.Sigmoid())
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def attention(self, output, hidden):
        # output  (batch_size, seq_len, hidden_size * num_direction)
        # hidden (batch_size, num_layers * num_direction, hidden_size)

        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]  # (batch_size, seq_len, hidden_size)

        hidden = torch.sum(hidden, dim=1)
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)

        atten_w = self.attention_layer(hidden)  # (batch_size, 1, hidden_size)
        m = nn.Tanh()(output)  # (batch_size, seq_len, hidden_size)

        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        softmax_w = F.softmax(atten_context, dim=-1)

        context = torch.bmm(softmax_w, output)

        return context.squeeze(1)

    def forward(self, inputs):
        inputs = self.embedding(inputs)

        # x (batch, seq_len, hidden_size)
        # hidden (num_layers *num_direction, batch_size, hidden_size)
        x, (hidden, _) = self.lstm(inputs, None)
        hidden = hidden.permute(1, 0, 2)  # (batch_size, num_layers *num_direction, hidden_size)

        # atten_out [batch_size, 1, hidden_dim]
        atten_out = self.attention(x, hidden)
        return self.classifier(atten_out)


def add_label(outputs, threshold=0.9):
    id = (outputs >= threshold) | (outputs < 1 - threshold)
    outputs[outputs >= threshold] = 1  # 大于等于 threshold 为正面
    outputs[outputs < 1 - threshold] = 0  # 小于 threshold 为负面
    return outputs.long(), id


def training(batch_size, n_epoch, lr, X_train, y_train, val_loader, model, device):
    # 输出模型总的参数数量、可训练的参数数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    loss = nn.BCELoss()  # 定义损失函数为二元交叉熵损失 binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer用Adam，设置适当的学习率lr
    total_loss, total_acc, best_acc = 0, 0, 0

    for epoch in range(n_epoch):
        print(X_train.shape)
        train_dataset = JDDataset(X=X_train, y=y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        total_loss, total_acc = 0, 0

        # training
        model.train()  # 将 model 的模式设为 train，这样 optimizer 就可以更新 model 的参数
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long)  # 因为 device 为 "cuda"，将 inputs 转成 torch.cuda.LongTensor
            labels = labels.to(device,
                               dtype=torch.float)  # 因为 device 为 "cuda"，将 labels 转成 torch.cuda.FloatTensor，loss()需要float

            optimizer.zero_grad()  # 由于 loss.backward() 的 gradient 会累加，所以每一个 batch 后需要归零
            outputs = model(inputs)  # 模型输入Input，输出output
            outputs = outputs.squeeze()  # 去掉最外面的 dimension，好让 outputs 可以丢进 loss()
            batch_loss = loss(outputs, labels)  # 计算模型此时的 training loss
            batch_loss.backward()  # 计算 loss 的 gradient
            optimizer.step()  # 更新模型参数

            accuracy = evaluation(outputs, labels)  # 计算模型此时的 training accuracy
            total_acc += (accuracy / batch_size)
            total_loss += batch_loss.item()
        print('Epoch | {}/{}'.format(epoch + 1, n_epoch))
        t_batch = len(train_loader)
        print('Train | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        model.eval()  # 将 model 的模式设为 eval，这样 model 的参数就会被固定住
        # self-training
        if val_loader is None:
            torch.save(model, "ckpt.model")
        else:
            with torch.no_grad():
                total_loss, total_acc = 0, 0

                for i, (inputs, labels) in enumerate(val_loader):
                    inputs = inputs.to(device, dtype=torch.long)  # 因为 device 为 "cuda"，将 inputs 转成 torch.cuda.LongTensor
                    labels = labels.to(device,
                                       dtype=torch.float)  # 因为 device 为 "cuda"，将 labels 转成 torch.cuda.FloatTensor，loss()需要float

                    outputs = model(inputs)  # 模型输入Input，输出output
                    outputs = outputs.squeeze()  # 去掉最外面的 dimension，好让 outputs 可以丢进 loss()
                    batch_loss = loss(outputs, labels)  # 计算模型此时的 training loss
                    accuracy = evaluation(outputs, labels)  # 计算模型此时的 training accuracy
                    total_acc += (accuracy / batch_size)
                    total_loss += batch_loss.item()

                v_batch = len(val_loader)
                print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
                if total_acc > best_acc:
                    # 如果 validation 的结果优于之前所有的結果，就把当下的模型保存下来，用于之后的testing
                    best_acc = total_acc
                    torch.save(model, "ckpt.model")
        print('-----------------------------------------------')


def testing(batch_size, test_loader, model, device):
    model.eval()  # 将 model 的模式设为 eval，这样 model 的参数就会被固定住
    ret_output = []  # 返回的output
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = torch.tensor(inputs, dtype=torch.long)
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
            outputs[outputs < 0.5] = 0  # 小于0.5为负面
            ret_output += outputs.int().tolist()

    return ret_output


if __name__ == "__main__":
    print("加载数据......")
    commentlist, labellist = loadDataSet()
    print("saving word2vec model......")
    model = train_word2vec(commentlist)
    model.save('w2v_all.model')
    # 通过 torch.cuda.is_available() 的值判断是否可以使用 GPU ，如果可以的话 device 就设为 "cuda"，没有的话就设为 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义句子长度、要不要固定 embedding、batch 大小、要训练几个 epoch、 学习率的值、 w2v的路径
    sen_len = 80
    fix_embedding = True  # fix embedding during training
    batch_size = 32
    epoch = 20
    lr = 2e-3
    w2v_path = 'w2v_all.model'

    print("loading data ...")

    # 对 input 跟 labels 做预处理
    preprocess = Preprocess(sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)

    train_x = preprocess.sentence_word2idx(commentlist)
    y = preprocess.labels_to_tensor(labellist)

    # train_x_no_label = preprocess.sentence_word2idx(train_x_no_label)

    # 把 data 分为 training data 和 validation data（将一部分 training data 作为 validation data）
    X_train, X_val, y_train, y_val = train_test_split(train_x, y, test_size=0.1, random_state=1, stratify=y)
    print('Train | Len:{} \nValid | Len:{}'.format(len(y_train), len(y_val)))

    val_dataset = JDDataset(X=X_val, y=y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义模型
    model = Atten_BiLSTM(embedding, embedding_dim=100, hidden_dim=96, num_layers=1, dropout=0.5,
                         fix_embedding=fix_embedding)
    model = model.to(device)  # device为 "cuda"，model 使用 GPU 来训练（inputs 也需要是 cuda tensor）
    # 开始训练
    training(batch_size, 10, lr, X_train, y_train, val_loader, model, device)
    # 测试模型并作预测
    test_x = preprocess.sentence_word2idx(X_val)
    test_dataset = JDDataset(X=test_x, y=y_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 读取模型
    # print('\nload model ...')
    # model = torch.load('ckpt.model')
    # # 测试模型
    # outputs = testing(batch_size, test_loader, model, device)
    #
    # # 保存为 csv
    # tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
    # print("save csv ...")
    # tmp.to_csv('predict.csv', index=False)
    # print("Finish Predicting")
