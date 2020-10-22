import pandas as pd
import re
import jieba


def getStopWords():
    path = "HGD_StopWords.txt"
    StopWords = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    return StopWords


def dataprocess():
    data = pd.read_csv('jdcomment.csv', encoding='GBK')
    commentlist = data['0'].tolist()
    labellist = data['1'].tolist()
    # for i in range(6):
    #     del commentlist[141]
    #     del labellist[141]
    cleanWord = '[\s"运行速度：""屏幕效果：""散热性能：""完形外观：""轻薄程度：""其他特色：""mx350""(&acute;,,&bull;?&bull;,,`)"~*a-zA-Z0-9."图一""图二""图三""图四""图五""联想""戴尔""小新""京东""顺丰到""顺丰""处理器""酷睿"' \
                '"快递""鲁大师""华为""东芝""惠普""小米""哈""笔记本""电脑""台式机""微软""宏碁""⊙﹏⊙"]'
    procComment = []
    for item in commentlist:
        d = re.sub(cleanWord, '', str(item))
        procComment.append(d)
    pd.DataFrame(procComment).to_csv('clearedComment.csv')
    return procComment, labellist


def SplitSentence(sentence, stopwordslist):
    splitedSentence = jieba.cut(sentence.strip(), cut_all=False)
    outstr = ""
    for words in splitedSentence:
        if words not in stopwordslist:
            if words not in ['\t', ' ', '\r', '\n', '\s']:
                outstr += (words + " ")
    return outstr.strip()


if __name__ == "__main__":
    comments = []
    commentlist, labellist = dataprocess()
    stopWordLists = getStopWords()
    pd.DataFrame(labellist).to_csv('label.csv')
    for item in commentlist:
        SplitedSentence = SplitSentence(item, stopWordLists)
        comments.append(SplitedSentence)
    pd.DataFrame(comments).to_csv('comments.csv')
