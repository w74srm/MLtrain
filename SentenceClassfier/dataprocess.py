import jieba


def getStopWords():
    path = "HGD_StopWords.txt"
    StopWords = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    return StopWords


def SplitSentence(sentence, stopwordslist):
    splitedSentence = jieba.cut(sentence.strip(), cut_all=False)
    outstr = ""
    for words in splitedSentence:
        if words not in stopwordslist:
            if words not in ['\t', ' ']:
                outstr += (words + " ")
    return outstr.strip()


if __name__ == "__main__":
    stopwords = getStopWords()
    inputfiles = open("ALL_Comment.txt", 'r', encoding='utf-8')
    outputfiles = open("ProcessedSentence.txt", 'w', encoding='utf-8')
    for line in inputfiles:
        writeline = SplitSentence(line, stopwords)
        outputfiles.writelines(writeline.strip() + '\n')
    inputfiles.close()
    outputfiles.close()
