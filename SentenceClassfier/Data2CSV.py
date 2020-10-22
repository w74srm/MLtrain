import csv
import pandas as pd


data = pd.DataFrame()
with open("ProcessedSentence.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    line = [i.strip() for i in lines]
data['comment'] = line

with open("All_label.txt", 'r', encoding='utf-8') as f:
    all_labels = f.readlines()[0]
    labels = all_labels.split(',')
data['labels'] = labels

data.to_csv('dataset.csv')