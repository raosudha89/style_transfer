import sys, pdb
import csv
from random import shuffle

if __name__ == '__main__':
    sentences_file = open(sys.argv[1])
    csv_file = open(sys.argv[2], 'w')
    csv_file.write('sentence_1,sentence_2,sentence_3,sentence_4,sentence_5\n')
    i = 0
    sentences = []
    for line in sentences_file.readlines():
        sentences.append(line.strip('\n'))
    shuffle(sentences)
    total = min(6000, len(sentences)/5)
    for i in range(total):
        line = ''
        for j in range(5):
            try:
                line += '\"' + sentences[i*5+j] + '",'
            except:
                pdb.set_trace()
        line = line[:-1] #remove last comma
        csv_file.write(line + '\n')
