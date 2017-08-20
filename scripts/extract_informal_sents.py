import sys, pdb
import csv
from random import shuffle

if __name__ == '__main__':
    answers_csv = open(sys.argv[1])
    answers_reader = csv.reader(answers_csv, delimiter='\t')
    informal_sentences_file = open(sys.argv[2], 'w')
    for row in answers_reader:
        # avg_score, scores, id, sentence = row
        if len(row) == 3:
            avg_score, scores, sentence = row
        elif len(row) == 2:
            avg_score, sentence = row
        if 'http' in sentence:
            continue
        sentence = sentence.strip()
        if len(sentence.split()) < 5 or len(sentence.split()) > 20:
            continue
        if float(avg_score) <= -1.0:
            informal_sentences_file.write(sentence+'\n')

            
