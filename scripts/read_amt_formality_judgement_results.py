import sys, pdb
import csv
import numpy
import matplotlib.pyplot as plt

def to_float_judgement(judgement):
    convert = {'1': -3.0, '2': -2.0, '3': -1.0, '4': 0.0, '5': 1.0, '6': 2.0, '7': 3.0}
    return convert[judgement]

if __name__ == '__main__':
    results_csv = open(sys.argv[1])
    results_reader = csv.reader(results_csv, delimiter=',')
    output_tsv = open(sys.argv[2], 'w')
    
    sentence_judgements = {}
    i = 0
    sentences_in_order = []
    for row in results_reader:
        if i == 0:
            print row[27:32]
            print row[33:38]
            i += 1
            continue
        curr_sentences = row[27:32]
        judgements = row[33:38]
        for curr_sentence in curr_sentences:
            if curr_sentence not in sentences_in_order:
                sentences_in_order.append(curr_sentence)
            curr_judgement = judgements[curr_sentences.index(curr_sentence)]
            if curr_judgement == '':
                continue
            try:
                sentence_judgements[curr_sentence].append(to_float_judgement(curr_judgement))
            except:
                sentence_judgements[curr_sentence] = [to_float_judgement(curr_judgement)]
    mean_judgements = []
    #for sentence, judgement in sentence_judgements.iteritems():
    for sentence in sentences_in_order:
        judgement = sentence_judgements[sentence]
        mean_judgements.append(numpy.mean(judgement))
        line = str(numpy.mean(judgement)) + '\t'
        line += str(judgement[0])
        for val in judgement[1:]:
            line += ',' + str(val)
        line += '\t' + sentence
        output_tsv.write(line+'\n')
    #plt.hist(mean_judgements, color='y')
    #plt.show()
            
