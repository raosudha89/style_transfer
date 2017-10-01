import sys, pdb
import csv
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict

def to_float_judgement(judgement):
    convert = {'Perfect': 5, 'Comprehensible': 4, 'Somewhat Comprehensible': 3, 'Incomprehensible': 2, 'Other': 1}
    return convert[judgement]

if __name__ == '__main__':
    results_csv = open(sys.argv[1])
    results_reader = csv.reader(results_csv, delimiter=',')
    output_tsv_prefix = sys.argv[2]
    
    i = 0
    model_sentences = {}
    model_sentence_judgements = {}
    for row in results_reader:
        if i == 0:
            print row[27:37]
            print row[37:42]
            i += 1
            continue
        for i in range(27, 37, 2):
            id, sent = row[i], row[i+1]
            no, model = id.split('_', 1)
            if model in ['ref0', 'ref1', 'ref2', 'ref3']:
                model = 'ref0_1_2_3'
            no = int(no)-1
            if model not in model_sentences:
                model_sentences[model] = [None]*500
                model_sentence_judgements[model] = [None]*500
            model_sentences[model][int(no)] = sent
            
            if not model_sentence_judgements[model][int(no)]:
                model_sentence_judgements[model][int(no)] = []
            if row[37+(i-27)/2] != '':
                model_sentence_judgements[model][no].append(to_float_judgement(row[37+(i-27)/2]))
    
    for model in model_sentence_judgements:
        output_tsv_file = open(output_tsv_prefix+'.'+model, 'w')
        for no in range(500):
            if model_sentence_judgements[model][no]:
                mean_judgement = numpy.mean(model_sentence_judgements[model][no])
            else:
                output_tsv_file.write('None\tNone,None,None\t%s\n' % (model_sentences[model][no]))
                continue
            line = '%.2f\t' % (mean_judgement)
            line += str(model_sentence_judgements[model][no][0])
            for val in model_sentence_judgements[model][no][1:]:
                line += ',' + str(val)
            line += '\t' + model_sentences[model][no]
            output_tsv_file.write(line+'\n')
            
    #plt.hist(mean_judgements, color='y')
    #plt.show()
            
