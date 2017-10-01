import sys, pdb
import csv
import numpy
import matplotlib.pyplot as plt
from collections import defaultdict

def to_float_judgement(judgement):
    convert = {'All meaning': 5, 'Most meaning': 4, 'Some meaning': 3, 'Little meaning': 2, 'None': 1}
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
            print row[27:47]
            print row[47:53]
            i += 1
            continue
        for i in range(27, 47, 4):
            id_a, sent_a = row[i], row[i+1]
            id_b, sent_b = row[i+2], row[i+3]
            no_a, model_a = id_a.split('_', 1)
            no_b, model_b = id_b.split('_', 1)
            if model_b in ['ref0', 'ref1', 'ref2', 'ref3']:
                model = 'ref0_1_2_3'
            else:
                model = model_b
            assert(no_a == no_b)
            no = int(no_a)-1
            sent = sent_b
            if model not in model_sentences:
                model_sentences[model] = [None]*500
                model_sentence_judgements[model] = [None]*500
            model_sentences[model][int(no)] = sent
            
            if not model_sentence_judgements[model][int(no)]:
                model_sentence_judgements[model][int(no)] = []
            if row[47+(i-27)/4] != '':
                model_sentence_judgements[model][no].append(to_float_judgement(row[47+(i-27)/4]))
    
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
            
