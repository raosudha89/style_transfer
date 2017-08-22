import sys, pdb
import csv
from random import shuffle

if __name__ == '__main__':
    informal_sentences_file = open(sys.argv[1])
    formal_sentences_file = open(sys.argv[2])
    fluency_csv_file = open(sys.argv[3], 'w')
    fluency_csv_file.write('id_1,sentence_1,id_2,sentence_2,id_3,sentence_3,id_4,sentence_4,id_5,sentence_5\n')
    meaning_csv_file = open(sys.argv[4], 'w')
    meaning_csv_file.write('id_1a,sentence_1a,id_1b,sentence_1b,'+ \
                           'id_2a,sentence_2a,id_2b,sentence_2b,' + \
                           'id_3a,sentence_3a,id_3b,sentence_3b,' + \
                           'id_4a,sentence_4a,id_4b,sentence_4b,' + \
                           'id_5a,sentence_5a,id_5b,sentence_5b\n')
    informal_sentences = informal_sentences_file.readlines()
    formal_sentences = formal_sentences_file.readlines()
    informal_formal_sentences = [None]*len(informal_sentences)
    for i in range(len(informal_formal_sentences)):
        informal_sentence = informal_sentences[i].strip('\n')
        formal_sentence = formal_sentences[i].strip('\n')
        id = i+1
        informal_formal_sentences[i] = ((str(id)+'a', informal_sentence), (str(id)+'b', formal_sentence))
    shuffle(informal_formal_sentences)
    output_line = ''
    all_sentences = []
    for i in range(len(informal_formal_sentences)):
        ((id_a, informal_sentence), (id_b, formal_sentence)) = informal_formal_sentences[i]
        all_sentences.append((id_a, informal_sentence))
        all_sentences.append((id_b, formal_sentence))
        if i%5 == 0 and i != 0:
            output_line = output_line[:-1] #remove last comma
            meaning_csv_file.write(output_line+'\n')
            output_line = ''
        output_line += id_a + ',' + '"' + informal_sentence.replace('\"', '') + '",' + \
                        id_b + ',' + '"' + formal_sentence.replace('\"', '') + '",'
        
    shuffle(all_sentences)
    output_line = ''
    for i in range(len(all_sentences)):
        id, sentence = all_sentences[i]
        if i%5 == 0 and i != 0:
            output_line = output_line[:-1] #remove last comma
            fluency_csv_file.write(output_line + '\n')
            output_line = ''
        output_line += id + ',' + '\"' + sentence.replace('\"','') + '",'
