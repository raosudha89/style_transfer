import sys, pdb
import csv
from collections import defaultdict

if __name__ == '__main__':
    results_csv = open(sys.argv[1])
    results_reader = csv.reader(results_csv, delimiter=',')
    informal_file = open(sys.argv[2], 'w')
    formal_file_ref1 = open(sys.argv[3]+'0', 'w')
    formal_file_ref2 = open(sys.argv[3]+'1', 'w')
    formal_file_ref3 = open(sys.argv[3]+'2', 'w')
    formal_file_ref4 = open(sys.argv[3]+'3', 'w')
    sentence_judgements = {}
    i = 0
    sentences_in_order = []
    too_short = 0
    missing = 0
    rewrites = {}
    for row in results_reader:
        if i == 0:
            print row[-7:-2]
            print row[-12:-7]
            i += 1
            continue
        HITId = row[0]
        try:
            rewrites[HITId]
        except:
            rewrites[HITId] = defaultdict(list)
        formal_sentences = row[-5:]
        informal_sentences = row[-10:-5]
        for j in range(5):
            informal = informal_sentences[j].strip('\n')
            formal = formal_sentences[j].strip('\n')
            if formal.strip() == '{}':
                missing += 1
                continue
            informal_basic = informal.replace('!', '').replace('.', '').replace('?','')
            #Ignore if the rewrite into formal is too short
            # i.e. ratio between (informal-formal)/informal is 
            if (len(informal_basic.split()) - len(formal.split()))*1.0/len(informal_basic.split()) > 0.5:
                # pdb.set_trace()
                # print informal
                # print formal
                too_short += 1
                continue
            rewrites[HITId][informal].append(formal)

    print '# of too short rewrites: %d' % too_short
    print '# of missing rewrites %d' % missing
    for HITId in rewrites.keys():
        for informal in rewrites[HITId]:
            if len(rewrites[HITId][informal]) < 4:
                continue
            informal_file.write(informal+'\n')
            formal_file_ref1.write(rewrites[HITId][informal][0]+'\n')
            formal_file_ref2.write(rewrites[HITId][informal][1]+'\n')
            formal_file_ref3.write(rewrites[HITId][informal][2]+'\n')
            formal_file_ref4.write(rewrites[HITId][informal][3]+'\n')
