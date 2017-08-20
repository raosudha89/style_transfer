import sys, pdb
import csv

if __name__ == '__main__':
    results_csv = open(sys.argv[1])
    results_reader = csv.reader(results_csv, delimiter=',')
    output_tsv = open(sys.argv[2], 'w')
    
    sentence_judgements = {}
    i = 0
    sentences_in_order = []
    too_short = 0
    missing = 0
    for row in results_reader:
        if i == 0:
            print row[-7:-2]
            print row[-12:-7]
            i += 1
            continue
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
                print informal
                print formal
                too_short += 1
                continue
            output_tsv.write(informal + '\t' + formal + '\n')
    print '# of too short rewrites: %d' % too_short
    print '# of missing rewrites %d' % missing
