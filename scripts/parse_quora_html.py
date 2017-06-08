import sys, pdb, os
from bs4 import BeautifulSoup
import nltk
from random import shuffle

if __name__ == '__main__':
    html_dir = sys.argv[1]
    output_file = open(sys.argv[2], 'w')
    output_file.write("sentence_1,sentence_2,sentence_3,sentence_4,sentence_5\n")
    quora_sents = []
    for file in os.listdir(html_dir):
        html = open(os.path.join(html_dir, file), 'r')
        parsed_html = BeautifulSoup(html, 'html.parser')
        for p in parsed_html.find_all('p', attrs={'class':'qtext_para'}):
            sent = p.text.strip('\n').strip()
            sents = nltk.sent_tokenize(sent)
            quora_sents += [s.replace('\n',' ').replace('\r',' ').replace('\"','') for s in sents if len(s.split()) > 5 and len(s.split()) < 15]
    shuffle(quora_sents)
    print len(quora_sents)
    for i in range(10):
        line = ''
        for j in range(5):
            line += '\"' + quora_sents[i*5+j].encode('utf-8') + '\",'
        line = line[:-1] #to remove the last comma
        output_file.write(line+'\n')
    output_file.close()