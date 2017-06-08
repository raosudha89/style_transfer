import sys, pdb, os
from bs4 import BeautifulSoup
import nltk
from random import shuffle

if __name__ == '__main__':
    html_dir = sys.argv[1]
    output_file = open(sys.argv[2], 'w')
    output_file.write("sentence_1,sentence_2,sentence_3,sentence_4,sentence_5\n")
    answer_sents = []
    for file in os.listdir(html_dir):
        html = open(os.path.join(html_dir, file), 'r')
        parsed_html = BeautifulSoup(html, 'html.parser')
        for div in parsed_html.find_all('div', attrs={'class':'answer-detail Fw-n'}):
            if div.span:
                answer = div.span.text.strip('\n').strip()
                sents = nltk.sent_tokenize(answer)
                answer_sents += [s.replace('\n',' ').replace('\r',' ').replace('\"','') for s in sents if len(s.split()) > 5 and len(s.split()) < 15]
    shuffle(answer_sents)
    print len(answer_sents)
    for i in range(10):
        line = ''
        for j in range(5):
            line += '\"' + answer_sents[i*5+j].encode('utf-8') + '\",'
        line = line[:-1] #to remove the last comma
        output_file.write(line+'\n')
    output_file.close()