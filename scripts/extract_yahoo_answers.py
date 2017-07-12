import sys, pdb, os
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')
from collections import defaultdict

if __name__ == '__main__':
    xml_file = sys.argv[1]
    output_dir = sys.argv[2]
    category_sents = defaultdict(list)
    output_sents = []
    for event, elem in ET.iterparse(xml_file, events=('start', 'end', 'start-ns', 'end-ns')):
        if event == 'start' and elem.tag == 'uri':
            uri = elem.text
        if event == 'start' and (elem.tag == 'maincat' and elem.text):
            curr_maincat = elem.text.strip()
            category_sents[curr_maincat] += output_sents
            output_sents = []
        if event == 'start' and (elem.tag == 'answer_item' or elem.tag == 'content'):
            text = elem.text
            if not text:
                continue
            try:
                text = text.replace('&lt;br /&gt;&#xa;', ' ').replace('\n', ' ').replace('<br />', ' ')
                text = text.replace('\n',' ').replace('\r',' ').replace('\"','')
                sents = nltk.sent_tokenize(text)
            except:
                continue
            output_sents += [s for s in sents if len(s.split()) > 5 and len(s.split()) < 20 and 'http' not in s and 'www' not in s and 'WWW' not in s]
    for category in category_sents.keys():
        output_file = open(os.path.join(output_dir,'_'.join(category.split())), 'w')
        for sent in category_sents[category]:
            output_file.write(sent + '\n')