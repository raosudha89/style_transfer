import sys, pdb
import xml.etree.ElementTree as ET
import nltk

if __name__ == '__main__':
    xml_file = sys.argv[1]
    output_file = open(sys.argv[2], 'w')
    for event, elem in ET.iterparse(xml_file, events=('start', 'end', 'start-ns', 'end-ns')):
        if event == 'start' and (elem.tag == 'answer_item' or elem.tag == 'content'):
            text = elem.text
            if not text:
                continue
            try:
                sents = nltk.sent_tokenize(text)
            except:
                pdb.set_trace()
            sents = [s.replace('\n',' ').replace('\r',' ').replace('\"','') for s in sents if len(s.split()) > 5 and len(s.split()) < 15]
            for sent in sents:
                if 'http' in sent:
                    continue
                sent = sent.replace('<br />', ' ')
                output_file.write(sent.encode('utf-8') +'\n')