import sys, pdb
import xml.etree.ElementTree as ET
import nltk

if __name__ == '__main__':
    xml_file = sys.argv[1]
    output_file = open(sys.argv[2], 'w')
    output_sents = []
    main_cat = False
    total_output_sents = 0
    curr_highest = 0
    try:
        for event, elem in ET.iterparse(xml_file, events=('start', 'end', 'start-ns', 'end-ns')):
            # if event == 'start' and (elem.tag == 'maincat' and elem.text == 'Entertainment & Music'):
            # if event == 'start' and (elem.tag == 'maincat' and elem.text == 'Travel'):
            if event == 'start' and (elem.tag == 'maincat' and elem.text == 'Business & Finance'):
                main_cat = True
            # if event == 'start' and (elem.tag == 'subcat' and elem.text == 'Celebrities' and main_cat):
            # if event == 'start' and (elem.tag == 'subcat' and elem.text == 'Travel (General)' and main_cat):
            if event == 'start' and (elem.tag == 'subcat' and elem.text == 'Careers & Employment' and main_cat):
                for sent in output_sents:
                    output_file.write(sent.encode('utf-8') +'\n')
                total_output_sents += len(output_sents)
                if total_output_sents > curr_highest+100:
                    print total_output_sents
                    curr_highest = total_output_sents
                output_sents = []
                main_cat = False
            if event == 'start' and (elem.tag == 'answer_item' or elem.tag == 'content'):
                output_sents = []
                main_cat = False
                text = elem.text
                if not text:
                    continue
                try:
                    sents = nltk.sent_tokenize(text)
                except:
                    continue
                sents = [s.replace('\n',' ').replace('\r',' ').replace('\"','') for s in sents if len(s.split()) > 5 and len(s.split()) < 15]
                for sent in sents:
                    if 'http' in sent:
                        continue
                    sent = sent.replace('<br />', ' ')
                    output_sents.append(sent)
    except:
        pass
