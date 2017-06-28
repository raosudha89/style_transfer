import sys, pdb
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')

if __name__ == '__main__':
    xml_file = sys.argv[1]
    output_sents = []
    main_cat = False
    total_output_sents = 0
    curr_highest = 0
    batch_no = 1
    output_file_name = sys.argv[2]+'.batch'+str(batch_no)
    for event, elem in ET.iterparse(xml_file, events=('start', 'end', 'start-ns', 'end-ns')):
        if event == 'start' and (elem.tag == 'maincat' and elem.text == 'Entertainment & Music'):
	# if event == 'start' and (elem.tag == 'maincat' and elem.text == 'Travel'):
        # if event == 'start' and (elem.tag == 'maincat' and elem.text == 'Business & Finance'):
            # main_cat = True
        # if event == 'start' and (elem.tag == 'subcat' and elem.text == 'Celebrities' and main_cat):
        # if event == 'start' and (elem.tag == 'subcat' and elem.text == 'Travel (General)' and main_cat):
        # if event == 'start' and (elem.tag == 'subcat' and elem.text == 'Careers & Employment' and main_cat):
            for sent in output_sents:
                output_file = open(output_file_name, 'a')
		output_file.write(sent.encode('utf-8') +'\n')
            total_output_sents += len(output_sents)
            if total_output_sents > curr_highest+5000:
		output_file.close()
		batch_no += 1
		output_file_name = sys.argv[2]+'.batch'+str(batch_no) 
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
	    	text = text.replace('&lt;br /&gt;&#xa;', ' ').replace('\n', ' ').replace('<br />', ' ')
		text = text.replace('\n',' ').replace('\r',' ').replace('\"','')
                sents = nltk.sent_tokenize(text)
            except:
                continue
            output_sents += [s for s in sents if len(s.split()) > 5 and len(s.split()) < 20 and 'http' not in s]
