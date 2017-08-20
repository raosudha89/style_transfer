import sys
import pdb

if __name__ == '__main__':
    bible_versions = []
    for i in range(1, len(sys.argv)):
        file = open(sys.argv[i], 'r')
        start = False
        for line in file.readlines():
            if line.split('\t')[0] == '#columns':
                start = True
                bible_version = {}
                continue
            if start:
                orig_book_index, orig_chapter, orig_verse, _, orig_subverse, text = line.split('\t')
                text = (text.strip('\n').strip('\r'))
                bible_version['_'.join([orig_book_index, orig_chapter, orig_verse])] = text
        bible_versions.append(bible_version)
    bible_versions_sentences = [None]*len(bible_versions)
    for i in range(len(bible_versions)):
        bible_versions_sentences[i] = []
    for index in bible_versions[0].keys():
        missing = False
        for i in range(1, len(bible_versions)):
            text_a = bible_versions[0][index]
            try:
                bible_versions[i][index]
            except:
                missing = True
        if not missing:
            for i in range(len(bible_versions)):
                bible_versions_sentences[i].append(bible_versions[i][index])
    for i in range(len(bible_versions)):
        print len(bible_versions_sentences[i])
        bible_sentences_file = open(sys.argv[i+1]+'.sents', 'w')
        for sentence in bible_versions_sentences[i]:
            bible_sentences_file.write(sentence+'\n')
        
                