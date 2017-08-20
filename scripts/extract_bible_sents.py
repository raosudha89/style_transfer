import sys
import pdb

if __name__ == '__main__':
    file = open(sys.argv[1], 'r')
    sents_file = open(sys.argv[2], 'w')
    start = False
    for line in file.readlines():
        if line.split('\t')[0] == '#columns':
            start = True
            continue
        if start:
            parts = line.split('\t')
            # pdb.set_trace()
            text = parts[5]
            sentence = (text.strip('\n').strip('\r'))
            sents_file.write(sentence + '\n')

        
                