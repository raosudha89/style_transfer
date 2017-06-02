import sys

if __name__ == '__main__':
    dataset = open(sys.argv[1], 'r')
    sentences_file = open(sys.argv[2], 'w')
    for line in dataset.readlines():
        _, _, _, sentence = line.split('\t')
        sentences_file.write(sentence + '\n')