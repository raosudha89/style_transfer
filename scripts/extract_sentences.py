import sys

if __name__ == '__main__':
    dataset = open(sys.argv[1], 'r')
    sentences_file = open(sys.argv[2], 'w')
    for line in dataset.readlines():
        parts = line.split('\t')
        sentence = parts[-1]
        sentences_file.write(sentence + '\n')