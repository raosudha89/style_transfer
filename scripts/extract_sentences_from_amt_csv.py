import sys
import csv

if __name__ == '__main__':
    csv_file = open(sys.argv[1])
    csv_reader = csv.reader(csv_file, delimiter=',')
    sentences_file = open(sys.argv[2], 'w')
    i = 0
    for row in csv_reader:
        if i == 0:
            i += 1
            continue
        for sentence in row:
            sentences_file.write("%s\n" % (sentence))
        
        