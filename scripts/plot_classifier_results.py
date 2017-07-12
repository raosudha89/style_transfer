import sys, pdb
import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    results_csv = open(sys.argv[1])
    results_reader = csv.reader(results_csv, delimiter='\t')
    scores = []
    i = 0
    for row in results_reader:
        score, sentence = row
        scores.append(float(score))
    plt.hist(scores, color='r')
    plt.show()
            