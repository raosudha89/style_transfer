import sys, pdb
import csv
import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    results_csv = open(sys.argv[1])
    results_reader = csv.reader(results_csv, delimiter='\t')
    scores = []
    i = 0
    for row in results_reader:
        splits = row
        score = splits[0]
        scores.append(float(score))
    plt.hist(numpy.array(scores), bins=[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0], color='y')
    plt.savefig(sys.argv[2])
            