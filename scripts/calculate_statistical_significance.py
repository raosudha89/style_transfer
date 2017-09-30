import sys
import scipy.stats

scores_1 = [float(score.strip('\n')) for score in open(sys.argv[1], 'r').readlines()]
scores_2 = [float(score.strip('\n')) for score in open(sys.argv[2], 'r').readlines()]

print scipy.stats.ttest_rel(scores_1, scores_2)
