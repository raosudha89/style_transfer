import sys
import scipy.stats

automatic_metric_scores_file = open(sys.argv[1], 'r')
human_judgment_scores_file = open(sys.argv[2], 'r')

automatic_metric_scores = [float(score.split()[0]) for score in automatic_metric_scores_file.readlines()]
human_judgment_scores = [score.split()[0] for score in human_judgment_scores_file.readlines()]

new_automatic_metric_scores = []
new_human_judgment_scores = []

for i in range(len(human_judgment_scores)):
	if human_judgment_scores[i] != 'None':
		new_automatic_metric_scores.append(automatic_metric_scores[i])
		new_human_judgment_scores.append(float(human_judgment_scores[i]))
print(scipy.stats.spearmanr(new_automatic_metric_scores, new_human_judgment_scores))
