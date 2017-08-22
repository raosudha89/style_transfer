import sys

if __name__ == "__main__":
	formality_predictions_file = open(sys.argv[1], 'r')
	total_score = 0
	count = 0
	for line in formality_predictions_file.readlines():
		score = line.split('\t')[0]
		total_score += float(score)
		count += 1
	print(total_score/count)
