import sys, pdb
import editdistance

if __name__ == "__main__":
	informal_file = open(sys.argv[1], 'r')
	formal_file = open(sys.argv[2], 'r')
	filtered_informal_file = open(sys.argv[3], 'w')
	filtered_formal_file = open(sys.argv[4], 'w')
	informal_sents = informal_file.readlines()
	formal_sents = formal_file.readlines()
	for i in range(len(informal_sents)):
		dist = editdistance.eval(informal_sents[i], formal_sents[i])
		if dist >= 10:
			filtered_informal_file.write(informal_sents[i])
			filtered_formal_file.write(formal_sents[i])
