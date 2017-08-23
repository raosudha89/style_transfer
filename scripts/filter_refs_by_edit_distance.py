import sys, pdb
import editdistance

if __name__ == "__main__":
	informal_file = open(sys.argv[1], 'r')
	formal_ref0 = open(sys.argv[2], 'r')
	formal_ref1 = open(sys.argv[3], 'r')
	formal_ref2 = open(sys.argv[4], 'r')
	formal_ref3 = open(sys.argv[5], 'r')
	new_formal_ref0 = open(sys.argv[6], 'w')
	new_formal_ref1 = open(sys.argv[7], 'w')
	new_formal_ref2 = open(sys.argv[8], 'w')
	new_formal_ref3 = open(sys.argv[9], 'w')
	informal_sents = informal_file.readlines()
	formal_ref_sents = [None]*4
	formal_ref_sents[0] = formal_ref0.readlines()
	formal_ref_sents[1] = formal_ref1.readlines()
	formal_ref_sents[2] = formal_ref2.readlines()
	formal_ref_sents[3] = formal_ref3.readlines()
	for i in range(len(informal_sents)):
		pairs = [None]*4
		for j in range(4):
			pairs[j] = [editdistance.eval(informal_sents[i], formal_ref_sents[j][i]), formal_ref_sents[j][i]]
		pairs = sorted(pairs)
		new_formal_ref0.write(pairs[3][1])
		new_formal_ref1.write(pairs[2][1])
		new_formal_ref2.write(pairs[1][1])
		new_formal_ref3.write(pairs[0][1])
