import sys
import pdb

if __name__ == "__main__":
	sentences = open(sys.argv[1], 'r')
	in_domain_scores = open(sys.argv[2], 'r')
	out_domain_scores = open(sys.argv[3], 'r')
	in_domain_name = sys.argv[4]
	sentences_lines = sentences.readlines()
	in_domain_lines = in_domain_scores.readlines()
	out_domain_lines = out_domain_scores.readlines()
	output_file = open(sys.argv[1]+'.'+in_domain_name+'.normdiff', 'w')
	for i in range(len(sentences_lines)):
		_, in_prob, _, in_oov = in_domain_lines[i].strip('\n').split()
		_, out_prob, _, out_oov = out_domain_lines[i].strip('\n').split()
		if int(in_oov) < 4 and int(out_oov) < 4:
			sentence = sentences_lines[i].strip('\n')
			num_toks = len(sentence.split())
			norm_diff = (float(in_prob) - float(out_prob))/num_toks
			output_file.write('%s\t%s\n' % (norm_diff, sentence))
