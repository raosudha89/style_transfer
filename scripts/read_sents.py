import sys

if __name__ == "__main__":
	org_informal = open(sys.argv[1], 'r').readlines()
	rewrite_formal = open(sys.argv[2], 'r').readlines()
	rule_based_formal = open(sys.argv[3], 'r').readlines()
	smt_formal = open(sys.argv[4], 'r').readlines()
	nmt_selftraining_formal = open(sys.argv[5], 'r').readlines()
	#nmt_glovefix_formal = open(sys.argv[6], 'r').readlines()
	#nmt_selftraining_formal = open(sys.argv[7], 'r').readlines()
	for i in range(100):
		print org_informal[i].strip('\n')
		print rewrite_formal[i].strip('\n')
		print rule_based_formal[i].strip('\n')
		print smt_formal[i].strip('\n')
		print nmt_selftraining_formal[i].strip('\n')
		#print nmt_glovefix_formal[i].strip('\n')
		print
