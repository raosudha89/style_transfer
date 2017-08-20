import sys, pdb
from nltk.tree import *

if __name__ == '__main__':
    lexparse_file = open(sys.argv[1], 'r')
    parse_string = ''
    for line in lexparse_file.readlines():
        if line.strip('\n') == '':
            parse_tree = Tree.fromstring(parse_string)
            pdb.set_trace()
        else:    
            parse_string += line.strip('\n')