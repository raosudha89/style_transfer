import sys

if __name__ == "__main__":
	output_file = open(sys.argv[1], 'r')
	translations_file = open(sys.argv[2], 'w')
	for line in output_file.readlines():
		if line[0] == 'H':
			_, _, translation = line.split('\t')
			translations_file.write(translation)
