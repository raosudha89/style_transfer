import sys

if __name__ == '__main__':
    model_outputs = [None]*(len(sys.argv)-1)
    for i in range(len(sys.argv)-1):
        model_outputs[i] = open(sys.argv[i+1], 'r').readlines()
    for i in range(len(model_outputs[0])):
        for j in range(len(model_outputs)):
            print model_outputs[j][i].strip('\n')
        print
