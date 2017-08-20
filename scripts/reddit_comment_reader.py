import sys, pdb
import json
import nltk
from random import shuffle

if __name__ == '__main__':
    reddit_comment_file = open(sys.argv[1], 'r')
    reddit_comment_sentences_file = open(sys.argv[2], 'w')
    reddit_comment_sentences_file.write('sentence_1,sentence_2,sentence_3,sentence_4,sentence_5\n')
    reddit_comments = []
    for line in reddit_comment_file.readlines():
        line = line.strip('\n')
        reddit_data = json.loads(line)
        comment = reddit_data['body'].replace('\n', ' ').replace('\r', ' ').replace('\"', '')
        sentences = nltk.sent_tokenize(comment)
        sentences = [s for s in sentences if len(s.split()) > 5 and len(s.split()) < 15 and 'http' not in s and '&' not in s]
        reddit_comments += sentences
    shuffle(reddit_comments)
    for i in range(2):
        comments = ""
        for j in range(5):
            comment = reddit_comments[i*10+j]
            comments += '\"' + comment.decode('utf-8', 'ignore') + '",'
        comments = comments[:-1] #to remove the last comma
        reddit_comment_sentences_file.write(comments + '\n')