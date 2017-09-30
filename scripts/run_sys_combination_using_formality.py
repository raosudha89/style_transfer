import sys
import numpy as np

model_names = ['rule_based', 'smt', 'smt_ensemble', 'nmt_yahoo_answers', 'nmt', 'nmt_ensemble']
model_files = [None]*6
model_files[0] = '/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.informal'
model_files[1] = '/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.smt.formal'
model_files[2] = '/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining6x.gt10editdist.lm_All_gt_0.smt.formal'
model_files[3] = '/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K_epoch14_29.77.formal'
model_files[4] = '/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.smt.model.selftraining10x_epoch8_31.89.formal'
model_files[5] = '/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.selftraining10x.usingnmtbacktranslate.nmtbpe.usinglm_epoch8_25.80.formal'
sys_comb_file = open('/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.sys_comb_by_formality.formal', 'w')
sys_comb_with_modelname_file = open('/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.sys_comb_by_formality.formal.withmodelname', 'w')

model_sentences = [None]*6
model_sentence_formality = [None]*6
for i in range(6):
	model_sentences[i] = open(model_files[i], 'r').readlines()
	model_sentence_formality[i] = [float(score.strip('\n').split()[0]) for score in open(model_files[i]+'.formality.predictions', 'r').readlines()]

for j in range(len(model_sentences[0])):
	scores = []
	sentences = []
	for i in range(6):
		sentences.append(model_sentences[i][j])
		scores.append(model_sentence_formality[i][j])
	sys_comb_file.write(sentences[np.argmax(scores)])
	sys_comb_with_modelname_file.write(model_names[np.argmax(scores)] + ':\t' +  sentences[np.argmax(scores)])
