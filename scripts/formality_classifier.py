import sys, argparse
from collections import defaultdict
from textblob import TextBlob
import pdb
import nltk
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

NER_TAGSET = ['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'MISC', 'NUMBER', 'PERCENT', \
			  'DATE', 'TIME', 'DURATION', 'SET', 'ORDINAL']
POS_TAGSET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
			  'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',\
			  'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',\
			  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',\
			  'WDT', 'WP', 'WP$', 'WRB']
UNIGRAM_SET = []
BIGRAM_SET = []
TRIGRAM_SET = []
DEPENDENCY_TUPLE_SET = []

class StanfordAnnotations:
	def __init__(self, token, lemma, pos, ner, head, depRel):
		self.token = token
		self.lemma = lemma
		self.pos = pos
		self.ner = ner
		self.head = head
		self.depRel = depRel

# FEATURE EXTRACTION FUNCTIONS

def get_case_features(sent_annotations, sentence):
	num_all_caps = 0
	for word_annotations in sent_annotations:
		if word_annotations.token.isupper():
			num_all_caps += 1
	if sentence.islower():
		is_sent_lower = 1
	else:
		is_sent_lower = 0
	if sent_annotations[0].token.isupper():
		is_first_word_caps = 1
	else:
		is_first_word_caps = 0
	return [num_all_caps, is_sent_lower, is_first_word_caps]

def get_dependency_tuples(sent_annotations):
	# (gov, typ, dep)  (gov, typ)  (typ, dep)  (gov, dep)
	global DEPENDENCY_TUPLE_SET
	dependency_tuples = []
	for word_annotations in sent_annotations:
		gov = sent_annotations[int(word_annotations.head)-1].pos
		typ = word_annotations.depRel
		dep = word_annotations.pos
		gov_typ_dep = [gov, typ, dep]
		if gov_typ_dep not in dependency_tuples:
			dependency_tuples.append(gov_typ_dep)
		gov_typ = [gov, typ]
		if gov_typ not in dependency_tuples:
			dependency_tuples.append(gov_typ)
		typ_dep = [typ, dep]
		if typ_dep not in dependency_tuples:
			dependency_tuples.append(typ_dep)
		gov_dep = [gov, dep]
		if gov_dep not in dependency_tuples:
			dependency_tuples.append(gov_dep)
	DEPENDENCY_TUPLE_SET = list(DEPENDENCY_TUPLE_SET + dependency_tuples)
	return dependency_tuples

def get_entity_features(sent_annotations):
	ner_tags = [0]*len(NER_TAGSET)
	person_mentions_total_len = 0
	for word_annotations in sent_annotations:
		if word_annotations.ner == 'O':
			continue
		if word_annotations.ner not in NER_TAGSET:
			pdb.set_trace()
		else:
			index = NER_TAGSET.index(word_annotations.ner)
			ner_tags[index] = 1
		if word_annotations.ner == 'PERSON':
			person_mentions_total_len += len(word_annotations.token)
	person_mentions_avg_len = person_mentions_total_len*1.0/len(sent_annotations)
	return ner_tags + [person_mentions_avg_len]

def get_lexical_features(words):
	num_contractions = 0
	total_word_len = 0
	for word in words:
		if '\'' in word:
			num_contractions += 1
		total_word_len += len(word)
	avg_num_contractions = num_contractions*1.0/len(words)
	avg_word_len = total_word_len*1.0/len(words)
	#TODO: avg word-log frequency acc to Google Ngram
	#TODO: avg formality score using Pavlick & Nenkova (2015)
	return [avg_num_contractions, avg_word_len]
	
def get_ngrams(sentence):
	blob = TextBlob(sentence)
	global UNIGRAM_SET, BIGRAM_SET, TRIGRAM_SET
	unigrams = sentence.split()
	bigrams = blob.ngrams(n=2)
	trigrams = blob.ngrams(n=3)
	for unigram in unigrams:
		if unigram not in UNIGRAM_SET:
			UNIGRAM_SET.append(unigram)
	for bigram in bigrams:
		if bigram not in BIGRAM_SET:
			BIGRAM_SET.append(bigram)
	for trigram in trigrams:
		if trigram not in TRIGRAM_SET:
			TRIGRAM_SET.append(trigram)	
	return unigrams, bigrams, trigrams	
				
def get_parse_features():
	pass
			
def get_POS_features(sent_annotations):
	pos_tag_ct = [0]*len(POS_TAGSET)
	for word_annotations in sent_annotations:
		try:
			pos_tag_ct[POS_TAGSET.index(word_annotations.pos)] += 1
		except:
			# print word_annotations.pos
			continue
	for i in range(len(pos_tag_ct)):
		pos_tag_ct[i] = pos_tag_ct[i]*1.0/len(sent_annotations)
	return pos_tag_ct

def get_punctuation_features(sentence):
	num_question_marks = sentence.count('?')
	num_ellipses = sentence.count('...')
	num_exclamations = sentence.count('!')
	return [num_question_marks, num_ellipses, num_exclamations]
				
def get_readability_features(sentence, words):
	num_words = len(words)
	num_chars = len(sentence) - sentence.count(' ')
	return [num_words, num_chars]

def get_subjectivity_features():
	pass
				
def get_word2vec_features():
	pass
				
def extract_features(corpus, stanford_annotations, args):
	features = {}
	
	for i in range(len(corpus)):
		id, sentence, rating = corpus[i]
		try:
			sent_annotations = stanford_annotations[id]
		except:
			# pdb.set_trace()
			# continue
			break
		words = sentence.split()
		
		#case features
		if args.case:
			case_features = get_case_features(sent_annotations, sentence)
		
		# dependency features
		if args.dependency:
			dependency_tuples = get_dependency_tuples(sent_annotations)
		
		# entity features
		if args.entity:
			entity_features = get_entity_features(sent_annotations)
		
		# lexical features
		if args.lexical:
			lexical_features = get_lexical_features(words)
		
		# ngram features
		if args.ngram:
			unigrams, bigrams, trigrams = get_ngrams(sentence)
				
		# parse features
		# if args.parse:
		# 	parse_features = get_parse_features()
		
		# POS features
		if args.POS:
			pos_features = get_POS_features(sent_annotations)
		
		# punctuation features
		if args.punctuation:
			punctuation_features = get_punctuation_features(sentence)
		
		# readability features
		if args.readability:
			readability_features = get_readability_features(sentence, words)
		
		# subjectivity features
		# if args.subjectivity:
		# 	 subjectivity_features = get_subjectivity_features()
		
		# word2vec features
		# if args.word2vec:
		# 	word2vec_features = get_word2vec_features()
	
		feature_set = case_features + entity_features + lexical_features + \
				pos_features + punctuation_features + readability_features
		features[id] = [feature_set, dependency_tuples, unigrams, bigrams, trigrams]
			
	for id in features.keys():
		[feature_set, dependency_tuples, unigrams, bigrams, trigrams] = features[id]
		dependency_tuples_feature = [0]*len(DEPENDENCY_TUPLE_SET)
		for dependency_tuple in dependency_tuples:
			dependency_tuples_feature[DEPENDENCY_TUPLE_SET.index(dependency_tuple)] = 1
		unigram_feature = [0]*len(UNIGRAM_SET)
		for unigram in unigrams:
			unigram_feature[UNIGRAM_SET.index(unigram)] = 1
		bigram_feature = [0]*len(BIGRAM_SET)
		for bigram in bigrams:
			bigram_feature[BIGRAM_SET.index(bigram)] = 1
		trigram_feature = [0]*len(TRIGRAM_SET)
		for trigram in trigrams:
			trigram_feature[TRIGRAM_SET.index(trigram)] = 1
		features[id] = feature_set + dependency_tuples_feature + unigram_feature + bigram_feature + trigram_feature
		
	return features

def read_data(dataset):
	corpus = []
	for line in dataset.readlines():
		rating, _, id, sentence = line.split('\t')
		corpus.append([id, sentence, rating])
	return corpus

def extract_annotations(dataset_stanford_annotations, corpus):
	stanford_annotations = {}
	id = corpus[0][0]
	k = 0
	sent_annotations = []
	for line in dataset_stanford_annotations.readlines():
		if line.strip('\n') == '':
			stanford_annotations[id] = sent_annotations
			k += 1
			if k == len(corpus):
				pdb.set_trace()
				break
			id = corpus[k][0]
			if id == 'JebBush_128518.0':
				pdb.set_trace()
			sent_annotations = []
		else:
			index, token, lemma, pos, ner, head, depRel = line.strip('\n').split('\t')
			if index == '1':
				if token != corpus[k][1].split()[0]:
					pdb.set_trace()
			word_annotations = StanfordAnnotations(token, lemma, pos, ner, head, depRel)
			sent_annotations.append(word_annotations)
	return stanford_annotations		

def main(args):
	dataset = open(args.dataset_file, 'r')
	dataset_stanford_annotations = open(args.dataset_stanford_annotations_file, 'r')
	corpus = read_data(dataset)
	stanford_annotations = extract_annotations(dataset_stanford_annotations, corpus)
	# pdb.set_trace()
	features = extract_features(corpus, stanford_annotations, args)
	ridge_regression = linear_model.Ridge(alpha = .5)
	feature_vectors = []
	labels = []
	for id, sentence, rating in corpus:
		if id == 'JebBush_128518.0':
			break
		feature_vectors.append(features[id])
		labels.append(float(rating))
	
	scores = cross_val_score(ridge_regression, feature_vectors, labels, cv=10)
	print scores
	# ridge_regression.fit(feature_vectors, labels)
	# predicted_labels = ridge_regression.predict(feature_vectors)
	# pdb.set_trace()
	
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--dataset_file", type = str)
	argparser.add_argument("--dataset_stanford_annotations_file", type = str)
	# argparser.add_argument("--glove_vectors", type = str)
	argparser.add_argument("--case", type=bool, default=True)
	argparser.add_argument("--dependency", type=bool, default=True)
	argparser.add_argument("--entity", type=bool, default=True)
	argparser.add_argument("--lexical", type=bool, default=True)
	argparser.add_argument("--ngram", type=bool, default=True)
	argparser.add_argument("--parse", type=bool, default=True)
	argparser.add_argument("--POS", type=bool, default=True)
	argparser.add_argument("--punctuation", type=bool, default=True)
	argparser.add_argument("--readability", type=bool, default=True)
	argparser.add_argument("--subjectivity", type=bool, default=True)
	argparser.add_argument("--word2vec", type=bool, default=True)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
