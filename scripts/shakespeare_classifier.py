import sys, argparse
from collections import defaultdict
from textblob import TextBlob
import pdb, time
import nltk, numpy
from nltk.tree import *
import gensim
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.metrics import make_scorer
import random

NER_TAGSET = ['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'MISC', 'NUMBER', 'PERCENT', \
			  'DATE', 'TIME', 'DURATION', 'SET', 'ORDINAL']
POS_TAGSET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
			  'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',\
			  'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',\
			  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',\
			  'WDT', 'WP', 'WP$', 'WRB']
FP_PRO_LIST = ['i', 'we', 'me', 'us', 'my', 'mine', 'our', 'ours']
TP_PRO_LIST = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs']

UNIGRAM_SET = []
BIGRAM_SET = []
TRIGRAM_SET = []
DEPENDENCY_TUPLE_SET = []
LEXPARSE_PRODUCTION_RULE_SET = []

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

def get_dependency_tuples(sent_annotations, is_test):
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
	if not is_test:
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
	
def get_ngrams(sentence, is_test):
	blob = TextBlob(sentence)
	global UNIGRAM_SET, BIGRAM_SET, TRIGRAM_SET
	unigrams = sentence.split()
	bigrams = blob.ngrams(n=2)
	trigrams = blob.ngrams(n=3)
	if not is_test:
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
				
def get_parse_features(stanford_parse_tree, sent_annotations, is_test):
	sent_len = len(sent_annotations)
	avg_depth = stanford_parse_tree.height()*1.0/sent_len
	productions = []
	for production in stanford_parse_tree.productions():
		if production.is_lexical():
			continue
		if not is_test:
			if production not in LEXPARSE_PRODUCTION_RULE_SET:
				LEXPARSE_PRODUCTION_RULE_SET.append(production)
		productions.append(production)
	avg_depth_feature = [avg_depth]
	return avg_depth_feature, list(productions)
			
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

def get_subjectivity_features(sent_annotations, sentence):
	subjectivity_features = []
	fp_pros = 0
	tp_pros = 0
	for word_annotations in sent_annotations:
		if word_annotations.token in FP_PRO_LIST:
			fp_pros += 1
		if word_annotations.token in TP_PRO_LIST:
			tp_pros += 1
	subjectivity_features.append(fp_pros*1.0/len(sent_annotations))
	subjectivity_features.append(tp_pros*1.0/len(sent_annotations))
	polarity, subjectivity = TextBlob(sentence).sentiment
	subjectivity_features.append(float(numpy.sign(polarity)))
	subjectivity_features.append(subjectivity)
	return subjectivity_features
				
def get_word2vec_features(sent_annotations, word2vec_model):
	word_vectors = []
	for word_annotations in sent_annotations:
		try:
			word_vector = word2vec_model[word_annotations.token]
			word_vectors.append(word_vector)
		except:
			# print word_annotations.token
			continue
	if len(word_vectors) == 0:
		avg_word_vectors = numpy.zeros(300)
	else:
		avg_word_vectors = numpy.transpose(numpy.mean(word_vectors, axis=0))
	return avg_word_vectors
				
def extract_features(sentences, stanford_annotations, stanford_parse_trees, args, word2vec_model, is_test=False):
	features = [None]*len(sentences)
	
	for i in range(len(sentences)):
		print i
		sentence = sentences[i]
		try:
			sent_annotations = stanford_annotations[i]
		except:
			# pdb.set_trace()
			# continue
			break
		words = sentence.split()
		
		feature_set = []
		
		#case features
		if args.case:
			case_features = get_case_features(sent_annotations, sentence)
			feature_set += case_features
		
		# dependency features
		if args.dependency:
			dependency_tuples = get_dependency_tuples(sent_annotations, is_test)
		else:
			dependency_tuples = None
		
		# entity features
		if args.entity:
			entity_features = get_entity_features(sent_annotations)
			feature_set += entity_features
		
		# lexical features
		if args.lexical:
			lexical_features = get_lexical_features(words)
			feature_set += lexical_features
		
		# ngram features
		if args.ngram:
			unigrams, bigrams, trigrams = get_ngrams(sentence.decode('utf-8', 'ignore'), is_test)
		else:
			unigrams, bigrams, trigrams = None, None, None
				
		# parse features
		if args.parse:
			avg_depth_feature, productions = get_parse_features(stanford_parse_trees[i], sent_annotations, is_test)
			feature_set += avg_depth_feature
		else:
			productions = None
		
		# POS features
		if args.POS:
			pos_features = get_POS_features(sent_annotations)
			feature_set += pos_features
		
		# punctuation features
		if args.punctuation:
			punctuation_features = get_punctuation_features(sentence)
			feature_set += punctuation_features
		
		# readability features
		if args.readability:
			readability_features = get_readability_features(sentence, words)
			feature_set += readability_features
		
		# subjectivity features
		if args.subjectivity:
			subjectivity_features = get_subjectivity_features(sent_annotations, sentence.decode('utf-8', 'ignore'))
			feature_set += subjectivity_features
		
		# word2vec features
		if args.word2vec:
			word2vec_features = get_word2vec_features(sent_annotations, word2vec_model)
			feature_set = numpy.concatenate((feature_set, word2vec_features), axis=0)
		
		features[i] = [feature_set, dependency_tuples, unigrams, bigrams, trigrams, productions]
			
	for i in range(len(features)):
		print i
		[feature_set, dependency_tuples, unigrams, bigrams, trigrams, productions] = features[i]
		features[i] = feature_set
		if args.dependency:
			dependency_tuples_feature = [0]*len(DEPENDENCY_TUPLE_SET)
			for dependency_tuple in dependency_tuples:
				try:
					dependency_tuples_feature[DEPENDENCY_TUPLE_SET.index(dependency_tuple)] = 1
				except:
					continue
			features[i] = numpy.concatenate((features[i], dependency_tuples_feature))
			
		if args.ngram:
			unigram_feature = [0]*len(UNIGRAM_SET)
			for unigram in unigrams:
				try:
					unigram_feature[UNIGRAM_SET.index(unigram)] = 1
				except:
					continue
			bigram_feature = [0]*len(BIGRAM_SET)
			for bigram in bigrams:
				try:
					bigram_feature[BIGRAM_SET.index(bigram)] = 1
				except:
					continue
			trigram_feature = [0]*len(TRIGRAM_SET)
			for trigram in trigrams:
				try:
					trigram_feature[TRIGRAM_SET.index(trigram)] = 1
				except:
					continue
			features[i] = numpy.concatenate((features[i], unigram_feature, bigram_feature, trigram_feature))	
				
		if args.parse:
			parse_feature = [0]*len(LEXPARSE_PRODUCTION_RULE_SET)
			for production in productions:
				try:
					parse_feature[LEXPARSE_PRODUCTION_RULE_SET.index(production)] += 1
				except:
					continue
			for i in range(len(parse_feature)):
				parse_feature[i] = parse_feature[i]*1.0/len(stanford_annotations[i])
			features[i] = numpy.concatenate((features[i], parse_feature))
				
	return features

def read_data(dataset):
	sentences = []
	for line in dataset.readlines():
		sentences.append(line.strip('\n'))
	return sentences

def extract_annotations(dataset_stanford_annotations):
	stanford_annotations = []
	sent_annotations = []
	for line in dataset_stanford_annotations.readlines():
		if line.strip('\n') == '':
			stanford_annotations.append(sent_annotations)
			sent_annotations = []
		else:
			index, token, lemma, pos, ner, head, depRel = line.strip('\n').split('\t')
			word_annotations = StanfordAnnotations(token, lemma, pos, ner, head, depRel)
			sent_annotations.append(word_annotations)
	return stanford_annotations		

def k_fold_cross_validation(features, labels, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	"""
	for k in xrange(K):
		train_features = [x for i, x in enumerate(features) if i % K != k]
		train_labels = [x for i, x in enumerate(labels) if i % K != k]
		test_features = [x for i, x in enumerate(features) if i % K == k]
		test_labels = [x for i, x in enumerate(labels) if i % K == k]
		yield train_features, train_labels, test_features, test_labels

def extract_parse(lexparse_file):
	parse_trees = []
	parse_string = ''
	for line in lexparse_file.readlines():
		if line.strip('\n') == '':
			parse_tree = Tree.fromstring(parse_string)
			parse_trees.append(parse_tree)
			parse_string = ''
		else:    
			parse_string += line.strip('\n')
	return parse_trees
		
def combine_shuffle(modern_features, original_features):
	features = original_features + modern_features
	labels = [0]*len(original_features) + [1]*len(modern_features)
	features_labels = list(zip(features, labels))
	random.shuffle(features_labels)
	features, labels = zip(*features_labels)
	return features, labels
		
def main(args):
	modern_sentences_file = open(args.modern_sentences_file, 'r')
	modern_stanford_annotations_file = open(args.modern_stanford_annotations_file, 'r')
	original_sentences_file = open(args.modern_sentences_file, 'r')
	original_stanford_annotations_file = open(args.original_stanford_annotations_file, 'r')
	
	modern_sentences = read_data(modern_sentences_file)
	modern_stanford_annotations = extract_annotations(modern_stanford_annotations_file)
	original_sentences = read_data(original_sentences_file)
	original_stanford_annotations = extract_annotations(original_stanford_annotations_file)
	
	if args.parse:
		modern_stanford_parse = open(args.modern_stanford_parse_file, 'r')
		modern_stanford_parse_trees = extract_parse(modern_stanford_parse)
		original_stanford_parse = open(args.original_stanford_parse_file, 'r')
		original_stanford_parse_trees = extract_parse(original_stanford_parse)
	else:
		modern_stanford_parse_trees = None
		original_stanford_parse_trees = None
	
	if args.word2vec:
		print 'Loading word2vec model...'
		start_time = time.time()
		word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_pretrained_model, binary=True)
		print time.time() - start_time
	else:
		word2vec_model = None
	
	# pdb.set_trace()
	print 'Extracting features...'
	start_time = time.time()
	modern_features = extract_features(modern_sentences, modern_stanford_annotations, modern_stanford_parse_trees, args, word2vec_model)
	original_features = extract_features(original_sentences, original_stanford_annotations, original_stanford_parse_trees, args, word2vec_model)
	print time.time() - start_time
	
	features, labels = combine_shuffle(modern_features, original_features)
	
	ridge_regression = linear_model.Ridge(alpha = .5)

	train_scores = []
	test_scores = []

	print 'Running 10 fold cross-validation...'
	start_time = time.time()
	for train_features, train_labels, test_features, test_labels in k_fold_cross_validation(features, labels, K=10):
		try:
			ridge_regression.fit(train_features, train_labels)
		except:
			pdb.set_trace()
		predicted_train_labels = ridge_regression.predict(train_features)
		predicted_test_labels = ridge_regression.predict(test_features)
		train_scores.append(stats.spearmanr(train_labels, predicted_train_labels)[0])
		test_scores.append(stats.spearmanr(test_labels, predicted_test_labels)[0])
	print time.time() - start_time
	print train_scores
	print numpy.mean(train_scores)
	print test_scores
	print numpy.mean(test_scores)
		
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--modern_sentences_file", type = str)
	argparser.add_argument("--modern_stanford_annotations_file", type = str)
	argparser.add_argument("--modern_stanford_parse_file", type = str)
	argparser.add_argument("--original_sentences_file", type = str)
	argparser.add_argument("--original_stanford_annotations_file", type = str)
	argparser.add_argument("--original_stanford_parse_file", type = str)
	argparser.add_argument("--word2vec_pretrained_model", type = str)
	argparser.add_argument("--case", dest='case', default=False, action='store_true')
	argparser.add_argument("--dependency", dest='dependency', default=False, action='store_true')
	argparser.add_argument("--entity", dest='entity', default=False, action='store_true')
	argparser.add_argument("--lexical", dest='lexical', default=False, action='store_true')
	argparser.add_argument("--ngram", dest='ngram', default=False, action='store_true')
	argparser.add_argument("--parse", dest='parse', default=False, action='store_true')
	argparser.add_argument("--POS", dest='POS', default=False, action='store_true')
	argparser.add_argument("--punctuation", dest='punctuation', default=False, action='store_true')
	argparser.add_argument("--readability", dest='readability', default=False, action='store_true')
	argparser.add_argument("--subjectivity", dest='subjectivity', default=False, action='store_true')
	argparser.add_argument("--word2vec", dest='word2vec', default=False, action='store_true')
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	
