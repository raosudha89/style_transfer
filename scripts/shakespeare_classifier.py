import sys, argparse
from collections import defaultdict
from textblob import TextBlob
import pdb, time
import nltk, numpy
from nltk.tree import *
import gensim
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.metrics import make_scorer
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

NER_TAGSET = ['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'MISC', 'NUMBER', 'PERCENT', \
			  'DATE', 'TIME', 'DURATION', 'SET', 'ORDINAL']
POS_TAGSET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
			  'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',\
			  'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',\
			  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',\
			  'WDT', 'WP', 'WP$', 'WRB']
FP_PRO_LIST = ['i', 'we', 'me', 'us', 'my', 'mine', 'our', 'ours']
TP_PRO_LIST = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs']

UNIGRAM_DICT = defaultdict(int)
BIGRAM_DICT = defaultdict(int)
TRIGRAM_DICT = defaultdict(int)

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
	dependency_tuple_dict = defaultdict(int)
	for word_annotations in sent_annotations:
		gov = sent_annotations[int(word_annotations.head)-1].pos
		typ = word_annotations.depRel
		dep = word_annotations.pos
		gov_typ_dep = '_'.join([gov, typ, dep])
		dependency_tuple_dict[gov_typ_dep] = 1
		gov_typ = '_'.join([gov, typ])
		dependency_tuple_dict[gov_typ] = 1
		typ_dep = '_'.join([typ, dep])
		dependency_tuple_dict[typ_dep] = 1
		gov_dep = '_'.join([gov, dep])
		dependency_tuple_dict[gov_dep] = 1
	return dependency_tuple_dict

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
	
def get_ngrams(sent_annotations, is_test):
	tokens = [w.token for w in sent_annotations]
	sentence = ' '.join(tokens).decode('utf-8', 'ignore')
	blob = TextBlob(sentence)
	unigrams = tokens
	bigrams = blob.ngrams(n=2)
	trigrams = blob.ngrams(n=3)
	unigram_dict = defaultdict(int)
	bigram_dict = defaultdict(int)
	trigram_dict = defaultdict(int)
	global UNIGRAM_DICT, BIGRAM_DICT, TRIGRAM_DICT
	for unigram in unigrams:
		unigram_dict[unigram] = 1
		UNIGRAM_DICT[unigram] += 1
	for bigram in bigrams:
		bigram_dict['_'.join(bigram)] = 1
		BIGRAM_DICT['_'.join(bigram)] += 1
	for trigram in trigrams:
		trigram_dict['_'.join(trigram)] = 1
		TRIGRAM_DICT['_'.join(trigram)] += 1
	return unigram_dict, bigram_dict, trigram_dict	
				
def get_parse_features(stanford_parse_tree, sent_annotations, is_test):
	sent_len = len(sent_annotations)
	avg_depth = stanford_parse_tree.height()*1.0/sent_len
	lexical_production_dict = defaultdict(int)
	for production in stanford_parse_tree.productions():
		if production.is_lexical():
			continue
		lexical_production_dict[production] += 1
	avg_depth_feature = [avg_depth]
	return avg_depth_feature, lexical_production_dict
			
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

def remove_less_frequent(dict, reference_dict, freq_cutoff):
	new_dict = defaultdict(int)
	for item,count in dict.iteritems():
		if reference_dict[item] > freq_cutoff:
			new_dict[item] = count
	return new_dict
				
def extract_features(sentences, stanford_annotations, stanford_parse_trees, args, word2vec_model, is_test=False):
	features = [None]*len(sentences)
	
	for i in range(len(sentences)):
		sentence = sentences[i]
		sent_annotations = stanford_annotations[i]
		words = sentence.split()
		
		feature_set = []
		
		#case features
		if args.case:
			case_features = get_case_features(sent_annotations, sentence)
			feature_set += case_features
		
		# dependency features
		if args.dependency:
			dependency_tuple_dict = get_dependency_tuples(sent_annotations, is_test)
		else:
			dependency_tuple_dict = None
		
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
			unigram_dict, bigram_dict, trigram_dict = get_ngrams(sent_annotations, is_test)
		else:
			unigram_dict, bigram_dict, trigram_dict = None, None, None
				
		# parse features
		if args.parse:
			avg_depth_feature, lexical_production_dict = get_parse_features(stanford_parse_trees[id], sent_annotations, is_test)
			feature_set += avg_depth_feature
		else:
			lexical_production_dict = None
		
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
		
		features[i] = [feature_set, dependency_tuple_dict, unigram_dict, bigram_dict, trigram_dict, lexical_production_dict]
	
	global UNIGRAM_DICT, BIGRAM_DICT, TRIGRAM_DICT
	
	dependency_tuple_feature_set = []
	unigram_feature_set = []
	bigram_feature_set = []
	trigram_feature_set = []
	lexical_production_feature_set = []
	other_feature_set = []
			
	for i in range(len(features)):
		[feature_set, dependency_tuple_dict, unigram_dict, bigram_dict, trigram_dict, lexical_production_dict] = features[i]
		dependency_tuple_feature_set.append(dependency_tuple_dict)
		unigram_feature_set.append(remove_less_frequent(unigram_dict, UNIGRAM_DICT, 1))
		bigram_feature_set.append(remove_less_frequent(bigram_dict, BIGRAM_DICT, 2))
		trigram_feature_set.append(remove_less_frequent(trigram_dict, TRIGRAM_DICT, 3))
		lexical_production_feature_set.append(lexical_production_dict)
		other_feature_set.append(feature_set)
	
	feature_vectors = other_feature_set
	
	v = DictVectorizer(sparse=False)
	if args.dependency:
		dependency_tuple_feature = v.fit_transform(dependency_tuple_feature_set)
		feature_vectors = numpy.concatenate((feature_vectors, dependency_tuple_feature), axis=1)
	if args.ngram:
		unigram_feature = v.fit_transform(unigram_feature_set)
		bigram_feature = v.fit_transform(bigram_feature_set)
		trigram_feature = v.fit_transform(trigram_feature_set)
		feature_vectors = numpy.concatenate((feature_vectors, unigram_feature, bigram_feature, trigram_feature), axis=1)
	if args.parse:
		lexical_production_feature = v.fit_transform(lexical_production_feature_set)
		feature_vectors = numpy.concatenate((feature_vectors, lexical_production_feature), axis=1)
				
	return feature_vectors

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
	k_fold = KFold(n_splits=K, shuffle=randomise)
	labels = numpy.array(labels)
	features = numpy.array(features)
	for train_index, test_index in k_fold.split(features):
		train_features, test_features = features[train_index], features[test_index]
		train_labels, test_labels = labels[train_index], labels[test_index]
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
		
def combine_shuffle(features, no_of_modern):
	features = numpy.array(features)
	labels = [0]*no_of_modern + [1]*(len(features) - no_of_modern)
	labels = numpy.array(labels)
	permutation = numpy.random.permutation(len(features))
	shuffled_features = features[permutation]
	shuffled_labels = labels[permutation]
	return shuffled_features, shuffled_labels
		
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
	sentences = modern_sentences + original_sentences
	stanford_annotations = modern_stanford_annotations + original_stanford_annotations
	if args.parse:
		stanford_parse_trees = modern_stanford_parse_trees + original_stanford_parse_trees
	else:
		stanford_parse_trees = None
	features = extract_features(sentences, stanford_annotations, stanford_parse_trees, args, word2vec_model)
	# modern_features = extract_features(modern_sentences, modern_stanford_annotations, modern_stanford_parse_trees, args, word2vec_model)
	# original_features = extract_features(original_sentences, original_stanford_annotations, original_stanford_parse_trees, args, word2vec_model)
	print time.time() - start_time
	
	features, labels = combine_shuffle(features, len(modern_sentences))
	
	# ridge_regression = linear_model.Ridge(alpha = .5)
	svm_classifier = svm.SVC()
	# train_scores = []
	test_scores = []

	print 'Running 10 fold cross-validation...'
	i = 1
	for train_features, train_labels, test_features, test_labels in k_fold_cross_validation(features, labels, K=10):
		print 'Fold no. %d' % (i)
		i += 1
		start_time = time.time()
		# ridge_regression.fit(train_features, train_labels)
		# predicted_train_labels = ridge_regression.predict(train_features)
		# predicted_test_labels = ridge_regression.predict(test_features)
		svm_classifier.fit(train_features, train_labels)
		predicted_train_labels = svm_classifier.predict(train_features)
		predicted_test_labels = svm_classifier.predict(test_features)
		train_score = stats.spearmanr(train_labels, predicted_train_labels)[0]
		train_scores.append(train_score)
		print train_score
		test_score = stats.spearmanr(test_labels, predicted_test_labels)[0]
		test_scores.append(test_score)
		print test_score
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
	
