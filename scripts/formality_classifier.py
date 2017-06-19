import sys, argparse
from collections import defaultdict
from textblob import TextBlob
import pdb, time
import nltk, numpy
from nltk.tree import *
import uuid
import gensim
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.metrics import make_scorer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

NER_TAGSET = ['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'MISC', 'NUMBER', 'PERCENT', \
			  'DATE', 'TIME', 'DURATION', 'SET', 'ORDINAL']
POS_TAGSET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
			  'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',\
			  'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',\
			  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',\
			  'WDT', 'WP', 'WP$', 'WRB']
FP_PRO_LIST = ['i', 'we', 'me', 'us', 'my', 'mine', 'our', 'ours']
TP_PRO_LIST = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs']

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
	# tokens = [w.token for w in sent_annotations]
	tokens = [w.lemma for w in sent_annotations]
	sentence = ' '.join(tokens).decode('utf-8', 'ignore')
	blob = TextBlob(sentence)
	unigrams = tokens
	bigrams = blob.ngrams(n=2)
	trigrams = blob.ngrams(n=3)
	unigram_dict = defaultdict(int)
	bigram_dict = defaultdict(int)
	trigram_dict = defaultdict(int)
	for unigram in unigrams:
		unigram_dict[unigram] = 1
	for bigram in bigrams:
		bigram_dict['_'.join(bigram)] = 1
	for trigram in trigrams:
		trigram_dict['_'.join(trigram)] = 1
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
		if word_annotations.lemma in FP_PRO_LIST:
			fp_pros += 1
		if word_annotations.lemma in TP_PRO_LIST:
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
			word_vector = word2vec_model[word_annotations.lemma]
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
				
def extract_features(corpus, stanford_annotations, stanford_parse_trees, args, word2vec_model, is_test=False):
	features = []
	
	for i in range(len(corpus)):
		# print i
		id, sentence, rating = corpus[i]
		try:
			sent_annotations = stanford_annotations[id]
		except:
			pdb.set_trace()
			# continue
			# break
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
		
		features.append([feature_set, dependency_tuple_dict, unigram_dict, bigram_dict, trigram_dict, lexical_production_dict])
	
	# global UNIGRAM_DICT, BIGRAM_DICT, TRIGRAM_DICT		
			
	dependency_tuple_feature_set = []
	unigram_feature_set = []
	bigram_feature_set = []
	trigram_feature_set = []
	lexical_production_feature_set = []
	other_feature_set = []
			
	for i in range(len(features)):
		[feature_set, dependency_tuple_dict, unigram_dict, bigram_dict, trigram_dict, lexical_production_dict] = features[i]
		dependency_tuple_feature_set.append(dependency_tuple_dict)
		unigram_feature_set.append(unigram_dict)
		bigram_feature_set.append(bigram_dict)
		trigram_feature_set.append(trigram_dict)
		lexical_production_feature_set.append(lexical_production_dict)
		other_feature_set.append(feature_set)
	
	feature_vectors = numpy.array(other_feature_set)
	
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
	
	print 'No. of unigram features %d ' % len(unigram_feature[0])
	print 'No. of bigram features %d ' % len(bigram_feature[0])
	print 'No. of trigram features %d ' % len(trigram_feature[0])
	print 'No. of other_features %d ' % len(other_feature_set[0])
	print 'Total no. of features %d ' % len(feature_vectors[0])

	return feature_vectors

def read_data(dataset):
	corpus = []
	index = 0
	for line in dataset.readlines():
		parts = line.split('\t')
		if len(parts) == 4:
			rating, _, id, sentence = parts
		else:
			rating, _, sentence = parts
			id = index
			index += 1
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
				break
			id = corpus[k][0]
			sent_annotations = []
		else:
			index, token, lemma, pos, ner, head, depRel = line.strip('\n').split('\t')
			word_annotations = StanfordAnnotations(token, lemma, pos, ner, head, depRel)
			sent_annotations.append(word_annotations)
	return stanford_annotations		

def k_fold_cross_validation(features, labels, K, randomise = False):
	k_fold = KFold(n_splits=K, shuffle=randomise)
	labels = numpy.array(labels)
	for train_index, test_index in k_fold.split(features):
		train_features, test_features = features[train_index], features[test_index]
		train_labels, test_labels = labels[train_index], labels[test_index]
		yield train_features, train_labels, test_features, test_labels

def extract_parse(lexparse_file, corpus):
	parse_trees = {}
	parse_string = ''
	id = corpus[0][0]
	k = 0	
	for line in lexparse_file.readlines():
		if line.strip('\n') == '':
			parse_tree = Tree.fromstring(parse_string)
			parse_trees[id] = parse_tree
			k += 1
			if k == len(corpus):
				break
			id = corpus[k][0]
			parse_string = ''
		else:    
			parse_string += line.strip('\n')
	return parse_trees

def read_test_data(test_dataset):
	test_corpus = []
	for line in test_dataset.readlines():
		sentence = line.strip('\n')
		id = str(uuid.uuid4())
		test_corpus.append([id, sentence, None])
	return test_corpus
		
def main(args):
	dataset = open(args.dataset_file, 'r')
	dataset_stanford_annotations = open(args.dataset_stanford_annotations_file, 'r')
	
	train_corpus = read_data(dataset)
	train_stanford_annotations = extract_annotations(dataset_stanford_annotations, train_corpus)
	
	if args.parse:
		dataset_stanford_parse = open(args.dataset_stanford_parse_file, 'r')
		train_stanford_parse_trees = extract_parse(dataset_stanford_parse, train_corpus)
	else:
		train_stanford_parse_trees = None
	
	if args.word2vec:
		print 'Loading word2vec model...'
		start_time = time.time()
		word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_pretrained_model, binary=True)
		print time.time() - start_time
	
	if args.test_dataset_file:
		test_dataset = open(args.test_dataset_file, 'r')
		test_corpus = read_test_data(test_dataset)
		test_dataset_stanford_annotations = open(args.test_dataset_stanford_annotations_file, 'r')
		test_stanford_annotations = extract_annotations(test_dataset_stanford_annotations, test_corpus)
		if args.parse:
			test_dataset_stanford_parse = open(args.test_dataset_stanford_parse_file, 'r')
			test_stanford_parse_trees = extract_parse(test_dataset_stanford_parse, test_corpus)
		else:
			test_stanford_parse_trees = None
		# test_dataset_feature_vectors, _ = extract_features(test_corpus, test_stanford_annotations, test_stanford_parse_trees, args, word2vec_model, is_test=True)

		corpus = train_corpus + test_corpus
		train_stanford_annotations.update(test_stanford_annotations)
		stanford_annotations = train_stanford_annotations
		train_stanford_parse_trees.update(test_stanford_parse_trees)
		stanford_parse_trees = train_stanford_parse_trees
	
	print 'Extracting features...'
	start_time = time.time()
	feature_vectors = extract_features(corpus, stanford_annotations, stanford_parse_trees, args, word2vec_model)
	
	train_feature_vectors = feature_vectors[:len(train_corpus)]
	test_feature_vectors = feature_vectors[len(train_corpus):]
	
	print time.time() - start_time
	ridge_regression = linear_model.Ridge(alpha = .5)
	# print len(feature_vectors)
	labels = []
	for id, sentence, rating in train_corpus:
		labels.append(float(rating))
	
	train_scores = []
	test_scores = []
	if not args.test_dataset_file:
		K = 10
		print 'Running 10 fold cross-validation...'
	else:
		K = 2
	i = 1
	for train_features, train_labels, test_features, test_labels in k_fold_cross_validation(train_feature_vectors, labels, K):
		print 'Fold no. %d' % (i)
		start_time = time.time()
		i += 1
		ridge_regression.fit(train_features, train_labels)
		predicted_train_labels = ridge_regression.predict(train_features)
		predicted_test_labels = ridge_regression.predict(test_features)
		train_score = stats.spearmanr(train_labels, predicted_train_labels)[0]
		train_scores.append(train_score)
		test_score = stats.spearmanr(test_labels, predicted_test_labels)[0]
		test_scores.append(test_score)
		print train_score, test_score
		print time.time() - start_time
	print train_scores
	print numpy.mean(train_scores)
	print test_scores
	print numpy.mean(test_scores)
	
	if args.test_dataset_file:
		ridge_regression.fit(train_feature_vectors, labels)
		predicted_test_dataset_labels = ridge_regression.predict(test_feature_vectors)
		i = 0
		test_dataset_predictions_output = open(args.test_dataset_predictions_output_file, 'w')
		for id, sentence, rating in test_corpus:
			test_dataset_predictions_output.write('%s\t%s\n' % (predicted_test_dataset_labels[i], sentence))
			i += 1
		
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--dataset_file", type = str)
	argparser.add_argument("--dataset_stanford_annotations_file", type = str)
	argparser.add_argument("--dataset_stanford_parse_file", type = str)
	argparser.add_argument("--test_dataset_file", type = str)
	argparser.add_argument("--test_dataset_stanford_annotations_file", type = str)
	argparser.add_argument("--test_dataset_stanford_parse_file", type = str)
	argparser.add_argument("--test_dataset_predictions_output_file", type = str)
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
	
