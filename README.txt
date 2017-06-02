1. Get formality data from Pavlick and Tetreault 2016  http://www.seas.upenn.edu/~nlp/resources/formality-corpus.tgz
	data-for-release --> data/formality
   This data contains four domains
	data/formality/answers
	data/formality/blog
	data/formality/email
	data/formality/news

2. Extract sentences from data for running stanford corenlp and stanford constituency parser 
	script/extract_sentences.py data/formality/answers data/formality/answers.sents

3. Run stanford corenlp on the sentences (this gives tokenization, POS, NER and dependencies)
	java -cp "*" -Xmx10g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -ssplit.eolonly -file data/formality/answers.sents -outputFormat conll > data/formality/answers.sents.conll
	
4. Run stanford constituency parser on the sentences
	java -Xmx2g -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data/formality/answers.sents > data/formality/answers.sents.lexparse

5. Run formality classifier
	python scripts/formality_classifier.py --dataset_file data/formality/answers --dataset_stanford_annotations_file data/formality/answers.sents.conll --word2vec_pretrained_model data/GoogleNews-vectors-negative300.bin --dataset_stanford_parse_file data/formality/answers.sents.lexparse --case --dependency --entity --lexical --ngram --parse --POS --punctuation --readability --word2vec 	
