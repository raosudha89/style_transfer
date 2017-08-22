import sys, os, pdb
reload(sys)  
sys.setdefaultencoding('utf8')
import nltk, re
import argparse
import subprocess

class StanfordAnnotations:
    def __init__(self, token, lemma, pos, ner, head, depRel):
        self.token = token
        self.lemma = lemma
        self.pos = pos
        self.ner = ner
        self.head = head
        self.depRel = depRel

def extract_annotations(stanford_annotations_file):
    stanford_annotations = []
    sent_annotations = []
    for line in stanford_annotations_file.readlines():
        if line.strip('\n') == '':
            stanford_annotations.append(sent_annotations)
            sent_annotations = []
        else:
            index, token, lemma, pos, ner, head, depRel = line.strip('\n').split('\t')
            word_annotations = StanfordAnnotations(token, lemma, pos, ner, head, depRel)
            sent_annotations.append(word_annotations)
    return stanford_annotations    

def expand_contractions(sentence, contractions_expansions):
    sentence = sentence.replace(' \'', '\'')
    sentence = sentence.replace(' n\'t', 'n\'t')
    words = sentence.split()
    for i in range(len(words)):
        for contraction, expansion in contractions_expansions.iteritems():
            if contraction == words[i]:
                words[i] = expansion
    sentence = ' '.join(words)    
    return sentence

def main(args):
    sentences_file = open(args.sentences_file, 'r')
    contractions_expansions_file = open(args.contractions_expansions_file, 'r')
    contractions_expansions = {}
    for line in contractions_expansions_file.readlines():
        contraction, expansion = line.strip('\n').split('\t')
        contractions_expansions[contraction] = expansion
    slangs_file = open(args.slangs_file, 'r')
    slangs = {}
    for line in slangs_file.readlines():
        s, l = line.strip('\n').split('\t')
        slangs[s] = l
    output_file = open(args.output_file, 'w')
    for line in sentences_file.readlines():
        original_sentence = line.strip('\n')
        original_sentence = original_sentence.encode("ascii", "ignore")
        words = [w.lower() if w.upper() == w else w for w in nltk.word_tokenize(original_sentence)]
        if words:
            words[0] = words[0].lower()
        
        # If using Grammarly spell checker then it will take care of capitalization
        # If not, then use POS tags to decide on capitalization
        if not args.grammarly_spellcheck:
            words = [w.capitalize() if t in ['NNP', 'NNPS'] else w for (w,t) in nltk.pos_tag(words)]
        
        sentence = ' '.join(words)
        sentence = expand_contractions(sentence, contractions_expansions)
        sentence = expand_contractions(sentence, slangs)
        
        # replace muliple ?/!/. to one
        sentence = re.sub(r'[\? \. \! ]+(?=[\? \. \! ])', '', sentence)
        sentence = re.sub(r'( u[m]+)', '', sentence)
        sentence = re.sub(r'(U[m]+)', '', sentence)
        sentence = re.sub(r'(^u[m]+)', '', sentence)
        sentence = re.sub(r'[y]+([y]+)', 'y', sentence)
        sentence = re.sub(r'[tt]+([t]+)', 'tt', sentence)
        sentence = re.sub(r'[oo]+([o]+)', 'oo', sentence)
        sentence = re.sub(r'[ss]+([s]+)', 'ss', sentence)
        sentence = re.sub(r'[ll]+([l]+)', 'll', sentence)
        
        if not sentence:
            output_file.write('\n')
            continue
        # capitalize first word
        sentence = sentence[0].upper() + sentence[1:]
        sentence = ', '.join([s for s in sentence.split('. ') if s])
        
        sentence = sentence.replace(' i ', ' I ')
        sentence = re.sub(" (?=[\.,'!?:;])", "", sentence)
        sentence = sentence.replace(':-rrb-', ' :)')
        sentence = sentence.replace('-rrb-', ')')
        sentence = sentence.replace('-lrb-', '(')
        if sentence[-1] not in ['.', '!', '?']:
            sentence = sentence + '.'
        
        # Replace swear words    
        sentence = sentence.replace('fuck', 'f***')
        sentence = sentence.replace('dick', 'd***')
        sentence = sentence.replace('suck', 's***')
        sentence = sentence.replace('bitch', 'b***')
        
        output_file.write(sentence+'\n')
        
    if args.grammarly_spellcheck:
        program = 'python3 /Users/sudharao/Documents/spellcheck-corpora/benchmark/grammarly/spellcheck_grammarly.py --addr prod.spellcheck.grammarly.com:12345 %s %s' % (args.output_file, args.output_file+'.spell')
        try:
            subprocess.check_call(program, shell=True)
        except:
            pdb.set_trace()
        spell_file = open(args.output_file+'.spell', 'r')
        output_spell_checked_file = open(args.output_file+'.spell_checked', 'w')
        for line in spell_file.readlines():
            sentence = line.strip('\n')
            sentence = re.sub(r'{.+?=>([^\|]+?)}', r'\1', sentence)
            sentence = re.sub(r'{.+?=>(.+?)\|.+?}', r'\1', sentence)
            # sentence = re.sub(r'{.+?=>(.+?)}', r'\1', sentence)
            output_spell_checked_file.write(sentence+'\n')
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--sentences_file", type = str)
    argparser.add_argument("--contractions_expansions_file", type = str)
    argparser.add_argument("--slangs_file", type = str)
    argparser.add_argument("--output_file", type = str)
    argparser.add_argument("--grammarly_spellcheck", dest='grammarly_spellcheck', default=False, action='store_true')
    args = argparser.parse_args()
    print args
    print ""
    main(args)
