#!/usr/bin/env bash

#train_data=/projects/style_transfer/new_classifier_data/answers.formality.batch1.batch2.entertainment_music.sents.batch.1_30.random5000
#train_data=/projects/style_transfer/new_classifier_data/answers.entertainment_music.sents.batch.1_30.random5000
#train_data=/projects/style_transfer/data/formality/answers
#train_data=/projects/style_transfer/new_classifier_data/answers.entertainment_music.sents.batch.1_30.random2600.Entertainment_Music.formal.random2600.shuffled

#declare -a files=("/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.org.tok.informal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.informal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.formal.filtered.ref0" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.formal.filtered.ref1" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.formal.filtered.ref2" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.formal.filtered.ref3" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining6x.gt10editdist.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.smt.bleuter.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.bleuter.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.lm_all_formal_gt0.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.lm_all_formal_gt0_moore_lewis.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining6x.gt10editdist.lm_All_gt_0.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.orginformal.smt.formal" 
#	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.bleuter.smt.formal" ); 

#declare -a files=("/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K_epoch14_29.77.formal" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K.gloveemb_epoch14_32.81.formal" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.selftraining6x.gt10editdist.filtered.smt.model_epoch14_31.17.formal" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.smt.model.selftraining10x_epoch8_31.89.formal" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.selftraining10x.filtered.smt.model.50K.bpe_epoch8_31.89.formal"
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.opennmt.filtered.selftraining10x.usingnmtbacktranslate_epoch8_26.06.formal" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.selftraining10x.usingnmtbacktranslate.nmtbpe.usinglm_epoch8_25.80.formal" );

#declare -a files=("/corpora/yahoo_answers/final_data/Entertainment_Music/test/final_human_eval/answers.Entertainment_Music.sents.batch040.org.tok.informal.part1" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/final_human_eval/answers.Entertainment_Music.sents.batch040.tok.informal.part1" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/final_human_eval/answers.Entertainment_Music.sents.batch040.filtered.smt.formal.part1" 
#		"/corpora/yahoo_answers/final_data/Entertainment_Music/test/final_human_eval/answers.Entertainment_Music.sents.batch040.filtered.smt.model.selftraining10x_epoch8_31.89.formal.part1" );

#declare -a files=("/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.sys_comb_by_bleu.formal");
declare -a files=("/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.sys_comb_by_formality.formal");

train_data=/projects/style_transfer/new_classifier_data/answers.entertainment_music.sents.batch.1_30.random5000.Entertainment_Music.formal.random2600.shuffled

for file in "${files[@]}"; do
	python scripts/formality_classifier.py --dataset_file $train_data --dataset_stanford_annotations_file $train_data.conll --dataset_stanford_parse_file $train_data.lexparse --word2vec_pretrained_model data/GoogleNews-vectors-negative300.bin --case --entity --lexical --ngram --parse --POS --punctuation --readability --word2vec --dependency --test_dataset_file $file --test_dataset_stanford_annotations_file $file.conll --test_dataset_stanford_parse_file $file.lexparse --test_dataset_predictions_output_file $file.formality.predictions
done
