declare -a files=("/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.org.tok.informal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.informal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining6x.gt10editdist.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.smt.bleuter.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.bleuter.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.lm_all_formal_gt0.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining10x.lm_all_formal_gt0_moore_lewis.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.selftraining6x.gt10editdist.lm_All_gt_0.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.orginformal.smt.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K_epoch14_29.77.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K.gloveemb_epoch14_32.81.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.filtered.smt.model.selftraining10x_epoch8_31.89.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.selftraining10x.filtered.smt.model.50K.bpe_epoch8_31.89.formal"
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.opennmt.filtered.selftraining10x.usingnmtbacktranslate_epoch8_26.06.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.selftraining10x.usingnmtbacktranslate.nmtbpe.usinglm_epoch8_25.80.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K.copy_acc_50.08_ppl_36.76_e11.formal" 
	"/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.model.50K.copy.ansemb_acc_51.85_ppl_28.93_e13.formal");

ref_prefix=/corpora/yahoo_answers/final_data/Entertainment_Music/test/answers.Entertainment_Music.sents.batch040.tok.formal.filtered.ref

for file in "${files[@]}"; do
	~/mosesdecoder/bin/sentence-bleu ${ref_prefix}0 ${ref_prefix}1 ${ref_prefix}2 ${ref_prefix}3 < $file > $file.sent_bleu_scores
done

