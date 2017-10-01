declare -a files=("/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.org.tok.informal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.tok.informal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.orginformal.smt.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.smt.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.smt.bleuter.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.selftraining10x.smt.formal"
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.selftraining10x.gt0editdit.smt.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.selftraining10x.lm_all_formal_gt0.smt.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.selftraining10x.lm_all_formal_gt0_moore_lewis.smt.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.selftraining10x.gt0editdit.lm_All_gt_0.smt.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.model.50K_epoch13_13.89.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.model.50K.gloveemb_epoch14_14.88.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.model.50K.copy_acc_56.62_ppl_14.69_e13.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.model.50K.copy.ansemb_acc_57.42_ppl_13.20_e13.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.smt.model.selftraining10x_epoch12_20.58.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.smt.Family_Relationships.model.selftraining10x.usingnmtbacktranslate_epoch5_15.21.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.smt.model.selftraining10x.nmtbpe_epoch5_22.58.formal" 
		"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.smt.Family_Relationships.model.selftraining10x.usingnmtbacktranslate.nmtbpe.usinglm_epoch5_15.21.formal" 
		#"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.tok.formal.ref0" 
		#"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.tok.formal.ref1" 
		#"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.tok.formal.ref2" 
		#"/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.tok.formal.ref3"
		); 

ref_prefix=/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.filtered.tok.formal.ref
for file in "${files[@]}"; do
	echo $file
	perl /projects/style_transfer/Shakespearizing-Modern-English/code/main/PINC.perl $ref_prefix < $file
done

