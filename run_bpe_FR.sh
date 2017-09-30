train_prefix=/corpora/yahoo_answers/final_data/Family_Relationships/train/answers.Family_Relationships.sents.batches1_36.10x.answers.Family_Relationships.sents.batches0036_1564.backtranslated.filtered.smt.tok
num_operations=50000
codes_file=$train_prefix.codes
vocab_prefix=$train_prefix.vocab

valid_src=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.2x.tok.informal
valid_src_bpe=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.2x.tok.BPE.informal
valid_tgt=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.filtered.ref0.ref1.tok.formal
valid_tgt_bpe=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.filtered.ref0.ref1.tok.BPE.formal

test_prefix=/corpora/yahoo_answers/final_data/Family_Relationships/test/answers.Family_Relationships.sents.batch0040.tok

./learn_joint_bpe_and_vocab.py --input $train_prefix.informal $train_prefix.formal -s $num_operations -o $codes_file --write-vocabulary $vocab_prefix.informal $vocab_prefix.formal

./apply_bpe.py -c $codes_file --vocabulary $vocab_prefix.informal --vocabulary-threshold 50 < $train_prefix.informal > $train_prefix.BPE.informal
./apply_bpe.py -c $codes_file --vocabulary $vocab_prefix.formal --vocabulary-threshold 50 < $train_prefix.formal > $train_prefix.BPE.formal

./apply_bpe.py -c $codes_file --vocabulary $vocab_prefix.informal --vocabulary-threshold 50 < $valid_src > $valid_src_bpe
./apply_bpe.py -c $codes_file --vocabulary $vocab_prefix.formal --vocabulary-threshold 50 < $valid_tgt > $valid_tgt_bpe

./apply_bpe.py -c $codes_file --vocabulary $vocab_prefix.informal --vocabulary-threshold 50 < $test_prefix.informal > $test_prefix.BPE.informal
