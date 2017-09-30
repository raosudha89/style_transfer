train_src=/corpora/yahoo_answers/final_data/Family_Relationships/train/answers.Family_Relationships.sents.batches1_36.tok.informal
train_tgt=/corpora/yahoo_answers/final_data/Family_Relationships/train/answers.Family_Relationships.sents.batches1_36.tok.formal
valid_src=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.2x.tok.informal
valid_tgt=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.filtered.ref0.ref1.tok.formal
save_data=/corpora/yahoo_answers/final_data/Family_Relationships/OpenNMT-py/answers.Family_Relationships.sents.batches1_36
emb_file=/corpora/yahoo_answers/All_Yahoo_Answers/All_Yahoo_Answers.vectors_300.bin

#python preprocess.py -train_src $train_src -train_tgt $train_tgt -valid_src $valid_src -valid_tgt $valid_tgt -save_data $save_data

python train.py -data $save_data -save_model $save_data.model.50K.copy.ansemb -gpuid 0 -epochs 30 -copy_attn_force -encoder_type brnn -src_word_vec_size 300 -tgt_word_vec_size 300 -pre_word_vecs_enc $emb_file -pre_word_vecs_dec $emb_file -fix_word_vecs_enc -fix_word_vecs_dec
