train_src=/corpora/yahoo_answers/final_data/Family_Relationships/train/answers.Family_Relationships.sents.batches1_36.tok.informal
train_tgt=/corpora/yahoo_answers/final_data/Family_Relationships/train/answers.Family_Relationships.sents.batches1_36.tok.formal
valid_src=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.2x.tok.informal
valid_tgt=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.filtered.ref0.ref1.tok.formal
save_data=/corpora/yahoo_answers/final_data/Family_Relationships/OpenNMT-lua/answers.Family_Relationships.sents.batches1_36

th preprocess.lua -train_src $train_src -train_tgt $train_tgt -valid_src $valid_src -valid_tgt $valid_tgt -save_data $save_data

th tools/embeddings.lua -embed_type glove -embed_file /projects/style_transfer/GloVe-1.2/glove.840B.300d.txt -dict_file $save_data.src.dict -save_data $save_data.src.glove.emb

th tools/embeddings.lua -embed_type glove -embed_file /projects/style_transfer/GloVe-1.2/glove.840B.300d.txt -dict_file $save_data.tgt.dict -save_data $save_data.tgt.glove.emb

th train.lua -encoder_type brnn -data $save_data-train.t7 -save_model $save_data.filtered.model.50K.gloveemb -pre_word_vecs_enc $save_data.src.glove.emb-embeddings-300.t7 -src_word_vec_size 300 -pre_word_vecs_dec $save_data.tgt.glove.emb-embeddings-300.t7 -tgt_word_vec_size 300 -fix_word_vecs_enc -fix_word_vecs_dec -end_epoch 30 -gpuid 1 