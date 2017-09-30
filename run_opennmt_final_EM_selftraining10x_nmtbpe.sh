train_src=/corpora/yahoo_answers/final_data/Entertainment_Music/train/answers.entertainment_music.batches_1_30.10x.answers.Entertainment_Music.sents.batch000_732.filtered.smt.tok.informal
train_tgt=/corpora/yahoo_answers/final_data/Entertainment_Music/train/answers.entertainment_music.batches_1_30.10x.answers.Entertainment_Music.sents.batch000_732.filtered.smt.tok.formal
valid_src=/corpora/yahoo_answers/final_data/Entertainment_Music/tune/answers.Entertainment_Music.sents.batch50_60.2x.tok.informal
valid_tgt=/corpora/yahoo_answers/final_data/Entertainment_Music/tune/answers.Entertainment_Music.sents.batch50_60.filtered.ref0.ref1.tok.formal
save_data=/corpora/yahoo_answers/final_data/Entertainment_Music/OpenNMT-lua/answers.entertainment_music.batches_1_30.10x.answers.Entertainment_Music.sents.batch000_732.filtered.smt

th preprocess.lua -train_src $train_src -train_tgt $train_tgt -valid_src $valid_src -valid_tgt $valid_tgt -save_data $save_data

th tools/embeddings.lua -embed_type glove -embed_file /corpora/yahoo_answers/All_Yahoo_Answers/All_Yahoo_Answers.vectors_300.txt -dict_file $save_data.src.dict -save_data $save_data.src.answer.emb

th tools/embeddings.lua -embed_type glove -embed_file /corpora/yahoo_answers/All_Yahoo_Answers/All_Yahoo_Answers.vectors_300.txt -dict_file $save_data.tgt.dict -save_data $save_data.tgt.answer.emb

th tools/learn_bpe.lua -save_bpe $train_src.bpe.model < $train_src

th tools/learn_bpe.lua -save_bpe $train_tgt.bpe.model < $train_tgt

th train.lua -encoder_type brnn -data $save_data-train.t7 -save_model $save_data.model.50K.bpe -pre_word_vecs_dec $save_data.tgt.answer.emb-embeddings-300.t7 -tgt_word_vec_size 300 -fix_word_vecs_enc -fix_word_vecs_dec -end_epoch 30 -gpuid 1 -tok_src_bpe_model $train_src.bpe.model -tok_tgt_bpe_model $train_tgt.bpe.model 
