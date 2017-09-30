train_src=/corpora/yahoo_answers/final_data/Entertainment_Music/train/answers.entertainment_music.batches_1_30.tok.informal
train_tgt=/corpora/yahoo_answers/final_data/Entertainment_Music/train/answers.entertainment_music.batches_1_30.tok.formal
valid_src=/corpora/yahoo_answers/final_data/Entertainment_Music/tune/answers.Entertainment_Music.sents.batch50_60.2x.tok.informal
valid_tgt=/corpora/yahoo_answers/final_data/Entertainment_Music/tune/answers.Entertainment_Music.sents.batch50_60.filtered.ref0.ref1.tok.formal
save_data=/corpora/yahoo_answers/final_data/Entertainment_Music/OpenNMT-py/answers.entertainment_music.batches_1_30

python preprocess.py -train_src $train_src -train_tgt $train_tgt -valid_src $valid_src -valid_tgt $valid_tgt -save_data $save_data

python train.py -data $save_data -save_model $save_data.model.50K.copy.ansemb -gpuid 0 -epochs 30 -copy_attn_force -encoder_type brnn -src_word_vec_size 300 -tgt_word_vec_size 300
