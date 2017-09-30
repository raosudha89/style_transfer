train_src=/projects/style_transfer/Shakespearizing-Modern-English/data/train.modern.nltktok
train_tgt=/projects/style_transfer/Shakespearizing-Modern-English/data/train.original.nltktok
valid_src=/projects/style_transfer/Shakespearizing-Modern-English/data/valid.modern.nltktok
valid_tgt=/projects/style_transfer/Shakespearizing-Modern-English/data/valid.original.nltktok
save_data=/projects/style_transfer/Shakespearizing-Modern-English/data/OpenNMT-lua/train.opennmt
test_src=/projects/style_transfer/Shakespearizing-Modern-English/data/test.modern.nltktok

python preprocess.py -train_src $train_src -train_tgt $train_tgt -valid_src $valid_src -valid_tgt $valid_tgt -save_data $save_data

python train.py -data $save_data -save_model $save_data.model -gpuid 0 -epochs 30 -encoder_type brnn -src_word_vec_size 300 -tgt_word_vec_size 300
