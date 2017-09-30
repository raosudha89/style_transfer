train=/corpora/yahoo_answers/All_Yahoo_Answers/All_Yahoo_Answers.gt_0.EM.rewrites.FR.rewrites.tok.formal
valid=/corpora/yahoo_answers/final_data/EM_tune.FR_tune.refs.tok.formal
save_data=/corpora/yahoo_answers/All_Yahoo_Answers/All_Yahoo_Answers.gt_0.EM.rewrites.FR.rewrites.tok.formal.lm

th preprocess.lua -data_type monotext -train $train -valid $valid -save_data $save_data

th train.lua -model_type lm -gpuid 1 -data $save_data-train.t7 -save_model $save_data.model -end_epoch 30 -max_batch_size 256
