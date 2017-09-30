train=/corpora/yahoo_answers/all_categories/Family_Relationships.tok.formal
valid=/corpora/yahoo_answers/final_data/Family_Relationships/tune/answers.Family_Relationships.sents.batch50_60.filtered.ref0.ref1.tok.formal
save_data=/corpora/yahoo_answers/final_data/Family_Relationships/OpenNMT-lua/Family_Relationships.tok.formal.lm

th preprocess.lua -data_type monotext -train $train -valid $valid -save_data $save_data

th train.lua -model_type lm -gpuid 1 -data $save_data-train.t7 -save_model $save_data.model -end_epoch 30 -max_batch_size 256
