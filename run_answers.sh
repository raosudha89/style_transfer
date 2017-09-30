~/mosesdecoder/scripts/tokenizer/tokenizer.perl -no-escape -l en < ~/corpus/full/answers.63078.formal > ~/corpus/full/answers.63078.tok.formal 
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -no-escape -l en < ~/corpus/full/answers.63078.informal > ~/corpus/full/answers.63078.tok.informal 
# No truecasing, since we need the case as is

sed -n '1,55000p' ~/corpus/full/answers.63078.tok.formal > ~/corpus/training/answers.55000.tok.formal
sed -n '55001,63065p' ~/corpus/full/answers.63078.tok.formal > ~/corpus/tuning/answers.8065.tok.formal
sed -n '1,55000p' ~/corpus/full/answers.63078.tok.informal > ~/corpus/training/answers.55000.tok.informal
sed -n '55001,63065p' ~/corpus/full/answers.63078.tok.informal > ~/corpus/tuning/answers.8065.tok.informal

~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ~/corpus/training/answers.63078.Entertainment_Music.sents.batch000_470.formal > ~/corpus/training/answers.63078.Entertainment_Music.sents.batch000_470.tok.formal
cd lm
~/mosesdecoder/bin/lmplz -o 5 < ~/corpus/training/answers.entertainment_music.batches_1_30.Entertainment_Music.sents.batch000_732.tok.formal > answers.entertainment_music.batches_1_30.Entertainment_Music.sents.batch000_732.arpa.formal
~/mosesdecoder/bin/build_binary answers.entertainment_music.batches_1_30.Entertainment_Music.sents.batch000_732.arpa.formal answers.entertainment_music.batches_1_30.Entertainment_Music.sents.batch000_732.bin.formal

mkdir ~/working
cd ~/working
# Training the Translation System
nohup nice ~/mosesdecoder/scripts/training/train-model.perl -root-dir train -corpus /corpora/yahoo_answers/entertainment_music_modeldata/training/answers.entertaient_music.batches_1_30.30x.Entertainment_Music.sents.batch000_732.tok -f informal -e formal -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:5:/home/sudha.rao/lm_entertainment_music/answers.entertainment_music.batches_1_30.Entertainment_Music.sents.batch000_732.bin.formal:8 -external-bin-dir ~/tools -mgiza -cores 6 &> training.out &

nohup nice ~/mosesdecoder/scripts/training/train-model.perl -root-dir train -corpus /corpora/yahoo_answers/Family_Relationships_modeldata/training/answers.Family_Relationships.sents.batches1_36.tok -f informal -e formal -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:5:/projects/style_transfer/lm_FR_gt1/All_Yahoo_Answers.gt_1.bin.formal:8 -external-bin-dir ~/tools -mgiza -cores 6 &> training.out &

# Tuning
nohup nice ~/mosesdecoder/scripts/training/mert-moses.pl /corpora/yahoo_answers/entertainment_music_modeldata/tuning/answers.entertainment_music.batches_1_30.tok.informal /corpora/yahoo_answers/entertainment_music_modeldata/tuning/answers.entertainment_music.batches_1_30.tok.formal ~/mosesdecoder/bin/moses train/model/moses.ini --mertdir ~/mosesdecoder/bin/ --decoder-flags="-threads 6" &> mert.out &

nohup nice ~/mosesdecoder/scripts/training/mert-moses.pl /corpora/yahoo_answers/Family_Relationships_modeldata/tuning/answers.Family_Relationships.sents.batches1_36.tok.informal /corpora/yahoo_answers/Family_Relationships_modeldata/tuning/answers.Family_Relationships.sents.batches1_36.tok.formal ~/mosesdecoder/bin/moses train/model/moses.ini --mertdir ~/mosesdecoder/bin/ --decoder-flags="-threads 6" &> mert.out &

# Tokenize test file
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -no-escape -l en < ~/corpus/testing/answers.Entertainment_Music.sents.batch000_732.rule_based.formal > ~/corpus/testing/answers.Entertainment_Music.sents.batch000_732.rule_based.tok.formal

#Binarize test file
nohup ~/mosesdecoder/scripts/training/filter-model-given-input.pl filtered-answers.entertainment_music mert-work/moses.ini /corpora/yahoo_answers/entertainment_music_modeldata/testing/answers.entertainment_music.batches_1_30.tok.informal -Binarizer ~/mosesdecoder/bin/processPhraseTableMin &

nohup ~/mosesdecoder/scripts/training/filter-model-given-input.pl filtered-answers.Family_Relationships mert-work/moses.ini /corpora/yahoo_answers/Family_Relationships_modeldata/testing/answers.Family_Relationships.sents.batches1_36.tok.informal -Binarizer ~/mosesdecoder/bin/processPhraseTableMin &

#Translate test file
nohup nice ~/mosesdecoder/bin/moses -f filtered-answers.entertainment_music/moses.ini < /corpora/yahoo_answers/entertainment_music_modeldata/testing/answers.entertainment_music.batches_1_30.tok.informal > /corpora/yahoo_answers/entertainment_music_modeldata/testing/answers.entertainment_music.batches_1_30.translated.tertuning.formal 2> answers.entertainment_music.batches_1_30.out &

nohup nice ~/mosesdecoder/bin/moses -f filtered-answers.Family_Relationships/moses.ini < /corpora/yahoo_answers/Family_Relationships_modeldata/testing/answers.Family_Relationships.sents.batches1_36.tok.informal > /corpora/yahoo_answers/Family_Relationships_modeldata/testing/answers.Family_Relationships.sents.batches1_36.translated.tertuning.formal 2> answers.Family_Relationships.sents.batches1_36.out
