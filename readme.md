# ClozEx

This is the official code and dataset of ClozEx: A Task toward Generation of English Cloze Explanation (EMNLP 2023). [Paper link](https://aclanthology.org/2023.findings-emnlp.347/).

## Dataset Creation

Because the [AG NEWS CORPUS](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) is not allowed to re-distribute with a different name (e.g. ClozEx), we provide scripts to build the dataset locally.
Using the default arguments could ensure consistency with the data we used in our paper.

### Generate new questions with initial explanations.

#### Affix/VerbTense questions

To generate Affix/VerbTense questions, please execute the `dataset_creation/parse_corpus.py` script firstly to obtain parsed sentences from news corpus.

Then execute the `dataset_creation/generate_affix_tense.py` script to generate corresponding quesitons with explanations. (#Affix = #verbTense = 50000).
Please indicate the save path on you own.

#### Prep. questions

Please follow the repo [BertPSD](https://github.com/dirkneuhaeuser/preposition-sense-disambiguation) to train a Preposition Sense Disambiguation (PSD) model, and save the model in `dataset_creation/material/BertPSD/model/`.

Then you could execute `dataset_creation/generate_prep.py` to generate questions/explanations. (#Prep. == 47043)
Please indicate the save path on you own.

#### Dataset separation

Please concatenate the previous generated data (concat lists in order of affix, verbtense, prep.).
Then use `sklearn.model_selection.train_test_split` to split into train, valid, and test set. (train:valid:test = 0.7:0.15:0.15, random seed = 42).

#### Paraphrase the initial explanation

Please indicate the path of generated questions previously, your information of openAI API, and output path in `dataset_creation/paraphrase_gpt.py` then execute it.