
# DATASETS
* NYT:
https://www.kaggle.com/datasets/tumanovalexander/nyt-articles-data

created a merged dataset:
https://www.kaggle.com/datasets/alphadraco/i535-new-dataset

* Wikipedia:
http://www.cs.toronto.edu/~mebrunet/simplewiki-20171103-pages-articles-multistream.xml.bz2

# TRAINED MODELS
* Trained Google News model:
https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

* CODE ALONGWITH THE DATASET AND TRAINED MODELS:
https://www.kaggle.com/datasets/alphadraco/i535project/


# To REPRODUCE RESULTS

## Notebooks are in notebooks folder
 * one notebook to create the merged dataset
 * one notebook to train word2vec
 * one notebook to train glove
 * one notebook to check bias manifold and bias evolution
 * one notebook to plot documents' bias score

## To run the code
 * To generate weat scores for wikipedia dataset using differential bias run bash_bg
 * To generate weat scores for wikipedia dataset using jackknife run bash_jk
 Both of these run the two drivers in the root directory.



