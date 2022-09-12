# ** Important **

There is a bug in the implementation of the ensemble methods, causing the ensembles to have incorrectly high accuracies. Most notably, the previously reported accuracy of 97.42% for the DV-ngrams-cosine + NB-weighted BON ensemble is incorrect and when implemented correctly results in an accuracy of 93.68%. For now, please refer to https://github.com/bgzh/dv_cosine_revisited for a correct implementation of the ensembles. 

In summary, the bug is caused by the incorrect concatenation of documents, so the non-ensemble methods are unaffected.  

# Code for the ACL-SRW 2019 paper: "Sentiment Classification using Document Embeddings trained with Cosine Similarity".

This repository contains Java code to train document embeddings using cosine similarity, simply run the project in order to do so. All hyperparameters that need adjusting are in the top of the file NeuralNetwork.java, default hyperparameters are the same as in the paper.

There are also options to train them using dot product and L2-regularized dot product.

Run ensemble.py in order to test the combination of document embeddings with NB-weighted bag of ngrams.


IMDB data:
[unigrams](https://drive.google.com/file/d/1qxueBhd7WTBP58ZOdDL5K1DB0Sj2o5bZ/view?usp=sharing), [unigrams+bigrams](https://drive.google.com/file/d/1tou6u3-PHE-ZQAU43rhgmD_8BfJ0QLl1/view?usp=sharing), [unigrams+bigrams+trigrams](https://drive.google.com/file/d/1GDttGJrnZh370Y0KNMbAMfRNU50La07R/view?usp=sharing)


Trained embeddings (using cosine similarity):
[train vectors](https://drive.google.com/file/d/1a-eOTfKXXqUpM19GepIxkZxI4N8ESSBJ/view?usp=sharing), [test vectors](https://drive.google.com/file/d/1GFpVVrA1AlXBsWVx2McOnlAWyNm47TCI/view?usp=sharing)
