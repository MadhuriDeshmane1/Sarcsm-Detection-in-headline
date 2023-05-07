# Sarcsm Detection in headlines using NLP

[This repository] perform[s] semantic modelling of sentences using neural networks for the task of sarcasm detection using Naive bayes, RNN, LSTM.

Tasks Performed : Transliteration,Word2vec, Naive bayes model, RNN, CNN

Pre-requisite nltk (TweetTokenizer) Keras Tensorflow numpy scipy gensim (if you are using word2vec) itertools

Tasks Performed in this repository:
 1) Data Cleaning
 2) Data visualization
 3) Tokenization
 4) Key phrase extraction
 5) word cloud
 6) Countvectorizer(Bag of words)
 7) TFIDF 
 8) Word2vec
 9) Converting unlabelled data to labelled data
 10) Silhouette Visualizer
 11) Silhouette score
 
 Libaraies used :
 from langdetect import detect

from googletrans import Translator

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.util import ngrams

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from string import punctuation

import contractions

import yake

from rake_nltk import Rake

from unidecode import unidecode

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from gensim.models import Word2Vec

import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer
 



