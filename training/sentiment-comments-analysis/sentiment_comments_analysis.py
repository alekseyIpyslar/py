import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm as lgb
import re
from tqdm import tqdm
from itertools import product

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_validate, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_rows', 40)

comments = pd.read_csv('C:/Users/aleks/PycharmProjects/py/training/sentiment-comments-analysis/toxic_comments.csv')
comments.sample(3)

comments.info()

comments.duplicated().sum()

# comments["toxic"] = comments["toxic"].astype(int)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# bar = sns.barplot(x=comments["toxic"].value_counts(normalize=True).index,
#                   y=comments["toxic"].value_counts(normalize=True),
#                   ax=ax)
#
# for index, frac in comments["toxic"].value_counts(normalize=True).items():
#     ax.text(index, frac + 0.01, round(frac, 3), fontsize=12, fontweight='bold', ha='center')
#
# plt.xlabel('Is toxic')
# plt.ylabel('Class fraction')
# plt.ylim([0, 1])
# plt.show()
#
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

def lemmatize_and_clean(text):
    cleaned = []
    corpus = list(text.str.lower())
    wnl = WordNetLemmatizer()
    for i in tqdm(range(len(corpus))):
        text = re.sub(r'[^a-zA-Z ]', ' ', corpus[i])
        lemm_list = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i)
                     for i, j in pos_tag(nltk.word_tokenize(text))]
        lemm_text = ' '.join(lemm_list)
        cleaned.append(lemm_text)
    return cleaned

cleaned = lemmatize_and_clean(comments['text'])


