"""
  Some utilities.
"""

from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(tweet, ascii=True, ignore_rt_char=True, ignore_url=True,
               ignore_mention=True, ignore_hashtag=True,
               letter_only=True, remove_stopwords=True, min_tweet_len=3):

  sword = stopwords.words('english')

  if ascii:  # maybe remove lines with ANY non-ascii character
    for c in tweet:
      if not (0 < ord(c) < 127):
        return ''

  tokens = tweet.lower().split()  # to lower, split
  res = []

  for token in tokens:
    if remove_stopwords and token in sword:
      continue
    if ignore_rt_char and token == 'rt':
      continue
    if ignore_url and token.startswith('https:'):
      continue
    if ignore_mention and token.startswith('@'):
      continue
    if ignore_hashtag and token.startswith('#'):
      continue
    if letter_only:
      if not token.isalpha():
        continue
    elif token.isdigit():
      token = '<num>'

    res += token,

  if min_tweet_len and len(res) < min_tweet_len:
    return ''
  else:
    return ' '.join(res)


def get_tfidf(tweet_list, top_n, max_features=5000):
  """ return the top n feature names and idf scores of a tweets list """
  tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
  tfidf_vectorizer.fit_transform(tweet_list)
  indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
  features = tfidf_vectorizer.get_feature_names()
  top_feature_name = [features[i] for i in indices[:top_n]]
  top_feautre_idf = tfidf_vectorizer.idf_[indices][:top_n]

  return top_feature_name, top_feautre_idf



