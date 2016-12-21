"""
  Train an lDA model on 20 newsgroups (training + test sets)
"""

import os
import argparse
import time
import lda
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool


if __name__ == '__main__':

  ##############################################################################
  # setup

  news_base_dir = '20newsgroups'
  if not os.path.exists(news_base_dir):
    os.makedirs(news_base_dir)

  parser = argparse.ArgumentParser()
  parser.add_argument('--n_topics', required=True, type=int, default=20,
                      help='number of topics')
  parser.add_argument('--n_iter', required=True, type=int, default=500,
                      help='number of iteration for LDA model training')
  parser.add_argument('--top_n', required=True, type=int, default=5,
                      help='number of keywords to show for each topic')
  parser.add_argument('--threshold', required=True, type=float, default=0.0,
                      help='threshold probability for topic assignment')
  args = parser.parse_args()

  # unpack
  n_topics = args.n_topics
  n_iter = args.n_iter
  n_top_words = args.top_n
  threshold = args.threshold

  t0 = time.time()

  ##############################################################################
  # train an LDA model

  remove = ('headers', 'footers', 'quotes')
  newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
  newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
  news = [' '.join(filter(unicode.isalpha, raw.lower().split())) for raw in
          newsgroups_train.data + newsgroups_test.data]

  cvectorizer = CountVectorizer(min_df=5, stop_words='english')
  cvz = cvectorizer.fit_transform(news)

  lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
  X_topics = lda_model.fit_transform(cvz)

  t1 = time.time()

  print '\n>>> LDA training done; took {} mins\n'.format((t1-t0)/60.)

  np.save('20newsgroups/lda_doc_topic_{}news_{}topics.npy'.format(
    X_topics.shape[0], X_topics.shape[1]), X_topics)

  np.save('20newsgroups/lda_topic_word_{}news_{}topics.npy'.format(
    X_topics.shape[0], X_topics.shape[1]), lda_model.topic_word_)

  print '\n>>> doc_topic & topic word written to disk\n'

  ##############################################################################
  # threshold and plot

  _idx = np.amax(X_topics, axis=1) > threshold  # idx of news that > threshold
  _topics = X_topics[_idx]

  num_example = len(_topics)

  # t-SNE: 50 -> 2D
  tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                    init='pca')
  tsne_lda = tsne_model.fit_transform(_topics[:num_example])

  # find the most probable topic for each news
  _lda_keys = []
  for i in xrange(_topics.shape[0]):
    _lda_keys += _topics[i].argmax(),

  # show topics and their top words
  topic_summaries = []
  topic_word = lda_model.topic_word_  # get the topic words
  vocab = cvectorizer.get_feature_names()
  for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

  # 20 colors
  colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
  ])

  # plot
  title = "[20 newsgroups] t-SNE visualization of LDA model trained on {} news, " \
          "{} topics, thresholding at {} topic probability, {} iter ({} data " \
          "points and top {} words)".format(
    X_topics.shape[0], n_topics, threshold, n_iter, num_example, n_top_words)

  plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                       title=title,
                       tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)

  plot_lda.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                   color=colormap[_lda_keys][:num_example],
                   source=bp.ColumnDataSource({
                     "content": news[:num_example],
                     "topic_key": _lda_keys[:num_example]
                     }))

  # randomly choose a news (in a topic) coordinate as the crucial words coordinate
  topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
  for topic_num in _lda_keys:
    if not np.isnan(topic_coord).any():
      break
    topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

  # plot crucial words
  for i in xrange(X_topics.shape[1]):
    plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

  # hover tools
  hover = plot_lda.select(dict(type=HoverTool))
  hover.tooltips = {"content": "@content - topic: @topic_key"}

  save(plot_lda, '20_news_tsne_lda_viz_{}_{}_{}_{}_{}_{}.html'.format(
    X_topics.shape[0], n_topics, threshold, n_iter, num_example, n_top_words))

  t2 = time.time()
  print '\n>>> whole process done; took {} mins\n'.format((t2 - t0) / 60.)