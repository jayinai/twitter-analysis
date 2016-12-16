"""
  Discover real-time twitter trend.
"""

import sys
import json
import time
import datetime
from twitter import Api
from functools import partial
from threading import Thread
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure
from tornado import gen

from utils import preprocess, get_tfidf


# cli param
tw = float(sys.argv[1]) # time window in min
top_n = int(sys.argv[2])
try:
  store_history = sys.argv[3]
except IndexError:
  store_history = False # default: don't store history


# twitter dev api
with open('credentials.json') as j:
  cred = json.load(j)

api = Api(cred['CONSUMER_KEY'], cred['CONSUMER_SECRET'],
          cred['ACCESS_TOKEN'], cred['ACCESS_TOKEN_SECRET'])

# start time for the 1st batch
batch_start_time = time.time()

# bokeh setup
source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
doc = curdoc()

# bokeh update
@gen.coroutine
def update(x, y, text):
  source.stream(dict(x=[x], y=[y], text=[text]), 10)  # last param controls right shift

# get live tweets
def get_tweets():
  global batch_start_time
  processed_tweet = []
  try:
    for line in api.GetStreamSample():
      if 'text' in line and line['lang'] == u'en':
        text = line['text'].encode('utf-8').replace('\n', ' ')
        p_t = preprocess(text) # process tweets
        if p_t:
          processed_tweet += p_t,
      if time.time() - batch_start_time >= tw * 60: # time is over for this batch
        return processed_tweet
    return processed_tweet # server-side interruption
  except:
    pass

# main logic for batch update
def blocking_task():
  global batch_start_time
  temp_batch_tweet = []
  history = {}
  start_t = None

  while True:
    try:
      tweets = get_tweets()
      if temp_batch_tweet: # some leftover due to interruption
        temp_batch_tweet.extend(tweets)
      else: # no interruption in this batch
        temp_batch_tweet = tweets

      utc_t = datetime.datetime.utcfromtimestamp(batch_start_time)
      time_x = int('{}{}{}{}{}'.format(
        utc_t.year, utc_t.month, utc_t.day, utc_t.hour, utc_t.minute))

      if start_t is None: # history file start time
        start_t = time_x

      if temp_batch_tweet:
        # get top features and idf scores
        top_feature_name, top_feature_idf = get_tfidf(
          temp_batch_tweet, top_n=top_n, max_features=int(5000./60*tw))

        # maybe store history
        if store_history:
          history[time_x] = list(top_feature_name), list(top_feature_idf)

        # reset start time and contain to hold next batch
        batch_start_time = time.time()
        temp_batch_tweet = []

        # feature name ans scores (words with tie score on same line)
        batch_dict = {}
        for feat, score in zip(top_feature_name, top_feature_idf):
          if score not in batch_dict:
            batch_dict[score] = feat
          else:
            batch_dict[score] += ', {}'.format(feat)
        for score, feat in batch_dict.iteritems(): # update
          doc.add_next_tick_callback(
            partial(update, x=time_x, y=score, text=feat))

    except KeyboardInterrupt: # manual stop
      print 'KeyboardInterrupt; aborted'
      sys.exit(1)

    # maybe dump history
    if store_history and history:
      with open('{}_{}_{}.json'.format(start_t, len(history), tw), 'w') as f:
        json.dump(history, f)


# bokeh figure
p = figure(plot_height=650, plot_width=1300, title='Twitter Trending Words',
           x_axis_label='UTC Time', y_axis_label='IDF Score')

# title font size
p.title.text_font_size='20pt'

# no grids
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# no scientific representation
p.xaxis[0].formatter.use_scientific = False

# set x-tick min 1
p.xaxis[0].ticker.desired_num_ticks=1

l = p.text(x='x', y='y', text='text', text_font_size="10pt",
           text_baseline="middle", text_align='center', source=source)

doc.add_root(p)
thread = Thread(target=blocking_task)
thread.start()
