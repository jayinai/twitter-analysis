# Twitter analysis


## vritualenv

First make sure `pip` and `virtualenv` are installed. Then create a virtual
environment in the root dir by running:

`virtualenv env`

then activate the virtual env with

`source env/bin/activate`

(to get out of the virtualenv, run `deactivate`)


## Dependencies

install all the dependencies with

`pip install -r requirements.txt`

also make sure to download nltk's corpus by running those line in python
interpreter:

```python
import nltk
nltk.download()
```

## Credentials

Rename `sample_credentials.json` to `credentials.json`, and fill in the four
credentials from your twitter app.


## Real-time twitter trend discovery

Run `bokeh serve --show real-time-twitter-trend-discovery.py --args <tw>
<top_n_words> <*save_history>`, where `<tw>` and `<top_n_words>` are arguments
representing within what time window we treat tweets as a batch, and how many
words with highest idf scores to show, while `<*save_history>`` is an optional
boolean value indicating whether we want to dump the history. Make sure API
credentials are properly stored in the credentials.json file.


## Topic modeling and t-SNE visualization: 20 Newsgroups

To train a topic model and visualize the news in 2-D space, run
`python topic_20news.py --n_topics <n_topics> --n_iter <n_iter>
--top_n <top_n> --threshold <threshold>`, where `<n_topics>` being the number
of topics we select (default 20), `<n_iter>` being the number of iterations
for training an LDA model (default 500), `<top_n>` being the number of top
keywords we display (default 5), and `<threshold>` being the threshold
probability for topic assignment (default 0.0).


## Scrape tweets and save them to disk

To scrape tweets and save them to disk for later use, run
`python scrape_tweets.py`. If the script is interrupted, just re-run the same
command so new tweets collected. The script gets ~1,000 English tweets per min,
or 1.5 million/day.

Make sure API credentials are properly stored in the credentials.json file.


## Topic modeling and t-SNE visualization: tweets

First make sure you accumulated some tweets, then run `python topic_tweets.py
--raw_tweet_dir <raw_tweet_dir> --num_train_tweet <num_train_tweet>
--n_topics <n_topics> --n_iter <n_iter> --top_n <top_n> --threshold <threshold>
--num_example <num_example>`, where `<raw_tweet_dir>` being a folder containing
raw tweet files, `<num_train_tweet>` being the number of tweets we use for
training an LDA model, `<n_topics>` being the number of topics we select
(default 20), `<n_iter>` being the number of iterations for training an LDA
model (default 500), `<top_n>` being the number of top keywords we display
(default 5), `<threshold>` being the threshold probability for topic assignment
(default 0.0), and `<num_example>` being number of tweets to show on the plot
(default 5000)