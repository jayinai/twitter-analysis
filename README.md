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


## real-time twitter trend discovery

Run `bokeh serve --show real-time-twitter-trend-discovery.py --args <tw>
<top_n_words> <*save_history>`, where<tw> and <top_n_words> are cli arguments
representing within what time window we treat tweets as a batch, and how many
words with highest idf scores to show, while <*save_history> is an optional
boolean value indicating whether we want to dump the history. Make sure API
credentials are properly stored in the credentials.json file.