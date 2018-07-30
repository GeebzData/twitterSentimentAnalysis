# General:
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
import re               # For regex functions
import time

#For sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Tweet keys
# Consumer:
CONSUMER_KEY    = 'XXXXXXXXX'
CONSUMER_SECRET = 'XXXXXXXXX'

# Access:
ACCESS_TOKEN  = 'XXXXXXXXX'
ACCESS_SECRET = 'XXXXXXXXX'


# api setup
def twitter_setup():
    """
    Utility function to setup the Twitter's API access keys
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

# extractor object
extractor = twitter_setup()

# tweet list > 200 max tweets from tweepy
tweets = extractor.user_timeline(screen_name="elonmusk", count=200)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# Creation of an initial pandas data frame
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

#Additional tweet data
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])

# Time series for Likes and RTs
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

# Likes vs retweets vizzie
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True)

# TextBlob sentiment analysis
def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

# We create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

# data prep for pie charts
    pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
    neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
    neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

# Pie chart vizzie
labels = ['positive', 'negative', 'neutral']
sizes = [len(pos_tweets), len(neg_tweets), len(neu_tweets)]
colors = ['green', 'red', 'lightskyblue']
explode = (0.1, 0, 0) # explode first slice
plt.title('TextBlob Twitter Sentiment Results')
plt.pie(sizes, labels=labels, explode=explode,colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

# Bar chart vizzie
y_pos = np.arange(len(labels))
performance = sizes
plt.bar(y_pos, performance, align='center', alpha = 0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Number of Tweets', fontdict=None, labelpad=None)
plt.title('TextBlob Twitter Sentiment Results')
plt.show()

# Vader sentiment analysis
analyser = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    return snt

data['Vader'] = np.array([ print_sentiment_scores(tweet) for tweet in data['Tweets'] ])

#create dataframe to parse out Vader results
results = []

for line in data['Tweets']:
    pol_score = analyser.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

print(results[:3])

df = pd.DataFrame.from_records(results)
df.head()

# dataframes merging prep
df.rename(columns={'headline':'Tweets'}, inplace=True)

# merge dataframes
data = pd.merge(data, df, on='Tweets')

# count up positive and negative sentiment
v_pos_count = 0
v_neg_count = 0
v_neut_count = 0

for line in data['compound']:
    if line >= 0.05:
        v_pos_count += 1
    elif line <= -0.05:
        v_neg_count += 1
    else:
        v_neut_count += 1

total = v_pos_count + v_neg_count + v_neut_count

# print results
print("TEXTBLOB RESULTS BREAKDOWN")
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print(" ")
print("VADER RESULTS BREAKDOWN")
print("Percentage of positive tweets: {}%".format((v_pos_count/total)*100))
print("Percentage of negative tweets: {}%".format((v_neg_count/total)*100))
print("Percentage of neutral tweets: {}%".format((v_neut_count/total)*100))

# accuracy of results
## TextBlob accuracy
tb_pos_count = 0
tb_pos_correct = 0

for line in data['Tweets']:
    analysis = TextBlob(line)
    if analysis.sentiment.polarity > 0:
        tb_pos_correct += 1
    tb_pos_count +=1
tb_neg_count = 0
tb_neg_correct = 0

for line in data['Tweets']:
    analysis = TextBlob(line)
    if analysis.sentiment.polarity <= 0:
        tb_neg_correct += 1
    tb_neg_count +=1

# VADER accuracy
v_pos_count = 0
v_pos_correct = 0

for line in data['Tweets']:
    vs = analyser.polarity_scores(line)
    if not vs['neg'] > 0.1:
        if vs['pos']-vs['neg'] > 0:
            v_pos_correct += 1
        v_pos_count +=1

v_neg_count = 0
v_neg_correct = 0

for line in data['Tweets']:
    vs = analyser.polarity_scores(line)
    if not vs['pos'] > 0.1:
        if vs['pos']-vs['neg'] <= 0:
            v_neg_correct += 1
        v_neg_count +=1

# print accuracy
print("VADER ACCURACY")
print("VADER positive accuracy = {}% ".format(v_pos_correct/v_pos_count*100.0))
print("VADER total positive count = {}".format(v_pos_count))
print("VADER negative accuracy = {}%".format(v_neg_correct/v_neg_count*100.0))
print("VADER total negative count = {}".format(v_neg_count))
print(" ")
print("TextBlob ACCURACY")
print("TextBlob positive accuracy = {}%".format(tb_pos_correct/tb_pos_count*100.0))
print("TextBlob total positive count = {}".format(tb_pos_count))
print("TextBlob negative accuracy = {}%".format(tb_neg_correct/tb_neg_count*100.0))
print("TextBlob total negative count = {}".format(tb_neg_count))
