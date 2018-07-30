# Twitter Sentiment Analysis
Elon Musk was the account I trialed with, this can be edited easily by changing line 45 to the user you want to analyze.

This file does a number of things:
* Pulls in 200 tweets from a user with tweepy (http://www.tweepy.org/)
* Adds additional information associated to the tweet, such as:
** length, id, date, source, # of Likes, # of RTs
* Performs sentiment analysis using both TextBlob and VADER Sentiment Analysis
* Visualizes results from TextBlob >> in the future I plan to visualize the VADER results as well
* Calculates the accuracy of the sentiment analysis, not surprisingly the VADER analsis did better out of the box. There are ways to raise the subjectivity and polarity scores to get better results, but for this experiment I wanted to see out of the box how do these products perform

Visualizations of results
![alt text](https://github.com/GeebzData/twitterSentimentAnalysis/blob/master/Screen%20Shot%202018-07-30%20at%204.30.08%20PM.png)
![alt text](https://github.com/GeebzData/twitterSentimentAnalysis/blob/master/Screen%20Shot%202018-07-30%20at%204.30.08%20PM.png)

Credit due where credit is earned:
- https://pythonprogramming.net/sentiment-analysis-python-textblob-vader/
- https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
- https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
* I can't remember the other sources at this moment.
