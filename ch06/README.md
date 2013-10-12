Chapter 6 - Classification II - Sentiment Analysis
==================================================

When doing last code sanity checks for the book, Twitter
was using the API 1.0, which did not require authentication.
With its switch to version 1.1, this has now changed.

If you don't have already created your personal Twitter
access keys and tokens, you might want to do so at
[https://dev.twitter.com/docs/auth/tokens-devtwittercom](https://dev.twitter.com/docs/auth/tokens-devtwittercom) and paste the keys/secrets into twitterauth.py

Note that some tweets might be missing when you are running install.py. 
We experimented a bit with with the tweet-fetch-rate and found that
max_tweets_per_hr=10000 works just fine, now that we are using OAuth. If you experience issues you might want to lower this value.
