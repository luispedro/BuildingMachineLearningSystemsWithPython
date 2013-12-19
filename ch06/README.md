Chapter 6 - Classification II - Sentiment Analysis
==================================================

When doing last code sanity checks for the book, Twitter
was using the API 1.0, which did not require authentication.
With its switch to version 1.1, this has now changed.

If you don't have already created your personal Twitter
access keys and tokens, you might want to do so at
[https://dev.twitter.com/docs/auth/tokens-devtwittercom](https://dev.twitter.com/docs/auth/tokens-devtwittercom) and paste the keys/secrets into twitterauth.py

According to [https://dev.twitter.com/docs/rate-limiting/1](https://dev.twitter.com/docs/rate-limiting/1) Twitter has a rate limit of fetching 350 tweets/h for authorized users.

Note that some tweets might be missing when you are running install.py (user got suspended, changed authorization, or tweet deleted) and thus you might get different results. We keep track of those tweet IDs in data/{missing,not_authorized}.tsv, so that they are not fetched when you run install.py.
