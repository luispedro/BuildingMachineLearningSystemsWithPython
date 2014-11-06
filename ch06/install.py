# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#
# Sanders-Twitter Sentiment Corpus Install Script
# Version 0.1
#
# Pulls tweet data from Twitter because ToS prevents distributing it directly.
#
#   - Niek Sanders
#     njs@sananalytics.com
#     October 20, 2011
#
#

# In Sanders' original form, the code was using Twitter API 1.0.
# Now that Twitter moved to 1.1, we had to make a few changes.
# Cf. twitterauth.py for the details.

# Regarding rate limiting, please check
# https://dev.twitter.com/rest/public/rate-limiting

import sys
import csv
import json
import os
import time

try:
    import twitter
except ImportError:
    print("""\
You need to ...
    pip install twitter
If pip is not found you might have to install it using easy_install.
If it does not work on your system, you might want to follow instructions
at https://github.com/sixohsix/twitter, most likely:
  $ git clone https://github.com/sixohsix/twitter
  $ cd twitter
  $ sudo python setup.py install
""")

    sys.exit(1)

from twitterauth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET
api = twitter.Twitter(auth=twitter.OAuth(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                                         token=ACCESS_TOKEN_KEY, token_secret=ACCESS_TOKEN_SECRET))

DATA_PATH = "data"

# for some reasons TWeets disappear. In this file we collect those
MISSING_ID_FILE = os.path.join(DATA_PATH, "missing.tsv")
NOT_AUTHORIZED_ID_FILE = os.path.join(DATA_PATH, "not_authorized.tsv")


def get_user_params(DATA_PATH):

    user_params = {}

    # get user input params
    user_params['inList'] = os.path.join(DATA_PATH, 'corpus.csv')
    user_params['outList'] = os.path.join(DATA_PATH, 'full-corpus.csv')
    user_params['rawDir'] = os.path.join(DATA_PATH, 'rawdata/')

    # apply defaults
    if user_params['inList'] == '':
        user_params['inList'] = './corpus.csv'
    if user_params['outList'] == '':
        user_params['outList'] = './full-corpus.csv'
    if user_params['rawDir'] == '':
        user_params['rawDir'] = './rawdata/'

    return user_params


def dump_user_params(user_params):

    # dump user params for confirmation
    print('Input:    ' + user_params['inList'])
    print('Output:   ' + user_params['outList'])
    print('Raw data: ' + user_params['rawDir'])


def read_total_list(in_filename):

    # read total fetch list csv
    fp = open(in_filename, 'rt')
    reader = csv.reader(fp, delimiter=',', quotechar='"')

    if os.path.exists(MISSING_ID_FILE):
        missing_ids = [line.strip()
                       for line in open(MISSING_ID_FILE, "r").readlines()]
    else:
        missing_ids = []

    if os.path.exists(NOT_AUTHORIZED_ID_FILE):
        not_authed_ids = [line.strip()
                          for line in open(NOT_AUTHORIZED_ID_FILE, "r").readlines()]
    else:
        not_authed_ids = []

    print("We will skip %i tweets that are not available or visible any more on twitter" % (
        len(missing_ids) + len(not_authed_ids)))

    ignore_ids = set(missing_ids + not_authed_ids)
    total_list = []

    for row in reader:
        if row[2] not in ignore_ids:
            total_list.append(row)

    return total_list


def purge_already_fetched(fetch_list, raw_dir):

    # list of tweet ids that still need downloading
    rem_list = []
    count_done = 0

    # check each tweet to see if we have it
    for item in fetch_list:

        # check if json file exists
        tweet_file = os.path.join(raw_dir, item[2] + '.json')
        if os.path.exists(tweet_file):

            # attempt to parse json file
            try:
                parse_tweet_json(tweet_file)
                count_done += 1
            except RuntimeError:
                print("Error parsing", item)
                rem_list.append(item)
        else:
            rem_list.append(item)

    print("We have already downloaded %i tweets." % count_done)

    return rem_list


def download_tweets(fetch_list, raw_dir):

    # ensure raw data directory exists
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # download tweets
    for idx in range(0, len(fetch_list)):
        # current item
        item = fetch_list[idx]
        print(item)

        print('--> downloading tweet #%s (%d of %d)' %
              (item[2], idx + 1, len(fetch_list)))

        try:
            #import pdb;pdb.set_trace()
            response = api.statuses.show(_id=item[2])

            if response.rate_limit_remaining <= 0:
                wait_seconds = response.rate_limit_reset - time.time()
                print("Rate limiting requests us to wait %f seconds" %
                      wait_seconds)
                time.sleep(wait_seconds+5)

        except twitter.TwitterError as e:
            fatal = True
            print(e)
            for m in json.loads(e.response_data.decode())['errors']:
                if m['code'] == 34:
                    print("Tweet missing: ", item)
                    with open(MISSING_ID_FILE, "at") as f:
                        f.write(item[2] + "\n")

                    fatal = False
                    break
                elif m['code'] == 63:
                    print("User of tweet '%s' has been suspended." % item)
                    with open(MISSING_ID_FILE, "at") as f:
                        f.write(item[2] + "\n")

                    fatal = False
                    break
                elif m['code'] == 88:
                    print("Rate limit exceeded.")
                    fatal = True
                    break
                elif m['code'] == 179:
                    print("Not authorized to view this tweet.")
                    with open(NOT_AUTHORIZED_ID_FILE, "at") as f:
                        f.write(item[2] + "\n")
                    fatal = False
                    break

            if fatal:
                raise
            else:
                continue

        with open(raw_dir + item[2] + '.json', "wt") as f:
            f.write(json.dumps(dict(response)) + "\n")

    return


def parse_tweet_json(filename):

    # read tweet
    fp = open(filename, 'r')

    # parse json
    try:
        tweet_json = json.load(fp)
    except ValueError as e:
        print(e)
        raise RuntimeError('error parsing json')

    # look for twitter api error msgs
    if 'error' in tweet_json or 'errors' in tweet_json:
        raise RuntimeError('error in downloaded tweet')

    # extract creation date and tweet text
    return [tweet_json['created_at'], tweet_json['text']]


def build_output_corpus(out_filename, raw_dir, total_list):

    # open csv output file
    fp = open(out_filename, 'wb')
    writer = csv.writer(fp, delimiter=',', quotechar='"', escapechar='\\',
                        quoting=csv.QUOTE_ALL)

    # write header row
    writer.writerow(
        ['Topic', 'Sentiment', 'TweetId', 'TweetDate', 'TweetText'])

    # parse all downloaded tweets
    missing_count = 0
    for item in total_list:

        # ensure tweet exists
        if os.path.exists(raw_dir + item[2] + '.json'):

            try:
                # parse tweet
                parsed_tweet = parse_tweet_json(raw_dir + item[2] + '.json')
                full_row = item + parsed_tweet

                # character encoding for output
                for i in range(0, len(full_row)):
                    full_row[i] = full_row[i].encode("utf-8")

                # write csv row
                writer.writerow(full_row)

            except RuntimeError:
                print('--> bad data in tweet #' + item[2])
                missing_count += 1

        else:
            print('--> missing tweet #' + item[2])
            missing_count += 1

    # indicate success
    if missing_count == 0:
        print('\nSuccessfully downloaded corpus!')
        print('Output in: ' + out_filename + '\n')
    else:
        print('\nMissing %d of %d tweets!' % (missing_count, len(total_list)))
        print('Partial output in: ' + out_filename + '\n')

    return


def main():
    # get user parameters
    user_params = get_user_params(DATA_PATH)
    print(user_params)
    dump_user_params(user_params)

    # get fetch list
    total_list = read_total_list(user_params['inList'])

    # remove already fetched or missing tweets
    fetch_list = purge_already_fetched(total_list, user_params['rawDir'])
    print("Fetching %i tweets..." % len(fetch_list))

    if fetch_list:
        # start fetching data from twitter
        download_tweets(fetch_list, user_params['rawDir'])

        # second pass for any failed downloads
        fetch_list = purge_already_fetched(total_list, user_params['rawDir'])
        if fetch_list:
            print('\nStarting second pass to retry %i failed downloads...' %
                  len(fetch_list))
            download_tweets(fetch_list, user_params['rawDir'])
    else:
        print("Nothing to fetch any more.")

    # build output corpus
    build_output_corpus(user_params['outList'], user_params['rawDir'],
                        total_list)


if __name__ == '__main__':
    main()
