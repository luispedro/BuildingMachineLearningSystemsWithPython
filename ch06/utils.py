# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import sys
import collections
import csv
import json

from matplotlib import pylab
import numpy as np


DATA_DIR = "data"
CHART_DIR = "charts"

if not os.path.exists(DATA_DIR):
    raise RuntimeError("Expecting directory 'data' in current path")

if not os.path.exists(CHART_DIR):
    os.mkdir(CHART_DIR)


def tweak_labels(Y, pos_sent_list):
    pos = Y == pos_sent_list[0]
    for sent_label in pos_sent_list[1:]:
        pos |= Y == sent_label

    Y = np.zeros(Y.shape[0])
    Y[pos] = 1
    Y = Y.astype(int)

    return Y


def load_sanders_data(dirname=".", line_count=-1):
    count = 0

    topics = []
    labels = []
    tweets = []

    with open(os.path.join(DATA_DIR, dirname, "corpus.csv"), "r") as csvfile:
        metareader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in metareader:
            count += 1
            if line_count > 0 and count > line_count:
                break

            topic, label, tweet_id = line

            tweet_fn = os.path.join(
                DATA_DIR, dirname, 'rawdata', '%s.json' % tweet_id)
            try:
                tweet = json.load(open(tweet_fn, "r"))
            except IOError:
                print(("Tweet '%s' not found. Skip." % tweet_fn))
                continue

            if 'text' in tweet and tweet['user']['lang'] == "en":
                topics.append(topic)
                labels.append(label)
                tweets.append(tweet['text'])

    tweets = np.asarray(tweets)
    labels = np.asarray(labels)

    return tweets, labels


def plot_pr(auc_score, name, phase, precision, recall, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.plot(recall, precision, lw=1)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R curve (AUC=%0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    pylab.savefig(os.path.join(CHART_DIR, "pr_%s_%s.png" %
                               (filename, phase)), bbox_inches="tight")


def show_most_informative_features(vectorizer, clf, n=20):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = list(zip(c_f[:n], c_f[:-(n + 1):-1]))
    for (c1, f1), (c2, f2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))


def plot_log():
    pylab.clf()
    pylab.figure(num=None, figsize=(6, 5))

    x = np.arange(0.001, 1, 0.001)
    y = np.log(x)

    pylab.title('Relationship between probabilities and their logarithm')
    pylab.plot(x, y)
    pylab.grid(True)
    pylab.xlabel('P')
    pylab.ylabel('log(P)')
    filename = 'log_probs.png'
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def plot_feat_importance(feature_names, clf, name):
    pylab.clf()
    coef_ = clf.coef_
    important = np.argsort(np.absolute(coef_.ravel()))
    f_imp = feature_names[important]
    coef = coef_.ravel()[important]
    inds = np.argsort(coef)
    f_imp = f_imp[inds]
    coef = coef[inds]
    xpos = np.array(list(range(len(coef))))
    pylab.bar(xpos, coef, width=1)

    pylab.title('Feature importance for %s' % (name))
    ax = pylab.gca()
    ax.set_xticks(np.arange(len(coef)))
    labels = ax.set_xticklabels(f_imp)
    for label in labels:
        label.set_rotation(90)
    filename = name.replace(" ", "_")
    pylab.savefig(os.path.join(
        CHART_DIR, "feat_imp_%s.png" % filename), bbox_inches="tight")


def plot_feat_hist(data_name_list, filename=None):
    pylab.clf()
    num_rows = 1 + (len(data_name_list) - 1) / 2
    num_cols = 1 if len(data_name_list) == 1 else 2
    pylab.figure(figsize=(5 * num_cols, 4 * num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            pylab.subplot(num_rows, num_cols, 1 + i * num_cols + j)
            x, name = data_name_list[i * num_cols + j]
            pylab.title(name)
            pylab.xlabel('Value')
            pylab.ylabel('Density')
            # the histogram of the data
            max_val = np.max(x)
            if max_val <= 1.0:
                bins = 50
            elif max_val > 50:
                bins = 50
            else:
                bins = max_val
            n, bins, patches = pylab.hist(
                x, bins=bins, normed=1, facecolor='green', alpha=0.75)

            pylab.grid(True)

    if not filename:
        filename = "feat_hist_%s.png" % name

    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def plot_bias_variance(data_sizes, train_errors, test_errors, name):
    pylab.clf()
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Data set size')
    pylab.ylabel('Error')
    pylab.title("Bias-Variance for '%s'" % name)
    pylab.plot(
        data_sizes, train_errors, "-", data_sizes, test_errors, "--", lw=1)
    pylab.legend(["train error", "test error"], loc="upper right")
    pylab.grid()
    pylab.savefig(os.path.join(CHART_DIR, "bv_" + name + ".png"))


def load_sent_word_net():

    sent_scores = collections.defaultdict(list)
    sentiwordnet_path = os.path.join(DATA_DIR, "SentiWordNet_3.0.0_20130122.txt")

    if not os.path.exists(sentiwordnet_path):
        print("Please download SentiWordNet_3.0.0 from http://sentiwordnet.isti.cnr.it/download.php, extract it and put it into the data directory")
        sys.exit(1)

    with open(sentiwordnet_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue

            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            # print POS,PosScore,NegScore,SynsetTerms
            for term in SynsetTerms.split(" "):
                # drop #number at the end of every term
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term.split("#")[0])
                sent_scores[key].append((float(PosScore), float(NegScore)))
    for key, value in sent_scores.items():
        sent_scores[key] = np.mean(value, axis=0)

    return sent_scores


def log_false_positives(clf, X, y, name):
    with open("FP_" + name.replace(" ", "_") + ".tsv", "w") as f:
        false_positive = clf.predict(X) != y
        for tweet, false_class in zip(X[false_positive], y[false_positive]):
            f.write("%s\t%s\n" %
                    (false_class, tweet.encode("ascii", "ignore")))


if __name__ == '__main__':
    plot_log()
