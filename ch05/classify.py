# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import time
start_time = time.time()

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn import neighbors

from data import chosen, chosen_meta
from utils import plot_pr
from utils import plot_feat_importance
from utils import load_meta
from utils import fetch_posts
from utils import plot_feat_hist
from utils import plot_bias_variance
from utils import plot_k_complexity

# question Id -> {'features'->feature vector, 'answers'->[answer Ids]}, 'scores'->[scores]}
# scores will be added on-the-fly as the are not in meta
meta, id_to_idx, idx_to_id = load_meta(chosen_meta)

import nltk

# The sorting below is only to ensure reproducable numbers. Further down
# we will occasionally skip a fold when it contains instances of only
# one label. The two lines below ensure that the behavior is exactly the
# same for different runs.
all_questions = sorted([q for q, v in meta.items() if v['ParentId'] == -1])
all_answers = sorted([q for q, v in meta.items() if v['ParentId'] != -1])

feature_names = np.array((
    'NumTextTokens',
    'NumCodeLines',
    'LinkCount',
    'AvgSentLen',
    'AvgWordLen',
    'NumAllCaps',
    'NumExclams',
    'NumImages'
))


def prepare_sent_features():
    for pid, text in fetch_posts(chosen, with_index=True):
        if not text:
            meta[pid]['AvgSentLen'] = meta[pid]['AvgWordLen'] = 0
        else:
            from platform import python_version
            if python_version().startswith('2'):
                text = text.decode('utf-8')
            sent_lens = [len(nltk.word_tokenize(
                sent)) for sent in nltk.sent_tokenize(text)]
            meta[pid]['AvgSentLen'] = np.mean(sent_lens)
            meta[pid]['AvgWordLen'] = np.mean(
                [len(w) for w in nltk.word_tokenize(text)])

        meta[pid]['NumAllCaps'] = np.sum(
            [word.isupper() for word in nltk.word_tokenize(text)])

        meta[pid]['NumExclams'] = text.count('!')


prepare_sent_features()


def get_features(aid):
    return tuple(meta[aid][fn] for fn in feature_names)

qa_X = np.asarray([get_features(aid) for aid in all_answers])

classifying_answer = "good"
#classifying_answer = "poor"

if classifying_answer == "good":
    # Score > 0 tests => positive class is good answer
    qa_Y = np.asarray([meta[aid]['Score'] > 0 for aid in all_answers])
elif classifying_answer == "poor":
    # Score <= 0 tests => positive class is poor answer
    qa_Y = np.asarray([meta[aid]['Score'] <= 0 for aid in all_answers])
else:
    raise Exception("classifying_answer='%s' is not supported" %
                    classifying_answer)

for idx, feat in enumerate(feature_names):
    plot_feat_hist([(qa_X[:, idx], feat)])

#plot_feat_hist([(qa_X[:, idx], feature_names[idx]) for idx in [1,0]], 'feat_hist_two.png')
#plot_feat_hist([(qa_X[:, idx], feature_names[idx]) for idx in [3,4,5,6]], 'feat_hist_four.png')

avg_scores_summary = []


def measure(clf_class, parameters, name, data_size=None, plot=False):
    start_time_clf = time.time()
    if data_size is None:
        X = qa_X
        Y = qa_Y
    else:
        X = qa_X[:data_size]
        Y = qa_Y[:data_size]

    cv = KFold(n=len(X), n_folds=10, indices=True)

    train_errors = []
    test_errors = []

    scores = []
    roc_scores = []
    fprs, tprs = [], []

    pr_scores = []
    precisions, recalls, thresholds = [], [], []

    for fold_idx, (train, test) in enumerate(cv):
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        only_one_class_in_train = len(set(y_train)) == 1
        only_one_class_in_test = len(set(y_test)) == 1
        if only_one_class_in_train or only_one_class_in_test:
            # this would pose problems later on
            continue

        clf = clf_class(**parameters)

        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        proba = clf.predict_proba(X_test)

        label_idx = 1
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, label_idx])
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba[:, label_idx])

        roc_scores.append(auc(fpr, tpr))
        fprs.append(fpr)
        tprs.append(tpr)

        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

        # This threshold is determined at the end of the chapter 5,
        # where we find conditions such that precision is in the area of
        # about 80%. With it we trade off recall for precision.
        threshold_for_detecting_good_answers = 0.59

        print("Clone #%i" % fold_idx)
        print(classification_report(y_test, proba[:, label_idx] >
              threshold_for_detecting_good_answers, target_names=['not accepted', 'accepted']))

    # get medium clone
    scores_to_sort = pr_scores  # roc_scores
    medium = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
    print("Medium clone is #%i" % medium)

    if plot:
        #plot_roc(roc_scores[medium], name, fprs[medium], tprs[medium])
        plot_pr(pr_scores[medium], name, precisions[medium],
                recalls[medium], classifying_answer + " answers")

        if hasattr(clf, 'coef_'):
            plot_feat_importance(feature_names, clf, name)

    summary = (name,
               np.mean(scores), np.std(scores),
               np.mean(roc_scores), np.std(roc_scores),
               np.mean(pr_scores), np.std(pr_scores),
               time.time() - start_time_clf)
    print(summary)
    avg_scores_summary.append(summary)
    precisions = precisions[medium]
    recalls = recalls[medium]
    thresholds = np.hstack(([0], thresholds[medium]))
    idx80 = precisions >= 0.8
    print("P=%.2f R=%.2f thresh=%.2f" % (precisions[idx80][0], recalls[
          idx80][0], thresholds[idx80][0]))

    return np.mean(train_errors), np.mean(test_errors)


def bias_variance_analysis(clf_class, parameters, name):
    #import ipdb;ipdb.set_trace()
    data_sizes = np.arange(60, 2000, 4)

    train_errors = []
    test_errors = []

    for data_size in data_sizes:
        train_error, test_error = measure(
            clf_class, parameters, name, data_size=data_size)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_bias_variance(data_sizes, train_errors,
                       test_errors, name, "Bias-Variance for '%s'" % name)


def k_complexity_analysis(clf_class, parameters):
    ks = np.hstack((np.arange(1, 20), np.arange(21, 100, 5)))

    train_errors = []
    test_errors = []

    for k in ks:
        parameters['n_neighbors'] = k
        train_error, test_error = measure(
            clf_class, parameters, "%dNN" % k, data_size=2000)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_k_complexity(ks, train_errors, test_errors)

for k in [5]:
# for k in [5, 10, 40]:
    #measure(neighbors.KNeighborsClassifier, {'n_neighbors': k}, "%iNN" % k)
    bias_variance_analysis(neighbors.KNeighborsClassifier, {
                           'n_neighbors': k}, "%iNN" % k)
    k_complexity_analysis(neighbors.KNeighborsClassifier, {'n_neighbors': k})

from sklearn.linear_model import LogisticRegression
for C in [0.1]:
# for C in [0.01, 0.1, 1.0, 10.0]:
    name = "LogReg C=%.2f" % C
    bias_variance_analysis(LogisticRegression, {'penalty': 'l2', 'C': C}, name)
    measure(LogisticRegression, {'penalty': 'l2', 'C': C}, name, plot=True)

print("=" * 50)
from operator import itemgetter
for s in reversed(sorted(avg_scores_summary, key=itemgetter(1))):
    print("%-20s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % s)

print("time spent:", time.time() - start_time)
