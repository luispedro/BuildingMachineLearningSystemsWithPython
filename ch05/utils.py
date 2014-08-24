# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

try:
    import ujson as json  # UltraJSON if available
except:
    import json

from matplotlib import pylab
import numpy as np

from data import CHART_DIR


def fetch_data(filename, col=None, line_count=-1, only_questions=False):
    count = 0

    for line in open(filename, "r"):
        count += 1
        if line_count > 0 and count > line_count:
            break

        data = Id, ParentId, IsQuestion, IsAccepted, TimeToAnswer, Score, Text, NumTextTokens, NumCodeLines, LinkCount, MisSpelledFraction = line.split(
            "\t")

        IsQuestion = int(IsQuestion)

        if only_questions and not IsQuestion:
            continue

        if col:
            if col < 6:
                val = int(data[col])
            else:
                val = data[col]

            yield val

        else:
            Id = int(Id)
            assert Id >= 0, line

            ParentId = int(ParentId)

            IsAccepted = int(IsAccepted)

            assert not IsQuestion == IsAccepted == 1, "%i %i --- %s" % (
                IsQuestion, IsAccepted, line)
            assert (ParentId == -1 and IsQuestion) or (
                ParentId >= 0 and not IsQuestion), "%i %i --- %s" % (ParentId, IsQuestion, line)

            TimeToAnswer = int(TimeToAnswer)
            Score = int(Score)
            NumTextTokens = int(NumTextTokens)
            NumCodeLines = int(NumCodeLines)
            LinkCount = int(LinkCount)
            MisSpelledFraction = float(MisSpelledFraction)
            yield Id, ParentId, IsQuestion, IsAccepted, TimeToAnswer, Score, Text, NumTextTokens, NumCodeLines, LinkCount, MisSpelledFraction


def fetch_posts(filename, with_index=True, line_count=-1):
    count = 0

    for line in open(filename, "r"):
        count += 1
        if line_count > 0 and count > line_count:
            break

        Id, Text = line.split("\t")
        Text = Text.strip()

        if with_index:

            yield int(Id), Text

        else:

            yield Text


def load_meta(filename):
    meta = json.load(open(filename, "r"))
    keys = list(meta.keys())

    # JSON only allows string keys, changing that to int
    for key in keys:
        meta[int(key)] = meta[key]
        del meta[key]

    # post Id to index in vectorized
    id_to_idx = {}
    # and back
    idx_to_id = {}

    for PostId, Info in meta.items():
        id_to_idx[PostId] = idx = Info['idx']
        idx_to_id[idx] = PostId

    return meta, id_to_idx, idx_to_id


def plot_roc(auc_score, name, fpr, tpr):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('Receiver operating characteristic (AUC=%0.2f)\n%s' % (
        auc_score, name))
    pylab.legend(loc="lower right")
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.fill_between(tpr, fpr, alpha=0.5)
    pylab.plot(fpr, tpr, lw=1)
    pylab.savefig(
        os.path.join(CHART_DIR, "roc_" + name.replace(" ", "_") + ".png"))


def plot_pr(auc_score, name, precision, recall, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    filename = name.replace(" ", "_")
    pylab.savefig(os.path.join(CHART_DIR, "pr_" + filename + ".png"))


def show_most_informative_features(vectorizer, clf, n=20):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = list(zip(c_f[:n], c_f[:-(n + 1):-1]))
    for (c1, f1), (c2, f2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))


def plot_feat_importance(feature_names, clf, name):
    pylab.figure(num=None, figsize=(6, 5))
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
    if len(data_name_list) > 1:
        assert filename is not None

    pylab.figure(num=None, figsize=(8, 6))
    num_rows = int(1 + (len(data_name_list) - 1) / 2)
    num_cols = int(1 if len(data_name_list) == 1 else 2)
    pylab.figure(figsize=(5 * num_cols, 4 * num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            pylab.subplot(num_rows, num_cols, 1 + i * num_cols + j)
            x, name = data_name_list[i * num_cols + j]
            pylab.title(name)
            pylab.xlabel('Value')
            pylab.ylabel('Fraction')
            # the histogram of the data
            max_val = np.max(x)
            if max_val <= 1.0:
                bins = 50
            elif max_val > 50:
                bins = 50
            else:
                bins = max_val
            n, bins, patches = pylab.hist(
                x, bins=bins, normed=1, alpha=0.75)

            pylab.grid(True)

    if not filename:
        filename = "feat_hist_%s.png" % name.replace(" ", "_")

    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def plot_bias_variance(data_sizes, train_errors, test_errors, name, title):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Data set size')
    pylab.ylabel('Error')
    pylab.title("Bias-Variance for '%s'" % name)
    pylab.plot(
        data_sizes, test_errors, "--", data_sizes, train_errors, "b-", lw=1)
    pylab.legend(["test error", "train error"], loc="upper right")
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.savefig(
        os.path.join(CHART_DIR, "bv_" + name.replace(" ", "_") + ".png"), bbox_inches="tight")


def plot_k_complexity(ks, train_errors, test_errors):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('k')
    pylab.ylabel('Error')
    pylab.title('Errors for for different values of $k$')
    pylab.plot(
        ks, test_errors, "--", ks, train_errors, "-", lw=1)
    pylab.legend(["test error", "train error"], loc="upper right")
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.savefig(
        os.path.join(CHART_DIR, "kcomplexity.png"), bbox_inches="tight")
