# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import sys

from matplotlib import pylab
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

# Put your directory to the different music genres here
GENRE_DIR = None
GENRE_LIST = ["classical", "jazz", "country", "pop", "rock", "metal"]

# Put your directory to the test dir here
TEST_DIR = None

if GENRE_DIR is None or TEST_DIR is None:
    print("Please set GENRE_DIR and TEST_DIR in utils.py")
    sys.exit(1)


def plot_confusion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.show()
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.savefig(
        os.path.join(CHART_DIR, "confusion_matrix_%s.png" % name), bbox_inches="tight")


def plot_pr(auc_score, name, precision, recall, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.plot(recall, precision, lw=1)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R curve (AUC = %0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "pr_" + filename + ".png"), bbox_inches="tight")


def plot_roc(auc_score, name, tpr, fpr, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.fill_between(fpr, tpr, alpha=0.5)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve (AUC = %0.2f) / %s' %
                (auc_score, label), verticalalignment="bottom")
    pylab.legend(loc="lower right")
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "roc_" + filename + ".png"), bbox_inches="tight")


def show_most_informative_features(vectorizer, clf, n=20):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = zip(c_f[:n], c_f[:-(n + 1):-1])
    for (c1, f1), (c2, f2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))


def plot_log():
    pylab.clf()

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
    xpos = np.array(range(len(coef)))
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
    pylab.grid(True)
    pylab.savefig(os.path.join(CHART_DIR, "bv_" + name + ".png"))
