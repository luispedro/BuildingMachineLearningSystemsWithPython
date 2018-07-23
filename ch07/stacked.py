import numpy as np
import load_ml100k
import regression
import corrneighbours
from sklearn import linear_model, metrics
import norm

def predict(train):
    tr_train,tr_test = load_ml100k.get_train_test(train, random_state=34)
    tr_predicted0 = regression.predict(tr_train)
    tr_predicted1 = regression.predict(tr_train.T).T
    tr_predicted2 = corrneighbours.predict(tr_train)
    tr_predicted3 = corrneighbours.predict(tr_train.T).T
    tr_predicted4 = norm.predict(tr_train)
    tr_predicted5 = norm.predict(tr_train.T).T
    stack_tr = np.array([
        tr_predicted0[tr_test > 0],
        tr_predicted1[tr_test > 0],
        tr_predicted2[tr_test > 0],
        tr_predicted3[tr_test > 0],
        tr_predicted4[tr_test > 0],
        tr_predicted5[tr_test > 0],
        ]).T

    lr = linear_model.LinearRegression()
    lr.fit(stack_tr, tr_test[tr_test > 0])

    stack_te = np.array([
        tr_predicted0.ravel(),
        tr_predicted1.ravel(),
        tr_predicted2.ravel(),
        tr_predicted3.ravel(),
        tr_predicted4.ravel(),
        tr_predicted5.ravel(),
        ]).T

    return lr.predict(stack_te).reshape(train.shape)


def main():
    train,test = load_ml100k.get_train_test(random_state=12)
    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 stacked: {:.2%}'.format(r2))

if __name__ == '__main__':
    main()
