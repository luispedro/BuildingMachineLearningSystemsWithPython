import numpy as np
import load_ml100k
import regression
import corrneighbours
from sklearn import metrics
import norm

def predict(train):
    predicted0 = regression.predict(train)
    predicted1 = regression.predict(train.T).T
    predicted2 = corrneighbours.predict(train)
    predicted3 = corrneighbours.predict(train.T).T
    predicted4 = norm.predict(train)
    predicted5 = norm.predict(train.T).T
    stack = np.array([
        predicted0,
        predicted1,
        predicted2,
        predicted3,
        predicted4,
        predicted5,
        ])
    return stack.mean(0)


def main():
    train,test = load_ml100k.get_train_test(random_state=12)
    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 averaged: {:.2%}'.format(r2))

if __name__ == '__main__':
    main()
