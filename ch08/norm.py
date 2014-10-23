
class NormalizePositive(object):
    
    def fit(self, features):
        binary = (features > 0)
        count0 = binary.sum(0)
        count0 += (count0 == 0)
        self.mean = features.sum(0)/count0
        diff = (features - self.mean) * binary
        diff **= 2
        diff += 1e-9
        self.std = 1 + diff.sum(0)/count0
        return self

    def fit_transform(self, features):
        return self.fit(features).transform(features)

    def transform(self, features):
        binary = (features > 0)
        features = features - self.mean
        features /= self.std
        features *= binary
        return features

    def inverse_transform(self, features, copy=True):
        if copy:
            features = features.copy()
        features *= self.std
        features += self.mean
        return features

def predict(train):
    norm = NormalizePositive()
    train = norm.fit_transform(train)
    return norm.inverse_transform(train * 0.)


def main(transpose_inputs=False):
    from load_ml100k import get_train_test
    from sklearn import metrics
    train,test = get_train_test(random_state=12)
    if transpose_inputs:
        train = train.T
        test = test.T
    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 score ({} normalization): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))
if __name__ == '__main__':
    main()
    main(transpose_inputs=True)
