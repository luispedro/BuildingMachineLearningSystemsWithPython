
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
