from load import load_dataset

def test_iris():
    features, labels = load_dataset('iris')
    assert len(features[0]) == 4
    assert len(features)
    assert len(features) == len(labels)
def test_seeds():
    features, labels = load_dataset('seeds')
    assert len(features[0]) == 7
    assert len(features)
    assert len(features) == len(labels)
