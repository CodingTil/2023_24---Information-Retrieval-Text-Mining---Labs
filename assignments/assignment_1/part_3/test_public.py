import pytest

import A1_3 as module


@pytest.fixture(scope="module")
def train_data():
    return module.load_data("data/train.tsv")


@pytest.fixture(scope="module")
def test_data():
    return module.load_data("data/test.tsv")


@pytest.fixture(scope="module")
def processed_train_data(train_data):
    train_contents, _ = train_data
    return module.preprocess_multiple(train_contents)


@pytest.fixture(scope="module")
def processed_test_data(test_data):
    test_contents, _ = test_data
    return module.preprocess_multiple(test_contents)


@pytest.fixture(scope="module")
def feature_vectors(processed_train_data, processed_test_data):
    return module.extract_features(processed_train_data, processed_test_data)


@pytest.fixture(scope="module")
def classifier(train_data, feature_vectors):
    _, train_labels = train_data
    train_vectors, _ = feature_vectors
    return module.train(train_vectors, train_labels)


def test_load_data(train_data):
    """This test is 0.6 points."""
    train_contents, train_labels = train_data
    assert len(train_contents) == 74628
    assert (
        train_contents[0][0:120]
        == "Received: from NAHOU-MSMBX01V ([192.168.110.39]) by NAHOU-MSMBX05V"
        ".corp.enron.com with Microsoft SMTPSVC(5.0.2195.1600);"
    )
    assert (
        train_contents[0][-121:]
        == "subject to the terms and conditions of the license agreement with "
        "BNA. Unauthorized access or distribution is prohibited."
    )

    assert len(train_labels) == 74628
    assert train_labels[55:65] == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_preprocess(train_data):
    """This test is 0.6 points."""
    train_contents, _ = train_data
    preprocessed_example = module.preprocess(train_contents[0]).split()
    assert "don't" not in preprocessed_example
    assert "solicitation" in preprocessed_example
    assert "," not in preprocessed_example


def test_preprocess_multiple(processed_train_data, processed_test_data):
    """This test is 0.1 points."""
    assert len(processed_train_data) == 74628
    assert len(processed_test_data) == 8278


def test_extract_features(feature_vectors):
    """This test is 0.6 points."""
    train_vectors, test_vectors = feature_vectors
    assert train_vectors.shape[0] == 74628
    assert train_vectors.shape[1] > 0
    assert test_vectors.shape[0] == 8278
    assert test_vectors.shape[1] > 0


def test_train(classifier, feature_vectors):
    """This test is 0.6 points."""
    _, test_vectors = feature_vectors
    validation_prediction = classifier.predict(test_vectors)
    assert len(validation_prediction) == 8278


def test_validation_evaluation():
    """This test is 0.5 points."""
    test_labels = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    ground_truth = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    recall, precision, f1, accuracy = module.evaluate(test_labels, ground_truth)
    assert recall == pytest.approx(0.8333, 1e-4)
    assert precision == pytest.approx(0.7143, 1e-4)
    assert f1 == pytest.approx(0.7692, 1e-4)
    assert accuracy == 0.75
