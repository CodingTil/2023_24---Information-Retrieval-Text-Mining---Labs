import multiprocessing as mp
import string
from typing import List, Tuple, Union
import re

from numpy import ndarray
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(path: str) -> Tuple[List[str], List[int]]:
    """Loads data from file. Each except first (header) is a datapoint
    containing ID, Label, Email (content) separated by "\t". Lables should be
    changed into integers with 1 for "spam" and 0 for "ham".

    Args:
        path: Path to file from which to load data

    Returns:
        List of email contents and a list of lobels coresponding to each email.
    """
    data = pd.read_csv(path, sep="\t")
    labels = data["Label"].apply(lambda x: 1 if x == "spam" else 0)
    return data["Email"].tolist(), labels.tolist()


re_html = re.compile("<[^>]+>")
def remove_punctuation(doc: str) -> str:
    """Removes punctuation and HTMl tags from a string.

    Args:
        doc: String comprising the contents of some email file.

    Returns:
        String comprising the corresponding text with punctuation removed.
    """
    text = re_html.sub(" ", doc)
    # Replace punctuation marks (including hyphens) with spaces.
    for c in string.punctuation:
        text = text.replace(c, " ")
    return text


def remove_stopwords(doc: str) -> str:
    """Removes stopwords from a string.

    Args:
        doc: String comprising the contents of some email file.

    Returns:
        String comprising the corresponding text with stopwords removed.
    """
    doc = doc.lower()
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(doc)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


url_pattern = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)
def normalize_URLs(doc: str) -> str:
    """Normalizes URLs in a string.

    Args:
        doc: String comprising the contents of some email file.

    Returns:
        String comprising the corresponding text with URLs normalized.
    """
    doc = url_pattern.sub('URL', doc)
    return doc


number_pattern = re.compile(r'\b\d+\b')
def normalize_numbers(doc: str) -> str:
    """Normalizes numbers in a string.

    Args:
        doc: String comprising the contents of some email file.

    Returns:
        String comprising the corresponding text with numbers normalized.
    """
    doc = number_pattern.sub('NUMBER', doc)
    return doc


lemmatizer = WordNetLemmatizer()
def lemmatize(doc: str) -> str:
    """Lemmatizes a string.

    Args:
        doc: String comprising the contents of some email file.

    Returns:
        String comprising the corresponding text with lemmas substituted for
            words.
    """
    word_tokens = word_tokenize(doc)
    lemmatized_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]
    return " ".join(lemmatized_sentence)


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    text = doc
    text = normalize_URLs(text)
    text = normalize_numbers(text)
    text = remove_punctuation(text)
    text = lemmatize(text)
    text = remove_stopwords(text)
    return text


def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    pool = mp.Pool(mp.cpu_count())
    preprocessed_docs = pool.map(preprocess, docs)
    pool.close()
    return preprocessed_docs


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for 
          training and testing dataset respectively.
    """
    tfidf_vectorizer = TfidfVectorizer()
    train_feature_vectors = tfidf_vectorizer.fit_transform(train_dataset)
    test_feature_vectors = tfidf_vectorizer.transform(test_dataset)

    return train_feature_vectors, test_feature_vectors


def train(X: ndarray, y: List[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    #classifier = MultinomialNB()
    classifier = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        random_state=43
    )
    classifier.fit(X, y)
    return classifier


def evaluate(y: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return recall, precision, f1, accuracy


if __name__ == "__main__":
    print("Installing nltk...")
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
