from typing import List, Set, Tuple

import A4 as module
import pytest


@pytest.fixture(scope="module")
def words() -> List[str]:
    return [
        "every",
        "morning",
        "we",
        "look",
        "for",
        "shells",
        "in",
        "the",
        "sand",
        "i",
        "found",
        "fifteen",
        "big",
        "last",
        "year",
        "put",
        "them",
        "a",
        "special",
        "place",
        "my",
        "room",
        "this",
        "want",
        "to",
        "learn",
        "surf",
        "it",
        "is",
        "hard",
        "but",
        "so",
        "much",
        "fun",
        "sister",
        "good",
        "surfer",
        "she",
        "says",
        "that",
        "can",
        "teach",
        "me",
        "hope",
        "do",
    ]


@pytest.fixture(scope="module")
def corpus() -> List[Tuple[str, int]]:
    return [("neural", 3), ("network", 2), ("neuron", 1), ("learning", 5)]


@pytest.fixture(scope="module")
def vocabulary() -> Set[str]:
    return set(
        [
            "##o",
            "##g",
            "##w",
            "##u",
            "##n",
            "##l",
            "l",
            "##k",
            "##i",
            "##r",
            "##t",
            "n",
            "##e",
            "##a",
        ]
    )


@pytest.fixture(scope="module")
def tokenized_corpus() -> List[Tuple[List[str], int]]:
    return [
        (["n", "##e", "##u", "##r", "##a", "##l"], 3),
        (["n", "##e", "##t", "##w", "##o", "##r", "##k"], 2),
        (["n", "##e", "##u", "##r", "##o", "##n"], 1),
        (["l", "##e", "##a", "##r", "##n", "##i", "##n", "##g"], 5),
    ]


@pytest.fixture(scope="module")
def vocabulary2() -> Set[str]:
    return set(
        [
            "##l",
            "##k",
            "##e",
            "##o",
            "l",
            "##u",
            "##tw",
            "##two",
            "n",
            "##al",
            "##i",
            "##n",
            "##a",
            "##g",
            "##t",
            "##w",
            "##r",
        ]
    )


@pytest.fixture(scope="module")
def tokenized_corpus2() -> List[Tuple[List[str], int]]:
    return [
        (["n", "##e", "##u", "##r", "##al"], 3),
        (["n", "##e", "##two", "##r", "##k"], 2),
        (["n", "##e", "##u", "##r", "##o", "##n"], 1),
        (["l", "##e", "##a", "##r", "##n", "##i", "##n", "##g"], 5),
    ]


def test_initialize_vocabulary(words: List[str]) -> None:
    """This test is 0.5 points"""
    initial_vocabulary = module.initialize_vocabulary(words)
    assert len(initial_vocabulary) == 37
    assert "every" not in initial_vocabulary
    assert "##y" in initial_vocabulary


def test_tokenize_corpus(
    corpus: List[Tuple[str, int]], vocabulary: Set[str]
) -> None:
    """This test is 0.5 points"""
    tokenized_corpus = module.tokenize_corpus(corpus, vocabulary)
    assert len(tokenized_corpus) == 4
    assert tokenized_corpus[0][0] == ["n", "##e", "##u", "##r", "##a", "##l"]
    assert tokenized_corpus[0][1] == 3


def test_train(corpus: List[Tuple[str, int]], vocabulary: Set[str]) -> None:
    """This test is 1 point"""
    vocabulary1 = module.train(corpus, vocabulary, num_iterations=4)
    assert len(vocabulary1) == 18
    assert "ne" in vocabulary1
    assert "le" not in vocabulary1

    vocabulary2 = module.train(
        corpus, vocabulary, num_iterations=20, max_vocab_size=25
    )
    assert len(vocabulary2) == 25
    assert "##tw" in vocabulary2


def test_get_new_subword_token1(
    tokenized_corpus: List[Tuple[List[str], int]], vocabulary: Set[str]
) -> None:
    """This test is 0.5 points"""
    new_subword_token, score = module.get_new_subword_token(
        tokenized_corpus, vocabulary
    )
    assert new_subword_token == "##tw"
    assert score == 0.5


def test_get_new_subword_token2(
    tokenized_corpus2: List[Tuple[List[str], int]], vocabulary2: Set[str]
) -> None:
    """This test is 0.5 points"""
    new_subword_token, score = module.get_new_subword_token(
        tokenized_corpus2, vocabulary2
    )
    assert new_subword_token == "ne"
    assert score == pytest.approx(0.091, rel=1e-3)
