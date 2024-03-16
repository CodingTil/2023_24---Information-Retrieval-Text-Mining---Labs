import A3_2 as module
import pytest

INDEX_NAME = "toy_index_{}"

INDEX_SETTINGS = {
    "properties": {
        "title": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
        "body": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
        "anchors": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
    }
}


@pytest.fixture(scope="module")
def doc_collection1():
    return {
        "d1": {"body": "t3 t3 t3 t6 t6"},
        "d2": {"body": "t1 t2 t3 t3 t6"},
        "d3": {"body": "t3 t3 t4 t5"},
        "d4": {"body": "t4 t5 t6 t6"},
        "d5": {"body": "t1 t2 t3 t5"},
    }


@pytest.fixture(scope="module")
def doc_collection2():
    return {
        "d1": {
            "title": "t1",
            "body": "t1 t2 t3 t1 t3",
            "anchors": "t2 t2",
        },
        "d2": {
            "title": "t4 t5",
            "body": "t1 t3 t4 t4 t4 t5",
            "anchors": "t5 t3",
        },
        "d3": {
            "title": "t1 t3 t5",
            "body": "t1 t1 t5 t3 t5 t3 t3",
            "anchors": "t1 t1 t5",
        },
    }


@pytest.fixture(scope="module", autouse=True)
def EScollection1(doc_collection1):
    collection = module.ElasticsearchCollection("toy_index_1")
    collection.index(doc_collection1, INDEX_SETTINGS)
    return collection


@pytest.fixture(scope="module", autouse=True)
def EScollection2(doc_collection2):
    collection = module.ElasticsearchCollection("toy_index_2")
    collection.index(doc_collection2, INDEX_SETTINGS)
    return collection


@pytest.fixture(scope="module")
def documents1(EScollection1: module.ElasticsearchCollection, doc_collection1):
    return [EScollection1.get_document(doc_id) for doc_id in doc_collection1.keys()]


@pytest.fixture(scope="module")
def documents2(EScollection2: module.ElasticsearchCollection, doc_collection2):
    return [EScollection2.get_document(doc_id) for doc_id in doc_collection2.keys()]


@pytest.fixture(scope="module")
def single_field_scorer(EScollection1):
    return module.SDMScorer(EScollection1)


@pytest.fixture(scope="module")
def scorer(EScollection2):
    return module.FSDMScorer(EScollection2, fields=["body", "title", "anchors"])


def test_sdm_ordered_bigram_matches(single_field_scorer, documents1):
    """This test is 0.5 points"""
    assert single_field_scorer.ordered_bigram_matches(
        ["t7", "t3", "t3"], documents1[0], documents1
    ) == pytest.approx(-1.6492, abs=1e-4)
    assert (
        single_field_scorer.ordered_bigram_matches(
            ["t6", "t2"], documents1[3], documents1
        )
        == 0
    )


def test_sdm_unordered_bigram_matches(single_field_scorer, documents1):
    """This test is 0.5 points"""
    assert single_field_scorer.unordered_bigram_matches(
        ["t7", "t3", "t3"], documents1[0], documents1
    ) == pytest.approx(-1.4064, abs=1e-4)
    assert (
        single_field_scorer.unordered_bigram_matches(
            ["t5", "t1"], documents1[2], documents1
        )
        == 0
    )


def test_sdm_score_collection(single_field_scorer):
    """This test is 1 points"""
    assert single_field_scorer.score_collection("t7 t3 t3")["d1"] == pytest.approx(
        -1.9031, abs=1e-4
    )
    assert single_field_scorer.score_collection("t3 t5 t2")["d4"] == pytest.approx(
        -5.2229, abs=1e-4
    )


def test_fsdm_ordered_bigram_matches(scorer, documents2):
    """This test is 0.5 points"""
    assert scorer.ordered_bigram_matches(
        ["t2", "t1", "t3"], documents2[0], documents2
    ) == pytest.approx(-1.97630, abs=1e-4)
    assert scorer.ordered_bigram_matches(["t6", "t2"], documents2[1], documents2) == 0


def test_fsdm_unordered_bigram_matches(scorer, documents2):
    """This test is 0.5 points"""
    assert scorer.unordered_bigram_matches(
        ["t1", "t3", "t3"], documents2[0], documents2
    ) == pytest.approx(-5.1396, abs=1e-4)
    assert scorer.unordered_bigram_matches(
        ["t5", "t1"], documents2[2], documents2
    ) == pytest.approx(-1.74725, abs=1e-4)


def test_fsdm_score_collection(scorer):
    """This test is 1 points"""
    assert scorer.score_collection("t2 t1 t3")["d1"] == pytest.approx(-5.4979, abs=1e-4)
    assert scorer.score_collection("t3 t5 t2")["d2"] == pytest.approx(-5.4874, abs=1e-4)
