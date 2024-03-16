from typing import List

import A3_1 as module
import pytest
from elasticsearch import Elasticsearch

INDEX_NAME = "musicians"


@pytest.fixture(scope="module")
def es():
    return Elasticsearch()


@pytest.fixture(scope="module")
def artists():
    return module.get_artists()


@pytest.fixture(scope="module", autouse=True)
def setup_module_fixture(es, artists):
    """This fixture re-indexes the data before running the tests."""
    module.reset_index(es)
    module.bulk_index(es, artists)
    yield
    module.reset_index(es)


@pytest.fixture(scope="module")
def clm():
    class ToyCollectionLM(module.CollectionLM):
        def __init__(self):
            super().__init__(None, [])
            self._probs = {
                "names": {
                    "t1": 1 / 3,
                    "t2": 0,
                    "t3": 1 / 9,
                    "t4": 1 / 9,
                    "t5": 1 / 3,
                    "gospel": 1 / 9,
                    "soul": 2 / 9,
                },
                "types": {
                    "t1": 5 / 18,
                    "t2": 1 / 18,
                    "t3": 1 / 3,
                    "t4": 1 / 6,
                    "t5": 1 / 6,
                    "gospel": 0,
                    "soul": 0,
                },
                "description": {
                    "t1": 1 / 20,
                    "t2": 1 / 20,
                    "t3": 1 / 10,
                    "t4": 1 / 10,
                    "t5": 2 / 10,
                    "gospel": 4 / 10,
                    "soul": 1 / 5,
                },
                "attributes": {},
                "related_entities": {
                    "t1": 1 / 8,
                    "t2": 0,
                    "t3": 0,
                    "t4": 1 / 4,
                    "t5": 0,
                    "gospel": 1 / 2,
                    "soul": 1 / 8,
                },
                "catch_all": {
                    "t1": 2 / 14,
                    "t2": 2 / 14,
                    "t3": 1 / 14,
                    "t4": 0,
                    "t5": 2 / 14,
                    "gospel": 3 / 14,
                    "soul": 4 / 14,
                },
            }

        def prob(self, field, term):
            return self._probs.get(field, {}).get(term, 0)

    return ToyCollectionLM()


def test_dict_based_entity(artists):
    """This test is worth 0.5 point"""
    artist_properties_1 = artists["Al Green"]
    entity_dict = module.dict_based_entity(artist_properties_1)

    assert entity_dict["http://xmlns.com/foaf/0.1/name"] == "Al Green"
    assert (
        "Al Green" in entity_dict["http://www.w3.org/2000/01/rdf-schema#label"]
    )
    assert sorted(entity_dict["http://dbpedia.org/property/occupation"]) == [
        "",
        "Singer",
        "pastor",
        "record producer",
        "songwriter",
    ]

    artist_properties_2 = artists["Deadmau5"]
    entity_dict_2 = module.dict_based_entity(artist_properties_2)

    assert entity_dict_2["http://xmlns.com/foaf/0.1/name"] == "Deadmau5"
    assert (
        "데드마우스" in entity_dict_2["http://www.w3.org/2000/01/rdf-schema#label"]
    )
    assert sorted(entity_dict_2["http://dbpedia.org/ontology/alias"]) == [
        "Halcyon441",
        "Karma K",
        "Testpilot",
    ]


def test_fielded_doc_entity(artists):
    """This test is worth 0.5 point"""
    artist_properties_1 = artists["Al Green"]
    artist_fielded_1 = module.fielded_doc_entity(artist_properties_1)
    assert "アル・グリーン" in artist_fielded_1["names"]
    assert "Composer109947232" in artist_fielded_1["types"]
    assert "MCA Records artists" in artist_fielded_1["types"]
    assert (
        "one of the most gifted purveyors of soul music"
        in artist_fielded_1["attributes"]
    )
    assert "Quiet Elegance" in artist_fielded_1["related_entities"]

    artist_properties_2 = artists["Deadmau5"]
    artist_fielded_2 = module.fielded_doc_entity(artist_properties_2)
    assert "Joel Thomas Zimmerman" in artist_fielded_2["names"]
    assert "Canadian house musicians" in artist_fielded_2["types"]
    assert (
        "a Canadian electronic music producer, DJ, and musician"
        in artist_fielded_2["attributes"]
    )
    assert "Mau5trap" in artist_fielded_2["related_entities"]


def test_term_frequency(es: Elasticsearch):
    """This test is worth 0.5 point"""
    tv_1 = es.termvectors(index=INDEX_NAME, id="Al Green", fields="catch_all")
    assert (
        tv_1["term_vectors"]["catch_all"]["terms"]["1946"]["term_freq"] == 34
    )

    tv_2 = es.termvectors(
        index=INDEX_NAME, id="Runhild Gammelsæter", fields="attributes"
    )
    assert (
        tv_2["term_vectors"]["attributes"]["terms"]["khlyst"]["term_freq"] == 1
    )

    tv_3 = es.termvectors(
        index=INDEX_NAME, id="Music of Belarus", fields="description"
    )
    assert (
        tv_3["term_vectors"]["description"]["terms"]["musiqu"]["term_freq"]
        == 7
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "gospel soul",
            [
                "Oleta Adams",
                "Dani Wilde",
                "Kim Richardson",
                "Michelle Sweeney",
                "Liberty Silver",
            ],
        ),
        (
            "country punk",
            [
                "Lydia Loveless",
                "Rae Spoon",
                "GG Allin",
                "Cassadee Pope",
                "Jason Webley",
            ],
        ),
        (
            "fun award trick",
            [
                "Heidi Newfield",
                "Cyndi Lauper",
                "The New Pornographers",
                "Jennifer Day",
                "Jacob Hoggard",
            ],
        ),
    ],
)
def test_baseline_retrieval(
    query: str, expected: List[str], es: Elasticsearch
):
    """This test is worth 0 point"""
    results_1 = module.baseline_retrieval(es, INDEX_NAME, query, 5)
    assert results_1 == expected


def test_collection_probability(es: Elasticsearch):
    """This test is worth 1 point"""
    clm_1 = module.CollectionLM(es, module.analyze_query(es, "gospel soul"))
    assert clm_1.prob("types", "gospel") == pytest.approx(0.0003, abs=1e-4)
    assert clm_1.prob("types", "soul") == pytest.approx(0.00036, abs=1e-5)
    assert clm_1.prob("description", "gospel") == pytest.approx(
        0.00013, abs=1e-5
    )
    assert clm_1.prob("description", "soul") == pytest.approx(
        0.00016, abs=1e-4
    )

    clm_2 = module.CollectionLM(
        es, module.analyze_query(es, "fun time madness")
    )
    assert clm_2.prob("catch_all", "fun") == pytest.approx(0.00003, abs=1e-5)
    assert clm_2.prob("catch_all", "time") == pytest.approx(0.0006, abs=1e-4)
    assert clm_2.prob("catch_all", "madness") == pytest.approx(0.0, abs=1e-1)


def test_term_mapping_prob(clm):
    """This test is worth 1 point"""
    Pf_t_3_1 = module.get_term_mapping_probs(clm, "gospel")
    assert Pf_t_3_1["description"] == pytest.approx(0.32642, abs=1e-5)
    assert Pf_t_3_1["attributes"] == pytest.approx(0, abs=1e-5)
    assert Pf_t_3_1["related_entities"] == pytest.approx(0.40803, abs=1e-5)

    Pf_t_3_2 = module.get_term_mapping_probs(clm, "soul")
    assert Pf_t_3_2["names"] == pytest.approx(0.26679, abs=1e-5)
    assert Pf_t_3_2["types"] == pytest.approx(0, abs=1e-5)
    assert Pf_t_3_2["catch_all"] == pytest.approx(0.34302, abs=1e-5)


@pytest.mark.parametrize(
    "prms_query, expected",
    [
        (
            "winter snow",
            [
                "Snow (musician)",
                "Kurt Winter",
                "Jim Black",
                "Dave Pierce",
                "Ken Saydak",
            ],
        ),
        (
            "summer sun",
            [
                "Sun Nan",
                "Yingdi Sun",
                "Liu Huan",
                "A-Sun (singer)",
                "Sam Kim",
            ],
        ),
    ],
)
def test_prms(es: Elasticsearch, prms_query: str, expected: List[str]):
    """This test is worth 1 point"""
    prms_retrieval = module.prms_retrieval(es, prms_query)
    assert prms_retrieval[:5] == expected
