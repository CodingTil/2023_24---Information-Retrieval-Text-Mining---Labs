import json
import math
from operator import itemgetter
from typing import Any, Dict, List, Union
from collections.abc import Iterable

from elasticsearch import Elasticsearch

TYPE_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
NAME_PREDICATES = set(
    [
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://xmlns.com/foaf/0.1/name",
        "http://xmlns.com/foaf/0.1/givenName",
        "http://xmlns.com/foaf/0.1/surname",
    ]
)
TYPE_PREDICATES = set([TYPE_PREDICATE, "http://purl.org/dc/terms/subject"])
COMMENT_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#comment"


INDEX_NAME = "musicians"
INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "names": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "description": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "attributes": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "related_entities": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "types": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "catch_all": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
        }
    }
}


def has_type(properties: Dict[str, Any], target_type: str) -> bool:
    """Check whether properties contain specific type

    Args:
        properties: Dictionary with properties
        target_type: Type to check for.

    Returns:
        True if target type in properties.
    """
    if TYPE_PREDICATE not in properties:
        return False
    for p in properties[TYPE_PREDICATE]:
        if p["value"] == target_type:
            return True
    return False


def resolve_uri(uri: str) -> str:
    """Resolves uri."""
    uri = uri.split("/")[-1].replace("_", " ")
    if uri.startswith("Category:"):
        uri = uri[len("Category:") :]
    return uri


def dict_based_entity(entity_properties: Dict[str, Any]) -> Dict[str, Union[str, List]]:
    """Create a simpler dictionary-based entity representation.

    Args:
        entity_id: The ID of the entity.

    Returns:
        A dictionary-based entity representation.
    """
    new_dict = {}
    for key, value in entity_properties.items():
        # if value is an iterable (list, tuple, set, frozenset, generator, etc.)
        if isinstance(value, Iterable):
            if len(value) == 0:
                continue
            elif len(value) == 1:
                new_dict[key] = str(value[0]["value"])
            else:
                new_dict[key] = [str(x["value"]) for x in value]
        else:
            new_dict[key] = str(value)
    return new_dict


def fielded_doc_entity(entity_properties: Dict[str, Any]) -> Dict[str, Any]:
    """fielded document representation should include the fields `names`,
    `description`, `attributes`, `related_entities`, `types`, and `catch_all`.


    They should contain the following:
        * `names`: the objects of `NAME_PREDICATES`,
        * `description`: the object(s) of `COMMENT_PREDICATE`,
        * `attributes`: objects that are literal values,
        * `related_entities`: objects that are entities,
        * `types`: the objects of `TYPE_PREDICATES`, and
        * `catch_all`: all of the above.

    NB! All fields except `catch_all` are mutually exclusive.

    Args:
        entity_id: The ID of the entity.
        **kwargs: Additional keyword arguments. Notably, session to provide to
            get_dbpedia_entity() function

    Returns:
        Dictionary with the above stated keys.
    """
    new_dict = {}
    new_dict["names"] = []
    new_dict["description"] = []
    new_dict["attributes"] = []
    new_dict["related_entities"] = []
    new_dict["types"] = []
    new_dict["catch_all"] = []

    for key, value in entity_properties.items():
        if key in NAME_PREDICATES:
            if isinstance(value, Iterable):
                for v in value:
                    new_dict["names"].append(v["value"] if isinstance(v, dict) else v)
            else:
                new_dict["names"].append(value["value"] if isinstance(value, dict) else value)
        elif key == COMMENT_PREDICATE:
            if isinstance(value, Iterable):
                for v in value:
                    new_dict["description"].append(v["value"] if isinstance(v, dict) else v)
            else:
                new_dict["description"].append(value["value"] if isinstance(value, dict) else value)
        elif key in TYPE_PREDICATES:
            if isinstance(value, Iterable):
                for v in value:
                    new_dict["types"].append(resolve_uri(v["value"] if isinstance(v, dict) else v))
            else:
                new_dict["types"].append(resolve_uri(value["value"] if isinstance(value, dict) else value))
        elif isinstance(value, dict) and value.get["type"] == "literal":
            new_dict["attributes"].append(value["value"])
        elif isinstance(value, dict) and value.get["type"] == "uri":
            new_dict["related_entities"].append(resolve_uri(value["value"]))
        elif isinstance(value, Iterable):
            for v in value:
                if isinstance(v, dict) and v["type"] == "literal":
                    new_dict["attributes"].append(v["value"])
                elif isinstance(v, dict) and v["type"] == "uri":
                    new_dict["related_entities"].append(resolve_uri(v["value"]))

    new_dict["catch_all"] = (
        new_dict["names"]
        + new_dict["description"]
        + new_dict["attributes"]
        + new_dict["related_entities"]
        + new_dict["types"]
    )

    # Convert to string
    new_dict["names"] = [str(x) for x in new_dict["names"]]
    new_dict["description"] = [str(x) for x in new_dict["description"]]
    new_dict["attributes"] = [str(x) for x in new_dict["attributes"]]
    new_dict["related_entities"] = [str(x) for x in new_dict["related_entities"]]
    new_dict["types"] = [str(x) for x in new_dict["types"]]
    new_dict["catch_all"] = [str(x) for x in new_dict["catch_all"]]

    # Remove any empty strings
    new_dict["names"] = [x for x in new_dict["names"] if x]
    new_dict["description"] = [x for x in new_dict["description"] if x]
    new_dict["attributes"] = [x for x in new_dict["attributes"] if x]
    new_dict["related_entities"] = [x for x in new_dict["related_entities"] if x]
    new_dict["types"] = [x for x in new_dict["types"] if x]
    new_dict["catch_all"] = [x for x in new_dict["catch_all"] if x]

    # Join with , and space
    new_dict["names"] = ", ".join(new_dict["names"])
    new_dict["description"] = ", ".join(new_dict["description"])
    new_dict["attributes"] = ", ".join(new_dict["attributes"])
    new_dict["related_entities"] = ", ".join(new_dict["related_entities"])
    new_dict["types"] = ", ".join(new_dict["types"])
    new_dict["catch_all"] = ", ".join(new_dict["catch_all"])

    return new_dict


def bulk_index(es: Elasticsearch, artists: Dict[str, Any]) -> None:
    """Iterate over artists, and index those that are of the
    right type.

    Args:
        es: Elasticsearch instance.
        artists: Dictionary with artist names and their properties.
    """
    for artist_name, artist_properties in artists.items():
        entity = fielded_doc_entity(artist_properties)
        if "MusicalArtist" in entity["types"]:
            entity["id"] = artist_name
            es.index(index=INDEX_NAME, id=entity["id"], body=entity)


def baseline_retrieval(
    es: Elasticsearch, index_name: str, query: str, k: int = 100
) -> List[str]:
    """Performs baseline retrival on index.

    Args:
        es: Elasticsearch instance.
        index_name: A string of text.
        query: A string of text, space separated terms.
        k: An integer.

    Returns:
        A list of entity IDs as strings, up to k of them, in descending order of
            scores.
    """
    results = []
    res = es.search(index=index_name, query={"match": {"catch_all": query}})
    results = [(x["_score"], x["_id"]) for x in res["hits"]["hits"]]
    results = sorted(results, key=itemgetter(0, 1), reverse=True)
    if len(results) > k:
        results = results[:k]
    results = [x[1] for x in results]
    return results


def analyze_query(es: Elasticsearch, query: str) -> List[str]:
    """Analyzes query and returns a list of query terms that can be found in
    the collection.

    Args:
        es: Elasticsearch instance
        query: Query to analyze

    Returns:
        List of query terms.
    """
    tokens = es.indices.analyze(
        index=INDEX_NAME, body={"text": query, "analyzer": "english"}
    )["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        query_terms.append(t["token"])
    return query_terms


class CollectionLM:
    def __init__(
        self,
        es: Elasticsearch,
        qterms: List[str],
        fields: List[str] = None,
    ) -> None:
        """This class is used for obtaining collection language modeling
        probabilities $P(t|C_i)$.

        Args:
            es: Elasticsearch instance
            qterms: List of query terms
            fields: List of entity fields
        """
        self._es = es
        self._probs = {}
        self._fields = fields or [
            "names",
            "description",
            "attributes",
            "related_entities",
            "types",
            "catch_all",
        ]
        # computing P(t|C_i) for each field and for each query term
        for field in self._fields:
            self._probs[field] = {}
            for t in qterms:
                self._probs[field][t] = self._get_prob(field, t)

    @property
    def fields(self):
        return self._fields

    def _get_prob(self, field: str, term: str) -> float:
        """computes the collection Language Model probability of a term for a
        given field.

        Args:
            field: Fields for which to get the probability
            term: Term for which to get the probability

        Returns:
            Collection LM probability.
        """
        # use a boolean query to find a document that contains the term
        hits = (
            self._es.search(
                index=INDEX_NAME,
                query={"match": {field: term}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        # ask for global term statistics when requesting the term vector of
        # that doc (`term_statistics=True`)
        if doc_id is None:
            return 0
        tv = (
            self._es.termvectors(
                index=INDEX_NAME,
                id=doc_id,
                fields=field,
                term_statistics=True,
            )
            .get("term_vectors", {})
            .get(field, {})
        )
        # compute the collection LM probability of the term
        # $P(t|C_i) = \frac{f_{t,C_i}}{|C_i|}$
        sum_ttf = tv.get("field_statistics", {}).get("sum_ttf", 0)
        if sum_ttf == 0:
            return 0
        return tv.get("terms", {}).get(term, {}).get("ttf", 0) / sum_ttf

    def prob(self, field: str, term: str) -> float:
        """Return probability for a given field and term.

        Args:
            field: Fields for which to get the probability
            term: Term for which to get the probability

        Returns:
            Collection LM probability.
        """
        return self._probs.get(field, {}).get(term, 0)


def get_term_mapping_probs(clm: CollectionLM, term: str) -> Dict[str, float]:
    """PRMS: For a single term, find their mapping probabilities for all fields.

    Args:
        clm: Collection language model instance.
        term: A single-term string.

    Returns:
        Dictionary of mapping probabilities for the fields.
    """
    Pf_t = {}
    sum = 0.0
    for field in clm.fields:
        Pf_t[field] = clm.prob(field, term)
        sum += Pf_t[field]
    for field in clm.fields:
        Pf_t[field] /= sum
    return Pf_t


def score_prms(es, clm: CollectionLM, qterms: List[str], doc_id: str, mu: int = 100):
    """Score PRMS."""
    # Getting term frequency statistics for the given document field from
    # Elasticsearch
    # Note that global term statistics are not needed (`term_statistics=False`)
    tv = es.termvectors(
        index=INDEX_NAME, id=doc_id, fields=clm.fields, term_statistics=False
    ).get("term_vectors", {})

    # compute field lengths $|d_i|$
    len_d_i = []  # document field length
    for i, field in enumerate(clm.fields):
        if field in tv:
            len_d_i.append(sum([s["term_freq"] for _, s in tv[field]["terms"].items()]))
        else:  # that document field may be empty
            len_d_i.append(0)

    # scoring the query
    score = 0  # log P(q|d)
    for t in qterms:
        Pt_theta_d = 0  # P(t|\theta_d)
        # Get field mapping probs.
        Pf_t = get_term_mapping_probs(clm, t)
        for i, field in enumerate(clm.fields):
            if field in tv:
                ft_di = tv[field]["terms"].get(t, {}).get("term_freq", 0)  # $f_{t,d_i}$
            else:  # that document field is empty
                ft_di = 0
            Pt_Ci = clm.prob(field, t)  # $P(t|C_i)$
            Pt_theta_di = (ft_di + mu * Pt_Ci) / (
                mu + len_d_i[i]
            )  # $P(t|\theta_{d_i})$ with Dirichlet smoothing
            Pt_theta_d += Pf_t[field] * Pt_theta_di
        score += math.log(Pt_theta_d)

    return score


def prms_retrieval(es: Elasticsearch, query: str):
    """PRMS retrieval."""
    # Analyze query
    query_terms = analyze_query(es, query)

    # Perform initial retrieval using ES
    res = es.search(
        index=INDEX_NAME, q=query, df="catch_all", _source=False, size=200
    ).get("hits", {})

    # Instantiate collectionLM class
    clm = CollectionLM(es, query_terms)

    # Rerank results using PRMS
    scores = {}
    for doc in res.get("hits", {}):
        doc_id = doc.get("_id")
        scores[doc_id] = score_prms(es, clm, query_terms, doc_id)

    return [x[0] for x in sorted(scores.items(), key=itemgetter(1, 0), reverse=True)]


def reset_index(es: Elasticsearch) -> None:
    """Reset Index"""
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, mappings=INDEX_SETTINGS["mappings"])


def get_artists() -> Dict[str, Any]:
    """Loads artists from file."""
    with open("data/artists.json", "r") as f:
        return json.load(f)


def main():
    """Index artists"""
    es = Elasticsearch()
    es.info()

    artists = get_artists()

    reset_index(es)
    bulk_index(es, artists)


if __name__ == "__main__":
    main()
