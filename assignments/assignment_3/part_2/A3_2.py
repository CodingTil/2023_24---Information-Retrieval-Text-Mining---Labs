import abc
import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import Elasticsearch

_DEFAULT_FIELD = "body"


class Entity:
    def __init__(self, doc_id: str, stats: Dict[str, Dict[str, Any]]):
        """Representation of an entity.

        Args:
          doc_id: Document id
          stats: Term vector stats from elasticsearch. Keys are field and
            values are term and field statistics.
        """
        self.doc_id = doc_id
        self._stats = stats
        self._terms = {}

    def term_stats(self, term: str, field: str = _DEFAULT_FIELD) -> Dict[str, Any]:
        """Term statistics including term frequency and total term frequency."""
        return self._stats[field]["terms"].get(term)

    def field_stats(self, field: str = _DEFAULT_FIELD):
        """Field statistics including sum of total term frequency."""
        return self._stats[field]["field"]

    def terms(self, field: str = _DEFAULT_FIELD) -> List[str]:
        """Reconstructed document field from indexed positional information."""
        if field in self._terms:
            return self._terms[field]

        pos = {
            token["position"]: term
            for term, tinfo in self._stats[field]["terms"].items()
            for token in tinfo["tokens"]
        }
        self._terms[field] = [None] * (max(pos.keys()) + 1)
        for p, term in pos.items():
            self._terms[field][p] = term
        return self._terms[field]

    def length(self, field: str = _DEFAULT_FIELD) -> int:
        """Length of the document field."""
        return sum(term["term_freq"] for term in self._stats[field]["terms"].values())


class ElasticsearchCollection:
    def __init__(self, index_name):
        """Interface to an Elasticsearch index.

        Args:
          index_name: Name of the index to use.
        """
        self._index_name = index_name
        self.es = Elasticsearch()

    def baseline_retrieval(
        self, query: str, k: int = 100, field: Optional[str] = None
    ) -> List[str]:
        """Performs baseline retrieval on index.

        Args:
          query: A string of text, space separated terms.
          k: Number of documents to return.
          field: If specified, match only on the specified field.

        Returns:
          A list of entity IDs as strings, up to k of them, in descending
          order of scores.
        """
        res = self.es.search(
            index=self._index_name,
            q=query if not field else None,
            query={"match": {field: query}} if field else None,
            size=k,
        )
        return [x["_id"] for x in res["hits"]["hits"]]

    def get_query_terms(self, text: str) -> List[str]:
        """Analyzes text with the same pipeline that was used for indexing
        documents. It returns None in place of a term if it was removed (e.g.,
        using stopword removal).

        Args:
          text: Text to analyze.

        Returns:
          List of terms.
        """
        tokens = self.es.indices.analyze(index=self._index_name, body={"text": text})[
            "tokens"
        ]
        query_terms = [None] * (
            max(tokens, key=lambda x: x["position"])["position"] + 1
        )
        for token in tokens:
            query_terms[token["position"]] = token["token"]
        return query_terms

    def get_document(self, doc_id: str) -> Entity:
        """Generates entity representation given document id."""
        tv = self.es.termvectors(
            index=self._index_name,
            id=doc_id,
            term_statistics=True,
        )["term_vectors"]

        return Entity(
            doc_id,
            stats={
                field: {
                    "terms": tv[field]["terms"],
                    "field": tv[field]["field_statistics"],
                }
                for field in tv
            },
        )

    def index(self, collection: Dict[str, Any], settings: Dict[str, Any]):
        if self.es.indices.exists(index=self._index_name):
            self.es.indices.delete(index=self._index_name)
        self.es.indices.create(index=self._index_name, mappings=settings)
        for doc_id, doc in collection.items():
            self.es.index(document=doc, id=doc_id, index=self._index_name)
        time.sleep(10)


class Scorer(abc.ABC):
    collection: ElasticsearchCollection # for LSP
    window: int # for LSP

    def __init__(
        self,
        collection: ElasticsearchCollection,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
    ):
        """Interface for the scorer class.

        Args:
          collection: Collection of documents. Needed to calculate document
            statistical information.
          feature_weights: Weights associated with each feature function
          mu: Smoothing parameter
          window: Window for unordered feature function.
        """
        if not sum(feature_weights) == 1:
            raise ValueError("Feature weights should sum to 1.")

        self.collection = collection
        self.feature_weights = feature_weights
        self.mu = mu
        self.window = window

    def score_collection(self, query: str, k: int = 100):
        """Scores all documents in the collection using document-at-a-time query
        processing.

        Args:
          query: Sequence (list) of query terms.
          k: Number of documents to return

        Returns:
          Dict with doc_ids as keys and retrieval scores as values. (It may be
          assumed that documents that are not present in this dict have a
          retrival score of 0.)
        """
        # TODO
        documents: List[Entity] = [self.collection.get_document(doc_id) for doc_id in self.collection.baseline_retrieval(query, k)]
        query_terms: List[str] = self.collection.get_query_terms(query)

        lT, lO, lU = self.feature_weights
        return {
            doc.doc_id: (
                lT * self.unigram_matches(query_terms, doc)
                + lO * self.ordered_bigram_matches(query_terms, doc, documents)
                + lU * self.unordered_bigram_matches(query_terms, doc, documents)
            )
            for doc in documents
        }

    def _get_term_ttf(self, term, field=_DEFAULT_FIELD):
        """Returns total term frequency for term."""
        hits = self.collection.baseline_retrieval(term, k=1, field=field)
        if len(hits) == 0:
            return 0

        doc = self.collection.get_document(hits[0])
        return doc.term_stats(term, field)["ttf"]

    @abc.abstractmethod
    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for ordered bigram matches for document
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for unordered bigram matches for document
        """
        raise NotImplementedError


class SDMScorer(Scorer):
    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        total = 0
        l_e = doc.length()
        sum_ttf = doc.field_stats()["sum_ttf"]

        for term, freq in Counter(query_terms).items():
            term_stats = doc.term_stats(term)
            if term_stats:
                cqe = term_stats["term_freq"]
                ttf = term_stats["ttf"]
            else:
                cqe = 0
                ttf = self._get_term_ttf(term)

            if ttf == 0:
                continue

            total += freq * math.log((cqe + self.mu * ttf / sum_ttf) / (l_e + self.mu))
        return total

    def ordered_bigram_matches(
        self,
        query_terms: List[str],
        doc: Entity,
        documents: List[Entity],
    ) -> float:
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for ordered bigram matches for document
        """
        total: float = 0.0
        l_e: int = doc.length()
        sum_ttf: int = doc.field_stats()["sum_ttf"]

        all_entities: List[Entity] = documents
        if doc not in all_entities:
            all_entities.append(doc)

        # Count the frequency of each bigram in each document
        for (q1, q2), freq in Counter(zip(query_terms[:-1], query_terms[1:])).items():
            freqs: Dict[str, int] = {
                entity.doc_id: 0
                for entity in all_entities
            }

            for entity in all_entities:
                terms: List[str] = entity.terms()
                for i in range(len(terms) - 1):
                    if (terms[i], terms[i+1]) == (q1, q2):
                        freqs[entity.doc_id] += 1

            c_q1_q2_e: int = freqs[doc.doc_id]
            P_q1_q2: float = sum(
                freqs[entity.doc_id]
                for entity in all_entities
            ) / sum_ttf
            if c_q1_q2_e == 0 and P_q1_q2 == 0:
                continue
            total += freq * math.log((c_q1_q2_e + self.mu * P_q1_q2) / (l_e + self.mu))

        return total
            

    def unordered_bigram_matches(
        self,
        query_terms: List[str],
        doc: Entity,
        documents: List[Entity],
    ) -> float:
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for unordered bigram matches for document
        """
        total: float = 0.0
        l_e: int = doc.length()
        sum_ttf: int = doc.field_stats()["sum_ttf"]

        all_entities: List[Entity] = documents
        if doc not in all_entities:
            all_entities.append(doc)

        # Count the frequency of each bigram in each document
        for (q1, q2), freq in Counter(zip(query_terms[:-1], query_terms[1:])).items():
            freqs: Dict[str, int] = {
                entity.doc_id: 0
                for entity in all_entities
            }

            for entity in all_entities:
                terms: List[str] = entity.terms()
                for i in range(len(terms) - 1):
                    if q1 == terms[i]:
                        freqs[entity.doc_id] += sum(1 for term in terms[i+1:min(len(terms), i+self.window)] if term == q2)
                    elif q2 == terms[i]:
                        freqs[entity.doc_id] += sum(1 for term in terms[i+1:min(len(terms), i+self.window)] if term == q1)

            c_q1_q2_e: int = freqs[doc.doc_id]
            P_q1_q2: float = sum(
                freqs[entity.doc_id]
                for entity in all_entities
            ) / sum_ttf
            if c_q1_q2_e == 0 and P_q1_q2 == 0:
                continue
            total += freq * math.log((c_q1_q2_e + self.mu * P_q1_q2) / (l_e + self.mu))

        return total
            

class FSDMScorer(Scorer):
    fields: List[str] # for LSP
    field_weights: List[float] # for LSP

    def __init__(
        self,
        collection: ElasticsearchCollection,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
        fields: List[str] = ["title", "body", "anchors"],
        field_weights: List[float] = [0.2, 0.7, 0.1],
    ):
        """Fielded version of an SDM scorer.

        Args:
          collection: Collection of documents. Needed to calculate document
            statistical information.
          feature_weights: Weights associated with each feature function
          mu: Smoothing parameter
          window: Window for unordered feature function.
          fields: A list of fields to use for the calculating the score
          field_weights: A list of weights to use for each field.
        """
        super().__init__(collection, feature_weights, mu, window)
        assert len(fields) == len(field_weights)
        self.fields = fields
        self.field_weights = field_weights

    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        total = 0

        for term, freq in Counter(query_terms).items():
            inner_total = 0
            for field, weight in zip(self.fields, self.field_weights):
                l_e = doc.length(field)
                sum_ttf = doc.field_stats(field)["sum_ttf"]

                term_stats = doc.term_stats(term, field)
                if term_stats:
                    cqe = term_stats["term_freq"]
                    ttf = term_stats["ttf"]
                else:
                    cqe = 0
                    ttf = self._get_term_ttf(term, field)

                if ttf == 0:
                    continue

                PtTheta = (cqe + self.mu * ttf / sum_ttf) / (l_e + self.mu)
                inner_total += weight * PtTheta
            total += freq * math.log(inner_total) if inner_total else 0
        return total

    def ordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for ordered bigram matches for document
        """
        total: float = 0.0

        all_entities: List[Entity] = documents
        if doc not in all_entities:
            all_entities.append(doc)

        # Count the frequency of each bigram in each document
        for (q1, q2), freq in Counter(zip(query_terms[:-1], query_terms[1:])).items():
            inner_total: float = 0.0

            for field, weight in zip(self.fields, self.field_weights):
                l_e: int = doc.length(field)
                sum_ttf: int = doc.field_stats(field)["sum_ttf"]

                freqs: Dict[str, int] = {
                    entity.doc_id: 0
                    for entity in all_entities
                }

                for entity in all_entities:
                    terms: List[str] = entity.terms(field=field)
                    for i in range(len(terms) - 1):
                        if (terms[i], terms[i+1]) == (q1, q2):
                            freqs[entity.doc_id] += 1

                c_q1_q2_e: int = freqs[doc.doc_id]
                P_q1_q2: float = sum(
                    freqs[entity.doc_id]
                    for entity in all_entities
                ) / sum_ttf
                if c_q1_q2_e == 0 and P_q1_q2 == 0:
                    continue
                inner_total += weight * (c_q1_q2_e + self.mu * P_q1_q2) / (l_e + self.mu)
            
            total += freq * math.log(inner_total) if inner_total != 0 else 0

        return total
            

    def unordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for unordered bigram matches for document
        """
        total: float = 0.0

        all_entities: List[Entity] = documents
        if doc not in all_entities:
            all_entities.append(doc)

        # Count the frequency of each bigram in each document
        for (q1, q2), freq in Counter(zip(query_terms[:-1], query_terms[1:])).items():
            inner_total: float = 0.0

            for field, weight in zip(self.fields, self.field_weights):
                l_e: int = doc.length(field)
                sum_ttf: int = doc.field_stats(field)["sum_ttf"]

                freqs: Dict[str, int] = {
                    entity.doc_id: 0
                    for entity in all_entities
                }

                for entity in all_entities:
                    terms: List[str] = entity.terms(field=field)
                    for i in range(len(terms) - 1):
                        if q1 == terms[i]:
                            freqs[entity.doc_id] += sum(1 for term in terms[i+1:min(len(terms), i+self.window)] if term == q2)
                        elif q2 == terms[i]:
                            freqs[entity.doc_id] += sum(1 for term in terms[i+1:min(len(terms), i+self.window)] if term == q1)

                c_q1_q2_e: int = freqs[doc.doc_id]
                P_q1_q2: float = sum(
                    freqs[entity.doc_id]
                    for entity in all_entities
                ) / sum_ttf
                if c_q1_q2_e == 0 and P_q1_q2 == 0:
                    continue
                inner_total += weight * (c_q1_q2_e + self.mu * P_q1_q2) / (l_e + self.mu)

            total += freq * math.log(inner_total) if inner_total != 0 else 0

        return total
