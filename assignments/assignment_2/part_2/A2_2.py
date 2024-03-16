import abc
from collections import Counter
from collections import UserDict as DictClass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import math

CollectionType = Dict[str, Dict[str, List[Tuple[str, int]]]]


class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_field_length(self, field: str) -> int:
        """Total number of terms in a field for all documents."""
        return sum(len(fields[field]) for fields in self.values())

    def avg_field_length(self, field: str) -> float:
        """Average number of terms in a field across all documents."""
        return self.total_field_length(field) / len(self)

    def get_field_documents(self, field: str) -> Dict[str, List[str]]:
        """Dictionary of documents for a single field."""
        return {
            doc_id: doc[field] for (doc_id, doc) in self.items() if field in doc
        }


class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        
        self.index: Dict[str, Dict[str, Dict[str, int]]] = {}
        for field, terms in index.items():
            self.index[field] = {}
            for term, postings in terms.items():
                self.index[field][term] = {}
                for posting in postings:
                    doc_id, freq = posting
                    self.index[field][term][doc_id] = freq

        if not (field or fields):
            raise ValueError("Either field or fields have to be defined.")

        self.field = field
        self.fields = fields

        # Score accumulator for the query that is currently being scored.
        self.scores = defaultdict(float)

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in query_term_freqs.items():
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError


class SimpleScorer(Scorer):
    def score_term(self, term: str, query_freq: int) -> None:
        for field_terms in self.index.values():
            for doc_id, freq in field_terms.get(term, {}).items():
                self.scores[doc_id] += freq * query_freq
            
class ScorerBM25(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index, field)
        self.b = b
        self.k1 = k1

        self.doc_lengths = {}
        for doc_id, doc in self.collection.items():
            self.doc_lengths[doc_id] = sum(len(field) for field in doc.values())

        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def idf(self, term: str) -> float:
        N = len(self.collection)
        if N == 0:
            return 0
        n_t = sum(1 for field_terms in self.index.values() for _ in field_terms.get(term, {}))
        if n_t == 0:
            return 0
        return math.log(N / n_t)

    def score_term(self, term: str, query_freq: int) -> None:
        idf = self.idf(term)
        for field_terms in self.index.values():
            for doc_id, freq in field_terms.get(term, {}).items():
                doc_length = self.doc_lengths.get(doc_id, 0)
                self.scores[doc_id] += idf * (
                    freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                )


class ScorerLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        smoothing_param: float = 0.1,
    ):
        super(ScorerLM, self).__init__(collection, index, field)
        self.smoothing_param = smoothing_param

        assert self.field is not None
        assert self.field == field

        self.doc_lengths = {}
        for doc_id, doc in self.collection.items():
            self.doc_lengths[doc_id] = len(doc.get(self.field, []))

        absolute_term_frequencies = {}
        for term, postings in self.index.get(self.field, {}).items():
            absolute_term_frequencies[term] = sum(freq for freq in postings.values())
        total_length = sum(absolute_term_frequencies.values())
        self.relative_term_frequencies = {term: freq / total_length for term, freq in absolute_term_frequencies.items()}

    def score_term(self, term: str, query_freq: int) -> None:
        assert self.field is not None
        for doc_id in self.collection:
            freq = self.index.get(self.field, {}).get(term, {}).get(doc_id, 0)
            doc_length = self.doc_lengths.get(doc_id, 0)
            self.scores[doc_id] += query_freq * math.log(
                (1 - self.smoothing_param) * freq / doc_length + self.smoothing_param * self.relative_term_frequencies.get(term, 0)
            )

            
class ScorerBM25F(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        bi: List[float] = [0.75, 0.75],
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25F, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.bi = bi
        self.k1 = k1

        assert self.fields is not None
        assert self.fields == fields

        self.doc_field_lengths = {} # field: doc_id: length
        for field in fields:
            self.doc_field_lengths[field] = {}
            for doc_id, doc in self.collection.items():
                self.doc_field_lengths[field][doc_id] = len(doc.get(field, []))
        self.avg_doc_field_lengths = {}
        for field in fields:
            self.avg_doc_field_lengths[field] = sum(self.doc_field_lengths[field].values()) / len(self.doc_field_lengths[field])
    

    def B_i(self, doc_id: str, field_id: int, field: str):
        return 1 - self.bi[field_id] + self.bi[field_id] * self.doc_field_lengths[field][doc_id] / self.avg_doc_field_lengths[field]

    def c_tilde(self, doc_id: str, term: str):
        val = 0
        assert self.fields is not None
        for field_id, field in enumerate(self.fields):
            val += self.field_weights[field_id] * self.index.get(field, {}).get(term, {}).get(doc_id, 0) / self.B_i(doc_id, field_id, field)
        return val

    def idf(self, term: str) -> float:
        N = len(self.collection)
        if N == 0:
            return 0
        # only on "body" field
        assert "body" in self.index
        n_t = sum(1 for _ in self.index.get("body", {}).get(term, {}))
        if n_t == 0:
            return 0
        return math.log(N / n_t)

    def score_term(self, term: str, query_freq: int) -> None:
        assert self.fields is not None
        idf = self.idf(term)
        for doc_id in self.collection:
            c_tilde = self.c_tilde(doc_id, term)
            self.scores[doc_id] += idf * c_tilde / (c_tilde + self.k1)


class ScorerMLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        smoothing_param: float = 0.1,
    ):
        super(ScorerMLM, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.smoothing_param = smoothing_param

        assert self.fields is not None
        assert self.fields == fields

        # length of document d in field i (|d_i|)
        doc_field_lengths = {} # field: doc_id: length
        for field in fields:
            doc_field_lengths[field] = {}
            for doc_id, doc in self.collection.items():
                doc_field_lengths[field][doc_id] = len(doc.get(field, []))

        # term's relative frequency in the document field (P(t | d_i) = f_{t,d_i} / |d_i|)
        relative_term_document_field_frequencies = {} # field: term: doc_id: freq
        for field in fields:
            relative_term_document_field_frequencies[field] = {}
            for term, postings in self.index.get(field, {}).items():
                relative_term_document_field_frequencies[field][term] = {doc_id: 0.0 for doc_id in self.collection}
                for doc_id, freq in postings.items():
                    relative_term_document_field_frequencies[field][term][doc_id] = freq / doc_field_lengths[field][doc_id]

        # term's relative frequency in that field across the entire collection (P(t | C_i) = (sum_d' f_{t,d'_i}) / (sum_d' |d'_i|))
        relative_term_collection_field_frequencies = {} # field: term: freq
        for field in fields:
            relative_term_collection_field_frequencies[field] = {}
            for term, postings in self.index.get(field, {}).items():
                relative_term_collection_field_frequencies[field][term] = sum(postings.values()) / sum(doc_field_lengths[field].values())

        # P(t | Omega_d_i) = (1 - lambda_i) * P(t | d_i) + lambda_i * P(t | C_i)
        self.term_probabilities = {} # field: term: doc_id: prob
        for field in fields:
            self.term_probabilities[field] = {}
            for term in self.index.get(field, {}):
                self.term_probabilities[field][term] = {}
                for doc_id in self.collection:
                    self.term_probabilities[field][term][doc_id] = (
                        (1 - self.smoothing_param) * relative_term_document_field_frequencies[field][term][doc_id]
                        + self.smoothing_param * relative_term_collection_field_frequencies[field][term]
                    )

    def score_term(self, term: str, query_freq: float) -> None:
        assert self.fields is not None
        for doc_id in self.collection:
            tmp = 0.0
            for field_id, field in enumerate(self.fields):
                tmp += self.field_weights[field_id] * self.term_probabilities.get(field, {}).get(term, {}).get(doc_id, 0.0)
            self.scores[doc_id] += query_freq * math.log(tmp)

