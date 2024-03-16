import json
from collections import defaultdict
import math
from typing import Callable, Dict, List, Set, Tuple, Optional

import numpy as np
from elasticsearch import Elasticsearch

FIELDS = ["title", "body"]

INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
            "body": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
        }
    }
}

FEATURES_QUERY = [
    "query_length",
    "query_sum_idf",
    "query_max_idf",
    "query_avg_idf",
]
FEATURES_DOC = ["doc_length_title", "doc_length_body"]
FEATURES_QUERY_DOC = [
    "unique_query_terms_in_title",
    "sum_TF_title",
    "max_TF_title",
    "avg_TF_title",
    "unique_query_terms_in_body",
    "sum_TF_body",
    "max_TF_body",
    "avg_TF_body",
]


def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = "toy_index"
) -> List[str]:
    """Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    """
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the term.
        hits = (
            es.search(
                index=index,
                query={"match": {field: t["token"]}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms


def get_doc_term_freqs(
    es: Elasticsearch, doc_id: str, field: str, index: str = "toy_index"
) -> Optional[Dict[str, int]]:
    """Gets the term frequencies of a field of an indexed document.

    Args:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed.
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    """
    tv = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    )
    if tv["_id"] != doc_id:
        return None
    if field not in tv["term_vectors"]:
        return None
    term_freqs = {}
    for term, term_stat in tv["term_vectors"][field]["terms"].items():
        term_freqs[term] = term_stat["term_freq"]
    return term_freqs


def extract_query_features(
    query_terms: List[str], es: Elasticsearch, index: str = "toy_index"
) -> Dict[str, float]:
    """Extracts features of a query.

    Args:
        query_terms: List of analyzed query terms.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
    Returns:
        Dictionary with keys 'query_length', 'query_sum_idf',
            'query_max_idf', and 'query_avg_idf'.
    """
    total_docs = es.count(index=index)["count"]

    query_features = {
        "query_length": float(len(query_terms)),
        "query_sum_idf": 0.0,
        "query_max_idf": 0.0,
        "query_avg_idf": 0.0
    }

    for term in query_terms:
        hits = es.search(
            index=index,
            body={"query": {"match": {"body": term}}, "_source": False, "size": 1}
        ).get("hits", {}).get("hits", [])
        if not hits:
            continue
        doc_id = hits[0]["_id"]
        term_stats = es.termvectors(
            index=index,
            id=doc_id,
            body={"fields": ["body"], "offsets": False, "positions": False, "term_statistics": True}
        )
        doc_freq = term_stats.get("term_vectors", {}).get("body", {}).get("terms", {}).get(term, {}).get("doc_freq", 0)
        idf = math.log(total_docs / doc_freq) if doc_freq else 0
        query_features["query_sum_idf"] += idf
        query_features["query_max_idf"] = max(query_features["query_max_idf"], idf)

    query_features["query_avg_idf"] = query_features["query_sum_idf"] / len(query_terms) if query_terms else 0

    return query_features


def extract_doc_features(
    doc_id: str, es: Elasticsearch, index: str = "toy_index"
) -> Dict[str, float]:
    """Extracts features of a document.

    Args:
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        Dictionary with keys 'doc_length_title', 'doc_length_body'.
    """
    doc_features = {
        "doc_length_title": 0.0,
        "doc_length_body": 0.0
    }

    # Get term frequencies of title and body fields.
    tv = es.termvectors(
        index=index,
        id=doc_id,
        fields=["title", "body"],
        term_statistics=True,
    )

    # Return default features if document ID does not match.
    if tv["_id"] != doc_id:
        return doc_features

    # Calculate document length for title field.
    title_terms = tv.get("term_vectors", {}).get("title", {}).get("terms", {})
    for term_stat in title_terms.values():
        doc_features["doc_length_title"] += term_stat["term_freq"]

    # Calculate document length for body field.
    body_terms = tv.get("term_vectors", {}).get("body", {}).get("terms", {})
    for term_stat in body_terms.values():
        doc_features["doc_length_body"] += term_stat["term_freq"]

    return doc_features


def extract_query_doc_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = "toy_index",
) -> Dict[str, float]:
    """Extracts features of a query and document pair.

    Args:
        query_terms: List of analyzed query terms.
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        Dictionary with keys 'unique_query_terms_in_title',
            'unique_query_terms_in_body', 'sum_TF_title', 'sum_TF_body',
            'max_TF_title', 'max_TF_body', 'avg_TF_title', 'avg_TF_body'.
    """
    query_doc_features = {
        "unique_query_terms_in_title": 0.0,
        "unique_query_terms_in_body": 0.0,
        "sum_TF_title": 0.0,
        "sum_TF_body": 0.0,
        "max_TF_title": 0.0,
        "max_TF_body": 0.0,
        "avg_TF_title": 0.0,
        "avg_TF_body": 0.0
    }

    # Get term frequencies of title and body fields.
    tv = es.termvectors(
        index=index,
        id=doc_id,
        fields=["title", "body"],
        term_statistics=True,
    )
    if tv["_id"] != doc_id:
        return query_doc_features

    for term in query_terms:
        # For title field
        if "title" in tv["term_vectors"] and term in tv["term_vectors"]["title"]["terms"]:
            term_stat = tv["term_vectors"]["title"]["terms"][term]
            query_doc_features["unique_query_terms_in_title"] += 1
            query_doc_features["sum_TF_title"] += term_stat["term_freq"]
            query_doc_features["max_TF_title"] = max(
                query_doc_features["max_TF_title"], term_stat["term_freq"]
            )

        # For body field
        if "body" in tv["term_vectors"] and term in tv["term_vectors"]["body"]["terms"]:
            term_stat = tv["term_vectors"]["body"]["terms"][term]
            query_doc_features["unique_query_terms_in_body"] += 1
            query_doc_features["sum_TF_body"] += term_stat["term_freq"]
            query_doc_features["max_TF_body"] = max(
                query_doc_features["max_TF_body"], term_stat["term_freq"]
            )

    query_doc_features["avg_TF_title"] = (
        query_doc_features["sum_TF_title"] / len(query_terms) if query_terms else 0
    )
    query_doc_features["avg_TF_body"] = (
        query_doc_features["sum_TF_body"] / len(query_terms) if query_terms else 0
    )

    return query_doc_features

def extract_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = "toy_index",
) -> List[float]:
    """Extracts query features, document features and query-document features
    of a query and document pair.

    Args:
        query_terms: List of analyzed query terms.
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        List of extracted feature values in a fixed order.
    """
    query_features = extract_query_features(query_terms, es, index=index)
    feature_vect = [query_features[f] for f in FEATURES_QUERY]

    doc_features = extract_doc_features(doc_id, es, index=index)
    feature_vect.extend([doc_features[f] for f in FEATURES_DOC])

    query_doc_features = extract_query_doc_features(
        query_terms, doc_id, es, index=index
    )
    feature_vect.extend([query_doc_features[f] for f in FEATURES_QUERY_DOC])

    return feature_vect


def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:
    """Indexes documents from JSONL file."""
    bulk_data = []
    with open(filepath, "r") as docs:
        for doc in docs:
            doc = json.loads(doc)
            bulk_data.append(
                {"index": {"_index": index, "_id": doc.pop("doc_id")}}
            )
            bulk_data.append(doc)
    es.bulk(index=index, body=bulk_data, refresh=True)


def reset_index(es: Elasticsearch, index: str) -> None:
    """Reset Index"""
    if es.indices.exists(index=index):
        es.indices.delete(index=index)

    es.indices.create(index=index, body=INDEX_SETTINGS)


def load_queries(filepath: str) -> Dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.

    This is an example query:

    ```
    <top>
    <num> Number: OHSU1
    <title> 60 year old menopausal woman without hormone replacement therapy
    <desc> Description:
    Are there adverse effects on lipids when progesterone is given with estrogen replacement therapy
    </top>

    ```

    Take as query ID the value (on the same line) after `<num> Number: `,
    and take as the query string the rest of the line after `<title> `. Omit
    newline characters.

    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.

    Returns:
        A dictionary with query IDs and corresponding query strings.
    """
    queries = {}

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("<num> Number:"):
                splitline = line.split(" ", 2)
                query_id = splitline[-1].rstrip()
            if line.startswith("<title>"):
                splitline = line.split(" ", 1)
                queries[query_id] = splitline[-1].rstrip()
    return queries


def load_qrels(filepath: str) -> Dict[str, List[str]]:
    """Loads query relevance judgments from a file.
    The qrels file has content with tab-separated values such as the following:

    ```
    MSH1	87056458
    MSH1	87056800
    MSH1	87058606
    MSH2	87049102
    MSH2	87056792
    ```

    Args:
        filepath: String (constructed using os.path) of the filepath to a
            file with queries.

    Returns:
        A dictionary with query IDs and a corresponding list of document IDs
            for documents judged relevant to the query.
    """
    qrels = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            splitline = line.split("\t")
            query_id = splitline[0]
            doc_id = splitline[1].rstrip()
            qrels[query_id].append(doc_id)
    return qrels


def prepare_ltr_training_data(
    query_ids: List[str],
    all_queries: Dict[str, str],
    all_qrels: Dict[str, List[str]],
    es: Elasticsearch,
    index: str,
) -> Tuple[List[List[float]], List[int]]:
    """Prepares feature vectors and labels for query and document pairs found
    in the training data.

        Args:
            query_ids: List of query IDs.
            all_queries: Dictionary containing all queries.
            all_qrels: Dictionary with keys as query ID and values as list of
                relevant documents.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            X: List of feature vectors extracted for each pair of query and
                retrieved or relevant document.
            y: List of corresponding labels.
    """
    X = []
    y = []

    for i, query_id in enumerate(query_ids):
        print(
            "Processing query {}/{} ID {}".format(
                i + 1, len(query_ids), query_id
            )
        )
        query = all_queries[query_id]
        query_terms = analyze_query(es, query, "body", index=index)
        if len(query_terms) == 0:
            continue

        # Add documents and relevance labels from ground truth.
        qrels = set(all_qrels[query_id])

        # Generate features for documents in first-pass retrieval.
        hits = es.search(
            index=index, q=" ".join(query_terms), size=100, _source=False
        )["hits"]["hits"]
        all_docs = qrels.union(hit["_id"] for hit in hits)
        for doc_id in all_docs:
            feature_vector = extract_features(
                query_terms, doc_id, es, index=index
            )
            X.append(feature_vector)
            y.append(1 if doc_id in qrels else 0)

    return X, y


class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor."""
        # scikit learn regressor
        from sklearn.linear_model import LogisticRegression
        # SVR
        from sklearn.svm import SVR
        # MLP Regressor
        from sklearn.neural_network import MLPRegressor
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(100, 125, 100), max_iter=100000, random_state=17
        )

    def _train(self, X: List[List[float]], y: List[float]) -> None:
        """Trains an LTR model.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(
        self, ft: List[List[float]], doc_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def get_rankings(
    ltr: PointWiseLTRModel,
    query_ids: List[str],
    all_queries: Dict[str, str],
    es: Elasticsearch,
    index: str,
    rerank: bool = False,
) -> Dict[str, List[str]]:
    """Generate rankings for each of the test query IDs.

    Args:
        ltr: A trained PointWiseLTRModel instance.
        query_ids: List of query IDs.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    """

    test_rankings = {}
    for i, query_id in enumerate(query_ids):
        print(
            "Processing query {}/{} ID {}".format(
                i + 1, len(query_ids), query_id
            )
        )
        # First-pass retrieval
        query_terms = analyze_query(
            es, all_queries[query_id], "body", index=index
        )
        if len(query_terms) == 0:
            print(
                "WARNING: query {} is empty after analysis; ignoring".format(
                    query_id
                )
            )
            continue
        hits = es.search(
            index=index, q=" ".join(query_terms), _source=True, size=100
        )["hits"]["hits"]
        test_rankings[query_id] = [hit["_id"] for hit in hits]

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            feature_vectors = []
            for doc_id in test_rankings[query_id]:
                feature_vectors.append(
                    extract_features(query_terms, doc_id, es, index=index)
                )
            doc_ids = test_rankings[query_id]
            prediction: List[Tuple[str, float]] = ltr.rank(
                feature_vectors, doc_ids
            )
            # Sort by relevance score in descending order.
            prediction.sort(key=lambda x: x[1], reverse=True)
            test_rankings[query_id] = [doc_id for doc_id, _ in prediction]
    return test_rankings


def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: List[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of document
            IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean is
            computed over.

    Returns:
        Mean evaluation measure (float).
    """
    sum_score = 0
    for query_id, system_ranking in system_rankings.items():
        sum_score += eval_function(system_ranking, ground_truths[query_id])
    return sum_score / len(system_rankings)


if __name__ == "__main__":
    index_name = "trec9_index"
    es = Elasticsearch(timeout=120)

    reset_index(es, index_name)
    index_documents("data/documents.jsonl", es, index_name)
