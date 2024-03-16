from typing import Callable, Dict, List, Set

import ir_datasets


def load_rankings(
    filename: str = "system_rankings.tsv",
) -> Dict[str, List[str]]:
    """Load rankings from file. Every row in the file contains query ID and
    document ID separated by a tab ("\t").

        query_id    doc_id
        646	        4496d63c-8cf5-11e3-833c-33098f9e5267
        646	        ee82230c-f130-11e1-adc6-87dfa8eff430
        646	        ac6f0e3c-1e3c-11e3-94a2-6c66b668ea55

    Example return structure:

    {
        query_id_1: [doc_id_1, doc_id_2, ...],
        query_id_2: [doc_id_1, doc_id_2, ...]
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and list of documents as values.
    """
    filename = f"data/{filename}"
    with open(filename, "r") as f:
        # Skip header
        f.readline()
        
        result: Dict[str, List[str]] = {}
        for line in f:
            query_id, doc_id = line.strip().split("\t")
            result.setdefault(query_id, []).append(doc_id)
        return result
    

def load_ground_truth(
    collection: str = "wapo/v2/trec-core-2018",
) -> Dict[str, Set[str]]:
    """Load ground truth from ir_datasets. Qrel is a namedtuple class with
    following properties:

        query_id: str
        doc_id: str
        relevance: int
        iteration: str

    relevance is split into levels with values:

        0	not relevant
        1	relevant
        2	highly relevant

    This function considers documents to be relevant for relevance values
        1 and 2.

    Generic structure of returned dictionary:

    {
        query_id_1: {doc_id_1, doc_id_3, ...},
        query_id_2: {doc_id_1, doc_id_5, ...}
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and sets of documents as values.
    """
    dataset = ir_datasets.load(collection)
    result: Dict[str, Set[str]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            result.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    return result


def get_precision(
    system_ranking: List[str], ground_truth: Set[str], k: int = 100
) -> float:
    """Computes Precision@k.

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
        k: Cutoff. Only consider system rankings up to k.

    Returns:
        P@K (float).
    """
    k_prime = min(k, len(system_ranking))
    considered_ranking = set(system_ranking[:k_prime])
    return len(considered_ranking.intersection(ground_truth)) / k_prime


def get_average_precision(system_ranking: List[str], ground_truth: Set[str]) -> float:
    """Computes Average Precision (AP).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        AP (float).
    """
    number_of_relevant_documents = len(ground_truth)
    if number_of_relevant_documents == 0:
        return 0

    sum_of_precisions = 0
    number_of_relevant_documents_found = 0
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            number_of_relevant_documents_found += 1
            sum_of_precisions += (
                number_of_relevant_documents_found / (i + 1)
            )
    return sum_of_precisions / number_of_relevant_documents


def get_reciprocal_rank(system_ranking: List[str], ground_truth: Set[str]) -> float:
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
    return 1 / (len(system_ranking) + 1)


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of
            document IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean
            is computed over.

    Returns:
        Mean evaluation measure (float).
    """
    sum_of_eval_measures = 0
    for query_id, system_ranking in system_rankings.items():
        ground_truth = ground_truths[query_id]
        sum_of_eval_measures += eval_function(system_ranking, ground_truth)
    return sum_of_eval_measures / len(system_rankings)


if __name__ == "__main__":
    system_rankings = load_rankings()
    ground_truths = load_ground_truth()

