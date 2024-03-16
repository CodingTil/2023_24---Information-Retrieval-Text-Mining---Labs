from typing import Dict, List, Optional, Set, Tuple

PREFIX = "##"
UNKNOWN = "<unk>"


def initialize_vocabulary(word_corpus: List[str]) -> Set[str]:
    """Initializes the vocabulary with characters present in the corpus.

    Args:
        word_corpus: Corpus of words.

    Returns:
        Initial vocabulary.
    """
    vocabulary: Set[str] = set()
    for word in word_corpus:
        first_char, *rest = word
        vocabulary.add(first_char)
        for char in rest:
            vocabulary.add(f"{PREFIX}{char}")
    return vocabulary


def tokenize(word: str, vocabulary: Set[str]) -> List[str]:
    """Tokenizes a word using the vocabulary.

    The tokenizer splits the word using the longest possible tokens in the
    vocabulary. For example, if the word is "surfing", and the vocabulary
    contains the tokens "sur", "surf", and "ing", then the tokenizer will
    return ["surf", "##ing"].
    Returns <unk> token if the word cannot be fully tokenized.

    Args:
        word: Word to tokenize.
        vocabulary: Vocabulary.

    Returns:
        List of tokens.
    """
    word = word.lower()
    tokens = []
    while word.lstrip(PREFIX):
        for i in range(len(word)):
            if word[: len(word) - i] in vocabulary:
                tokens.append(word[: len(word) - i])
                word = PREFIX + word[len(word) - i :]
                break
        else:
            return [UNKNOWN]
    return tokens


def score(
    pair_freq: int, subword_token1_freq: int, subword_token2_freq: int
) -> float:
    """Calculates the score for merging two subword tokens.

    Args:
        pair_freq: Frequency of the pair.
        subword_token1_freq: Frequency of the first subword token.
        subword_token2_freq: Frequency of the second subword token.

    Returns:
        Score.
    """
    return pair_freq / (subword_token1_freq * subword_token2_freq)


def get_new_subword_token(
    data: List[Tuple[List[str], int]], vocabulary: Set[str]
) -> Tuple[str, float]:
    """Finds the new subword token to add to the vocabulary.

    The new subword token is the pair of tokens that maximizes the score. In
    case of ties, the pair that appears first in the vocabulary is chosen.

    Args:
        data: List of tokenized words and their frequencies.
        vocabulary: Vocabulary.

    Returns:
        New subword token and its score.
    """
    single_freq_dict: Dict[str, int] = {}
    pair_freq_dict: Dict[Tuple[str, str], int] = {}

    for word, freq in data:
        if UNKNOWN in word:
            continue
        for i in range(len(word) - 1):
            token_1, token_2 = word[i], word[i + 1]
            single_freq_dict[token_1] = single_freq_dict.get(token_1, 0) + freq
            pair_freq_dict[(token_1, token_2)] = pair_freq_dict.get(
                (token_1, token_2), 0
            ) + freq
        if len(word) > 1:
            single_freq_dict[word[-1]] = single_freq_dict.get(word[-1], 0) + freq
    
    scores_dict: Dict[Tuple[str, str], float] = {}
    for (token_1, token_2), pair_freq in pair_freq_dict.items():
        scores_dict[(token_1, token_2)] = score(
            pair_freq,
            single_freq_dict[token_1],
            single_freq_dict[token_2],
        )
    
    if len(scores_dict) == 0:
        return UNKNOWN, 0.0

    sorted_scores: List[Tuple[Tuple[str, str], float]] = sorted(
        scores_dict.items(), key=lambda x: x[1], reverse=True
    )
    # remove ones without max score
    max_score = sorted_scores[0][1]
    sorted_scores = [x for x in sorted_scores if x[1] == max_score]
    # Sort based on if the pair is in the vocabulary (if yes, make first)
    sorted_scores = sorted(
        sorted_scores,
        key=lambda x: 0 if x[0] in vocabulary else 1,
    )
    (token_1, token_2), score_value = sorted_scores[0]
    return f"{token_1}{token_2.replace(PREFIX, '', 1)}", score_value

def train(
    word_corpus: List[Tuple[str, int]],
    vocabulary: Set[str],
    num_iterations: Optional[int] = 4,
    max_vocab_size: Optional[int] = None,
) -> Set[str]:
    """Executes the WordPiece training algorithm.

    The algorithm iteratively merges subword tokens to create new ones. It stops
    when the number of iterations is reached or when the vocabulary reaches
    the maximum size.

    Args:
        word_corpus: Corpus of words and their frequencies.
        vocabulary: Vocabulary.
        num_iterations: Number of iterations to train the vocabulary. Defaults
            to 4.
        max_vocab_size: Maximum size of the vocabulary. Defaults to None.

    Returns:
        Vocabulary.
    """
    data: List[Tuple[List[str], int]] = tokenize_corpus(word_corpus, vocabulary)
    iteration: int = 0
    while True:
        new_subword_token, score = get_new_subword_token(data, vocabulary)
        if new_subword_token == UNKNOWN and score == 0.0:
            break
        vocabulary.add(new_subword_token)
        data = tokenize_corpus(word_corpus, vocabulary)
        iteration += 1
        if iteration == num_iterations:
            break
        if max_vocab_size and len(vocabulary) == max_vocab_size:
            break
    return vocabulary


def tokenize_corpus(
    corpus: List[Tuple[str, int]], vocabulary: Set[str]
) -> List[Tuple[List[str], int]]:
    """Tokenizes the corpus using the vocabulary.

    Args:
        corpus: Corpus of words and their frequencies.
        vocabulary: Vocabulary.

    Returns:
        List of tokenized words and their frequencies.
    """
    result: List[Tuple[List[str], int]] = []
    for word, freq in corpus:
        result.append((tokenize(word, vocabulary), freq))
    return result

