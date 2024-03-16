from typing import Dict, List

def tokenize(doc: str) -> List[str]:
    """Tokenizes a document.

    Args:
        doc: Document content given as a string.

    Returns:
        List of tokens.
    """
    punctuation = [',', '.', ':', ';', '?', '!']
    punctuation = ''.join(punctuation)
    # Replace any punctuation with a space
    translator = str.maketrans(punctuation, ' '*len(punctuation))
    doc = doc.translate(translator)
    # Split the document into tokens
    tokens = doc.split()
    # Convert all tokens to lowercase
    tokens = [token.lower() for token in tokens]
    return tokens


def get_word_frequencies(doc: str) -> Dict[str, int]:
    """Extracts word frequencies from a document.

    Args:
        doc: Document content given as a string.

    Returns:
        Dictionary with words as keys and their frequencies as values.
    """
    # Get the tokens
    tokens = tokenize(doc)
    # Count the number of occurrences of each token
    word_frequencies = {}
    for token in tokens:
        if token in word_frequencies:
            word_frequencies[token] += 1
        else:
            word_frequencies[token] = 1
    return word_frequencies


def get_word_feature_vector(
    word_frequencies: Dict[str, int], vocabulary: List[str]
) -> List[int]:
    """Creates a feature vector for a document, comprising word frequencies
        over a vocabulary.

    Args:
        word_frequencies: Dictionary with words as keys and frequencies as
            values.
        vocabulary: List of words.

    Returns:
        List of length `len(vocabulary)` with respective frequencies as values.
    """
    # Initialize the feature vector
    feature_vector = [0] * len(vocabulary)
    # Iterate over the vocabulary
    for i, word in enumerate(vocabulary):
        # If the word is in the document, add its frequency to the feature vector
        if word in word_frequencies:
            feature_vector[i] = word_frequencies[word]
    return feature_vector
