# Assignment A4: Neural IR

## Task

You will implement the bottom-up version of WordPiece algorithm.  

## Specific steps

For this assignment, **only packages that are part of the standard Anaconda distribution are allowed**.
If you do not follow these restrictions, you may potentially lose all marks for the assignment.

## Assignment scoring

Complete each function and method according to instructions. There is a test for each coding section. Make sure that your code passes all the tests.
Passing the tests that you can see will mean you should get points for the assignment.

## WordPiece algorithm

The WordPiece algorithm is a subword tokenization algorithm. It has two versions: top-down (see E10-1) and bottom-up. In this assignment, you will implement the bottom-up version of the algorithm.

The algorithm is as follows [[1]](#1):

1. Initialize the vocabulary with all the characters in the training set.
2. Build a language model on the training set.
3. Generate a new subword token by merging two subword tokens that maximize the likelihood of the training set.
4. Repeat from step 2 until the vocabulary size reaches the maximum size or the maximum number of iterations is reached.

### Vocabulary initialization

You need to implement the function `initialize_vocabulary`. This function takes a list of words and returns a vocabulary. The vocabulary should be initialized with all the characters in the training set. Note that a prefix `##` should be added to tokens that are within a word. For example, for the word `hello`, the initial vocabulary should contain the following tokens: `["h", "##e", "##l", "##l", "##o"]`.

### Training

You need to implement the function `train`. This function takes a list of words and a vocabulary and returns a vocabulary. The function should build a language model on the training set and generate a new subword token by merging two subword tokens that maximize the likelihood of the training set. The function should repeat the process until the vocabulary size reaches the maximum size or the maximum number of iterations is reached.
The function `get_new_subword_token` should be used to find the new subword token. The algorithm computes a score for each pair of subword tokens and returns the pair with the highest score. The score is computed as follows:

$$score(a,b) = { c_{ab} \over c_a * c_b} $$

where $c_x$ represents the frequency of the subword token $x$.

## References

<a id="1">[1]</a> Schuster, Mike, and Kaisuke Nakajima. "Japanese and korean voice search." 2012 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2012.
