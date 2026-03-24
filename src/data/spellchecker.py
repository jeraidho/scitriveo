import re
from collections import Counter
import Levenshtein


class SpellChecker:
    def __init__(self, word_frequencies: dict, max_distance: int = 2, min_word_length: int = 4):
        """
        Initialize the spell checker with word frequencies.

        Parameters:
        - word_frequencies: vocabulary frequency dictionary
        - max_distance: maximum allowed Levenshtein distance for correction
        - min_word_length: do not correct very short words
        """
        self.word_frequencies = {word.lower(): freq for word, freq in word_frequencies.items()}
        self.max_distance = max_distance
        self.min_word_length = min_word_length

    def _get_candidates(self, word):
        """
        Find candidate corrections based on Levenshtein distance.
        Returns only candidates within max_distance.
        """
        candidates = []

        for vocab_word, freq in self.word_frequencies.items():
            # Small optimization: skip words with very different length
            if abs(len(vocab_word) - len(word)) > self.max_distance:
                continue

            distance = Levenshtein.distance(word, vocab_word)
            if distance <= self.max_distance:
                candidates.append((vocab_word, distance, freq))

        # Sort by:
        # 1. smaller edit distance
        # 2. higher frequency
        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[:3]

    def correct(self, word: str) -> str:
        """
        Correct a single word if it is misspelled.
        """
        if not isinstance(word, str) or not word:
            return word

        word = word.lower()

        # Do not touch very short tokens
        if len(word) < self.min_word_length:
            return word

        # Do not touch tokens with digits or non-alpha content
        if not word.isalpha():
            return word

        # Word is already known
        if word in self.word_frequencies:
            return word

        candidates = self._get_candidates(word)
        if not candidates:
            return word

        return candidates[0][0]

    def correct_text(self, text: str) -> str:
        """
        Correct all words in a text while preserving separators.
        """
        if not isinstance(text, str) or not text:
            return ""

        parts = re.split(r'([^a-zA-Z]+)', text)
        corrected_parts = []

        for part in parts:
            if not part:
                continue

            if re.fullmatch(r'[a-zA-Z]+', part):
                corrected_parts.append(self.correct(part))
            else:
                corrected_parts.append(part)

        return ''.join(corrected_parts)


def build_word_frequencies(texts):
    """
    Build a frequency dictionary from a list of texts.
    """
    word_freq = Counter()

    for text in texts:
        if not isinstance(text, str):
            continue
        words = re.findall(r'\w+', text.lower())
        word_freq.update(words)

    return word_freq


if __name__ == "__main__":
    texts = [
        "Hello world",
        "How are you",
        "Scientific research in machine learning",
        "Natural language processing",
        "Neural networks for information retrieval"
    ]

    word_frequencies = build_word_frequencies(texts)
    spellchecker = SpellChecker(word_frequencies)

    print(spellchecker.correct("helllo"))      # hello
    print(spellchecker.correct("scientfic"))   # scientific
    print(spellchecker.correct("nlp"))         # nlp