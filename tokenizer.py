from collections import Counter
import torch

class Tokenizer:
    def __init__(self, texts, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.build_vocab(texts)

    def build_vocab(self, texts):
        words = [word for text in texts for word in text.split()]
        word_counts = Counter(words)
        for word, _ in word_counts.most_common(self.vocab_size - 2):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word

    def encode(self, text):
        return [self.word_to_index.get(word, 1) for word in text.split()]

    def decode(self, indices):
        return " ".join([self.index_to_word.get(idx, "<UNK>") for idx in indices])