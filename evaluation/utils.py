class Tokenizer13a:
    """Simple tokenizer approximating WMT13a tokenization using whitespace split."""
    def __call__(self, text):
        return text.split()
