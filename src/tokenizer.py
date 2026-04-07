from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass
class TokenizerConfig:
    max_length: int = 32
    min_freq: int = 1


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int], config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self.vocab = vocab
        self.inv_vocab = {idx: token for token, idx in vocab.items()}

        self.pad_id = vocab[PAD_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]
        self.bos_id = vocab[BOS_TOKEN]
        self.eos_id = vocab[EOS_TOKEN]

    @classmethod
    def build(cls, texts: Iterable[str], config: TokenizerConfig | None = None) -> "SimpleTokenizer":
        config = config or TokenizerConfig()
        counts: dict[str, int] = {}

        for text in texts:
            for token in cls.basic_tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        vocab = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }

        for token in sorted(counts):
            if counts[token] >= config.min_freq and token not in vocab:
                vocab[token] = len(vocab)

        return cls(vocab=vocab, config=config)

    @staticmethod
    def basic_tokenize(text: str) -> List[str]:
        return text.lower().strip().split()

    def encode(self, text: str) -> List[int]:
        tokens = [self.bos_id]
        for token in self.basic_tokenize(text):
            tokens.append(self.vocab.get(token, self.unk_id))
        tokens.append(self.eos_id)

        if len(tokens) > self.config.max_length:
            tokens = tokens[: self.config.max_length]
            tokens[-1] = self.eos_id

        return tokens

    def batch_encode(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def pad(self, token_ids: Sequence[int]) -> List[int]:
        padded = list(token_ids[: self.config.max_length])
        if len(padded) < self.config.max_length:
            padded.extend([self.pad_id] * (self.config.max_length - len(padded)))
        return padded

    def batch_pad(self, batch_tokens: Sequence[Sequence[int]]) -> List[List[int]]:
        return [self.pad(tokens) for tokens in batch_tokens]

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.inv_vocab.get(int(token_id), UNK_TOKEN)
            if token in {PAD_TOKEN, BOS_TOKEN}:
                continue
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
