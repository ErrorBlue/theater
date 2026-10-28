from typing import Protocol


class Tokenizer(Protocol):
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...
