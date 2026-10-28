from .base import Tokenizer


class ByteTokenizer(Tokenizer):
    pad_id = 256
    bos_id = 257
    eos_id = 258
    eot_id = 259
    vocab_size = 260

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_ids = [i for i in ids if 0 <= i < 256]
        return bytes(byte_ids).decode("utf-8", errors="replace")
