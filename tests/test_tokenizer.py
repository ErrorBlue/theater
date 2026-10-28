import pytest

from theater.tokenizer import ByteTokenizer


@pytest.fixture
def tok():
    return ByteTokenizer()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "Hello, world!",
        "你好，世界",
        "emoji: 😀🚀",
        "a\nb\tc",
        "mixed 中文 and English 😀",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_without_special_tokens(tok):
    text = "hello 世界"
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.decode(ids) == text


def test_encode_adds_bos_eos(tok):
    ids = tok.encode("hi")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert ids[1:-1] == list(b"hi")


def test_encode_no_special_tokens(tok):
    assert tok.encode("hi", add_special_tokens=False) == list(b"hi")


def test_encode_empty(tok):
    assert tok.encode("") == [tok.bos_id, tok.eos_id]
    assert tok.encode("", add_special_tokens=False) == []


def test_byte_ids_in_range(tok):
    for i in tok.encode("abc 😀"):
        assert 0 <= i < 256 or i in {tok.pad_id, tok.bos_id, tok.eos_id, tok.eot_id}


def test_decode_ignores_special_tokens(tok):
    ids = [tok.bos_id, *b"ok", tok.eos_id, tok.pad_id, tok.eot_id]
    assert tok.decode(ids) == "ok"


def test_decode_invalid_utf8_is_replaced(tok):
    assert tok.decode([0xFF]) == "\ufffd"


def test_unknown_ids_are_ignored(tok):
    assert tok.decode([9999, *b"ok"]) == "ok"


def test_tokenizer_is_stateless():
    a, b = ByteTokenizer(), ByteTokenizer()
    assert a.encode("hi") == b.encode("hi")
