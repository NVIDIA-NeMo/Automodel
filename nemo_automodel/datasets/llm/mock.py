from datasets import Dataset, Features, Sequence, Value
import random

def make_vocab(vocab_size:int=100):
    """
    Build a trivial vocab; index 0=<pad>, 1=<eos>, rest = tok_i
    """
    vocab = {"<pad>": 0, "<eos>": 1}
    for i in range(2, vocab_size):
        vocab[f"tok_{i}"] = i
    return vocab

def gen_sentence_ids(vocab, mean_len:float, std_len:float, max_len:int):
    """ Sentence generator with Gaussian length control """
    words = list(vocab.values())[2:]     # exclude <pad>, <eos>
    L = max(1, min(max_len, int(random.gauss(mean_len, std_len))))
    return random.choices(words, k=L) + [vocab["<eos>"]]

def build_unpacked_dataset(
        *,
        num_sentences: int       = 10,
        mean_len: float          = 20.0,
        std_len: float           = 6.0,
        vocab_size: int          = 100,
        max_sentence_len: int    = 64,
        seed: int                = 0,
):
    """
    Build a dataset where each example is one sentence (variable length).
    Returns:
      - a HuggingFace Dataset with fields:
          input_ids:     Sequence(int64)
          attention_mask:Sequence(int8)
          labels:        Sequence(int64)
          position_ids:  Sequence(int64)
    """
    random.seed(seed)
    vocab = make_vocab(vocab_size)
    eos_id = vocab["<eos>"]

    examples = []
    for _ in range(num_sentences):
        sent = gen_sentence_ids(vocab, mean_len, std_len, max_sentence_len)
        # build position_ids just like flush_block would
        pos_ids = []
        pos = 0
        for tid in sent:
            pos_ids.append(pos)
            pos = 0 if tid == eos_id else pos + 1

        examples.append({
            "input_ids":      sent,
            "attention_mask": [1] * len(sent),
            "labels":         sent.copy(),
            "position_ids":   pos_ids,
        })

    features = Features({
        "input_ids":      Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int8")),
        "labels":         Sequence(Value("int64")),
        "position_ids":   Sequence(Value("int64")),
    })
    ds = Dataset.from_list(examples, features=features)
    return ds

if __name__ == "__main__":
    ds = build_unpacked_dataset(
        num_sentences   = 5,
        mean_len        = 12.0,
        std_len         = 3.0,
        vocab_size      = 50,
        max_sentence_len= 20,
    )
    print(ds)
    # Show lengths of each field for the first example
    print({k: len(v) for k, v in ds[0].items()})
    print("ds[0]:", ds[0])
