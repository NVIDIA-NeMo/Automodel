# Tokenization: A Beginner's Guide

Large language models do not read text the way humans do. Before a model can process any input, the text must be converted into a sequence of numbers called **tokens**. This conversion process is called **tokenization**, and the component that performs it is a **tokenizer**.

This guide explains what tokenization is, how different tokenizers work, what special tokens are, and how these concepts connect to chat templates and fine-tuning.

---

## From Text to Numbers

A model's vocabulary is a fixed table that maps every known piece of text to an integer ID. The tokenizer looks up each piece in this table to produce the sequence of IDs the model actually receives.

```
         Tokenizer
  Text ──────────────> Token IDs ──────────────> Model

  "Hello, world!"      [15496, 11, 995, 0]       ┌─────────┐
                                                   │   LLM   │
                                                   └─────────┘
```

The reverse operation — converting token IDs back to text — is called **decoding**.

---

## What Is a Token?

A token is a unit of text that the model treats as a single element. Depending on the tokenizer, a token might be:

- a whole word (`"hello"`)
- a subword (`"to"` + `"ken"` + `"ize"`)
- a single character (`"a"`)
- a special marker (`<|end_of_text|>`)

Most modern tokenizers split text into **subwords** — pieces that are shorter than full words but longer than individual characters. This lets the model handle rare or unseen words by composing them from familiar parts.

```
  Input:    "unhappiness"

  Word-level token:     [ unhappiness ]           ← 1 token (must be in vocabulary)

  Subword tokens:       [ un | happi | ness ]     ← 3 tokens (composed from common parts)

  Character tokens:     [ u | n | h | a | p | ... ]  ← 11 tokens (always works, but slow)
```

Subword tokenization hits the sweet spot: common words stay as single tokens (efficient), while rare words are split into recognizable pieces (flexible).

---

## Types of Tokenizers

There are several algorithms for deciding how to split text into subwords. You do not need to choose one — the model you use ships with its tokenizer already trained — but understanding the landscape helps when reading documentation.

| Algorithm | Used by | How it works |
|-----------|---------|-------------|
| **BPE** (Byte Pair Encoding) | GPT, Llama, Qwen | Starts with characters, repeatedly merges the most frequent pair into a new token |
| **WordPiece** | BERT | Similar to BPE, but picks merges that maximize the likelihood of the training data |
| **SentencePiece** | T5, Gemma | Operates on raw bytes (no pre-tokenization step), supports both BPE and Unigram modes |
| **Unigram** | XLNet, ALBERT | Starts with a large vocabulary and prunes tokens that contribute least |

### BPE in Action

BPE is the most common algorithm for modern LLMs. Here is a simplified view of how it builds a vocabulary:

```
  Training corpus:   "low lower lowest"

  Step 0 — start with characters:
    l o w   l o w e r   l o w e s t

  Step 1 — most frequent pair is (l, o), merge into "lo":
    lo w   lo w e r   lo w e s t

  Step 2 — most frequent pair is (lo, w), merge into "low":
    low   low e r   low e s t

  Step 3 — most frequent pair is (e, r), merge into "er":
    low   low er   low e s t

  ... and so on until the desired vocabulary size is reached.
```

At inference time, the tokenizer applies these learned merge rules to split any new text into tokens from its vocabulary.

---

## Special Tokens

Beyond ordinary text tokens, every tokenizer defines a set of **special tokens** — markers with a reserved meaning that the model was trained to recognize. They are never part of the user's actual text; instead, they provide structural signals to the model.

Common special tokens:

| Token | Purpose |
|-------|---------|
| `<\|begin_of_text\|>` | Marks the start of the input (beginning of sequence) |
| `<\|end_of_text\|>` | Marks the end of the input (end of sequence) |
| `<\|pad\|>` | Fills unused positions when batching sequences of different lengths |
| `<\|start_header_id\|>` | Opens a role header (Llama-3 style) |
| `<\|end_header_id\|>` | Closes a role header (Llama-3 style) |
| `<\|im_start\|>` | Opens a message turn (ChatML / Qwen style) |
| `<\|im_end\|>` | Closes a message turn (ChatML / Qwen style) |

Special tokens are critical for fine-tuning: if the training data does not include the same special tokens the model expects, the model will behave unpredictably at inference time.

### How Special Tokens Appear in Practice

Here is what a single-turn conversation looks like after tokenization with Llama-3 style tokens:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ <|begin_of_text|>                                                   │  ← start of sequence
  │ <|start_header_id|> user <|end_header_id|>                          │  ← role header
  │                                                                     │
  │ What is the capital of France?                                      │  ← user content
  │ <|start_header_id|> assistant <|end_header_id|>                     │  ← role header
  │                                                                     │
  │ The capital of France is Paris.                                     │  ← assistant content
  │ <|end_of_text|>                                                     │  ← end of sequence
  └─────────────────────────────────────────────────────────────────────┘
```

And the same conversation with Qwen / ChatML style tokens:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ <|im_start|>user                                                    │  ← role + turn open
  │ What is the capital of France?<|im_end|>                            │  ← content + turn close
  │ <|im_start|>assistant                                               │  ← role + turn open
  │ The capital of France is Paris.<|im_end|>                           │  ← content + turn close
  └─────────────────────────────────────────────────────────────────────┘
```

Every model family defines its own set of special tokens. The tokenizer knows which ones to use.

---

## Chat Templates: Connecting It All

A [chat template](chat-template.md) is the bridge between a structured conversation (a list of messages with roles) and the flat token sequence the model consumes. It is a template string — most commonly written in [Jinja](https://jinja.palletsprojects.com/) — that inserts the correct special tokens around each message.

```
  Input (structured):                    Output (flat token sequence):

  [                                      <|im_start|>user
    {"role": "user",           ──>       What is 2+2?<|im_end|>
     "content": "What is 2+2?"},         <|im_start|>assistant
    {"role": "assistant",                4<|im_end|>
     "content": "4"}
  ]

         messages list          chat template          tokenized text
```

When you call `tokenizer.apply_chat_template(messages)`, the tokenizer:

1. Reads the template string (from `tokenizer.chat_template`).
2. Renders it with your messages, producing text with special tokens.
3. Tokenizes the rendered text into token IDs.

This is why matching the chat template during fine-tuning matters — the model learned to associate these special tokens with conversational structure during its original training.

---

## Think Tokens

Some models support **thinking** or **reasoning** before producing a visible response. In these models, the tokenizer includes additional special tokens that delimit an internal reasoning block:

```
  <|im_start|>assistant
  <think>
  The user is asking about the capital of France.
  I know it is Paris.
  </think>
  The capital of France is Paris.<|im_end|>
```

The tokens between `<think>` and `</think>` (the exact markers vary by model) represent the model's chain-of-thought reasoning. Depending on the deployment setup, this block may be:

- **Visible** to the developer for debugging but hidden from the end user.
- **Stripped** entirely before the response is returned.
- **Included** in the training data so the model learns to reason before answering.

Think tokens are ordinary special tokens from the tokenizer's perspective — they are part of the vocabulary and processed like any other token. Their special behavior comes from how the model was trained to use them, not from the tokenizer itself.

---

## Putting It All Together

Here is how the full pipeline flows from a user's message to model input:

```
  User's message          Chat template              Tokenizer             Model
  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌─────────┐
  │ "What is     │       │ Inserts      │       │ Splits text  │       │         │
  │  2+2?"       │──────>│ special      │──────>│ into token   │──────>│   LLM   │
  │              │       │ tokens       │       │ IDs          │       │         │
  └──────────────┘       └──────────────┘       └──────────────┘       └─────────┘

  "What is 2+2?"    "<|im_start|>user\n       [151644, 872, 198,      [  ...  ]
                     What is 2+2?              3838, 374, 220, 17,
                     <|im_end|>\n               10, 17, 30, 151645,
                     <|im_start|>              198, 151644, 77091]
                     assistant\n"
```

| Concept | What it means |
|---------|--------------|
| **Tokenize** | Convert text into token IDs that the model can process |
| **Tokenizer** | The component that performs tokenization (and decoding) |
| **Special tokens** | Reserved markers (e.g., `<\|im_start\|>`) that signal structure, not content |
| **Chat template** | A template that wraps messages with the correct special tokens for a given model |
| **Think tokens** | Special tokens that delimit an internal reasoning block in the model's output |

---

## Further Reading

- [Chat Template Configuration](chat-template.md) — how to enable and customize chat templates for fine-tuning in NeMo AutoModel.
- [Fine-Tuning Guide](finetune.md) — end-to-end guide for fine-tuning LLMs.
- [Integrate Your Own Text Dataset](dataset.md) — for datasets that do not use chat template formatting.
