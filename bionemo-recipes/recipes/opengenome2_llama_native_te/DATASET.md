# Dataset and Tokenization for OpenGenome2

This document describes how data is loaded, tokenized, and shuffled for the OpenGenome2 Llama
recipe: data sources, sharding, tokenization (windowed vs pre-chunked), the streaming buffer
shuffle, why we reshuffled and resharded the original OpenGenome2 data, and how configs map to
these choices.

## Data sources

Two modes are supported:

- **Streaming** — HuggingFace `load_dataset(..., streaming=True)` with `data_files` pointing at
  JSONL (e.g. `data_metagenomics_train_*.jsonl.gz`). Each file is a shard; data is read
  sequentially per shard.
- **Pre-chunked Parquet** — A directory of Parquet shards (e.g. produced by a reshard script).
  Each row is a fixed-length sequence (e.g. 8190 characters → 8192 tokens with BOS/EOS). No
  windowing is applied at tokenization time.

## Sharding

- **Streaming** — When `world_size <= num_shards`, we use `dataset.shard(num_shards=world_size, index=rank)` so each rank gets a disjoint subset of files. When `world_size > num_shards`, we use
  `split_dataset_by_node` so that data is split across nodes without duplicating full shards. With
  few files (e.g. 80) and many ranks (e.g. 48), each rank sees only 1–2 files, which can create
  strong temporal locality within a rank.
- **Pre-chunked** — The same HuggingFace streaming API is used over many Parquet shards. Using
  more shards (e.g. on the order of 10× world_size) gives each rank a larger mix of files and
  better batch diversity.

Implementation: [dataset.py](dataset.py) in `create_tokenized_dataset`.

## Tokenization

- **Windowed** (`stride` not `None`) — Long sequences are chunked with the tokenizer’s
  `return_overflowing_tokens=True`, `max_length=max_seq_length`, and `stride=stride` (overlap in
  tokens). Default `stride=200`. One sequence can produce many tokenized windows.
- **Pre-chunked** (`stride` `None`) — No windowing; each row is tokenized once with BOS/EOS
  added and no truncation. Used when the dataset is already pre-windowed (e.g. each Parquet row is
  one window).

Reference: [dataset.py](dataset.py) `create_tokenized_dataset`.

## Streaming buffer shuffle

After tokenization, the stream is shuffled with
`tokenized_dataset.shuffle(seed=42, buffer_size=buffer_size)`. This is a reservoir-style shuffle:
the buffer holds up to `buffer_size` tokenized windows; each new item can replace a randomly
chosen element in the buffer. The shuffle seed can be updated per epoch (e.g. via the dataloader’s
`set_epoch`) so that different epochs see different orderings. Ordering is randomized only within a
sliding window of size `buffer_size`, not globally.

![Streaming buffer and THD flow](assets/dataset_streaming_buffer.png)

*Placeholder: add your figure as `assets/dataset_streaming_buffer.png` (files → shard → tokenize → buffer shuffle → THD pack).*

## Why we reshuffled and resharded

The original OpenGenome2 JSONL data had relatively few files (e.g. 80) and was ordered (e.g. by
sequence length or similarity). With many ranks (e.g. 48), each rank saw only 1–2 files, so:

- Batches had strong temporal locality (consecutive samples from the same file/source).
- Validation loss could be worse than training loss (overfitting to local order).
- Matching the Megatron/ShardedEden baseline was difficult until data order and precision were
  fixed.

We addressed this by:

1. **Global shuffle** — Shuffling all sequences (or windows) before writing to disk.
2. **More shards** — Writing to many Parquet shards (e.g. 480+) so that each rank streams from
   many files and sees a representative mix.
3. **Pre-chunked path** — Using the pre-chunked Parquet path so that each row is one window and
   tokenization is a simple BOS+text+EOS pass.

![Original file order vs reshuffled shards](assets/dataset_original_vs_reshuffled.png)

*Placeholder: add your figure as `assets/dataset_original_vs_reshuffled.png`.*

## Scripts

Resharding and inspection scripts may live in the broader repo (e.g. under
`bionemo-recipes/recipes/llama3_native_te/scripts/` or similar):

- **reshard_jsonl_to_parquet** — Converts JSONL (or HuggingFace cache Arrow files) to globally
  shuffled Parquet shards. Run before using the pre-chunked config.
- **check_shard_stats** — Prints per-shard and aggregate stats: sequence counts, window counts,
  total characters, THD micro-steps, buffer coverage, etc.

If your deployment uses a different path for these scripts, point your preprocessing pipeline
there.

## Config mapping

| Config                          | Data source         | Tokenization        | stride          | buffer_size |
| ------------------------------- | ------------------- | ------------------- | --------------- | ----------- |
| `og2_7b_thd_gqa`                | Streaming JSONL     | Windowed            | 200             | 50_000      |
| `og2_7b_thd_gqa_global_shuffle` | Pre-chunked Parquet | Direct (no windows) | 200 (no effect) | 10_000      |

For `og2_7b_thd_gqa`, `load_dataset_kwargs` typically points at `path: "json"` and
`data_files: ".../data_metagenomics_train_*.jsonl.gz"`. For `og2_7b_thd_gqa_global_shuffle`,
`load_dataset_kwargs.path` points at the directory of Parquet shards (e.g.
`/data/opengenome2/parquet_split`).
