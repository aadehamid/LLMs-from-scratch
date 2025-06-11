from pathlib import Path
from rich import print
from rich.traceback import install
from rich import print_json
from rich import pretty
from rich.panel import Panel
from rich.console import Console, Group
import logging
import logging.config as config
from yaml import safe_load
from typing import Dict, Tuple
import json

install()
console = Console()
pretty.install()
# ge the path to the data we will use to build the tokenizer
text_path: Path = Path(r"../ch02/01_main-chapter-code/the-verdict.txt")
logger_config_path: Path = Path(r"../logger_config.yaml")
logger = logging.getLogger("tokenizer")


def configure_logger(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = safe_load(f)
        logging.config.dictConfig(config)
        # print_json(json.dumps(config, indent=4))


def read_data(path: Path) -> str:
    logger.info("Reading data...")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    # console.print(text[:1090])
    logger.info("Finished reading data....")
    return text

def pair_freq(tokens: list) -> dict:
    counts: dict = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list, pair: tuple, idx: int) -> list:
    """Merge a pair of adjacent tokens into a single token given by idx.

    ids is a list of token ids.
    pair is a tuple of two token ids.
    idx is the id of the new merged token.

    Returns a new list of token ids where all instances of `pair` have been
    replaced with `idx`.
    """
    newids: list = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def decode(ids, merges, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text




def main():
    configure_logger(logger_config_path)
    text: str = read_data(text_path)
    # encode text into utf-8 to get the byte representation of the text
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens)) # convert raw bytes to integers
    # console.print(f"text_byte: {tokens}\ntokens_length: {len(tokens)}")
    print("-----------------")
    print(f"Original_text_length: {len([i for i in text])}")
    counts_stats = pair_freq(tokens)
    sorted_token = sorted(((v,k) for k, v in counts_stats.items()), reverse=True)
    print("-----------------")
    top_pair = max(counts_stats, key=counts_stats.get)
    print(Panel(f"Pair with the highest freq: [italic red] {top_pair} [/italic red]"))
    merged_tokens = merge(tokens, top_pair, 256)
    print(Panel(f"{merged_tokens}", title="Merged tokens"))

    # Nw let us run the functions to iterate over all of the text
    vocab_size = 276
    num_merges: int = vocab_size - 256
    ids = list(tokens) # get a copy of the list that is separate from the original list

    merges: Dict[Tuple[int, int], int] = {}
    for i in range(num_merges):
        stats = pair_freq(ids)
        max_pair = max(stats, key=stats.get)
        idx = 256 + i
        logger.info(f"merging pair {max_pair} into a new token {idx}.")
        ids = merge(ids, max_pair, idx)
        merges[max_pair] = idx

    # Compute the compression ratio
    print(Panel(f"Original_text_length: {len(tokens)}\nMerged_text_length: {len(ids)}\nCompression Ration: {len(tokens)/len(ids):.2f}x",
                title="Compression Ratio"))

    # decode the ids into a string
    vocab = {idx: bytes([idx]) for idx in range(256)}  # set the vocab variable.
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    print(Panel(f"Decoded Text: {decode(ids, merges, vocab)}", title="[red bold]Decoded Text[/red bold]"))






if __name__ == "__main__":
    main()

