import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import httpx
    from rich import print
    import re
    import tiktoken


@app.cell
def _():
    # url = ("https://raw.githubusercontent.com/aadehamid/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    # raw_text = httpx.get(url).text
    # text_length = len(raw_text)

    file_path = Path("/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/LLMs-from-scratch/ch02/01_main-chapter-code/the-verdict.txt")

    with open(file_path, "r") as f:
        raw_text = f.read()
    text_length = len(raw_text)

    # print(text_length)
    # print(raw_text[:99])
    raw_preprocessed = re.split(r'([(){}_?!:,.;:"\'}]|--|\s)', raw_text)
    raw_preprocessed = [item.strip() for item in raw_preprocessed if item.strip()]

    mo.md(
        f"""# Load the Text data
        Total number of characters in The Verdict: {text_length}\n
        The total number of tokens using regex is: {len(raw_preprocessed)}
    
        """
    )


    return (raw_preprocessed,)


@app.cell
def _():



    mo.md(f"""
    # Convert Tokens to ID
    - take a set of all tokensnto remove duplicate
    - sort the set in alphabetical order
    - assign interger position to each token in the sorted set
    """)
    return


@app.cell
def _(raw_preprocessed):
    len(raw_preprocessed)
    return


@app.cell
def _(raw_preprocessed):
    raw_preprocessed_sorted = sorted(set(raw_preprocessed) )
    mo.md(f"""
    Vocab length = {len(raw_preprocessed_sorted)}

    """)

    return (raw_preprocessed_sorted,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Build the Vocab
    """)
    return


@app.cell
def _(raw_preprocessed_sorted):
    vocab = {token:ID for ID, token in enumerate(raw_preprocessed_sorted)}

    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
    return (vocab,)


@app.class_definition
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.token_int = vocab
        self.int_token = {ID: token for token, ID in vocab.items()}

    def encode(self, text):
        preprocessed_text = re.split(r'([(){}_?!:,.;:"\'}]|--|\s)', text)
        preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
        tokenized_text = [self.token_int[token] for token in preprocessed_text]
        return tokenized_text

    def decode(self, ids):
        text = " ".join([self.int_token[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


@app.cell
def _(vocab):
    initialized_tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
    tokenized_text = initialized_tokenizer.encode(text)
    print(tokenized_text)

    return initialized_tokenizer, tokenized_text


@app.cell
def _(initialized_tokenizer, tokenized_text):
    decoded_text = initialized_tokenizer.decode(tokenized_text)
    print(decoded_text)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test Out of Vocabulary Word
    """)
    return


@app.cell
def _(initialized_tokenizer):
    oov_text = "Hello, do you like tea?"
    print(initialized_tokenizer.encode(oov_text))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We need to figure out a way to use a special token such as **<|unk|>** to represent unknown word. We also use **<|endoftext|>** to demarcate where one docmuments begin and the other starts if we are ingesting data from multiple documents.
    """)
    return


@app.cell
def _(raw_preprocessed):
    len(sorted(list(set(raw_preprocessed) )))
    return


@app.cell
def _(raw_preprocessed):
    raw_preprocessed_sorted2 = sorted(list(set(raw_preprocessed) ))
    raw_preprocessed_sorted2.extend(["<|endoftext|>", "<|unk|>"])
    extended_vocab = {token:ID for ID, token in enumerate(raw_preprocessed_sorted2)}
    print(f"Length of the Extended Vocab is {len(extended_vocab)}")
    return (extended_vocab,)


@app.cell
def _(extended_vocab):
    for iex, item_ex in enumerate(list(extended_vocab.items())[-5:]):
        print(item_ex)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Update the V1 of the tokenizer
    """)
    return


@app.class_definition
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


@app.cell
def _():
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text_combo = " <|endoftext|> ".join((text1, text2))
    print(text_combo)
    return (text_combo,)


@app.cell
def _(extended_vocab, text_combo):
    enhanced_tokenizer = SimpleTokenizerV2(extended_vocab)
    print(enhanced_tokenizer.encode(text_combo))
    return (enhanced_tokenizer,)


@app.cell
def _(enhanced_tokenizer, text_combo):
    print(enhanced_tokenizer.decode(enhanced_tokenizer.encode(text_combo)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BPE
    """)
    return


@app.cell
def _():
    tokenizer = tiktoken.get_encoding('gpt2')
    tik_text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )

    tik_ids = tokenizer.encode(tik_text, allowed_special={"<|endoftext|>"})
    print(tik_ids)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
