import argparse
import json
import os
import re
import unicodedata
from typing import Optional

import pandas as pd
import spacy
from loguru import logger
from sklearn.model_selection import train_test_split
from spacy.cli import download

# default values
INPUT_DATA = "data/raw/intern_screening_dataset.csv"
OUTPUT_DIR = "data/cleaned/"
NUM_EXAMPLES = 1000
MAX_ANSWER_LEN = 250


def is_valid_question(text: str) -> bool:
    """ Simple function to identify if a str is a valid question

    Args:
        text (str): text to validate

    Returns:
        bool: is question or not
    """

    doc = nlp(text.strip())
    if not doc:
        return False
    # if first token is what why etc it may be a question
    if doc[0].tag_ in ["WP", "WRB", "MD", "VBP"]:
        return True
    # if ends with ?
    if text.strip().endswith("?"):
        return True
    return False


def filter_valid_data(df: pd.DataFrame, limit: Optional[int] = None, max_answer_len = 250) -> pd.DataFrame:
    """ Function to filter a dataset and get a cleaned, smaller subset
        We use basic logic to filter based on the findings on our EDA:
            - remove questions w/o answers
            - remove answers that are questions (maked as invalid_answer)
            - ignore large responses
        In the future and with more time we could enhance this function to cover other cases.

    Args:
        df (pd.DataFrame): original DF
        limit (Optional[int]): if specified the filtered DF will be truncated to the <limit> rows
        max_answer_len (int): default 250 words (75% of answers have less than 250 words based on out EDA)

    Returns:
        pd.DataFrame: filtered DF
    """

    filtered_df = df[~df['answer'].isnull()].head(limit * 3).copy() # just a sanity copy jic we need the df
    filtered_df['answer_words'] = filtered_df['answer'].str.split().apply(len)
    filtered_df['valid_question'] = filtered_df['question'].apply(is_valid_question)
    # some answers are long using spacy on all will be expensive, better a simple check here
    filtered_df['valid_answer'] = ~filtered_df['answer'].str.strip().str.endswith('?')

    filtered_df = filtered_df[
        (filtered_df["answer_words"] < max_answer_len)
        & (filtered_df["valid_question"])
        & (filtered_df["valid_answer"])]

    if limit:
        # if we specify a limit get a sample subset of the filtered DF
        return filtered_df.sample(limit)
    return filtered_df


def clean_text(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def format_df(df: pd.DataFrame) -> list:
    pairs = [
        {
            "instruction": clean_text(q),
            "input": "",
            "output": clean_text(a)
        } for q, a in zip(df.question, df.answer)
        if 5 <= len(clean_text(q).split()) <= 512
    ]
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="prepare dataset to finetune llm")
    parser.add_argument("--input_data", type=str, default=INPUT_DATA)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--max_answer_len", type=int, default=MAX_ANSWER_LEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # ensure the spacy model is downloaded
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    logger.info("Filtering base dataset")
    df = filter_valid_data(
        df = pd.read_csv(args.input_data),
        limit = args.num_examples,
        max_answer_len = args.max_answer_len
        )

    # split train test sets:
    logger.info("Splitting filtered dataset into train, test sets")
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=111,
    )

    logger.info(f"training dataset size: {len(train_df)}, test dataset size: {len(val_df)}")

    # write DFs to disk
    logger.info("Persisting train, test sets")
    # create directory if doesnt exists
    os.makedirs(args.output_dir, exist_ok=True)

    # format data
    train_pairs = format_df(train_df)
    with open(f"{args.output_dir}/train.jsonl", "w") as f:
        for row in train_pairs:
            json.dump(row, f)
            f.write("\n")

    val_pairs = format_df(val_df)
    with open(f"{args.output_dir}/val.jsonl", "w") as f:
        for row in val_pairs:
            json.dump(row, f)
            f.write("\n")

    train_df.to_parquet(f"{args.output_dir}/train_dataset.parquet")
    val_df.to_parquet(f"{args.output_dir}/validation_dataset.parquet")
