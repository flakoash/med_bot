import argparse

from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from loguru import logger

# default values
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "medquad_faiss"
NUM_EXAMPLES = 5000


def get_dataset(num_samples: int) -> list[dict]:
    """Funtion to download the medquad open dataset. This will be used to create our Vector DB for the RAG app

    Args:
        num_samples (int): number of records we want to include in the vector DB

    Returns:
        list[dict]: returns a list of dicts with page_content, and metadata columns
    """
    ds = (load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
          .shuffle(seed=111)
          .select(range(num_samples)))

    def to_doc(row: dict) -> dict:
        return {
            "page_content": row["Answer"],
            "metadata": {"type": row["qtype"]}
        }
    return [to_doc(item) for item in ds]


def split_documents(docs: list[dict], chunk_size: int = 512, chunk_overlap: int = 64) -> list[dict]:
    """
    Args:
        docs (list[dict]): documents to get chunked
        chunk_size (int, optional): how long each chunk will be. Defaults to 512.
        chunk_overlap (int, optional): if we are going to have overlap. Defaults to 64.

    Returns:
        list[dict]: a list of chunked Documents
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    contents = [d["page_content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    return splitter.create_documents(contents, metadatas=metadatas)


def create_vector_DB(chunks: list[dict], embd_name: str, output_dir: str) -> None:
    """Function to create vector DB with specified doc chunks

    Args:
        chunks (list[dict]): the chunsk to insert
        embd_name (str): the embedding model to use (name from HuggingFaceEmbeddings)
        output_dir (str): location where the local vector DB will be saved
    """
    embedder = HuggingFaceEmbeddings(model_name=embd_name)
    vector_db = FAISS.from_documents(chunks, embedder)
    vector_db.save_local(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="create a local FAISS vector DB")
    parser.add_argument("--emb_model_name", type=str, default=EMB_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        docs = get_dataset(args.num_examples)
        chunks = split_documents(docs)
        create_vector_DB(chunks, args.emb_model_name, args.output_dir)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)


if __name__ == "__main__":
    main()
