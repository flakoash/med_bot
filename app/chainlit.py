import os
import uuid
from operator import itemgetter
from typing import Sequence

import chainlit as cl
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                                    SystemMessagePromptTemplate)
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnableWithMessageHistory)
from loguru import logger

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("MODEL_NAME")


_history_store: dict[str, ChatMessageHistory] = {}


def get_history(session_id: str) -> ChatMessageHistory:
    return _history_store.setdefault(session_id, ChatMessageHistory())


def format_docs(docs: Sequence[Document]) -> str:
    """helper function to format documents from the retriever to str including id and metadata

    Args:
        docs (Document): the retrived documents

    Returns:
        str: formated documents
    """
    return "\n\n".join(
        f"[{i + 1}] doc_id:{d.id} doc_type:({d.metadata.get('type')}) {d.page_content[:500]}"
        for i, d in enumerate(docs)
    )


# create llm as an OpenAI wrapper for vLLM
# (we can still call open AI models if we configure the ENVs)
llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
)

# load local embedder, as an improvement we can have a separate vllm serving this in next iterations
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# load local vector DB, again, as an improvement we can use a proper vector DB in next iterations
vectorstore = FAISS.load_local("../medquad_faiss", embedder, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a helpful medical assistant. Answer ONLY from the context "
            "and cite sources with [1] (doc_type: doc_id), "
            "including the doc_id and type."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ]
)

rag_answer = (
    RunnableParallel(
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )
    | prompt
    | llm
)

rag_graph = RunnableParallel(
    {
        "answer": rag_answer,
        "source_documents": itemgetter("question") | retriever ,
    }
)

chat_chain = RunnableWithMessageHistory(
    rag_graph,
    get_history,
    input_key="question",
    # history_key="chat_history",
    history_messages_key="chat_history"
)


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("chain", chat_chain)
    cl.user_session.set("session_id", session_id)
    await cl.Message(content="Hi I am MedBot, your assistant to ask anything related with medicine, "
                     "do you have any questions?").send()


@cl.on_message
async def on_message(msg: cl.Message):
    chain = cl.user_session.get("chain")
    session_id = cl.user_session.get("session_id")

    result = chain.invoke(
        {"question": msg.content},
        config={"configurable": {"session_id": session_id}},
    )
    answer = result["answer"]
    sources = result["source_documents"]
    await cl.Message(content=answer.content).send()
