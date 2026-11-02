from functools import lru_cache
import config
from langchain_groq import ChatGroq
from langchain_community.embeddings import JinaEmbeddings

@lru_cache(maxsize=1)
def get_chat_model() ->ChatGroq:
    """
    Create (or reuse) the Groq chat model used for every LLM call.

    The chat model does two jobs in this project: writing alternative versions
    of the user's question, and writing the final answer from the retrieved
    context.

    Returns:
        A configured LangChain ChatGroq model.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
    """
    config.check_groq_key()

    return ChatGroq(
        model = config.GROQ_CHAT_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> JinaEmbeddings:
    """
    Create (or reuse) the OpenAI embedding model used to build the vector index.

    An embedding turns a piece of text into a list of numbers (a vector).
    Texts with similar meaning get similar vectors, which is what makes
    "search by meaning" possible.

    Returns:
        A configured LangChain OpenAIEmbeddings model.

    Raises:
        ValueError: If OPENAI_API_KEY is missing.
    """
    config.check_groq_key()

    return JinaEmbeddings(
        model = config.JINA_EMBEDDING_MODEL,
        api_key = config.JINA_API_KEY,
    )