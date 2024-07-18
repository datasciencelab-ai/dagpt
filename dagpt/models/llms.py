from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dagpt.utils.utils import BaseLogger

# Load environment variables from a .env file
load_dotenv()


def load_embedding_model(
    embedding_model_name: str = "openai", logger: BaseLogger = BaseLogger()
):
    """
    Load the specified embedding model.

    Args:
        embedding_model_name (str): The name of the embedding model to load. Defaults to "openai".
        logger (BaseLogger): The logger instance to use for logging information. Defaults to a new BaseLogger instance.

    Returns:
        An instance of the specified embedding model.

    Raises:
        ValueError: If an unknown embedding model name is provided.
    """
    if embedding_model_name == "openai":
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        dimension = 1536
        logger.info(f"Embedding: Using OpenAI with {dimension} dimensions.")
    elif embedding_model_name == "google-genai":
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        dimension = 512
        logger.info(
            f"Embedding: Using Google Generative AI with {dimension} dimensions."
        )
    else:
        raise ValueError(
            "Unknown embedding model API. Please choose from 'openai' or 'google-genai'."
        )

    return embedding


def load_llm(llm_name: str, logger: BaseLogger = BaseLogger()):
    """
    Load the specified language model.

    Args:
        llm_name (str): The name of the language model to load.
        logger (BaseLogger): The logger instance to use for logging information. Defaults to a new BaseLogger instance.

    Returns:
        An instance of the specified language model.

    Raises:
        ValueError: If an unknown language model name is provided.
    """
    if llm_name == "gpt-4":
        logger.info("INFO: Using GPT-4")
        return ChatOpenAI(
            model_name="gpt-4", streaming=True, temperature=0, max_tokens=1000
        )
    elif llm_name == "gpt-3.5":
        logger.info("INFO: Using GPT-3.5 Turbo")
        return ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0)
    elif llm_name == "gemini-pro":
        logger.info("INFO: Using Gemini")
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
    elif llm_name == "gemini-flask":
        logger.info("INFO: Using Gemini")
        return ChatGoogleGenerativeAI(model="gemini-flask", temperature=0.0)
    else:
        raise ValueError(
            "Unknown LLM. Please choose from ['gpt-4', 'gpt-3.5', 'gemini-pro', 'gemini-flask']."
        )
