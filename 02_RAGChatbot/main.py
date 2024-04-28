# Opens a chat session with the model. Uses RAG (Retrieval Augmented Generation) to expand model knowledge.
import os
import sys
from configparser import ConfigParser
from langchain import hub
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma


CONFIG_PATH = '../'
CONFIG_FILENAME = 'config.toml'
DOCUMENTS_LIMIT = 3


def get_client() -> ChatOpenAI:
    return ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", temperature=0.7, streaming=False)


def get_retriever(config : ConfigParser, client : ChatOpenAI) -> MultiQueryRetriever:
    embedding_function : HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=config.get('COMMON', 'EMBEDDINGS_MODEL'), model_kwargs={'device': config.get('COMMON', 'DEVICE')})
    vector_db : Chroma = Chroma(persist_directory=get_normalized_path(config.get('COMMON', 'CHROMA_DATABASE_PATH'), True), embedding_function=embedding_function)
    return MultiQueryRetriever.from_llm(retriever = vector_db.as_retriever(), llm=client)
    

def start_chat(client : ChatOpenAI, mq_retriever : MultiQueryRetriever):
    prompt = hub.pull("rlm/rag-prompt")
    history = [
        ("system", "You are expert in Computer Science. Respond briefly to questions using your knowledge and context information."),
        ("user", "Hello, introduce yourself to someone opening this program for the first time. Be concise."),
    ]
    while True:
        response : AIMessage = client.invoke(history)

        print(response.content)
        history.append(("assistant", response.content))

        user_input = input("> ")
        docs = mq_retriever.invoke(user_input)
        context :str = "\n".join([doc.page_content for doc in docs])

        user_message = prompt.invoke({"context": context, "question": user_input}).to_messages()
        history.append(("user", user_message[0].content))


def get_normalized_path(relative_path : str, is_config : bool = False) -> str:
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    if is_config:
        relative_path = CONFIG_PATH + relative_path
    raw_full_path = os.path.join(script_path, relative_path)
    return os.path.normpath(raw_full_path)


def load_config() -> ConfigParser:
    config_file_path : str = get_normalized_path(CONFIG_FILENAME, True)

    config : ConfigParser = ConfigParser()
    config.read(config_file_path)

    return config


def main():
    config : ConfigParser = load_config()

    client : ChatOpenAI = get_client()
    retriever : MultiQueryRetriever = get_retriever(config, client)
    start_chat(client, retriever)

    print('End')


if __name__ == "__main__":
    main()
