# Insert documents from different sources in a vector database for RAG (Retrieval Augmented Generation)
import os
import sys
from configparser import ConfigParser
from langchain.text_splitter import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document


CONFIG_PATH = "../"
CONFIG_FILENAME = "config.toml"
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]
PATH_TO_DOCUMENTSOURCES = "./document_sources/"
PATH_TO_PDFS = f"{PATH_TO_DOCUMENTSOURCES}pdfs/"
PATH_TO_MARKDOWN = f"{PATH_TO_DOCUMENTSOURCES}markdown/"

def load_pdfs(config: ConfigParser):
    config = config
    print("-- Source: PDF files --")

    print("  - Loading files...")
    normalized_path = get_normalized_path(PATH_TO_PDFS)
    loader : DirectoryLoader = DirectoryLoader(
        normalized_path,
        glob="**/*.pdf",
        use_multithreading=True,
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    print("  - Splitting content...")
    splitted_docs : list[Document] = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n"]))
    
    print("  - Creating embbedings...")
    insert_documents_to_db(config, splitted_docs)


def load_markdown(config: ConfigParser):
    print("-- Source: Markdown files --")

    print("  - Loading files...")
    text_loader_kwargs = {"autodetect_encoding": True}
    normalized_path = get_normalized_path(PATH_TO_MARKDOWN)
    loader : DirectoryLoader = DirectoryLoader(
        normalized_path,
        glob="**/*.md",
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs,
        show_progress=True
    )
    docs : list[Document] = loader.load()

    print("  - Splitting content...")
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n"])
    splitted_docs : list[Document] = []
    for doc in docs:
        md_splitter_docs = markdown_splitter.split_text(doc.page_content)
        splitted_docs.extend(recursive_splitter.split_documents(md_splitter_docs))

    print("  - Creating embbedings...")
    insert_documents_to_db(config, splitted_docs)


def insert_documents_to_db(config: ConfigParser, input_documents: list[Document]):
    embedding_function: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name=config.get("COMMON", "EMBEDDINGS_MODEL"),
        model_kwargs={"device": config.get("COMMON", "DEVICE")},
    )
    Chroma.from_documents(
        documents=input_documents,
        embedding=embedding_function,
        persist_directory=get_normalized_path(
            config.get("COMMON", "CHROMA_DATABASE_PATH"), True
        ),
    )


def get_normalized_path(relative_path: str, is_config: bool = False) -> str:
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    if is_config:
        relative_path = CONFIG_PATH + relative_path
    raw_full_path = os.path.join(script_path, relative_path)
    return os.path.normpath(raw_full_path)


def load_config() -> ConfigParser:
    config_file_path: str = get_normalized_path(CONFIG_FILENAME, True)

    config: ConfigParser = ConfigParser()
    config.read(config_file_path)

    return config


def main():
    config: ConfigParser = load_config()

    load_pdfs(config)
    load_markdown(config)

    print("End")


if __name__ == "__main__":
    main()
