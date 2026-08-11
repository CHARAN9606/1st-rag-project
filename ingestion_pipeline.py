import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load documents from docs folder
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""

    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your files."
        )

    # Load all txt files
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. Please add documents."
        )

    # Display document information
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Preview: {doc.page_content[:100]}...")
        print(f" Metadata: {doc.metadata}")

    return documents



# Split documents into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks"""

    print("\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # Show sample chunks
    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(chunk.page_content)
        print("-" * 50)

    return chunks



# Create Chroma vector database
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""

    print("\nCreating embeddings and storing in ChromaDB...")

    # Free Hugging Face embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    print("\n--- Finished creating vector store ---")
    print(f"Vector store saved at: {persist_directory}")

    return vectorstore



# Main ingestion pipeline
def main():

    print("=== RAG Document Ingestion Pipeline ===\n")


    docs_path = "docs"
    persistent_directory = "db/chroma_db"


    # If database already exists, load it
    if os.path.exists(persistent_directory):

        print("✅ Vector store already exists. Loading existing database...")


        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={
                "hnsw:space": "cosine"
            }
        )


        print(
            f"Loaded vector store with "
            f"{vectorstore._collection.count()} documents"
        )

        return vectorstore



    print("Creating new vector store...\n")


    # Step 1: Load documents
    documents = load_documents(docs_path)


    # Step 2: Split documents
    chunks = split_documents(documents)


    # Step 3: Create vector database
    vectorstore = create_vector_store(
        chunks,
        persistent_directory
    )


    print(
        "\n✅ Ingestion complete! "
        "Documents are ready for RAG queries."
    )


    return vectorstore



if __name__ == "__main__":
    main()