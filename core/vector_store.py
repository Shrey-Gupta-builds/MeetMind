import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# NOTE: No CHROMA_DIR — we intentionally use an in-memory (ephemeral) client.
# Each build_vector_store() call creates a fresh, isolated in-memory ChromaDB
# instance, which avoids all Windows SQLite file-lock issues and guarantees
# that Video B's retriever never sees chunks from Video A.
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript: str) -> Chroma:
    """Build a brand-new, fully isolated in-memory vector store for *this* transcript.

    Using chromadb.EphemeralClient() (in-memory) instead of PersistentClient:
    - Every call produces a completely empty ChromaDB instance in RAM.
    - There are no SQLite files on disk, so Windows file-lock issues cannot occur.
    - When the caller replaces its reference (e.g. st.session_state.result) with
      a new video's data, Python's GC automatically reclaims the old in-memory
      client and all of its embeddings — zero cross-video contamination.
    """
    print("Building vector store (in-memory, isolated)")

    # Fresh in-memory client — completely empty, no shared state with any previous call.
    client = chromadb.EphemeralClient()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )


