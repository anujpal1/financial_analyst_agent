import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config

# 1. Initialize the ultra-fast local embedding model
embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)

# 2. Connect to the local Vector Database folder
vector_store = Chroma(
    collection_name="financial_research_memory",
    embedding_function=embeddings,
    persist_directory=config.VECTOR_DB_DIR
)

def memorize_document(text: str, source_title: str) -> str:
    """Chops a massive document into bite-sized chunks and memorizes them in the vector DB."""
    try:
        # Split massive text into 1000-character chunks with slight overlap so context isn't lost
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)
        
        # Convert chunks into LangChain Document objects
        docs = [Document(page_content=chunk, metadata={"source": source_title}) for chunk in chunks]
        
        # Save them into ChromaDB
        vector_store.add_documents(docs)
        return f"Successfully memorized {len(docs)} chunks from '{source_title}' into Long-Term Memory."
    except Exception as e:
        return f"Failed to memorize document: {str(e)}"

def recall_from_memory(query: str, max_results: int = 3) -> str:
    """Searches the agent's long-term memory for the most relevant historical chunks."""
    try:
        results = vector_store.similarity_search(query, k=max_results)
        
        if not results:
            return "No relevant data found in long-term memory for this query."
            
        memory_strings = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            memory_strings.append(f"[Memory Chunk {i} from {source}]:\n{doc.page_content}")
            
        return "\n\n".join(memory_strings)
    except Exception as e:
        return f"Failed to recall from memory: {str(e)}"