
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from src.document_loader import load_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store


def create_rag_pipeline():

    # 1. Load financial documents
    documents = load_documents()

    if not documents:
        raise ValueError("Please add PDF files to data/raw.")

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # 3. Load embedding model
    embedding_model = get_embedding_model()

    # 4. Create FAISS vector database
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # 5. Connect Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    # 6. Define prompt
    prompt = ChatPromptTemplate.from_template("""
    You are Finexa, a financial operations assistant.

    Answer the question using only the provided context.

    If the answer is not available in the context,
    say that you do not have enough information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    # 7. Create question-answering function
    def ask_question(question):

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        relevant_docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in relevant_docs
        )

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        return response.content

    return ask_question
