
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from src.intent_classifier import classify_intent
from src.document_loader import load_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store


# -----------------------------------------
# RELEVANCE FILTER SETTINGS
# -----------------------------------------

POLICY_TOP_K = 5
OPERATIONS_TOP_K = 3

# FAISS L2 distance: lower score means closer match.
# Initial threshold based on current retrieval testing.
POLICY_DISTANCE_THRESHOLD = 2.5
OPERATIONS_DISTANCE_THRESHOLD = 2.5


def create_rag_pipeline():

    # 1. Load financial documents
    documents = load_documents()

    if not documents:
        raise ValueError(
            "Please add PDF files to data/raw."
        )

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

    # 6. Define RAG prompt
    prompt = ChatPromptTemplate.from_template("""
    You are Finexa, a Financial Operations Assistant.

    Answer the question using only the provided context.

    Rules:
    1. Give clear and accurate financial information.
    2. Never invent policy details.
    3. If the answer is not available in the context,
       say you do not have enough information.
    4. Mention policy IDs when available.
    5. Do not use outside knowledge.
    6. If multiple policies are relevant, explain them separately.
    7. Do not treat unrelated retrieved documents as relevant.
    8. Do not combine conflicting metadata values as if
       they are all simultaneously active.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    # -----------------------------------------
    # QUESTION ANSWERING FUNCTION
    # -----------------------------------------

    def ask_question(question):

        # 1. Intent classification
        intent = classify_intent(question)

        print("\nDetected Intent:", intent)

        # 2. Greeting route
        if intent == "greeting":
            return (
                "Hello! I'm Finexa, your Financial "
                "Operations Assistant. How can I help you?"
            )

        # 3. Out-of-domain route
        if intent == "out_of_domain":
            return (
                "I'm Finexa, a Financial Operations "
                "Assistant. I can help with financial "
                "policies and operations. Your question "
                "is outside my supported domain."
            )

        # 4. Select retrieval settings
        if intent == "policy":

            print("Routing to: Financial Policy Retrieval")

            top_k = POLICY_TOP_K
            threshold = POLICY_DISTANCE_THRESHOLD

        elif intent == "financial_operations":

            print("Routing to: Financial Operations Retrieval")

            top_k = OPERATIONS_TOP_K
            threshold = OPERATIONS_DISTANCE_THRESHOLD

        else:
            return "Unable to identify the financial intent."

        # 5. Retrieve documents with similarity scores
        results = vector_store.similarity_search_with_score(
            question,
            k=top_k
        )

        if not results:
            return (
                "I do not have enough information "
                "to answer this question."
            )

        # 6. Apply relevance filtering
        relevant_docs = []

        print("\n----- RETRIEVAL RESULTS -----")

        for doc, score in results:

            distance = float(score)

            print("Similarity Distance:", distance)
            print(
                "Retrieved Content:",
                doc.page_content[:200]
            )

            if distance <= threshold:

                relevant_docs.append(doc)

                print("Relevance Status: ACCEPTED")

            else:

                print("Relevance Status: REJECTED")

            print("-----------------------------")

        print("\nTotal Retrieved:", len(results))
        print("Relevant Documents:", len(relevant_docs))
        print("Distance Threshold:", threshold)

        # 7. Check whether relevant documents exist
        if not relevant_docs:

            print("No relevant documents found.")

            return (
                "I couldn't find sufficiently relevant "
                "information in the financial documents "
                "to answer your question."
            )

        # 8. Prepare context from relevant documents only
        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )

        # 9. Generate answer using Groq LLM
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        # 10. Return final answer
        return response.content

    # Return question-answering function
    return ask_question
