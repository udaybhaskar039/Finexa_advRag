

import re

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from src.intent_classifier import classify_intent
from src.document_loader import load_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store


# -----------------------------------------
# RETRIEVAL SETTINGS
# -----------------------------------------

POLICY_TOP_K = 10
OPERATIONS_TOP_K = 8

FINAL_CONTEXT_DOCS = 4

# FAISS L2 distance: lower means closer.
# These are starting values, not universally validated thresholds.
POLICY_DISTANCE_THRESHOLD = 1.5
OPERATIONS_DISTANCE_THRESHOLD = 1.5


# -----------------------------------------
# TEXT PROCESSING
# -----------------------------------------

STOPWORDS = {
    "what", "when", "where", "which", "who",
    "how", "does", "do", "is", "are", "the",
    "a", "an", "and", "or", "to", "of", "in",
    "on", "for", "with", "if", "should", "can",
    "could", "would", "will", "be", "it", "this",
    "that", "my", "your", "their", "customer",
    "happens", "much", "time", "within"
}


def tokenize(text):

    words = re.findall(r"\b[a-z0-9]+\b", text.lower())

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def expand_query(question):

    expanded_terms = tokenize(question)

    # POS terminology
    if re.search(r"\bpos\b|point of sale", question, re.I):

        expanded_terms.update([
            "pos",
            "point",
            "sale",
            "merchant",
            "settlement",
            "authorization",
            "duplicate",
            "batch",
            "debit",
            "reversal"
        ])

    # Duplicate transaction terminology
    if "duplicate" in question.lower():

        expanded_terms.update([
            "duplicate",
            "debit",
            "transaction",
            "secondary",
            "refund",
            "reversed"
        ])

    # KYC terminology
    if re.search(r"\bkyc\b", question, re.I):

        expanded_terms.update([
            "kyc",
            "verification",
            "overdue",
            "freeze",
            "documents",
            "video"
        ])

    return expanded_terms


# -----------------------------------------
# KEYWORD-BASED RERANKING
# -----------------------------------------

def calculate_keyword_score(question, document):

    question_lower = question.lower()
    content = document.page_content.lower()

    query_terms = expand_query(question)

    document_terms = tokenize(content)

    if not query_terms:
        return 0

    matched_terms = query_terms.intersection(document_terms)

    score = len(matched_terms) / len(query_terms)

    # Give additional weight to exact topic phrases.
    important_phrases = [
        "point of sale",
        "pos duplicate",
        "duplicate merchant debit",
        "duplicate batch",
        "primary authorization",
        "secondary debit",
        "network retry",
        "dual switch",
        "beneficiary settlement",
        "bank settlement cycle",
        "video-kyc",
        "officially valid documents"
    ]

    for phrase in important_phrases:

        if phrase in question_lower and phrase in content:
            score += 0.25

    # Give a bonus when the question explicitly asks for a policy ID.
    requested_ids = re.findall(
        r"\bPOL\d{3}\b",
        question.upper()
    )

    for policy_id in requested_ids:

        if policy_id.lower() in content:
            score += 1.0

    return score


# -----------------------------------------
# POLICY-AWARE RETRIEVAL
# -----------------------------------------

def retrieve_relevant_documents(
    vector_store,
    question,
    top_k,
    threshold
):

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    if not results:
        return []

    candidates = []

    print("\n----- INITIAL FAISS RETRIEVAL -----")

    for rank, (doc, distance) in enumerate(results, start=1):

        distance = float(distance)

        policy_ids = re.findall(
            r"\bPOL\d{3}\b",
            doc.page_content.upper()
        )

        keyword_score = calculate_keyword_score(
            question,
            doc
        )

        print(f"\nCandidate Rank: {rank}")
        print("FAISS Distance:", round(distance, 4))
        print("Keyword Score:", round(keyword_score, 4))
        print(
            "Policy IDs:",
            list(dict.fromkeys(policy_ids))
        )
        print("Content:", doc.page_content[:250])

        if distance <= threshold:

            candidates.append({
                "document": doc,
                "distance": distance,
                "keyword_score": keyword_score,
                "vector_rank": rank
            })

            print("Status: ACCEPTED")

        else:

            print("Status: REJECTED")

        print("----------------------------------")

    # First prioritize lexical/topic relevance.
    # Use FAISS distance to break ties.
    candidates.sort(
        key=lambda item: (
            -item["keyword_score"],
            item["distance"]
        )
    )

    selected = candidates[:FINAL_CONTEXT_DOCS]

    print("\n----- FINAL RERANKED DOCUMENTS -----")

    for rank, item in enumerate(selected, start=1):

        doc = item["document"]

        policy_ids = re.findall(
            r"\bPOL\d{3}\b",
            doc.page_content.upper()
        )

        print(f"\nFinal Rank: {rank}")
        print("FAISS Distance:", round(item["distance"], 4))
        print("Keyword Score:", round(item["keyword_score"], 4))
        print("Policy IDs:", list(dict.fromkeys(policy_ids)))
        print("Content:", doc.page_content[:250])

    print("\nTotal FAISS Results:", len(results))
    print("Accepted Candidates:", len(candidates))
    print("Final Context Documents:", len(selected))

    return [
        item["document"]
        for item in selected
    ]


# -----------------------------------------
# MAIN RAG PIPELINE
# -----------------------------------------

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

    print(f"\nTotal chunks created: {len(chunks)}")

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

    # 6. Strict financial policy prompt
    prompt = ChatPromptTemplate.from_template("""
You are Finexa, a Financial Operations Assistant.

Answer using ONLY the supplied financial document context.

STRICT RULES:

1. Use only information present in the context.
2. Never invent policy IDs, policy names, timeframes,
   procedures, or financial rules.
3. Preserve the exact policy ID associated with the
   retrieved policy content.
4. Never assign a policy ID based only on similar wording.
5. Do not transfer information from one policy to another.
6. Keep different policies separate when multiple policies
   are relevant.
7. Do not combine conflicting policy details.
8. If the context does not clearly establish the policy ID,
   do not guess it.
9. If the answer is missing, clearly say that there is
   insufficient information.
10. Answer the actual question directly.
11. Mention the relevant policy ID and policy name
    when explicitly available in the context.
12. Do not use external knowledge.

IMPORTANT:
If two retrieved documents contain similar descriptions
but different policy IDs, use the document that most directly
answers the question. Do not assume that the IDs are
interchangeable.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

    # -----------------------------------------
    # QUESTION ANSWERING FUNCTION
    # -----------------------------------------

    def ask_question(question):

        # 1. Classify intent
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

            print(
                "Routing to: Financial Policy Retrieval"
            )

            top_k = POLICY_TOP_K
            threshold = POLICY_DISTANCE_THRESHOLD

        elif intent == "financial_operations":

            print(
                "Routing to: Financial Operations Retrieval"
            )

            top_k = OPERATIONS_TOP_K
            threshold = OPERATIONS_DISTANCE_THRESHOLD

        else:

            return "Unable to identify the financial intent."

        # 5. Retrieve and rerank documents
        relevant_docs = retrieve_relevant_documents(
            vector_store=vector_store,
            question=question,
            top_k=top_k,
            threshold=threshold
        )

        # 6. Check retrieved context
        if not relevant_docs:

            print("\nNo relevant documents found.")

            return (
                "I couldn't find sufficiently relevant "
                "information in the financial documents "
                "to answer your question."
            )

        # 7. Build context with document metadata
        context_parts = []

        for index, doc in enumerate(relevant_docs, start=1):

            metadata = doc.metadata or {}

            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown page")

            context_parts.append(
                f"""
DOCUMENT {index}
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
            )

        context = "\n\n".join(context_parts)

        # 8. Generate answer
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        # 9. Return answer
        return response.content

    return ask_question
