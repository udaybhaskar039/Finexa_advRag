

import re


def classify_intent(question):

    question = question.lower().strip()

    # Remove extra spaces
    question = re.sub(r"\s+", " ", question)

    # Financial policy questions
    policy_keywords = [
        "policy",
        "kyc",
        "chargeback",
        "dispute",
        "duplicate transaction",
        "duplicate debit",
        "pos transaction",
        "refund",
        "transaction limit",
        "credit limit",
        "beneficiary",
        "compliance",
        "suspicious transaction",
        "high value transaction",
        "account closure",
        "emi",
        "auto-debit",
        "auto debit",
        "blocked card limit"
    ]

    # Financial operations questions
    operations_keywords = [
        "account",
        "payment",
        "transaction",
        "failed",
        "pending",
        "settlement",
        "bank",
        "debit",
        "credit",
        "transfer",
        "balance",
        "withdrawal",
        "deposit"
    ]

    # Greeting keywords
    greeting_keywords = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    # Check greetings using whole-word matching
    if any(
        re.search(r"\b" + re.escape(word) + r"\b", question)
        for word in greeting_keywords
    ):
        return "greeting"

    # Check financial policy intent
    if any(
        re.search(r"\b" + re.escape(word) + r"\b", question)
        for word in policy_keywords
    ):
        return "policy"

    # Check financial operations intent
    if any(
        re.search(r"\b" + re.escape(word) + r"\b", question)
        for word in operations_keywords
    ):
        return "financial_operations"

    # Unsupported questions
    return "out_of_domain"
