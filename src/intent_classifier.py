
import re


def classify_intent(question):

    question = question.lower().strip()

    # Financial policy questions
    policy_keywords = [
        "policy", "kyc", "chargeback", "dispute",
        "duplicate transaction", "refund",
        "transaction limit", "credit limit",
        "beneficiary", "compliance"
    ]

    # Financial operations questions
    operations_keywords = [
        "account", "payment", "transaction",
        "failed", "pending", "settlement",
        "bank", "debit", "credit", "transfer"
    ]

    # Greetings
    greeting_keywords = [
        "hello", "hi", "hey", "good morning",
        "good evening"
    ]

    if any(word in question for word in greeting_keywords):
        return "greeting"

    if any(word in question for word in policy_keywords):
        return "policy"

    if any(word in question for word in operations_keywords):
        return "financial_operations"

    return "out_of_domain"
