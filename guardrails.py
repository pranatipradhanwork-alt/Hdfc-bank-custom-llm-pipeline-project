"""Deterministic input, context and output controls shared by data cleaning, the gateway and evaluation.

Each check returns findings rather than raising, so the caller decides whether to mask, block or escalate
and every decision can be recorded in the response and the trace log.
"""
import re

# --- Sensitive data (also used by clean_data.py, so training data and live traffic are masked the same way) ---
ACCOUNT_OR_CARD = re.compile(r"\b\d{10,16}\b")
INDIAN_MOBILE = re.compile(r"\b[6-9]\d{9}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Mask only when a value follows ("PIN 1234", "OTP is 5678", "password: abc"),
# so ordinary mentions like "change my PIN" keep their meaning.
SECRET = re.compile(r"(?i)\b(?:otp|cvv|pin|password)\b(?:\s*(?:is\s+)?[:=-]?\s*\S*\d\S*|\s*[:=]\s*\S+)")

# Phone numbers first: a 10-digit mobile number also fits the account pattern
MASKS = (
    (INDIAN_MOBILE, "[MASKED_PHONE_NUMBER]"),
    (ACCOUNT_OR_CARD, "[MASKED_ACCOUNT_OR_CARD]"),
    (EMAIL, "[MASKED_EMAIL]"),
    (SECRET, "[MASKED_SECRET]"),
)


MASK_TOKEN = re.compile(r"\[MASKED_[A-Z_]+\]")


def search_text(masked_text):
    # Mask placeholders are noise to the embedding model and pushed real questions below the scope threshold
    return re.sub(r"\s+", " ", MASK_TOKEN.sub(" ", masked_text)).strip()


def mask_sensitive(text):
    """Return the text with identifiers and secrets masked, plus the labels of what was masked."""
    found = []
    for pattern, label in MASKS:
        text, count = pattern.subn(label, text)
        if count:
            found.append(label.strip("[]"))
    return text, found


# --- Prompt injection, in customer input or in retrieved source content ---
INJECTION = re.compile(
    r"(?i)\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instructions?|rules|prompt|above|previous)\b"
    r"|\b(system|developer)\s+(prompt|message|mode)\b"
    r"|\breveal\b[^.\n]{0,30}\b(prompt|instructions?)\b"
    r"|\byou are now\b|\bact as (an?|the)\b|\bjailbreak\b"
)

# --- Requests the assistant must not act on: it answers questions, it never moves money or changes accounts.
# Only direct requests match; "How do I transfer money?" is an information question and is answered normally.
ACTION = r"(transfer|send|pay|withdraw|move|close|block|unblock|activate|cancel)"
TRANSACTION_REQUEST = re.compile(
    rf"(?i)^\s*(please\s+)?{ACTION}\b"
    rf"|\b(can|could|will|would)\s+you\s+(please\s+)?{ACTION}\b"
    r"|\b(for me|on my behalf)\b"
)

# --- Output: the assistant asking the customer to hand over credentials ---
CREDENTIAL_REQUEST = re.compile(
    r"(?i)\b(share|send|tell|give|provide|type|reply with)\b[^.\n]{0,40}\b(otp|pin|cvv|password|mpin|ipin)\b"
)
NEGATION = re.compile(r"(?i)\b(never|not|don't|do not|no one|nobody)\b")

# Amounts, rates, periods and phone-like numbers
SPECIFIC = re.compile(
    r"(?i)(?:rs\.?|₹|inr)\s*\d[\d,.]*|\d[\d,.]*\s*(?:%|lakhs?|crores?|days?|months?|years?|hours?)"
    r"|\d{3,5}(?:[\s-]\d{3,4}){2}|\d{7,}"
)


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
# "Zero balance account" / "free of charge" support an answer that says "Rs. 0"
ZERO_PHRASES = re.compile(r"(?i)\b(free|nil|no (minimum|charges?|fees?|balance))\b")


def invented_specifics(prediction, source):
    """Amounts, rates, periods and phone-like numbers in the answer whose number never appears in the source."""
    source_numbers = set(re.findall(r"\d+", source.replace(",", "")))
    source_numbers |= {NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", source.lower()) if w in NUMBER_WORDS}
    if ZERO_PHRASES.search(source):
        source_numbers.add("0")
    invented = []
    for match in SPECIFIC.findall(prediction):
        numbers = re.findall(r"\d+", match.replace(",", ""))
        if numbers and not all(n in source_numbers for n in numbers):
            invented.append(match.strip().rstrip(".,"))
    return invented


def asks_for_credentials(answer):
    # A sentence that asks the customer to share a credential, and is not a warning against doing so
    return any(CREDENTIAL_REQUEST.search(s) and not NEGATION.search(s) for s in re.split(r"(?<=[.!?])\s+", answer))


def check_input(question):
    masked, pii = mask_sensitive(question)
    flags = [f"input_contains_{label.lower()}" for label in pii]
    if INJECTION.search(question):
        flags.append("prompt_injection")
    if TRANSACTION_REQUEST.search(question):
        flags.append("transaction_request")
    return masked, flags


def is_injected(text):
    return bool(INJECTION.search(text))


def check_output(answer, context):
    # Mask first, so a leaked identifier is reported as PII rather than as an unsupported figure
    masked, pii = mask_sensitive(answer)
    flags = [f"output_contains_{label.lower()}" for label in pii]
    if asks_for_credentials(masked):
        flags.append("asks_for_credentials")
    unsupported = invented_specifics(masked, context)
    if unsupported:
        flags.append("unsupported_specifics")
    return masked, flags, unsupported
