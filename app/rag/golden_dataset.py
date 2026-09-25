"""
Golden Q/A dataset for RAGAS evaluation of the PulseTrack KB pipeline.

12 straightforward questions (one grounded in each policy fact across the
three docs) plus 2 deliberate edge cases:
  - AMBIGUOUS: "refunds" means different things in billing.txt (subscription
    refunds) vs. shipping.txt (device return refunds) - a good pipeline
    should surface both.
  - OUT_OF_SCOPE: a real PulseTrack-sounding question the KB genuinely does
    not cover (device hardware features aren't documented anywhere in
    data/kb_docs) - the correct behavior is to say so, not to guess.
"""

GOLDEN_DATASET: list[dict] = [
    {
        "question": "How long is a password reset link valid for?",
        "ground_truth": "The password reset link is valid for 60 minutes.",
    },
    {
        "question": "How many failed login attempts trigger an account lockout, and for how long?",
        "ground_truth": "5 failed login attempts trigger a temporary lockout of 15 minutes.",
    },
    {
        "question": "What must support verify before disabling 2FA for a user locked out due to a lost device?",
        "ground_truth": "Support must confirm the account email and the last 4 digits of the payment method on file.",
    },
    {
        "question": "What happens during the 30 days after a user requests account deletion?",
        "ground_truth": "The account enters a 30-day grace period during which the user can cancel the deletion by logging back in; after 30 days the deletion becomes permanent.",
    },
    {
        "question": "How much does PulseTrack Plus cost per month and per year?",
        "ground_truth": "PulseTrack Plus costs $7.99 per month or $69 per year.",
    },
    {
        "question": "How many days does PulseTrack retry a failed renewal payment before downgrading the account?",
        "ground_truth": "PulseTrack retries a failed payment automatically for 3 days before downgrading the account to PulseTrack Free.",
    },
    {
        "question": "Are monthly PulseTrack Plus subscriptions refundable?",
        "ground_truth": "No. Monthly subscriptions are non-refundable once a billing cycle has started, though users can cancel anytime to stop future charges.",
    },
    {
        "question": "Does cancelling PulseTrack Plus delete my workout history?",
        "ground_truth": "No, cancelling does not delete workout history or account data.",
    },
    {
        "question": "What is the flat rate for expedited shipping?",
        "ground_truth": "Expedited shipping is a flat rate of $12.99, with delivery in 1-2 business days.",
    },
    {
        "question": "How long do I have to report a damaged package to get a free replacement?",
        "ground_truth": "Damaged packages must be reported within 7 days of delivery for a free replacement.",
    },
    {
        "question": "How many days do I have to return a PulseTrack device for a full refund?",
        "ground_truth": "PulseTrack devices can be returned within 30 days of delivery for a full refund.",
    },
    {
        "question": "How long after a return is received does it take to issue the refund?",
        "ground_truth": "Refunds are issued within 5-7 business days after the returned item is received and inspected.",
    },
    {
        "question": "What's your policy on refunds?",
        "ground_truth": (
            "It depends what's being refunded. Annual PulseTrack Plus subscriptions "
            "are eligible for a full refund within 14 days of purchase; monthly "
            "subscriptions are non-refundable once a billing cycle has started. "
            "PulseTrack devices can be returned within 30 days of delivery for a "
            "full refund."
        ),
        "note": "AMBIGUOUS - spans billing.txt (subscription refunds) and shipping.txt (device return refunds)",
    },
    {
        "question": "Does the PulseTrack device track blood oxygen (SpO2) levels?",
        "ground_truth": (
            "This isn't documented anywhere in the knowledge base - it only covers "
            "account/login, billing, and shipping policies, not device hardware "
            "features. The correct answer is that this information isn't available "
            "here, not a guess."
        ),
        "note": "OUT_OF_SCOPE - no doc in data/kb_docs mentions device sensors/features at all",
    },
]
