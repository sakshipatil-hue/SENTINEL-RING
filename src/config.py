"""
Central config for Sentinel Ring.
Keep volume targets, seeds, and paths here so every script agrees.
"""

import os

# Reproducibility
RANDOM_SEED = 42

# Volume targets (see docs/schema.md for reasoning)
NUM_ACCOUNTS = 2500
NUM_TRANSACTIONS = 10000
NUM_RINGS = 6
RING_SIZE_MIN = 4
RING_SIZE_MAX = 10
INDIVIDUAL_FRAUD_RATE = 0.04  # lone bad actors, NOT part of any ring

# Ring injection patterns
RING_PATTERNS = [
    "shared_device_staggered",
    "shared_payment_burst",
    "address_cluster_sequential",
]

# Agent bounds
MAX_AGENT_TOOL_CALLS = 6

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACCOUNTS_PATH = os.path.join(DATA_DIR, "accounts.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
RINGS_PATH = os.path.join(DATA_DIR, "rings.json")
