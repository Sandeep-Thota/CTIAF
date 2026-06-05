"""CTIAF reference implementation.

Reference implementation of the Cross-Platform Trust and Intent Alignment
Framework as described in the paper (Section 3, Algorithm 1). Provided for
transparency; it implements the published method but is not a turnkey
reproduction of the exact reported numbers, which depend on the full ShareChat
preprocessing pipeline and trained checkpoints.
"""
from .model import CTIAF, Conversation, combined_loss
from .modules import TARS, CPAE, DTG, trust_proxies, intent_satisfaction
from .features import build_conversation, platform_profile

__all__ = ["CTIAF", "Conversation", "combined_loss", "TARS", "CPAE", "DTG",
           "trust_proxies", "intent_satisfaction", "build_conversation",
           "platform_profile"]
__version__ = "0.1.0"
