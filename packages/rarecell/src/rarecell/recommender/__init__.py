"""rarecell.recommender — Recommender protocol + concrete implementations.

ClaudeRecommender is re-exported here for convenience but lives in
rarecell.agent.recommender. It requires the [agent] extra.
"""

from rarecell.recommender.base import Recommendation, Recommender
from rarecell.recommender.basic import BasicRecommender

try:
    from rarecell.agent.recommender import ClaudeRecommender  # noqa: F401

    _has_claude = True
except ImportError:
    _has_claude = False

__all__ = ["BasicRecommender", "Recommendation", "Recommender"]
if _has_claude:
    __all__.append("ClaudeRecommender")
