"""Profile drafting — thin re-export of rarecell.agent.draft.

Drafting requires the [agent] extra. Import from rarecell.profile.draft for
convenience; the implementation lives in rarecell.agent.draft so the core
profile module stays LLM-free.
"""

try:
    from rarecell.agent.draft import draft_profile_from_prompt
except ImportError as e:
    raise ImportError(
        "Profile drafting requires the [agent] extra. Install with: pip install rarecell[agent]"
    ) from e

__all__ = ["draft_profile_from_prompt"]
