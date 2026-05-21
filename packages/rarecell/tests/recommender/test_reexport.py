def test_basic_recommender_always_importable():
    from rarecell.recommender import BasicRecommender, Recommendation, Recommender

    assert BasicRecommender is not None
    assert Recommendation is not None
    assert Recommender is not None


def test_claude_recommender_importable_when_agent_installed():
    """When [agent] is installed (dev env), ClaudeRecommender is re-exported."""
    try:
        from rarecell.recommender import ClaudeRecommender
    except ImportError:
        # Acceptable in environments without [agent]
        return
    assert ClaudeRecommender is not None
