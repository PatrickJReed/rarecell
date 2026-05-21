"""Canned Anthropic API responses for ClaudeRecommender tests."""

RECOMMENDATIONS_RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "content": [
        {
            "type": "text",
            "text": """```json
{
  "recommendations": [
    {
      "cluster_id": "0",
      "recommendation": "keep",
      "confidence": 0.9,
      "evidence_summary": {"score_pan_t_mean": 2.0, "is_contaminant_frac": 0.02},
      "reasoning": "Strong positive signal and low contamination.",
      "citations": []
    },
    {
      "cluster_id": "1",
      "recommendation": "drop",
      "confidence": 0.85,
      "evidence_summary": {"score_pan_t_mean": 0.1, "is_contaminant_frac": 0.5},
      "reasoning": "Weak positive and heavy contamination.",
      "citations": []
    },
    {
      "cluster_id": "2",
      "recommendation": "purify",
      "confidence": 0.55,
      "evidence_summary": {"score_pan_t_mean": 1.5, "is_contaminant_frac": 0.15},
      "reasoning": "Mixed signal — recommend surgical subclustering.",
      "citations": []
    }
  ]
}
```""",
        }
    ],
}
