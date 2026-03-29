"""
Integration export for the crospeleo-automation dossier format.

Wraps OriginPredictor output into the manual_choice_field schema
expected by the dossier system.
"""

from __future__ import annotations

MODEL_SOURCE = "ml_model_v1"


def format_for_dossier(prediction: dict) -> dict:
    """
    Convert a prediction dict (from OriginPredictor.predict()) into
    the dossier-compatible manual_choice_field format.

    Args:
        prediction: dict returned by OriginPredictor.predict()

    Returns:
        dict matching the manual_choice_field schema
    """
    return {
        "key":                    "podrijetlo_imena",
        "label":                  "Podrijetlo imena",
        "required":               True,
        "predicted_value":        prediction["predicted_value"],
        "confidence":             prediction["confidence"],
        "top_k":                  prediction["top_k"],
        "explanation":            prediction["explanation"],
        "similar_examples":       prediction["similar_examples"],
        "needs_user_confirmation": prediction["needs_user_confirmation"],
        "source":                 MODEL_SOURCE,
    }


def format_batch_for_dossier(predictions: list[dict]) -> list[dict]:
    """Convenience wrapper for batch processing."""
    return [format_for_dossier(p) for p in predictions]
