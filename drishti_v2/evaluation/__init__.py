"""Evaluation utilities."""

from drishti_v2.evaluation.metrics import detection_metrics

__all__ = ["DRISHTIEvaluator", "detection_metrics"]


def __getattr__(name: str):
    if name == "DRISHTIEvaluator":
        from drishti_v2.evaluation.evaluator import DRISHTIEvaluator

        return DRISHTIEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
