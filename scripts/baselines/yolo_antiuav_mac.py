"""Run the Anti-UAV410 YOLO pipeline locally with Apple Silicon settings."""

from unittest.mock import patch

import torch
from ultralytics import YOLO

import yolo_antiuav
from small_object_benchmark import complete_nms


class MacYOLO(YOLO):
    """Fit the existing Anti-UAV410 prediction batches into unified memory."""

    def train(self, **kwargs):
        return super().train(amp=False, **kwargs)

    def predict(self, source=None, stream=False, **kwargs):
        predict = super().predict
        kwargs["half"] = False
        batch_size = 8
        if stream and isinstance(source, list) and len(source) > batch_size:
            return (
                result
                for start in range(0, len(source), batch_size)
                for result in predict(source=source[start : start + batch_size], stream=True, **kwargs)
            )
        results = predict(source=source, stream=stream, **kwargs)
        if not stream and kwargs.get("device") == "mps":
            torch.mps.synchronize()
        return results


if __name__ == "__main__":
    with patch.object(yolo_antiuav, "load_model", MacYOLO), complete_nms():
        yolo_antiuav.main()
