import numpy as np

from human_vision import Vision, VisionConfig
from human_vision.results import VisionResult


def test_vision_analyze_smoke(monkeypatch):
    config = VisionConfig()
    # disable missing deps by turning off capabilities using non-available backends
    for cap_conf in config.capabilities.values():
        cap_conf.enabled = False
    config.capabilities[next(iter(config.capabilities))].enabled = True
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with Vision(config) as vision:
        res = vision.analyze(frame)
    assert isinstance(res, VisionResult)
    assert res.debug is not None
