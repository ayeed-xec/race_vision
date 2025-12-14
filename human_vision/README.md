# human_vision

`human_vision` is a Pythonic computer-vision library tailored for humanoid robots. It focuses on predictable APIs, stable behavior, and observability-friendly outputs so robotics engineers can build reliable perception pipelines.

## Installation

```bash
pip install human_vision
# optional extras
pip install human_vision[vision,mediapipe,tflite,onnx]
```

## Quickstart

```python
import numpy as np
from human_vision import Vision, VisionConfig

frame = np.zeros((480, 640, 3), dtype=np.uint8)
config = VisionConfig()

with Vision(config) as vision:
    result = vision.analyze(frame)
    print(result.model_dump())
```

## Model caching

Models are cached under the platform-specific cache directory (e.g., `~/.cache/human_vision`). Use the CLI to list and ensure models:

```bash
human-vision models list
human-vision models ensure movenet-lightning
```

## Repo model sync

You can sync models from a GitHub repository containing a `models/` folder:

```bash
human-vision models sync-repo --repo https://github.com/example/repo --branch main
```

## ONNX drop-in

Place your ONNX weights in the cache directory under `repo_models/<name>/models`. The registry will surface manual/placeholder models and the ONNX backend will gracefully skip missing weights while providing debug notes.

## Concurrency

`Vision` supports sequential, threaded, and multiprocess scheduling. Threaded mode is the default and works well for CPU-bound robotics workloads. Tune `max_workers` and backend-specific thread counts to avoid oversubscription.
