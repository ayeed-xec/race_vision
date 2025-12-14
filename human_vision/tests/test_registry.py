from human_vision.models.specs_builtin import REGISTRY
from human_vision.core.types import Capability


def test_registry_contains_required_ids():
    required = [
        "movenet-lightning",
        "movenet-thunder",
        "handlandmark-full",
        "blazeface",
        "selfie",
        "mb3-centernet",
        "liveness",
    ]
    model_ids = {spec.model_id for spec in REGISTRY.list_models()}
    for mid in required:
        assert mid in model_ids


def test_resolve_default_per_capability():
    for cap in Capability:
        spec = REGISTRY.resolve_default(cap)
        assert spec is None or spec.capability == cap
