from hub.adapters.cylinder3d_adapter import Cylinder3DAdapter
from hub.adapters.pointtransformer_adapter import PointTransformerAdapter


REGISTRY = {
    "cylinder3d": Cylinder3DAdapter,
    "pointtransformer": PointTransformerAdapter,
}


def get_adapter(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return REGISTRY[name]()