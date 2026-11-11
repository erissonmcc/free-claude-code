"""Google Vertex AI provider package."""

__all__ = ["VertexProvider"]


def __getattr__(name: str):
    if name != "VertexProvider":
        raise AttributeError(name)
    from .client import VertexProvider

    return VertexProvider
