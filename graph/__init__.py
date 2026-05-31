__all__ = ["build_graph"]


def __getattr__(name):
    if name == "build_graph":
        from graph.workflow import build_graph
        return build_graph
    raise AttributeError(f"module 'graph' has no attribute {name!r}")
