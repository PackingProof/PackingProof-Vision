from __future__ import annotations

import onnx


def sort_onnx_graph(graph: onnx.GraphProto) -> None:
    available = {
        value.name
        for value in [*graph.input, *graph.initializer]
        if value.name
    }
    pending = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        ready = [
            node
            for node in pending
            if all(not name or name in available for name in node.input)
        ]
        if not ready:
            blocked = ", ".join(node.name or node.op_type for node in pending[:5])
            raise ValueError(f"ONNX graph contains a dependency cycle near: {blocked}")
        for node in ready:
            ordered.append(node)
            available.update(name for name in node.output if name)
        ready_ids = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in ready_ids]

    graph.ClearField("node")
    graph.node.extend(ordered)
