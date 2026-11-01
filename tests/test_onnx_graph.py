from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from onnx_graph import sort_onnx_graph


def test_sort_onnx_graph_orders_dependencies() -> None:
    cast = helper.make_node(
        "Cast",
        ["input"],
        ["graph_input_cast_0"],
        to=TensorProto.FLOAT16,
    )
    relu = helper.make_node("Relu", ["graph_input_cast_0"], ["output"])
    graph = helper.make_graph(
        [relu, cast],
        "reversed-graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    expect_invalid = False
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        expect_invalid = True
    assert expect_invalid

    sort_onnx_graph(model.graph)
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Cast", "Relu"]
