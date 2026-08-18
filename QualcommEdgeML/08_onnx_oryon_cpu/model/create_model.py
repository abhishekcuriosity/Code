from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "tiny_cnn_fp32.onnx"
rng = np.random.default_rng(42)

def init(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)

input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,1,8,8])
output_info = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1,3])

conv1_w = init("conv1_w", rng.normal(0, 0.15, (4,1,3,3)))
conv1_b = init("conv1_b", rng.normal(0, 0.05, (4,)))
conv2_w = init("conv2_w", rng.normal(0, 0.15, (8,4,3,3)))
conv2_b = init("conv2_b", rng.normal(0, 0.05, (8,)))
fc_w = init("fc_w", rng.normal(0, 0.15, (8,3)))
fc_b = init("fc_b", rng.normal(0, 0.05, (3,)))

nodes = [
    helper.make_node("Conv", ["input","conv1_w","conv1_b"], ["conv1_out"], name="Conv1",
                     kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1]),
    helper.make_node("Relu", ["conv1_out"], ["relu1_out"], name="Relu1"),
    helper.make_node("MaxPool", ["relu1_out"], ["pool1_out"], name="MaxPool1",
                     kernel_shape=[2,2], strides=[2,2]),
    helper.make_node("Conv", ["pool1_out","conv2_w","conv2_b"], ["conv2_out"], name="Conv2",
                     kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1]),
    helper.make_node("Relu", ["conv2_out"], ["relu2_out"], name="Relu2"),
    helper.make_node("GlobalAveragePool", ["relu2_out"], ["gap_out"], name="GlobalAveragePool"),
    helper.make_node("Flatten", ["gap_out"], ["flatten_out"], name="Flatten", axis=1),
    helper.make_node("Gemm", ["flatten_out","fc_w","fc_b"], ["logits"], name="Classifier"),
    helper.make_node("Softmax", ["logits"], ["probabilities"], name="Softmax", axis=1),
]

graph = helper.make_graph(
    nodes, "QualcommTinyCNN", [input_info], [output_info],
    initializer=[conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b]
)
model = helper.make_model(graph, producer_name="QualcommEdgeML_Experiment08",
                          opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 10
onnx.checker.check_model(model)
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, MODEL_PATH)

params = sum(numpy_helper.to_array(t).size for t in model.graph.initializer)
bytes_ = sum(numpy_helper.to_array(t).nbytes for t in model.graph.initializer)
print("==============================================")
print("Experiment 08 - ONNX model generated")
print("==============================================")
print("Model        :", MODEL_PATH)
print("Graph        :", model.graph.name)
print("Nodes        :", len(model.graph.node))
print("Initializers :", len(model.graph.initializer))
print("Parameters   :", params)
print("Weight bytes :", bytes_)
print("Input        : [1, 1, 8, 8]")
print("Output       : [1, 3]")
