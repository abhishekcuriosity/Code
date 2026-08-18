from pathlib import Path
import sys
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
model_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "model" / "tiny_cnn_fp32.onnx"
model = onnx.load(model_path)
graph = model.graph

def shape_of(v):
    out = []
    for d in v.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append("?")
    return out

shape_map = {}
for v in list(graph.input) + list(graph.value_info) + list(graph.output):
    shape_map[v.name] = shape_of(v)

print("\nMODEL")
print("-----")
print("File      :", model_path)
print("IR version:", model.ir_version)
print("Producer  :", model.producer_name)
print("Graph     :", graph.name)
for x in model.opset_import:
    print("Opset     :", x.domain if x.domain else "ai.onnx", x.version)

print("\nINPUTS")
for v in graph.input:
    print(v.name, shape_map.get(v.name))

print("\nINITIALIZERS")
total_p = total_b = 0
for t in graph.initializer:
    a = numpy_helper.to_array(t)
    total_p += a.size
    total_b += a.nbytes
    print(f"{t.name:12s} shape={str(list(a.shape)):16s} params={a.size:4d} bytes={a.nbytes:4d}")
print("Total parameters  :", total_p)
print("Total weight bytes:", total_b)

print("\nNODES")
for i, n in enumerate(graph.node):
    print(f"\n[{i}] {n.name}")
    print("  op      :", n.op_type)
    print("  inputs  :", list(n.input))
    print("  outputs :", list(n.output))
    for o in n.output:
        if o in shape_map:
            print("  shape   :", shape_map[o])
    for a in n.attribute:
        print("  attr    :", a.name, "=", onnx.helper.get_attribute_value(a))

print("\nOUTPUTS")
for v in graph.output:
    print(v.name, shape_map.get(v.name))
