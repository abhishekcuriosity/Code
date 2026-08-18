from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model" / "tiny_cnn_fp32.onnx"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

x = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(1,1,8,8)
model = onnx.load(MODEL)
y = np.asarray(ReferenceEvaluator(model).run(None, {"input": x})[0], dtype=np.float32)

x.tofile(DATA / "input.bin")
y.tofile(DATA / "golden.bin")

print("Input [1,1,8,8]:")
print(x.reshape(8,8))
print("\nGolden output [1,3]:")
for i, v in enumerate(y.reshape(-1)):
    print(f"class[{i}] = {v:.9f}")
print(f"Probability sum = {float(y.sum()):.9f}")
print("\nGenerated:")
print(DATA / "input.bin")
print(DATA / "golden.bin")
