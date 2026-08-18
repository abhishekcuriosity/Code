# Experiment 08 — ONNX Anatomy & Native ARM64 Inference on Qualcomm Oryon

> **Goal:** Understand how a hardware-neutral ONNX graph becomes native ARM64 inference through ONNX Runtime's CPU Execution Provider, and measure what ONNX Runtime graph optimization changes on Qualcomm Oryon.

---

## Experiment Snapshot

| Item | Configuration |
|---|---|
| Platform | Snapdragon X laptop |
| OS | Windows 11 ARM64 |
| CPU | Qualcomm Oryon |
| Runtime | ONNX Runtime 1.29.0 |
| Execution Provider | CPU EP |
| Build | Release / ARM64 |
| IDE | Visual Studio |
| Model | FP32 Tiny CNN |
| Parameters | 363 |
| Approx. major MACs | 6,936 |
| Golden verification | **PASS** |
| ORT opt OFF median | **8.047 µs** |
| ORT opt ON median | **7.344 µs** |
| Measured speedup | **1.096×** |

---

## Why This Experiment Exists

Experiments 1–7 stayed close to the machine.

```text
Scalar Dot Product
        │
        ▼
     NEON SIMD
        │
        ▼
       SDOT
        │
        ▼
Fully Connected Layer
```

In those experiments, we explicitly controlled how the arithmetic was implemented.

Experiment 08 changes the abstraction level.

```text
Neural-Network Graph
        │
        ▼
      ONNX
        │
        ▼
   ONNX Runtime
        │
        ▼
CPU Execution Provider
        │
        ▼
 Optimized CPU Kernels
        │
        ▼
   ARM64 / Oryon
```

The objective is not simply:

> "Can ONNX Runtime run a model?"

The objective is:

> "What exists between a graph operator such as `Conv` or `Gemm` and the actual ARM64 execution underneath it?"

---

# 1. The Core Conceptual Transition

Earlier experiments:

```text
Programmer chooses implementation
        │
        ├── Scalar
        ├── NEON
        └── SDOT
               │
               ▼
        ARM64 execution
```

Experiment 08:

```text
Programmer provides graph
        │
        ▼
      ONNX
        │
        ▼
  ONNX Runtime
        │
        ▼
Execution Provider
        │
        ▼
 Runtime-selected kernels
        │
        ▼
    ARM64 execution
```

This is the key transition from **hand-written machine-oriented kernels** to a **runtime-driven Edge AI software stack**.

---

# 2. End-to-End Architecture

```text
┌────────────────────────────────────┐
│         tiny_cnn_fp32.onnx         │
│     Hardware-neutral ML graph      │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│          ONNX Runtime 1.29.0       │
│                                    │
│  Parse                             │
│  Validate                          │
│  Optimize                          │
│  Assign Execution Provider         │
│  Select kernels                    │
│  Plan memory                       │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│       CPU Execution Provider       │
│                                    │
│        CPU operator kernels        │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│        Native ARM64 Execution      │
│                                    │
│         Qualcomm Oryon CPU         │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│          3-class FP32 output       │
│                                    │
│      compared with golden.bin      │
└────────────────────────────────────┘
```

---

# 3. Project Layout

```text
08_onnx_oryon_cpu/
│
├── main.cpp
├── README.md
├── 08_onnx_oryon_cpu.vcxproj
├── 08_onnx_oryon_cpu.vcxproj.filters
│
├── model/
│   ├── create_model.py
│   ├── inspect_model.py
│   ├── run_reference.py
│   └── tiny_cnn_fp32.onnx
│
├── data/
│   ├── input.bin
│   └── golden.bin
│
└── scripts/
```

ONNX Runtime is integrated through NuGet:

```text
Microsoft.ML.OnnxRuntime 1.29.0
```

The repository should keep the project metadata / package declaration, but not commit the generated NuGet `packages/` cache.

---

# 4. Model Architecture

A deliberately tiny CNN is used so that every operator remains understandable.

```text
Input
[1,1,8,8]
   │
   ▼
Conv
1 → 4
3×3
pad=1
stride=1
   │
   ▼
[1,4,8,8]
   │
   ▼
ReLU
   │
   ▼
MaxPool 2×2
   │
   ▼
[1,4,4,4]
   │
   ▼
Conv
4 → 8
3×3
pad=1
stride=1
   │
   ▼
[1,8,4,4]
   │
   ▼
ReLU
   │
   ▼
GlobalAveragePool
   │
   ▼
[1,8,1,1]
   │
   ▼
Flatten
   │
   ▼
[1,8]
   │
   ▼
Gemm
8 → 3
   │
   ▼
[1,3]
   │
   ▼
Softmax
   │
   ▼
3 probabilities
```

Logical node sequence:

```text
1. Conv
2. ReLU
3. MaxPool
4. Conv
5. ReLU
6. GlobalAveragePool
7. Flatten
8. Gemm
9. Softmax
```

---

# 5. Parameter Count

## Conv1

```text
Input channels  = 1
Output channels = 4
Kernel          = 3×3

Weights = 4 × 1 × 3 × 3 = 36
Bias    = 4

Total   = 40 parameters
```

## Conv2

```text
Input channels  = 4
Output channels = 8
Kernel          = 3×3

Weights = 8 × 4 × 3 × 3 = 288
Bias    = 8

Total   = 296 parameters
```

## Gemm

```text
Input features  = 8
Output features = 3

Weights = 8 × 3 = 24
Bias    = 3

Total   = 27 parameters
```

## Entire Model

```text
40 + 296 + 27 = 363 parameters
```

FP32 parameter storage:

```text
363 × 4 bytes = 1452 bytes
```

So the learned parameter payload is only about **1.4 KB**.

---

# 6. Approximate MAC Count

## Conv1

```text
Output spatial size = 8×8
Output channels     = 4
MACs/output         = 1×3×3 = 9

Total = 8×8×4×9
      = 2304 MACs
```

## Conv2

```text
Output spatial size = 4×4
Output channels     = 8
MACs/output         = 4×3×3 = 36

Total = 4×4×8×36
      = 4608 MACs
```

## Gemm

```text
8 × 3 = 24 MACs
```

## Approximate Total

```text
2304 + 4608 + 24 = 6936 major MACs
```

The network is intentionally tiny. This experiment is about **visibility into the runtime stack**, not model scale.

---

# 7. Connecting Conv Back to Dot Product

A convolution output value is fundamentally a dot product.

For one 3×3 single-channel patch:

```text
Input patch
┌───┬───┬───┐
│x0 │x1 │x2 │
├───┼───┼───┤
│x3 │x4 │x5 │
├───┼───┼───┤
│x6 │x7 │x8 │
└───┴───┴───┘

      dot

Kernel
┌───┬───┬───┐
│w0 │w1 │w2 │
├───┼───┼───┤
│w3 │w4 │w5 │
├───┼───┼───┤
│w6 │w7 │w8 │
└───┴───┴───┘
```

Conceptually:

```text
x0*w0 +
x1*w1 +
...
x8*w8
    │
    ▼
one convolution output
```

Therefore:

```text
ONNX Conv
    │
    ▼
Convolution kernel
    │
    ▼
Many structured dot products
    │
    ▼
Multiply-accumulate work
```

This is why the earlier scalar / NEON / SDOT experiments matter.

---

# 8. What ONNX Actually Contains

ONNX is not executable ARM64 code.

It does **not** directly contain:

```text
NEON intrinsics
SDOT instructions
Oryon-specific instructions
QNN API calls
GPU shaders
Hexagon instructions
```

It is a hardware-independent graph representation.

Conceptually:

```text
ModelProto
   │
   └── GraphProto
         │
         ├── Inputs
         ├── Outputs
         ├── NodeProto[]
         ├── TensorProto[] initializers
         └── shape / type metadata
```

A `Conv` node represents:

```text
Operator : Conv

Inputs:
    activation tensor
    weights
    bias

Attributes:
    kernel
    padding
    stride

Output:
    activation tensor
```

It describes **what computation must happen**.

It does not prescribe **how the CPU must implement it**.

---

# 9. Why We Build the ONNX Graph Manually

The model is constructed directly in Python rather than exported from PyTorch or TensorFlow.

Run:

```cmd
python model\create_model.py
```

This creates:

```text
model\tiny_cnn_fp32.onnx
```

That design is intentional.

If we simply exported a model from a high-level framework, we could run the model without really understanding what was serialized.

Here we explicitly create:

```text
nodes
initializers
tensor shapes
attributes
opset
graph inputs
graph outputs
```

so ONNX itself becomes part of the learning.

---

# 10. Inspecting the Graph

Run:

```cmd
python model\inspect_model.py model\tiny_cnn_fp32.onnx
```

The script exposes:

```text
IR version
opset version
graph name
inputs
outputs
initializers
initializer shapes
parameter counts
operator names
node inputs
node outputs
attributes
inferred shapes
```

The important mental model is:

```text
.onnx file
   ≠
opaque binary blob

.onnx file
   =
serialized computation graph
```

---

# 11. Independent Golden Reference

Before benchmarking the native runtime, a reference result is generated independently.

Run:

```cmd
python model\run_reference.py
```

The input is deterministic:

```text
64 FP32 values
linearly spaced from -1 to +1
reshaped to [1,1,8,8]
```

The script produces:

```text
data\input.bin
data\golden.bin
```

The purpose is:

```text
input.bin
    │
    └── same input consumed by C++

golden.bin
    │
    └── expected reference output
```

This lets us separate two questions:

```text
Is the runtime correct?
        vs
Is the runtime fast?
```

---

# 12. Native ARM64 Build

Visual Studio configuration:

```text
Configuration : Release
Platform      : ARM64
```

The application reports:

```text
Build architecture : ARM64
```

The intended execution chain is:

```text
Visual Studio
     │
     ▼
MSVC ARM64 build
     │
     ▼
ARM64 executable
     │
     ▼
Windows ARM64
     │
     ▼
Qualcomm Oryon
```

This gives a native Oryon CPU baseline before moving to Qualcomm accelerator backends.

---

# 13. What ONNX Runtime Does

ONNX Runtime is the software layer between the graph and the execution backend.

Conceptually:

```text
ONNX model
   │
   ▼
Parse graph
   │
   ▼
Validate graph
   │
   ▼
Optimize graph
   │
   ▼
Assign nodes to Execution Provider
   │
   ▼
Select operator kernels
   │
   ▼
Plan memory
   │
   ▼
Execute graph
```

For Experiment 08:

```text
Execution Provider = CPU
```

So:

```text
ONNX
  │
  ▼
ONNX Runtime
  │
  ▼
CPU EP
  │
  ▼
Oryon CPU
```

---

# 14. What Is an Execution Provider?

The Execution Provider abstraction lets ONNX Runtime delegate graph execution to different backends.

Conceptually:

```text
                 ┌──────────────► CPU EP ─────► Oryon
                 │
ONNX Runtime ────┼──────────────► GPU EP ─────► GPU
                 │
                 └──────────────► QNN EP ─────► Qualcomm QNN
```

This is one of the most important concepts before learning QNN.

The graph-facing application can remain similar while the backend implementation changes.

---

# 15. Native C++ Runtime Flow

The C++ application performs:

```text
Create ORT environment
        │
        ▼
Create SessionOptions
        │
        ▼
Load model
        │
        ▼
Create [1,1,8,8] input tensor
        │
        ▼
Session::Run()
        │
        ▼
Receive [1,3] output
        │
        ▼
Compare with golden.bin
        │
        ▼
Benchmark repeated inference
```

Two runtime configurations are compared.

---

# 16. Optimization OFF

Configuration:

```text
GraphOptimizationLevel = ORT_DISABLE_ALL
```

Measured results:

```text
Trial 1 : 8.024 us/inference
Trial 2 : 8.053 us/inference
Trial 3 : 8.047 us/inference
Trial 4 : 8.078 us/inference
Trial 5 : 8.087 us/inference
Trial 6 : 8.020 us/inference
Trial 7 : 8.031 us/inference
```

Median:

```text
8.047 us
```

This serves as the baseline.

---

# 17. Optimization ON

Configuration:

```text
GraphOptimizationLevel = ORT_ENABLE_ALL
```

Measured results:

```text
Trial 1 : 7.209 us/inference
Trial 2 : 7.269 us/inference
Trial 3 : 7.238 us/inference
Trial 4 : 7.344 us/inference
Trial 5 : 7.441 us/inference
Trial 6 : 7.357 us/inference
Trial 7 : 7.404 us/inference
```

Median:

```text
7.344 us
```

---

# 18. Benchmark Summary

| Runtime mode | Median latency |
|---|---:|
| ORT optimization OFF | 8.047 µs |
| ORT optimization ON | **7.344 µs** |
| Speedup | **1.096×** |

Approximate latency improvement:

```text
8.047 - 7.344 = 0.703 us
```

Relative improvement:

```text
≈ 9.6%
```

The important conclusion is:

```text
Same model
Same input
Same CPU
Same Execution Provider
        │
        ▼
Different runtime graph optimization policy
        │
        ▼
Different measured latency
```

---

# 19. Correctness Verification

Both configurations produced:

```text
class[0] = 0.316699147
class[1] = 0.331867039
class[2] = 0.351433784
```

Maximum absolute difference from the golden output:

```text
2.980232239e-08
```

Result:

```text
Verification : PASS
```

That is effectively FP32-scale numerical variation.

So:

```text
Graph optimization changed execution performance
                │
                ▼
but preserved model output
```

---

# 20. Original Model vs Runtime-Optimized Model

The experiment now has two important graph artifacts:

```text
tiny_cnn_fp32.onnx
```

and:

```text
tiny_cnn_optimized.onnx
```

Conceptually:

```text
What we authored
      │
      ▼
tiny_cnn_fp32.onnx
      │
      ▼
ONNX Runtime optimizer
      │
      ▼
tiny_cnn_optimized.onnx
      │
      ▼
What the runtime chose to execute
```

This distinction becomes extremely important later for:

```text
QNN graph mapping
subgraph partitioning
unsupported-op fallback
backend-specific graph rewriting
accelerator scheduling
```

---

# 21. NCHWc Warning — Why It Matters

When optimization was enabled, ONNX Runtime printed:

```text
Serializing optimized model with Graph Optimization level greater
than ORT_ENABLE_EXTENDED and the NchwcTransformer enabled.

The generated model may contain hardware specific optimizations,
and should only be used in the same environment the model was
optimized in.
```

The original model uses the usual layout:

```text
N C H W
```

For example:

```text
[1,4,8,8]
```

CPU implementations may choose blocked channel layouts for better vector/cache behavior.

Conceptually:

```text
NCHW
  │
  ▼
CPU-oriented layout transform
  │
  ▼
NCHWc
```

Important:

> The warning alone does not prove that every tensor or every operator was rewritten.

The optimized graph must be inspected before making a precise claim about what changed.

---

# 22. Inspect the Optimized Graph

Original model:

```cmd
python model\inspect_model.py model\tiny_cnn_fp32.onnx
```

Optimized model:

```cmd
python model\inspect_model.py ..\ARM64\Release\tiny_cnn_optimized.onnx
```

The comparison should answer:

```text
Did node count change?
Were Conv + ReLU fused?
Did operator names change?
Were new layout-transform nodes inserted?
Were initializers changed?
Were tensor names rewritten?
Are CPU-specific operators visible?
```

Also inspect both graphs side by side in Netron.

---

# 23. Graph Optimization vs Compiler Optimization

These are different abstraction layers.

## Earlier experiments

```text
C++ loop
   │
   ▼
MSVC optimizer
   │
   ▼
ARM64 instructions
```

Examples:

```text
scalar
auto-vectorized
NEON
SDOT
```

## Experiment 08

```text
ONNX graph
   │
   ▼
ORT graph optimizer
   │
   ▼
Execution Provider / kernel selection
   │
   ▼
native CPU implementation
   │
   ▼
ARM64 machine code
```

So the stack is approximately:

```text
Model graph optimization
        │
        ▼
Runtime scheduling / kernel selection
        │
        ▼
Native CPU kernel implementation
        │
        ▼
Compiler-generated machine code
        │
        ▼
Oryon execution units
```

---

# 24. Model-Level vs Implementation-Level Optimization

Do not confuse these two classes of optimization.

## Model / architecture change

Example:

```text
Standard Conv
      vs
Depthwise-Separable Conv
```

This changes the network structure itself.

## Runtime implementation change

Examples:

```text
Direct convolution
im2col + GEMM
implicit GEMM
blocked tensor layout
specialized vector kernel
```

These can implement the same mathematical operator in different ways.

Therefore:

```text
Depthwise separable
    = model-level decision

im2col
    = implementation-level strategy
```

---

# 25. Why im2col Connects to the Earlier Experiments

A convolution can be rearranged into matrix multiplication.

```text
Sliding input patches
        │
        ▼
Rearrange / flatten patches
        │
        ▼
Matrix
        │
        ×
Kernel matrix
        │
        ▼
GEMM
```

This is one reason why dot-product and fully connected experiments are foundational.

Many high-level neural-network operators ultimately reduce to structured multiply-accumulate work.

The ONNX graph does not force one implementation strategy.

That choice lives below the graph abstraction.

---

# 26. Troubleshooting We Hit

## 26.1 Visual Studio compiled the wrong `main.cpp`

At one point Visual Studio referenced a source file from Downloads rather than the actual project directory.

Key lesson:

```text
Folder View
     =
physical filesystem

Solution View / .vcxproj
     =
what Visual Studio actually builds
```

The correct source file was added from:

```text
C:\EmbeddedAI\Qualcomm\QualcommEdgeML\08_onnx_oryon_cpu\main.cpp
```

---

## 26.2 Missing ONNX Runtime header

Initial failure:

```text
Cannot open include file:
'onnxruntime_cxx_api.h'
```

Resolved by installing:

```text
Microsoft.ML.OnnxRuntime 1.29.0
```

through NuGet.

---

## 26.3 Windows `min` / `max` macro collision

The compile issue was fixed with:

```cpp
#define NOMINMAX
#include <onnxruntime_cxx_api.h>
#include <windows.h>
```

This prevents Windows macros from breaking standard C++ usages such as:

```cpp
std::max(...)
```

---

## 26.4 ONNX Runtime version API mismatch

Incorrect:

```cpp
Ort::GetApi().GetVersionString()
```

Working call:

```cpp
Ort::GetVersionString()
```

---

# 27. Build Discipline

For Experiment 08, rebuild only:

```text
08_onnx_oryon_cpu
```

instead of rebuilding the entire solution.

Earlier experiments intentionally emit vectorization diagnostics, so `Rebuild All` makes the output unnecessarily noisy.

Visual Studio build verbosity:

```text
Tools
→ Options
→ Projects and Solutions
→ Build and Run
→ MSBuild project build output verbosity
→ Minimal
```

---

# 28. What We Actually Proved

Experiment 08 demonstrates this full chain:

```text
Manually constructed ONNX graph
        │
        ▼
Portable FP32 model
        │
        ▼
Independent reference evaluation
        │
        ▼
Deterministic input + golden output
        │
        ▼
Native ARM64 C++ application
        │
        ▼
ONNX Runtime 1.29.0
        │
        ▼
CPU Execution Provider
        │
        ▼
Qualcomm Oryon
        │
        ▼
3-class FP32 output
        │
        ▼
Golden comparison
        │
        ▼
PASS
```

We then measured:

```text
ORT_DISABLE_ALL
      │
      ▼
8.047 us median
```

versus:

```text
ORT_ENABLE_ALL
      │
      ▼
7.344 us median
```

with:

```text
1.096× speedup
```

and the same verified model output.

---

# 29. What This Experiment Does NOT Yet Prove

We have **not yet proven**:

```text
which exact ARM64 instructions each ORT Conv kernel uses
which exact nodes were fused
which tensors were changed to blocked layouts
whether a given kernel uses NEON
whether a given kernel uses SDOT
how QNN represents this graph
how Adreno executes this graph
how Hexagon / HTP executes this graph
```

Those require additional graph inspection, profiling, disassembly, and later Qualcomm-specific experiments.

This separation between **measured facts** and **assumptions about internal implementation** is important.

---

# 30. Why Experiment 08 Matters for Qualcomm

Current baseline:

```text
ONNX
  │
  ▼
ONNX Runtime
  │
  ▼
CPU EP
  │
  ▼
Oryon
```

Later progression:

```text
ONNX
  │
  ▼
ONNX Runtime
  │
  ▼
QNN EP
  │
  ▼
Qualcomm QNN
  │
  ▼
Qualcomm backend
```

That opens the door to:

```text
Adreno
Hexagon / HTP
QNN
QAIRT
graph partitioning
operator support
fallback
profiling
context binaries
heterogeneous execution
```

Experiment 08 therefore establishes the **portable CPU runtime baseline** before introducing Qualcomm-specific acceleration.

---

# 31. Current Completion Status

```text
[✓] Tiny CNN designed
[✓] ONNX graph constructed manually
[✓] FP32 model serialized
[✓] Shapes and operators known
[✓] Independent reference output generated
[✓] Native ARM64 C++ app built
[✓] ONNX Runtime integrated
[✓] CPU Execution Provider used
[✓] ARM64 execution confirmed
[✓] Golden verification PASS
[✓] Optimization OFF benchmarked
[✓] Optimization ON benchmarked
[✓] Optimized model serialized
[ ] Original vs optimized graph compared node-by-node
[ ] Optimized graph visually inspected
```

---

# 32. Interview-Ready Explanation

> In Experiment 08 I moved from manually optimized ARM64 kernels to a hardware-neutral ONNX graph. I built a small CNN directly using ONNX primitives, generated an independent golden output, and ran it from native ARM64 C++ using ONNX Runtime's CPU Execution Provider on Qualcomm Oryon. I compared graph optimization disabled and enabled. Median latency improved from 8.047 microseconds to 7.344 microseconds, about 1.096×, while the native output matched the golden result with a maximum FP32 error of about 3e-8. ONNX Runtime also reported the NCHWc transformer while serializing the optimized graph, which highlighted the distinction between a portable ONNX model and a runtime-optimized, potentially environment-specific representation. The key learning was the boundary between the model graph, runtime optimizer, Execution Provider, CPU kernels, and actual ARM64 execution.

A shorter version:

> Earlier I manually controlled scalar, NEON and SDOT implementations. In Experiment 08 I instead provided ONNX Runtime with a graph and let the runtime decide how the CPU Execution Provider should execute it. That gives me the baseline needed to later study Qualcomm QNN, graph partitioning, backend selection and accelerator mapping.

---

# 33. Key Takeaway

The most important result is not only:

```text
8.047 us
   ↓
7.344 us
```

The deeper transition is:

```text
I write machine-oriented kernels
            │
            ▼
I describe a neural-network graph
            │
            ▼
A runtime decides how that graph executes
```

That is the bridge from low-level ARM64 optimization to modern Edge AI runtime architecture.

---

## Final Baseline

```text
Experiment           : 08 — ONNX Anatomy + Oryon CPU Baseline
Runtime              : ONNX Runtime 1.29.0
Execution Provider   : CPU
Architecture         : ARM64
Model                : FP32 Tiny CNN
Parameters           : 363
Approx. major MACs   : 6936
ORT optimization OFF : 8.047 us
ORT optimization ON  : 7.344 us
Speedup              : 1.096×
Max error            : 2.980232239e-08
Verification         : PASS
Graph diff analysis  : PENDING
```
