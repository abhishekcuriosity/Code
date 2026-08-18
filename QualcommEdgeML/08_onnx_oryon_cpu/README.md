Experiment 08 — ONNX Anatomy and Native ARM64 CPU Inference on Qualcomm Oryon

Qualcomm Edge ML Learning Track

Target: Windows 11 ARM64 on Snapdragon X
CPU: Qualcomm Oryon
Runtime: ONNX Runtime 1.29.0
Execution Provider: CPU EP
Build: Release | ARM64
IDE: Visual Studio
Model: FP32 ONNX tiny CNN

1. Why this experiment exists

Experiments 1–7 stayed close to the hardware:

scalar dot product
    ↓
NEON
    ↓
SDOT
    ↓
fully connected layer

Experiment 08 introduces the runtime abstraction:

Neural-network graph
        ↓
      ONNX
        ↓
 ONNX Runtime
        ↓
CPU Execution Provider
        ↓
optimized CPU kernels
        ↓
ARM64 / Oryon

The goal is not merely to run an ONNX model. The goal is to understand what exists between a graph operator such as Conv or Gemm and the ARM64 execution underneath it.

2. Learning objectives

This experiment should make the following distinctions clear:

Model architecture
      vs
ONNX serialized graph
      vs
runtime optimization / execution plan
      vs
machine instructions

It also introduces the Execution Provider abstraction, independent golden-output verification, and graph optimization OFF vs ON.

3. Model architecture

The model is intentionally tiny so every node can be inspected:

Input [1,1,8,8]
      ↓
Conv 1→4, 3×3, pad=1, stride=1
      ↓
[1,4,8,8]
      ↓
ReLU
      ↓
MaxPool 2×2
      ↓
[1,4,4,4]
      ↓
Conv 4→8, 3×3, pad=1, stride=1
      ↓
[1,8,4,4]
      ↓
ReLU
      ↓
GlobalAveragePool
      ↓
[1,8,1,1]
      ↓
Flatten
      ↓
[1,8]
      ↓
Gemm 8→3
      ↓
[1,3]
      ↓
Softmax
      ↓
3 probabilities

Logical node count:

1. Conv
2. ReLU
3. MaxPool
4. Conv
5. ReLU
6. GlobalAveragePool
7. Flatten
8. Gemm
9. Softmax

4. Parameter count

Conv1

Weights = 4 × 1 × 3 × 3 = 36
Bias    = 4
Total   = 40

Conv2

Weights = 8 × 4 × 3 × 3 = 288
Bias    = 8
Total   = 296

Gemm

Weights = 8 × 3 = 24
Bias    = 3
Total   = 27

Total

40 + 296 + 27 = 363 parameters
363 × 4 bytes = 1452 bytes of FP32 parameter data

Approximate major MACs:

Conv1 = 8×8×4×9    = 2304
Conv2 = 4×4×8×36   = 4608
Gemm  = 8×3        = 24

Total ≈ 6936 MACs

The model is tiny by design; observability is more important here than workload size.

5. Connection to Experiments 1–7

A convolution output is fundamentally built from dot products.

For a simple 3×3 single-channel convolution:

9 input values
      ·
9 kernel weights
      ↓
dot product
      ↓
one output value

With multiple channels, the dot product becomes larger.

Therefore the progression is:

scalar dot product
      ↓
NEON dot product
      ↓
SDOT
      ↓
fully connected layer
      ↓
convolution
      ↓
ONNX Conv operator

The new abstraction does not replace the earlier learning; it sits above it.

6. What ONNX is

ONNX is a hardware-neutral model representation.

It does not contain ARM64, NEON, SDOT, QNN, GPU shader, or Hexagon instructions.

Conceptually an ONNX model contains:

ModelProto
   └── GraphProto
         ├── graph inputs
         ├── graph outputs
         ├── NodeProto[]
         ├── TensorProto[] initializers
         └── shape/type metadata

A Conv node says what operation must happen. It does not prescribe the exact machine implementation.

7. What ONNX Runtime does

ONNX Runtime sits between the portable graph and hardware execution:

ONNX model
    ↓
parse
    ↓
validate / infer metadata
    ↓
graph optimization
    ↓
Execution Provider assignment
    ↓
kernel selection
    ↓
memory planning
    ↓
execution

In Experiment 08:

Execution Provider = CPU

so the execution path is:

ONNX
  ↓
ONNX Runtime
  ↓
CPU EP
  ↓
Oryon CPU

This is the baseline before Qualcomm-specific backends are introduced later.

8. Execution Provider concept

The application talks to ONNX Runtime while the backend can vary.

Conceptually:

                 ┌── CPU EP ──► Oryon
ONNX Runtime ────┼── GPU EP ──► GPU
                 └── QNN EP ──► Qualcomm QNN backend

Understanding this abstraction is essential before studying QNN graph mapping, accelerator support, partitioning, and fallback.

9. Project layout

08_onnx_oryon_cpu/
│
├── main.cpp
├── README.md
├── 08_onnx_oryon_cpu.vcxproj
├── 08_onnx_oryon_cpu.vcxproj.filters
├── 08_onnx_oryon_cpu.vcxproj.user
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

ONNX Runtime is installed through NuGet:

Microsoft.ML.OnnxRuntime 1.29.0

This removes the need to manually maintain ORT include/lib folders.

10. Native ARM64 build

Visual Studio configuration:

Configuration : Release
Platform      : ARM64

The executable itself reports:

Build architecture : ARM64

The intended chain is:

Visual Studio
    ↓
ARM64-targeted MSVC build
    ↓
ARM64 executable
    ↓
Windows ARM64
    ↓
Qualcomm Oryon

11. Model creation

The model is constructed manually in Python rather than exported from PyTorch or TensorFlow.

Run:

python model\create_model.py

This creates:

model\tiny_cnn_fp32.onnx

Manual construction forces us to understand the graph itself instead of treating ONNX export as a black box.

12. Inspecting the ONNX model

Run:

python model\inspect_model.py model\tiny_cnn_fp32.onnx

The script exposes:

IR version
opset
graph name
inputs / outputs
initializer shapes
parameter counts
node names
operator types
node inputs / outputs
attributes
inferred shapes

This is the first important learning artifact of the experiment.

13. Independent golden reference

Run:

python model\run_reference.py

The deterministic input is:

64 FP32 values linearly spaced from -1 to +1
reshaped to [1,1,8,8]

The script creates:

data\input.bin
data\golden.bin

The native C++ program consumes the same input and compares its result to golden.bin.

This separates runtime correctness from benchmark performance.

14. Native C++ inference

The C++ application:

creates ORT environment
      ↓
creates SessionOptions
      ↓
loads ONNX model
      ↓
creates [1,1,8,8] tensor
      ↓
Session::Run()
      ↓
gets [1,3] output
      ↓
compares against golden.bin
      ↓
benchmarks repeated inference

It runs two configurations:

ORT_DISABLE_ALL
ORT_ENABLE_ALL

The optimized case also serializes:

tiny_cnn_optimized.onnx

15. Benchmark methodology

For each configuration the application performs:

initial run
    ↓
warm-up runs
    ↓
7 measured trials
    ↓
many inference iterations per trial
    ↓
median latency

The median is used to reduce sensitivity to occasional scheduling noise.

16. Actual measured results

Runtime:

Build architecture : ARM64
ONNX Runtime       : 1.29.0
Execution Provider : CPU

Optimization OFF

Trial 1 : 8.024 us/inference
Trial 2 : 8.053 us/inference
Trial 3 : 8.047 us/inference
Trial 4 : 8.078 us/inference
Trial 5 : 8.087 us/inference
Trial 6 : 8.020 us/inference
Trial 7 : 8.031 us/inference

Median  : 8.047 us

Optimization ON

Trial 1 : 7.209 us/inference
Trial 2 : 7.269 us/inference
Trial 3 : 7.238 us/inference
Trial 4 : 7.344 us/inference
Trial 5 : 7.441 us/inference
Trial 6 : 7.357 us/inference
Trial 7 : 7.404 us/inference

Median  : 7.344 us

Summary

Optimization OFF : 8.047 us
Optimization ON  : 7.344 us
Speedup          : 1.096x

The optimized runtime path improved median latency by roughly 9.6% for this tiny model.

The absolute speedup is not the main point. The key observation is:

same model
same input
same hardware
same CPU EP
        ↓
different graph optimization policy
        ↓
different execution performance

17. Correctness

Both modes produced:

class[0] = 0.316699147
class[1] = 0.331867039
class[2] = 0.351433784

Golden comparison:

Max error    : 2.980232239e-08
Verification : PASS

The difference is only FP32-scale numerical variation.

Therefore optimization changed execution behavior/performance without materially changing model semantics.

18. Original vs optimized graph

Two artifacts now exist:

tiny_cnn_fp32.onnx
tiny_cnn_optimized.onnx

The first is the portable graph authored by us.

The second is serialized after ONNX Runtime optimization.

This lets us compare:

what the model author expressed
            vs
what the runtime transformed

That distinction will become central later when studying QNN graph preparation and backend mapping.

19. NCHWc warning

With ORT_ENABLE_ALL, ONNX Runtime printed:

Serializing optimized model with Graph Optimization level greater
than ORT_ENABLE_EXTENDED and the NchwcTransformer enabled.

The generated model may contain hardware specific optimizations,
and should only be used in the same environment the model was
optimized in.

This is valuable information.

The portable model uses the usual CNN layout:

N C H W

CPU kernels may benefit from blocked channel arrangements, conceptually:

NCHW
  ↓
CPU-oriented layout transformation
  ↓
NCHWc

However, the warning alone does not prove that every operator or tensor was converted.

The optimized graph must be inspected before making a precise claim.

20. Inspect both graphs

Original:

python model\inspect_model.py model\tiny_cnn_fp32.onnx

Optimized:

python model\inspect_model.py ..\ARM64\Release\tiny_cnn_optimized.onnx

Compare:

node count
operator types
fusions
removed nodes
new nodes
layout transformations
initializer changes
tensor names
shape changes
runtime-specific operators

Also inspect both files side-by-side in Netron.

21. Graph optimization vs compiler optimization

These are different layers.

Earlier experiments

C++ loop
   ↓
MSVC optimization
   ↓
ARM64 instructions

Experiment 08

ONNX nodes
   ↓
ORT graph optimizer
   ↓
EP/kernel selection
   ↓
native CPU kernels
   ↓
ARM64 machine code

The complete stack is therefore approximately:

ONNX graph optimization
        ↓
runtime scheduling / kernel selection
        ↓
native CPU implementation
        ↓
compiler-generated ARM64 code
        ↓
Oryon execution

22. Model-level vs implementation-level optimization

Do not confuse:

standard convolution
vs
depthwise-separable convolution

with:

direct convolution
im2col + GEMM
implicit GEMM
blocked layouts
specialized vector kernels

The first category changes the model architecture.

The second changes how the same mathematical operator is implemented.

For example:

depthwise separable = model/algorithm choice
im2col             = implementation strategy

23. Why im2col connects to the earlier work

A convolution can be transformed into matrix multiplication:

sliding input patches
      ↓
flatten / rearrange
      ↓
matrix
      ×
kernel matrix
      ↓
GEMM

That explains why the dot-product and fully connected experiments matter.

Many high-level neural-network operators eventually reduce to structured multiply-accumulate work.

The ONNX graph does not dictate which convolution strategy the runtime must use.

24. Troubleshooting encountered

Wrong main.cpp location

Visual Studio initially referenced a file under Downloads instead of the project directory.

Lesson:

Folder View = physical filesystem
Solution View / .vcxproj = what Visual Studio actually builds

The correct file was added from:

C:\EmbeddedAI\Qualcomm\QualcommEdgeML\08_onnx_oryon_cpu\main.cpp

Missing ONNX Runtime header

Initial error:

Cannot open include file: 'onnxruntime_cxx_api.h'

Resolved by installing:

Microsoft.ML.OnnxRuntime 1.29.0

through NuGet.

Windows min/max macro collision

The fix was:

#define NOMINMAX
#include <onnxruntime_cxx_api.h>
#include <windows.h>

ORT version API

Incorrect:

Ort::GetApi().GetVersionString()

Working C++ wrapper call:

Ort::GetVersionString()

25. Build-output discipline

For this experiment, rebuild only:

08_onnx_oryon_cpu

instead of Rebuild All, because earlier projects intentionally emit compiler/vectorization diagnostics.

Visual Studio build output verbosity can be set to:

Minimal

under:

Tools
→ Options
→ Projects and Solutions
→ Build and Run
→ MSBuild project build output verbosity

26. What Experiment 08 proved

We demonstrated:

manually constructed ONNX graph
        ↓
portable FP32 model
        ↓
independent reference inference
        ↓
deterministic input / golden output
        ↓
native ARM64 C++ application
        ↓
ONNX Runtime 1.29.0
        ↓
CPU Execution Provider
        ↓
Qualcomm Oryon
        ↓
verified probabilities
        ↓
PASS

We also measured graph optimization:

8.047 us
   ↓
7.344 us

1.096× speedup

with the same verified output.

27. What this experiment does NOT yet prove

It does not yet prove:

which exact ARM64 instructions each ORT Conv kernel uses
which exact nodes were fused
which tensors were transformed to blocked layouts
whether a particular kernel uses NEON or SDOT
how QNN represents this graph
how Adreno executes it
how Hexagon/HTP executes it

Those require graph inspection, profiling, disassembly, and later Qualcomm experiments.

Measured facts must remain separate from assumptions about internal implementation.

28. Why this experiment matters for Qualcomm

The current baseline is:

ONNX
  ↓
ONNX Runtime
  ↓
CPU EP
  ↓
Oryon

Later the path will evolve toward:

ONNX
  ↓
ONNX Runtime
  ↓
QNN EP
  ↓
Qualcomm QNN
  ↓
Qualcomm backend

and direct QNN experiments will expose even more of the stack.

This experiment therefore establishes the neutral runtime baseline before studying:

Adreno
Hexagon / HTP
QNN
QAIRT
graph partitioning
backend selection
fallback
profiling
context binaries
heterogeneous execution

29. Completion status

[✓] ONNX graph constructed manually
[✓] Known tensor shapes and operators
[✓] FP32 ONNX model created
[✓] Independent reference generated
[✓] Native ARM64 C++ application built
[✓] ONNX Runtime integrated
[✓] CPU EP used
[✓] ARM64 execution confirmed
[✓] Golden verification PASS
[✓] Optimization OFF measured
[✓] Optimization ON measured
[✓] Optimized graph serialized
[ ] Original and optimized graphs compared node-by-node
[ ] Optimized graph inspected visually

30. Interview-ready explanation

In Experiment 08 I moved from manually optimized ARM64 kernels to a hardware-neutral ONNX graph. I built a small CNN directly with ONNX, generated an independent golden output, and executed it using native ARM64 C++ with ONNX Runtime's CPU Execution Provider on Qualcomm Oryon. I compared graph optimization disabled and enabled. Median latency improved from 8.047 microseconds to 7.344 microseconds, about 1.096×, while matching the golden output with only about 3e-8 FP32 maximum error. ORT also reported the NCHWc transformer while serializing the optimized model, which showed that the runtime may make CPU/environment-specific graph decisions. The main learning was the boundary between the portable ONNX graph, runtime graph optimization, Execution Provider kernel selection, and actual ARM64 execution.

Key takeaway

Experiments 1–7 asked:

How do I make the arithmetic itself fast on ARM64?

Experiment 08 asks:

What happens when I provide a neural-network graph
and allow a runtime to decide how it executes?

The key transition is:

I write machine-oriented kernels
            ↓
I describe a neural-network graph
            ↓
the runtime decides how it executes

That is the bridge from low-level ARM64 optimization to modern Edge AI runtime architecture.

Current baseline

Experiment           : 08 — ONNX Anatomy + Oryon CPU Baseline
Runtime              : ONNX Runtime 1.29.0
Execution Provider   : CPU
Architecture         : ARM64
Parameters           : 363
Approx. major MACs   : 6936
ORT optimization OFF : 8.047 us
ORT optimization ON  : 7.344 us
Speedup              : 1.096×
Max error            : 2.980232239e-08
Verification         : PASS
Status               : FUNCTIONALLY WORKING
Graph diff analysis  : PENDING