# Project 04 — Scalar vs NEON vs SDOT Benchmark

## Goal

Measure the real performance difference between three implementations of the same INT8 dot product on a native Windows ARM64 system using a Qualcomm Oryon CPU:

1. Scalar ARM64 using ordinary signed multiply-accumulate.
2. General NEON SIMD using `SMULL + SADALP`.
3. NEON SDOT using the dedicated signed INT8 dot-product instruction.

The mathematical operation is identical in all three cases:

```text
result = Σ input[i] × weight[i]
```

This experiment answers:

> After understanding the assembly, how much speedup do SIMD and the specialized `SDOT` instruction actually provide on the real CPU?

---

## Experiment progression

```text
Project 01
Scalar INT8 dot product
        ↓
LDRSB + MADD

Project 02
General NEON SIMD
        ↓
SMULL + SADALP + ADDV

Project 03
Specialized NEON dot product
        ↓
SDOT + ADDV

Project 04
Benchmark all three
        ↓
Measure real execution-time improvement
```

---

## Project structure

```text
04_dot_product_benchmark/
│
├── kernels.h
├── dot_scalar.cpp
├── dot_neon.cpp
├── dot_sdot.cpp
└── main.cpp
```

The kernels are intentionally placed in separate translation units so that each one can be compiled with its own ISA target.

```text
main.cpp
   │
   ├── dot_scalar.cpp  → scalar ARM64
   ├── dot_neon.cpp    → general NEON
   └── dot_sdot.cpp    → NEON SDOT
```

---

## Common kernel interface

`kernels.h`

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

std::int32_t dot_scalar_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);

std::int32_t dot_neon_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);

std::int32_t dot_sdot_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);
```

All three implement the same contract:

```text
INT8 input vector
+
INT8 weight vector
+
element count
        ↓
INT32 dot-product result
```

---

# Implementation 1 — Scalar ARM64

The scalar implementation uses ordinary C++ and explicitly disables loop vectorization:

```cpp
std::int32_t dot_scalar_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count)
{
    std::int32_t accumulator = 0;

#pragma loop(no_vector)
    for (std::size_t i = 0; i < count; ++i)
    {
        accumulator +=
            static_cast<std::int32_t>(input[i]) *
            static_cast<std::int32_t>(weights[i]);
    }

    return accumulator;
}
```

Expected assembly signature:

```asm
ldrsb
ldrsb
madd
```

Conceptually:

```text
load one signed INT8 input
        ↓
load one signed INT8 weight
        ↓
multiply + accumulate
        ↓
repeat for every element
```

---

# Implementation 2 — General NEON SIMD

The general SIMD implementation loads 16 signed bytes at a time and builds the dot product from ordinary NEON vector primitives.

Core operations:

```cpp
const int8x16_t input_vector =
    vld1q_s8(input + i);

const int8x16_t weight_vector =
    vld1q_s8(weights + i);

const int16x8_t products_low =
    vmull_s8(
        vget_low_s8(input_vector),
        vget_low_s8(weight_vector));

const int16x8_t products_high =
    vmull_s8(
        vget_high_s8(input_vector),
        vget_high_s8(weight_vector));

accumulator =
    vpadalq_s16(accumulator, products_low);

accumulator =
    vpadalq_s16(accumulator, products_high);
```

Expected assembly signature:

```asm
ldr q...
ldr q...
smull
smull
sadalp
sadalp
...
addv
```

Conceptually:

```text
16 INT8 inputs
+
16 INT8 weights
        ↓
SMULL + SMULL
        ↓
16 INT16 products
        ↓
SADALP + SADALP
        ↓
4 INT32 partial accumulators
        ↓
ADDV
        ↓
scalar result
```

---

# Implementation 3 — NEON SDOT

The SDOT implementation uses the Arm signed byte dot-product intrinsic:

```cpp
accumulator =
    vdotq_s32(
        accumulator,
        input_vector,
        weight_vector);
```

Expected assembly signature:

```asm
ldr q...
ldr q...
sdot
...
addv
```

`SDOT` directly performs grouped signed INT8 dot products and accumulates them into four INT32 lanes.

```text
16 INT8 inputs
+
16 INT8 weights
        ↓
       SDOT
        ↓
4 INT32 partial accumulators
        ↓
       ADDV
        ↓
1 scalar result
```

---

# Per-file ISA configuration

The files are compiled separately so that the baseline implementations remain clean.

```text
dot_scalar.cpp → /arch:armv8.0
dot_neon.cpp   → /arch:armv8.0
dot_sdot.cpp   → /arch:armv8.4
```

This gives the intended separation:

```text
dot_scalar.cpp
    ↓
MADD

dot_neon.cpp
    ↓
SMULL + SADALP

dot_sdot.cpp
    ↓
SDOT
```

---

# Benchmark configuration

```text
Elements per dot product : 4096
Iterations per trial     : 10000
Trials                   : 7
Warm-up                  : enabled
Result selection         : median trial
```

Each array contains:

```text
4096 × 1 byte = 4096 bytes
```

So input + weights occupy roughly:

```text
8 KiB
```

This intentionally keeps the working set small so the experiment primarily exposes compute-kernel behavior rather than large-memory streaming effects.

---

# Why benchmark many iterations?

Timing one tiny kernel call would make timer overhead and operating-system noise too significant.

Instead:

```text
START TIMER

dot product
dot product
dot product
...
10000 times

STOP TIMER
```

Then:

```text
time per dot
=
total elapsed time / 10000
```

---

# Why use a median?

Seven trials are collected and sorted, then the middle result is selected.

This reduces sensitivity to occasional noise from:

```text
scheduler activity
interrupts
background applications
frequency changes
other operating-system activity
```

---

# Preventing dead-code elimination

The benchmark accumulates kernel results into a `volatile` sink.

This prevents the optimizer from removing the benchmarked calculations because their results appear unused.

---

# Runtime SDOT capability check

Before invoking the SDOT implementation:

```cpp
IsProcessorFeaturePresent(
    PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
```

is used to verify runtime support.

Observed:

```text
SDOT supported : YES
```

---

# Correctness result

All three kernels produced the same answer:

```text
Scalar result : -9188
NEON result   : -9188
SDOT result   : -9188

Verification  : PASS
```

Therefore the timing comparison is between mathematically equivalent implementations.

---

# Measured performance

Observed:

```text
General NEON vs Scalar : 9.326x
SDOT vs Scalar         : 22.388x
SDOT vs General NEON   : 2.400x
```

Normalized view:

| Implementation | Relative performance |
|---|---:|
| Scalar ARM64 | 1.000x |
| General NEON | 9.326x |
| NEON SDOT | 22.388x |

Visualization:

```text
Scalar ARM64
MADD
│
│  9.326×
▼
General NEON
SMULL + SADALP
│
│  2.400×
▼
NEON SDOT
```

Overall:

```text
Scalar → SDOT = 22.388×
```

---

# Interpretation 1 — SIMD gives a major speedup

The scalar kernel processes one input-weight pair at a time:

```text
pair 0    → MADD
pair 1    → MADD
pair 2    → MADD
...
pair 4095 → MADD
```

The general NEON implementation processes blocks of 16 signed INT8 values using 128-bit vector registers.

Measured improvement:

```text
General NEON vs Scalar = 9.326×
```

A 128-bit vector can hold 16 INT8 values, but that does not imply an automatic 16× application speedup.

Execution still includes:

```text
vector loads
loop control
instruction dependencies
pairwise reduction
horizontal reduction
pipeline limits
instruction throughput limits
```

Therefore:

```text
SIMD width ≠ automatic speedup
```

---

# Interpretation 2 — Specialized SIMD matters even after vectorization

The most interesting result is:

```text
SDOT vs General NEON = 2.400×
```

Both implementations already use 128-bit NEON SIMD.

The difference is the arithmetic abstraction.

General NEON constructs the dot product from:

```asm
smull
smull
sadalp
sadalp
```

while the specialized implementation uses:

```asm
sdot
```

Conceptually:

```text
GENERAL NEON

16 INT8 pairs
    ↓
SMULL
SMULL
    ↓
INT16 products
    ↓
SADALP
SADALP
    ↓
INT32 partial sums
```

versus:

```text
SDOT

16 INT8 pairs
    ↓
SDOT
    ↓
INT32 partial sums
```

The measured additional gain from the specialized operation was:

```text
2.400× over general NEON
```

---

# Why SDOT is not simply 4× faster than general NEON

It is tempting to count:

```text
SMULL
SMULL
SADALP
SADALP
```

versus:

```text
SDOT
```

and expect roughly 4× improvement.

Real execution is more complicated.

Both implementations still require:

```text
input loads
weight loads
pointer/index updates
loop branches
accumulator dependencies
final horizontal reduction
```

Only part of the loop has been compressed into the specialized instruction.

Therefore the measured result is more meaningful than static instruction counting.

---

# Assembly verification

For the benchmark to be considered valid, inspect all three kernel assembly listings.

## Scalar

Expected:

```asm
ldrsb
ldrsb
madd
```

The intended main arithmetic path should not contain:

```asm
smull
sadalp
sdot
```

## General NEON

Expected:

```asm
ldr q...
ldr q...
smull
smull
sadalp
sadalp
```

Important:

```text
NO SDOT in the intended main SIMD path
```

## SDOT

Expected:

```asm
ldr q...
ldr q...
sdot
```

This gives the evidence chain:

```text
Scalar source
    ↓
MADD

General NEON source
    ↓
SMULL + SADALP

SDOT source
    ↓
SDOT
```

---

# Windows build

Recommended:

```text
Release | ARM64
```

Optimization:

```text
/O2
```

Whole Program Optimization:

```text
Disabled
```

Keeping the kernels as separate translation units makes the comparison easier to inspect and reason about.

---

# Generate assembly in Visual Studio

For each kernel source file:

```text
Right-click source file
→ Properties
→ C/C++
→ Output Files
→ Assembler Output
→ Assembly-Only Listing (/FA)
```

Recommended listing location:

```text
$(IntDir)%(Filename).asm
```

Expected files:

```text
dot_scalar.asm
dot_neon.asm
dot_sdot.asm
```

---

# What this experiment teaches

The progression is now measurable:

```text
C++ scalar loop
        ↓
LDRSB + MADD
        ↓
baseline

128-bit general SIMD
        ↓
SMULL + SADALP
        ↓
9.326× faster

specialized INT8 SIMD
        ↓
SDOT
        ↓
22.388× faster than scalar
        ↓
2.400× faster than general NEON
```

The key lesson is:

> Performance does not improve only by making vectors wider. It can improve further when the ISA provides operations that closely match the structure of the workload.

This is an important bridge toward understanding optimized ML kernels.

---

# Connection to neural-network kernels

A neural-network kernel is not one magic instruction.

For a fully connected layer:

```text
output neuron 0
    ↓
dot product
    ↓
SDOT + SDOT + SDOT ...

output neuron 1
    ↓
dot product
    ↓
SDOT + SDOT + SDOT ...

...
```

The specialized instruction accelerates the inner mathematical primitive.

The surrounding kernel still organizes:

```text
tensor traversal
multiple outputs
bias
requantization
activation/clamping
tail processing
memory access
output storage
```

This is the bridge from an ISA-level operation toward an optimized neural-network kernel.

---

# Important limitation

These measurements describe this specific benchmark configuration:

```text
Native Windows ARM64
Qualcomm Oryon CPU
4096-element INT8 dot product
hot/reused small working set
/O2 optimized Release build
10000 iterations per trial
median of 7 trials
```

The result should therefore be reported as:

> For this benchmark configuration, general NEON achieved 9.326× speedup over the scalar implementation, while SDOT achieved 22.388× over scalar and 2.400× over the general NEON implementation.

It should not be interpreted as a universal claim that SDOT is always 22.388× faster than scalar code.

Different tensor sizes, cache states, memory layouts, compiler versions, CPU frequencies, and surrounding kernel work can change the result.

---

# Next experiment

Project 05 will examine compiler auto-vectorization.

The question becomes:

> If ordinary scalar C++ is written without explicit NEON intrinsics, how much SIMD optimization can the compiler discover on its own?

That will help separate:

```text
what the programmer explicitly requested
```

from:

```text
what the compiler can automatically infer
```

and clarify when handwritten NEON intrinsics are useful.
