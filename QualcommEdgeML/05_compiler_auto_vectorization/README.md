# Project 05 — Compiler Auto-Vectorization on ARM64

## Goal

Understand what Microsoft Visual C++ (MSVC) generates from a plain C++ INT8 dot-product loop when the programmer does **not** use NEON intrinsics.

This experiment compares three versions of the same mathematical operation:

```text
result = Σ input[i] × weight[i]
```

The three implementations are:

1. **Forced scalar**
   - Plain C++
   - `#pragma loop(no_vector)`
   - Compiled for Armv8.0

2. **Auto-vectorized Armv8.0**
   - Plain C++
   - No NEON intrinsics
   - Compiler decides whether/how to vectorize
   - Compiled for Armv8.0

3. **Auto-vectorized Armv8.4**
   - Same plain C++ loop
   - Compiler decides whether/how to vectorize
   - Compiled for Armv8.4

The central question is:

> If the compiler is allowed to vectorize a simple INT8 dot product, will it merely generate general NEON instructions, or will it recognize the dot-product pattern and emit `SDOT`?

---

# Experiment progression

This project builds directly on the earlier instruction-level experiments:

```text
Project 01
Scalar C++
    ↓
LDRSB + MADD

Project 02
Explicit NEON intrinsics
    ↓
SMULL + SADALP

Project 03
Explicit dot-product intrinsic
    ↓
SDOT

Project 04
Benchmark scalar vs NEON vs SDOT
    ↓
Measure real speedup

Project 05
Plain C++ again
    ↓
Let MSVC decide
    ↓
Inspect vectorizer report + assembly + timing
```

---

# Project structure

```text
05_compiler_auto_vectorization/
│
├── kernels.h
├── dot_forced_scalar.cpp
├── dot_auto_v80.cpp
├── dot_auto_v84.cpp
└── main.cpp
```

The files are separated so that different compiler ISA targets can be applied per translation unit.

---

# Common kernel interface

`kernels.h`

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

std::int32_t dot_forced_scalar_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);

std::int32_t dot_auto_v80_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);

std::int32_t dot_auto_v84_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);
```

All three functions compute the same INT8 dot product and return an INT32 result.

---

# 1. Forced scalar implementation

`dot_forced_scalar.cpp`

```cpp
std::int32_t accumulator = 0;

#pragma loop(no_vector)
for (std::size_t i = 0; i < count; ++i)
{
    accumulator +=
        static_cast<std::int32_t>(input[i]) *
        static_cast<std::int32_t>(weights[i]);
}
```

The key line is:

```cpp
#pragma loop(no_vector)
```

This explicitly tells MSVC not to vectorize the loop.

Expected scalar assembly pattern:

```asm
ldrsb
ldrsb
madd
```

Conceptually:

```text
input[i]
   ↓
LDRSB

weight[i]
   ↓
LDRSB

input × weight + accumulator
   ↓
MADD
```

This serves as the control implementation.

---

# 2. Auto-vectorized Armv8.0 implementation

`dot_auto_v80.cpp`

```cpp
std::int32_t accumulator = 0;

for (std::size_t i = 0; i < count; ++i)
{
    accumulator +=
        static_cast<std::int32_t>(input[i]) *
        static_cast<std::int32_t>(weights[i]);
}
```

There are intentionally:

```text
NO NEON intrinsics
NO arm64_neon.h
NO vectorization pragma
```

The compiler is free to recognize the reduction pattern and generate SIMD instructions.

Per-file compiler target:

```text
/arch:armv8.0
```

---

# 3. Auto-vectorized Armv8.4 implementation

`dot_auto_v84.cpp`

The source loop is intentionally the same as the Armv8.0 version:

```cpp
std::int32_t accumulator = 0;

for (std::size_t i = 0; i < count; ++i)
{
    accumulator +=
        static_cast<std::int32_t>(input[i]) *
        static_cast<std::int32_t>(weights[i]);
}
```

The important difference is only the compiler target:

```text
/arch:armv8.4
```

This makes newer instructions available to the compiler, including the Arm dot-product extension.

However:

```text
Instruction available
        ≠
compiler must select it
```

That distinction is the core lesson of this experiment.

---

# Compiler vectorization report

The project was built with:

```text
/Qvec-report:2
```

This reports:

```text
C5001 → loop vectorized
C5002 → loop not vectorized
```

Observed build output:

```text
dot_forced_scalar.cpp(20)
info C5002: loop not vectorized due to reason '1400'

dot_auto_v84.cpp(19)
info C5001: loop vectorized

dot_auto_v80.cpp(19)
info C5001: loop vectorized
```

Therefore:

```text
Forced scalar
    ↓
NOT vectorized

Auto Armv8.0
    ↓
vectorized

Auto Armv8.4
    ↓
vectorized
```

For the forced scalar version, reason `1400` is expected because the source explicitly contains:

```cpp
#pragma loop(no_vector)
```

---

# Important distinction: vectorized does not mean SDOT

The vectorizer report only answers:

> Did the compiler transform the loop into vector code?

It does **not** answer:

> Which vector instructions did the compiler use?

So the experiment has two compiler-observation layers:

```text
/Qvec-report:2
      ↓
Did MSVC vectorize?

/FA assembly output
      ↓
How did MSVC vectorize?
```

---

# Observed Armv8.4 assembly

The generated `dot_auto_v84_q7()` assembly did **not** contain `SDOT`.

The important vectorized portion was:

```asm
cmp         x2,#8
blo         scalar_tail

and         x10,x2,#-8

movi        v18.4s,#0
movi        v17.4s,#0

vector_loop:

ldr         s16,[x8,x1]
add         x11,x8,x1
add         x9,x8,x3

ldr         s21,[x11,#4]
ldr         s20,[x9,#4]

sxtl        v19.8h,v16.8b

ldr         s16,[x8,x3]

add         x8,x8,#8
cmp         x8,x10

sxtl        v16.8h,v16.8b
smlal       v18.4s,v16.4h,v19.4h

sxtl        v19.8h,v21.8b
sxtl        v16.8h,v20.8b
smlal       v17.4s,v16.4h,v19.4h

blo         vector_loop

add         v17.4s,v18.4s,v17.4s
addv        s17,v17.4s
fmov        w0,s17
```

Then the compiler used a scalar cleanup loop:

```asm
ldrsb       w10,[x8,x3]
ldrsb       w9,[x8,x1]

add         x8,x8,#1
cmp         x8,x2

madd        w0,w10,w9,w0
blo         scalar_tail
```

---

# What the Armv8.4 assembly is doing

The compiler selected an 8-element SIMD strategy.

Conceptually:

```text
8 input-weight pairs
        │
        ├───────────────┐
        │               │
        ▼               ▼
     first 4          next 4
        │               │
      SXTL            SXTL
        │               │
      INT16           INT16
        │               │
      SMLAL           SMLAL
        │               │
        ▼               ▼
     v18.4s          v17.4s
        │               │
        └───────┬───────┘
                ▼
            vector ADD
                ↓
              ADDV
                ↓
          scalar INT32 result
```

---

# `SXTL`

Example:

```asm
sxtl v16.8h,v16.8b
```

`SXTL` means:

```text
Signed eXTend Long
```

It converts signed INT8 values into signed INT16 values.

Conceptually:

```text
INT8
 ↓
SXTL
 ↓
INT16
```

This widening is necessary because the chosen multiply-accumulate path operates on wider elements.

---

# `SMLAL`

Example:

```asm
smlal v18.4s,v16.4h,v19.4h
```

`SMLAL` means:

```text
Signed Multiply-Accumulate Long
```

Conceptually:

```text
4 × INT16 inputs
      ×
4 × INT16 weights
      ↓
4 × INT32 products
      +
4 × INT32 accumulator
```

Equivalent idea:

```text
v18[0] += input0 × weight0
v18[1] += input1 × weight1
v18[2] += input2 × weight2
v18[3] += input3 × weight3
```

The second `SMLAL` handles the next four elements.

---

# Two vector accumulators

MSVC uses:

```asm
movi v18.4s,#0
movi v17.4s,#0
```

So two independent 4-lane accumulators are maintained.

Conceptually:

```text
elements 0..3  → v18
elements 4..7  → v17
```

At the end:

```asm
add v17.4s,v18.4s,v17.4s
```

combines them.

Then:

```asm
addv s17,v17.4s
```

horizontally reduces the four INT32 lanes into one scalar sum.

Finally:

```asm
fmov w0,s17
```

moves the result into the ARM64 integer return register.

---

# Scalar tail

If the element count is not divisible by 8, the remaining values are processed using:

```asm
ldrsb
ldrsb
madd
```

So the generated function contains:

```text
main SIMD path
    ↓
8 elements per iteration
    ↓
SXTL + SMLAL

then

scalar remainder
    ↓
LDRSB + MADD
```

For the benchmark size of 4096:

```text
4096 % 8 = 0
```

so the scalar tail is not used during the main benchmark path.

---

# Critical finding: no SDOT

Even though the Armv8.4 file was compiled with a target where dot-product instructions are available, MSVC generated:

```asm
sxtl
smlal
```

and did **not** generate:

```asm
sdot
```

Therefore:

```text
Plain C++ dot-product loop
        ↓
MSVC recognizes SIMD opportunity
        ↓
general NEON vectorization
        ↓
SXTL + SMLAL

NOT

Plain C++ dot-product loop
        ↓
SDOT
```

This is one of the main conclusions of Project 05.

---

# Benchmark configuration

```text
Elements / dot     : 4096
Iterations / trial : 10000
Trials             : 7
```

The same deterministic input and weight vectors are used for every implementation.

---

# Correctness result

Observed runtime output:

```text
Forced scalar result : -9188
Auto Armv8.0 result   : -9188
Auto Armv8.4 result   : -9188

Verification : PASS
```

All three implementations therefore compute the same mathematical result.

---

# Performance result

Observed:

```text
Auto v8.0 vs Scalar : 8.194x
Auto v8.4 vs Scalar : 8.244x
Auto v8.4 vs v8.0   : 1.006x
```

Normalized:

| Implementation | Relative performance |
|---|---:|
| Forced scalar | 1.000x |
| Auto Armv8.0 | 8.194x |
| Auto Armv8.4 | 8.244x |

---

# What the performance result means

The compiler's automatic vectorization was highly effective.

Without writing a single NEON intrinsic:

```text
plain C++
    ↓
MSVC auto-vectorizer
    ↓
~8.2× faster than forced scalar
```

This demonstrates that a modern compiler can recognize a reduction loop and transform it into SIMD code automatically.

---

# Why Armv8.4 is almost identical to Armv8.0

Measured:

```text
Auto v8.4 vs v8.0 = 1.006x
```

That is only about a 0.6% difference.

The observed Armv8.4 assembly explains why.

The compiler did not switch to the specialized `SDOT` instruction.

Instead it used general vector operations:

```asm
sxtl
smlal
```

Therefore the Armv8.4 implementation remained very similar in character to ordinary NEON vectorization.

---

# Compare with Project 04

Project 04 measured:

```text
General NEON vs Scalar : 9.326x
SDOT vs Scalar         : 22.388x
SDOT vs General NEON   : 2.400x
```

Project 05 measured:

```text
Auto v8.0 vs Scalar : 8.194x
Auto v8.4 vs Scalar : 8.244x
```

Conceptually:

```text
Forced scalar
    │
    ├── compiler auto-vectorization
    │        ↓
    │      ~8.2×
    │
    ├── explicit general NEON
    │        ↓
    │      ~9.3×
    │
    └── explicit SDOT
             ↓
           ~22.4×
```

The exact numbers come from separate benchmark runs, so they should not be treated as a laboratory-grade direct comparison.

However, the trend is extremely useful:

```text
auto-vectorization
        ↓
finds SIMD parallelism

explicit specialized intrinsic
        ↓
can expose a more workload-specific instruction
```

---

# Why `/arch:armv8.4` did not force SDOT

The compiler option tells MSVC which instructions it is allowed to generate.

It does not mean:

```text
"replace every possible dot-product loop with SDOT"
```

There are two separate ideas:

```text
ISA availability
    ↓
SDOT is legal to use

pattern recognition / instruction selection
    ↓
compiler decides whether the source maps to SDOT
```

In this experiment:

```text
SDOT available
    ↓
MSVC still selected SXTL + SMLAL
```

Therefore:

```text
available instruction
    ≠
selected instruction
```

---

# How to guarantee SDOT

To explicitly request the signed INT8 dot-product operation, use the NEON intrinsic:

```cpp
accumulator =
    vdotq_s32(
        accumulator,
        input_vector,
        weight_vector);
```

with:

```cpp
#include <arm64_neon.h>
```

and compile with an ISA target that supports the dot-product extension.

That approach was already explored in Projects 03 and 04.

Project 05 deliberately does **not** use this intrinsic because the purpose here is to study compiler auto-vectorization.

---

# The key lesson

Project 05 demonstrates a very important distinction:

```text
Compiler auto-vectorization
        ↓
"I see independent work that can run in SIMD"

                ≠

Specialized idiom recognition
        ↓
"I recognize this exact operation as an INT8 dot product"
        ↓
"Use SDOT"
```

MSVC successfully performed the first step.

For this source pattern and compiler build, it did not perform the second.

---

# Why optimized neural-network kernels still matter

This experiment helps explain why libraries such as optimized neural-network kernels exist.

A general-purpose compiler may successfully produce SIMD code from plain C++.

But an optimized kernel author can deliberately structure computation around:

```text
specialized instructions
multiple accumulators
blocking
tiling
memory layout
tail handling
requantization
activation
cache behavior
```

For an INT8 dot product:

```text
plain C++
    ↓
compiler
    ↓
general SIMD

versus

optimized kernel
    ↓
explicit workload knowledge
    ↓
specialized SIMD such as SDOT
```

This becomes increasingly important as we move from a single dot product toward real Fully Connected and convolution kernels.

---

# Experiment 05 conclusion

The complete evidence chain is now:

```text
SOURCE
  ↓
plain scalar-looking C++

COMPILER REPORT
  ↓
C5001: loop vectorized

ASSEMBLY
  ↓
SXTL + SMLAL
NO SDOT

PERFORMANCE
  ↓
~8.2× faster than forced scalar
```

The main takeaway is:

> MSVC successfully auto-vectorized the plain INT8 dot-product loop, but simply compiling for Armv8.4 did not make it select `SDOT`. The compiler generated general NEON SIMD using `SXTL + SMLAL`. Explicitly expressing the dot-product operation with `vdotq_s32()` remains the reliable way to request `SDOT`.

---

# Next experiment

Project 06 moves from a single dot product to a real neural-network building block:

```text
Fully Connected Layer
```

Instead of calculating:

```text
one dot product
```

we will calculate:

```text
multiple output neurons
    ↓
each neuron = dot product + bias
```

This is where an optimized dot-product primitive starts becoming a true neural-network kernel.
