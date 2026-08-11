# Project 03 — NEON SDOT INT8 Dot Product

## Goal

Compute the same signed INT8 dot product using the ARM integer dot-product extension.

```text
Project 1
Scalar MADD
     ↓
Project 2
SMULL + SADALP
     ↓
Project 3
SDOT
```

The mathematics remains:

```text
result = Σ input[i] × weight[i]
```

# Core C++ intrinsic

```cpp
accumulator =
    vdotq_s32(
        accumulator,
        input_vector,
        weight_vector);
```

The compiler should map this to the signed byte dot-product instruction when the selected ISA supports it.

# What SDOT means

A 128-bit input vector contains sixteen signed bytes:

```text
a0 a1 a2 a3 | a4 a5 a6 a7 |
a8 a9 a10 a11 | a12 a13 a14 a15
```

The weight vector contains:

```text
b0 b1 b2 b3 | b4 b5 b6 b7 |
b8 b9 b10 b11 | b12 b13 b14 b15
```

One vector `SDOT` updates four INT32 accumulator lanes:

```text
lane 0 += a0*b0 + a1*b1 + a2*b2 + a3*b3

lane 1 += a4*b4 + a5*b5 + a6*b6 + a7*b7

lane 2 += a8*b8 + a9*b9 + a10*b10 + a11*b11

lane 3 += a12*b12 + a13*b13 + a14*b14 + a15*b15
```

So conceptually:

```text
16 signed INT8 multiplications
        ↓
four groups of four
        ↓
4 INT32 accumulated partial sums
```

# Assembly exploration

The key proof for this project is the presence of:

```asm
sdot
```

inside `dot_sdot_q7`.

A typical main path should look similar to:

```asm
movi    v16.4s,#0

ldr     q17,[...]
ldr     q18,[...]

sdot    v16.4s,v17.16b,v18.16b

...

addv    s16,v16.4s
```

Exact register numbers and scheduling can differ between compilers.

## `movi ...4s,#0`

Initializes:

```text
[A0,A1,A2,A3] = [0,0,0,0]
```

Equivalent intrinsic:

```cpp
vdupq_n_s32(0)
```

## `ldr q...`

Each vector load brings in:

```text
16 × INT8 = 128 bits
```

The main loop therefore normally loads:

```text
16 inputs
16 weights
```

before `SDOT`.

# `SDOT`

Canonical form:

```asm
sdot vAcc.4s,vInput.16b,vWeight.16b
```

Interpretation:

```text
destination:
4 × INT32 accumulator lanes

source 1:
16 × signed INT8 values

source 2:
16 × signed INT8 values
```

One `SDOT` represents:

```text
16 signed INT8 multiplications
+
grouped accumulation into 4 INT32 lanes
```

# `ADDV`

After repeated `SDOT` operations:

```text
vAcc.4s = [A0,A1,A2,A3]
```

If the function wants one scalar result:

```asm
addv s16,vAcc.4s
```

conceptually computes:

```text
A0 + A1 + A2 + A3
```

# Project 2 vs Project 3

Project 2 arithmetic core:

```asm
smull
smull
sadalp
sadalp
```

Meaning:

```text
construct the dot product
from general SIMD operations
```

Project 3 arithmetic core:

```asm
sdot
```

Meaning:

```text
use a specialized signed INT8
dot-product primitive
```

`SDOT` does not replace loads, loop control, tail handling, or final reduction. It replaces the core arithmetic pattern.

# Main data flow

```text
16 INT8 inputs
+
16 INT8 weights
        ↓
LDR Q
LDR Q
        ↓
SDOT
        ↓
4 INT32 partial accumulators
        ↓
ADDV
        ↓
1 scalar result
```

# Evidence to preserve

After compiling/running, save the actual generated assembly:

```text
03_sdot_int8_dot_product/
└── evidence/
    └── dot_sdot_q7.asm
```

Then replace the expected assembly block in this README with the exact observed MSVC output.

The learning chain should be:

```text
C++ intrinsic
    ↓
actual compiler output
    ↓
actual SDOT instruction
```

# Windows

Use:

```text
Release | ARM64
```

Compile for an ISA level supporting dot product, for example:

```text
/arch:armv8.4
```

Runtime support can also be checked on Windows with:

```cpp
IsProcessorFeaturePresent(
    PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
```

Generate assembly:

```text
C/C++ → Output Files → Assembly-Only Listing (/FA)
```

Search for:

```text
dot_sdot_q7
```

and then:

```text
sdot
```

# Linux

Build on hardware supporting the extension:

```bash
g++ \
  -O2 \
  -std=c++20 \
  -march=armv8.2-a+dotprod \
  main.cpp \
  -o sdot_dot
```

Generate assembly:

```bash
g++ \
  -O2 \
  -std=c++20 \
  -march=armv8.2-a+dotprod \
  -S \
  -fverbose-asm \
  main.cpp \
  -o sdot_dot.s
```

Verify:

```bash
grep -i "sdot" sdot_dot.s
```

# Three-project assembly progression

```text
PROJECT 1
────────────────────
LDRSB
LDRSB
MADD

scalar
1 pair at a time


        ↓


PROJECT 2
────────────────────
LDR Q
LDR Q
SMULL
SMULL
SADALP
SADALP
ADDV

general NEON SIMD
build the dot product


        ↓


PROJECT 3
────────────────────
LDR Q
LDR Q
SDOT
ADDV

specialized NEON SIMD
dedicated INT8 dot product
```

# Cortex-M55 / Helium connection

Conceptually:

```text
Cortex-M55 / MVE              ARM64 / NEON
──────────────────────────────────────────
128-bit vector                128-bit vector

VMLADAVA.S8                   SDOT
    │                           │
    ▼                           ▼
scalar-oriented              4 INT32 partial
accumulation                 accumulators
                                │
                                ▼
                               ADDV
```

They are not identical instructions, but both illustrate ISA support for low-precision multiply-accumulate workloads.

# Core takeaway

```text
scalar arithmetic
    ↓
general vector arithmetic
    ↓
specialized vector dot-product arithmetic
```

Project 4 will benchmark what these instruction-level differences mean on the actual Oryon CPU.
