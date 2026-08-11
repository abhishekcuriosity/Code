# Project 02 — General NEON INT8 Dot Product

## Goal

Compute the same INT8 dot product as Project 1 using explicit 128-bit ARM NEON SIMD intrinsics.

```text
result = Σ input[i] × weight[i]
```

The mathematics does not change. The execution strategy does.

```text
Project 1:
1 input × 1 weight

Project 2:
16 inputs and 16 weights loaded into 128-bit vector registers
```

# Core C++ intrinsics

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

std::int32_t sum =
    vaddvq_s32(accumulator);
```

# Assembly exploration

Observed main MSVC ARM64 SIMD path:

```asm
|?dot_neon_q7@@YAHPEBC0_K@Z| PROC

    mov         x8,#0
    movi        v16.4s,#0

    cmp         x2,#0x10
    blo         |$LN37@dot_neon_q|

|$LL4@dot_neon_q|

    ldr         q20,[x8,x0]
    mov         x9,x8

    ldr         q19,[x8,x1]

    add         x9,x9,#0x20
    add         x8,x8,#0x10
    cmp         x9,x2

    dup         d17,v20.d[1]
    dup         d18,v19.d[1]

    smull       v18.8h,v17.8b,v18.8b
    smull       v17.8h,v20.8b,v19.8b

    sadalp      v16.4s,v17.8h
    sadalp      v16.4s,v18.8h

    bls         |$LL4@dot_neon_q|

|$LN37@dot_neon_q|

    addv        s16,v16.4s

    ...
```

The full compiler output also included a compiler-vectorized remainder path and a final scalar cleanup path.

## `movi v16.4s,#0`

```asm
movi v16.4s,#0
```

Creates four INT32 accumulator lanes:

```text
v16.4s = [0,0,0,0]
```

Equivalent intrinsic:

```cpp
vdupq_n_s32(0)
```

## `cmp x2,#0x10`

```asm
cmp x2,#0x10
```

`0x10` = 16.

The main SIMD loop operates only when a complete group of 16 elements is available.

## 128-bit vector loads

```asm
ldr q20,[x8,x0]
ldr q19,[x8,x1]
```

Each `q` register is 128 bits.

Therefore:

```text
q20 = 16 × INT8 inputs
q19 = 16 × INT8 weights
```

This is the first obvious proof that NEON SIMD is active.

## Why `dup` appears

```asm
dup d17,v20.d[1]
dup d18,v19.d[1]
```

The source requested the upper eight bytes using `vget_high_s8`.

MSVC copied the upper 64-bit halves and then used ordinary `SMULL`.

Another compiler might instead use `SMULL2`.

Important lesson:

```text
intrinsic semantics are portable
exact instruction selection is compiler-dependent
```

# `SMULL`

```asm
smull v18.8h,v17.8b,v18.8b
smull v17.8h,v20.8b,v19.8b
```

`SMULL` = Signed Multiply Long.

Conceptually:

```text
8 × INT8
    ×
8 × INT8
    ↓
8 × INT16 products
```

Two `SMULL` instructions cover all sixteen pairs:

```text
SMULL 1 → products 0..7
SMULL 2 → products 8..15
```

# `SADALP`

```asm
sadalp v16.4s,v17.8h
sadalp v16.4s,v18.8h
```

Mnemonic:

```text
S   Signed
AD  Add
A   Accumulate
L   Long / widen
P   Pairwise
```

If:

```text
[p0 p1 p2 p3 p4 p5 p6 p7]
```

is the source vector, `SADALP` conceptually performs:

```text
acc[0] += p0 + p1
acc[1] += p2 + p3
acc[2] += p4 + p5
acc[3] += p6 + p7
```

So:

```text
8 × INT16 products
        ↓
pairwise add + widen + accumulate
        ↓
4 × INT32 accumulators
```

The second `SADALP` adds products 8..15 into the same accumulator lanes.

# `ADDV`

```asm
addv s16,v16.4s
```

If:

```text
v16.4s = [A,B,C,D]
```

then:

```text
ADDV → A+B+C+D
```

This reduces the vector accumulator to one scalar dot-product result.

# Main SIMD data flow

```text
16 INT8 inputs
+
16 INT8 weights
        ↓
LDR Q
LDR Q
        ↓
SMULL
SMULL
        ↓
16 INT16 products
        ↓
SADALP
SADALP
        ↓
4 INT32 partial sums
        ↓
ADDV
        ↓
1 scalar result
```

# Compiler-generated remainder path

The source cleanup loop was written as ordinary scalar C++ and did not have `#pragma loop(no_vector)`.

MSVC therefore optimized part of it again.

The full assembly contained instructions such as:

```asm
sxtl
smlal
addv
```

before the last few elements fell back to:

```asm
ldrsb
ldrsb
madd
```

So the actual function has three zones:

```text
1. Explicit 16-element NEON loop
2. Compiler-generated smaller SIMD remainder loop
3. Final scalar remainder loop
```

For the original 16-element test vector, the main 16-element SIMD path is the important path.

# Why this proves general NEON SIMD

Vector registers:

```text
q19 q20
v16 v17 v18
```

Vector arithmetic:

```asm
smull
sadalp
addv
```

No specialized:

```asm
sdot
```

Therefore Project 2 is:

```text
general SIMD instructions
used to construct a dot product
```

# C++ → assembly map

| C++ intrinsic / intent | Observed instruction |
|---|---|
| `vdupq_n_s32(0)` | `movi` |
| `vld1q_s8(input+i)` | `ldr q...` |
| `vld1q_s8(weights+i)` | `ldr q...` |
| upper-half extraction | `dup` |
| `vmull_s8(...)` | `smull` |
| `vpadalq_s16(...)` | `sadalp` |
| `vaddvq_s32(...)` | `addv` |
| final scalar cleanup | `ldrsb` + `madd` |

# Project 1 vs Project 2

```text
PROJECT 1                     PROJECT 2
─────────────────────────────────────────────

Scalar                        NEON SIMD

1 pair                        16 values loaded

LDRSB                         LDR Q
LDRSB                         LDR Q
MADD                          SMULL
                              SMULL
                              SADALP
                              SADALP
                              ADDV
```

# Portable include

```cpp
#if defined(_MSC_VER) && defined(_M_ARM64)
#include <arm64_neon.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#else
#error This experiment requires ARM64/AArch64.
#endif
```

# Windows

Use:

```text
Release | ARM64
```

Assembly:

```text
C/C++ → Output Files → Assembly-Only Listing (/FA)
```

Search:

```text
dot_neon_q7
```

# Linux

```bash
g++ \
  -O2 \
  -std=c++20 \
  -march=armv8-a+simd \
  main.cpp \
  -o neon_dot
```

Generate assembly:

```bash
g++ \
  -O2 \
  -std=c++20 \
  -march=armv8-a+simd \
  -S \
  -fverbose-asm \
  main.cpp \
  -o neon_dot.s
```

Inspect:

```bash
grep -Ei "smull|sadalp|addv" neon_dot.s
```

# Core takeaway

```text
INT8 vectors
    ↓
SMULL
    ↓
INT16 products
    ↓
SADALP
    ↓
INT32 partial sums
    ↓
ADDV
    ↓
scalar result
```

Project 3 asks whether ARM provides an instruction specifically for the dot-product pattern itself.
