# Project 01 — Scalar INT8 Dot Product

## Goal

Build a signed INT8 dot product and inspect exactly what ARM64 executes when SIMD vectorization is disabled.

```text
result = Σ input[i] × weight[i]
```

This is the scalar baseline for all later experiments.

## C++ implementation

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
        const std::int32_t input_value =
            static_cast<std::int32_t>(input[i]);

        const std::int32_t weight_value =
            static_cast<std::int32_t>(weights[i]);

        accumulator += input_value * weight_value;
    }

    return accumulator;
}
```

`#pragma loop(no_vector)` is important: it stops MSVC from silently generating NEON for the scalar baseline.

## Expected output

```text
Dot-product result : -793
Verification       : PASS
```

# Assembly exploration

Observed MSVC ARM64 function:

```asm
|?dot_scalar_q7@@YAHPEBC0_K@Z| PROC

    mov         x3,x0

    mov         w0,#0
    mov         x8,#0

    cbz         x2,|$LN16@dot_scalar|

|$LL4@dot_scalar|

    ldrsb       w10,[x8,x3]
    ldrsb       w9,[x8,x1]

    add         x8,x8,#1
    cmp         x8,x2

    madd        w0,w10,w9,w0
    blo         |$LL4@dot_scalar|

|$LN16@dot_scalar|

    ret
    ENDP
```

## Register mapping

```text
x0 = input pointer
x1 = weights pointer
x2 = count

x3 = saved input pointer
x8 = loop index
w0 = accumulator and return value
w10 = input[i]
w9 = weights[i]
```

`x0` and `w0` are two views of the same architectural register. The compiler copies the input pointer to `x3` so that `w0` can become the accumulator and eventual return register.

## Opcode-by-opcode

### `mov x3,x0`

```asm
mov x3,x0
```

Save the input pointer.

### `mov w0,#0`

```asm
mov w0,#0
```

Equivalent to:

```cpp
accumulator = 0;
```

### `mov x8,#0`

```asm
mov x8,#0
```

Equivalent to:

```cpp
i = 0;
```

### `cbz x2,...`

```asm
cbz x2,exit
```

`CBZ` = Compare and Branch if Zero.

Equivalent:

```cpp
if (count == 0)
    return 0;
```

### `ldrsb`

```asm
ldrsb w10,[x8,x3]
ldrsb w9,[x8,x1]
```

`LDRSB` = Load Register Signed Byte.

Each instruction:

```text
loads one INT8 value
        ↓
sign-extends it to INT32
```

Equivalent:

```cpp
input_value  = static_cast<int32_t>(input[i]);
weight_value = static_cast<int32_t>(weights[i]);
```

### `add x8,x8,#1`

```asm
add x8,x8,#1
```

Equivalent:

```cpp
++i;
```

### `cmp x8,x2`

```asm
cmp x8,x2
```

Compares the updated index against `count`.

### `madd`

```asm
madd w0,w10,w9,w0
```

`MADD` = Multiply Add.

```text
w0 = w10 × w9 + w0
```

Equivalent:

```cpp
accumulator += input_value * weight_value;
```

Important: this is still scalar. One `MADD` handles one input-weight pair.

### `blo`

```asm
blo loop
```

`BLO` = Branch if Lower, using an unsigned comparison because `size_t` is unsigned.

Equivalent:

```cpp
if (i < count)
    continue;
```

### `ret`

```asm
ret
```

The final accumulator already lives in `w0`, the 32-bit return register.

## One iteration

```text
input[i]    = 3
weight[i]   = 2
accumulator = 10
```

Then:

```text
LDRSB → w10 = 3
LDRSB → w9  = 2
MADD  → w0  = 3 × 2 + 10 = 16
```

## Why this proves scalar execution

Only general-purpose registers appear:

```text
x0 x1 x2 x3 x8
w0 w9 w10
```

No vector registers:

```text
v0-v31
q0-q31
```

No SIMD arithmetic instructions:

```text
SMULL
SADALP
SDOT
ADDV
```

Therefore:

```text
1 input-weight pair
        ↓
      MADD
        ↓
 scalar accumulator
```

## C++ → assembly map

| C++ intent | ARM64 |
|---|---|
| preserve input pointer | `mov x3,x0` |
| accumulator = 0 | `mov w0,#0` |
| i = 0 | `mov x8,#0` |
| count == 0 | `cbz` |
| load signed input | `ldrsb` |
| load signed weight | `ldrsb` |
| ++i | `add` |
| compare i/count | `cmp` |
| multiply-accumulate | `madd` |
| repeat | `blo` |
| return | `ret` |

# Windows

Use:

```text
Release | ARM64
```

Generate assembly:

```text
C/C++ → Output Files → Assembler Output → Assembly-Only Listing (/FA)
```

or use:

```text
Debug → Windows → Disassembly
```

Search for:

```text
dot_scalar_q7
```

# Linux

On native AArch64:

```bash
g++ \
  -O2 \
  -std=c++20 \
  -fno-tree-vectorize \
  -fno-tree-slp-vectorize \
  main.cpp \
  -o scalar_dot
```

Generate assembly:

```bash
g++ \
  -O2 \
  -std=c++20 \
  -fno-tree-vectorize \
  -fno-tree-slp-vectorize \
  -S \
  -fverbose-asm \
  main.cpp \
  -o scalar_dot.s
```

Inspect:

```bash
grep -Ei "ldrsb|madd" scalar_dot.s
```

Exact register allocation may differ from MSVC.

# Core takeaway

```text
Scalar C++
    ↓
LDRSB
    ↓
MADD
    ↓
one product at a time
```
