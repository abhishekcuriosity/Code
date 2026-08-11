# Qualcomm ARM64 Edge ML Lab

A bottom-up learning repository for understanding how low-precision neural-network arithmetic maps from C++ to ARM64 instructions, then to SIMD kernels, ML runtimes, and eventually Qualcomm accelerator backends.

## Method

Every experiment preserves four layers:

```text
C++ source
    ↓
compiler/intrinsic intent
    ↓
generated ARM64 assembly
    ↓
hardware interpretation
```

The assembly exploration is intentionally part of each project README.

## Current projects

| Project | Execution style | Assembly focus |
|---|---|---|
| `01_scalar_int8_dot_product` | Scalar ARM64 | `LDRSB`, `MADD`, loop control |
| `02_neon_int8_dot_product` | General 128-bit NEON | vector loads, `SMULL`, `SADALP`, `ADDV`, tail handling |
| `03_sdot_int8_dot_product` | Specialized NEON dot product | `SDOT`, accumulator lanes, `ADDV` |

## Learning progression

```text
Project 1
Scalar C++
    ↓
LDRSB + MADD

Project 2
NEON intrinsics
    ↓
SMULL + SADALP + ADDV

Project 3
Dot-product intrinsic
    ↓
SDOT + ADDV

Project 4
Benchmark
    ↓
measure actual Oryon performance
```

## Recommended evidence folder

```text
project/
├── README.md
├── main.cpp
└── evidence/
    ├── output.txt
    └── important_function.asm
```

The goal is to preserve not only working code but the exact compiler-generated evidence used to understand it.
