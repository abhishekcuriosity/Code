#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <arm64_neon.h>

#if !defined(_M_ARM64)
#error This experiment must be compiled for native ARM64.
#endif

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

NOINLINE std::int32_t dot_scalar_q7(
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

NOINLINE std::int32_t dot_neon_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count)
{
    std::size_t i = 0;

    // Four parallel INT32 accumulator lanes.
    int32x4_t accumulator = vdupq_n_s32(0);

    // Process 16 signed INT8 values per iteration.
    for (; i + 16 <= count; i += 16)
    {
        // Load 16 INT8 inputs.
        const int8x16_t input_vector =
            vld1q_s8(input + i);

        // Load 16 INT8 weights.
        const int8x16_t weight_vector =
            vld1q_s8(weights + i);

        // Multiply the lower eight INT8 lanes.
        // Results are widened to eight INT16 values.
        const int16x8_t products_low =
            vmull_s8(
                vget_low_s8(input_vector),
                vget_low_s8(weight_vector));

        // Multiply the upper eight INT8 lanes.
        const int16x8_t products_high =
            vmull_s8(
                vget_high_s8(input_vector),
                vget_high_s8(weight_vector));

        // Pairwise-add the INT16 products,
        // widen to INT32 and accumulate.
        accumulator =
            vpadalq_s16(accumulator, products_low);

        accumulator =
            vpadalq_s16(accumulator, products_high);
    }

    // Add the four INT32 accumulator lanes.
    std::int32_t sum = vaddvq_s32(accumulator);

    // Handle any elements remaining after the SIMD loop.
    for (; i < count; ++i)
    {
        sum +=
            static_cast<std::int32_t>(input[i]) *
            static_cast<std::int32_t>(weights[i]);
    }

    return sum;
}

int main()
{
    constexpr std::array<std::int8_t, 16> input = {
          3,  -2,   5,   7,
         -4,   6,  -8,   9,
         10, -11,  12,  13,
        -14,  15,  16, -17
    };

    constexpr std::array<std::int8_t, 16> weights = {
          2,   4,  -3,   5,
          6,  -7,   8,   9,
        -10,  11,  12, -13,
         14,  15, -16,  17
    };

    const std::int32_t scalar_result =
        dot_scalar_q7(
            input.data(),
            weights.data(),
            input.size());

    const std::int32_t neon_result =
        dot_neon_q7(
            input.data(),
            weights.data(),
            input.size());

    constexpr std::int32_t expected = -793;

    std::cout << "Qualcomm Edge ML - Experiment 2\n";
    std::cout << "Build target       : Native ARM64\n";
    std::cout << "Execution unit     : Qualcomm Oryon CPU\n";
    std::cout << "SIMD technology    : NEON 128-bit\n";
    std::cout << "Elements           : " << input.size() << '\n';
    std::cout << "Scalar result      : " << scalar_result << '\n';
    std::cout << "NEON result        : " << neon_result << '\n';

    if (scalar_result != expected)
    {
        std::cerr << "Scalar verification: FAIL\n";
        return 1;
    }

    if (neon_result != expected)
    {
        std::cerr << "NEON verification  : FAIL\n";
        return 1;
    }

    if (scalar_result != neon_result)
    {
        std::cerr << "Result comparison  : FAIL\n";
        return 1;
    }

    std::cout << "Scalar verification: PASS\n";
    std::cout << "NEON verification  : PASS\n";
    std::cout << "Result comparison  : PASS\n";

    return 0;
}