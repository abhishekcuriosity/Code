#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <windows.h>

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


NOINLINE std::int32_t dot_sdot_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count)
{
    std::size_t i = 0;

    int32x4_t accumulator =
        vdupq_n_s32(0);

    for (; i + 16 <= count; i += 16)
    {
        const int8x16_t input_vector =
            vld1q_s8(input + i);

        const int8x16_t weight_vector =
            vld1q_s8(weights + i);

        accumulator =
            vdotq_s32(
                accumulator,
                input_vector,
                weight_vector);
    }

    std::int32_t sum =
        vaddvq_s32(accumulator);

#pragma loop(no_vector)
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


    const bool sdot_supported =
        IsProcessorFeaturePresent(
            PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != FALSE;


    std::cout
        << "Qualcomm Edge ML - Experiment 3\n";

    std::cout
        << "Execution unit     : Qualcomm Oryon CPU\n";

    std::cout
        << "SIMD               : NEON SDOT\n";

    std::cout
        << "SDOT supported     : "
        << (sdot_supported ? "YES" : "NO")
        << '\n';


    if (!sdot_supported)
    {
        std::cerr
            << "SDOT not available.\n";

        return 1;
    }


    const std::int32_t scalar_result =
        dot_scalar_q7(
            input.data(),
            weights.data(),
            input.size());


    const std::int32_t sdot_result =
        dot_sdot_q7(
            input.data(),
            weights.data(),
            input.size());


    constexpr std::int32_t expected = -793;


    std::cout
        << "Scalar result      : "
        << scalar_result
        << '\n';

    std::cout
        << "SDOT result        : "
        << sdot_result
        << '\n';


    if (scalar_result != expected ||
        sdot_result != expected)
    {
        std::cout
            << "Verification       : FAIL\n";

        return 1;
    }


    std::cout
        << "Verification       : PASS\n";

    return 0;
}