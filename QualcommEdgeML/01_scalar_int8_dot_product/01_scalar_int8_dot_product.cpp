#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#if !defined(_M_ARM64)
#error This experiment must be compiled for native ARM64.
#endif

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

// q7 means signed 8-bit integer data.
NOINLINE std::int32_t dot_scalar_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count)
{
    std::int32_t accumulator = 0;

    // This is our scalar baseline.
    // Prevent MSVC from silently converting the loop to NEON SIMD.
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

    const std::int32_t result = dot_scalar_q7(
        input.data(),
        weights.data(),
        input.size());

    constexpr std::int32_t expected = -793;

    std::cout << "Qualcomm Edge ML - Experiment 1\n";
    std::cout << "Build target       : Native ARM64\n";
    std::cout << "Execution unit     : Qualcomm Oryon CPU\n";
    std::cout << "Implementation     : Scalar INT8\n";
    std::cout << "Elements           : " << input.size() << '\n';
    std::cout << "Dot-product result : " << result << '\n';

    if (result != expected)
    {
        std::cerr << "Verification       : FAIL\n";
        std::cerr << "Expected           : " << expected << '\n';
        return 1;
    }

    std::cout << "Verification       : PASS\n";
    return 0;
}