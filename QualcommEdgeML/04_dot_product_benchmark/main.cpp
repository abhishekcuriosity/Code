#include "kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include <windows.h>


#if !defined(_M_ARM64)
#error This experiment requires native ARM64.
#endif


using DotFunction =
std::int32_t(*)(
    const std::int8_t*,
    const std::int8_t*,
    std::size_t);


// Prevent benchmark results from becoming dead code.
volatile std::int64_t g_sink = 0;


// ------------------------------------------------------------
// Time one implementation.
//
// We deliberately time MANY dot products together.
// We do not put the timer around every individual function call.
// ------------------------------------------------------------

double benchmark_ns_per_dot(
    DotFunction function,
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count,
    std::size_t iterations)
{
    // Warm-up
    std::int64_t checksum = 0;

    constexpr std::size_t warmup_iterations = 500;


    for (std::size_t i = 0;
        i < warmup_iterations;
        ++i)
    {
        checksum +=
            function(
                input,
                weights,
                count);
    }


    g_sink = checksum;


    checksum = 0;


    const auto start =
        std::chrono::steady_clock::now();


    for (std::size_t i = 0;
        i < iterations;
        ++i)
    {
        checksum +=
            function(
                input,
                weights,
                count);
    }


    const auto end =
        std::chrono::steady_clock::now();


    g_sink = checksum;


    const double elapsed_ns =
        std::chrono::duration<double, std::nano>(
            end - start)
        .count();


    return
        elapsed_ns /
        static_cast<double>(iterations);
}


// ------------------------------------------------------------
// Run seven trials and return the median.
// ------------------------------------------------------------

double median_of_seven(
    DotFunction function,
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count,
    std::size_t iterations)
{
    std::array<double, 7> samples{};


    for (double& sample : samples)
    {
        sample =
            benchmark_ns_per_dot(
                function,
                input,
                weights,
                count,
                iterations);
    }


    std::sort(
        samples.begin(),
        samples.end());


    return samples[3];
}


int main()
{
    constexpr std::size_t element_count =
        4096;


    constexpr std::size_t iterations =
        10'000;


    std::vector<std::int8_t> input(
        element_count);


    std::vector<std::int8_t> weights(
        element_count);


    // Deterministic values.
    // Keep the magnitude moderate so INT32 accumulation remains safe.
    for (std::size_t i = 0;
        i < element_count;
        ++i)
    {
        input[i] =
            static_cast<std::int8_t>(
                static_cast<int>((i * 17) % 127)
                - 63);


        weights[i] =
            static_cast<std::int8_t>(
                static_cast<int>((i * 29) % 127)
                - 63);
    }


    const bool sdot_supported =
        IsProcessorFeaturePresent(
            PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
        != FALSE;


    std::cout
        << "Qualcomm ARM64 Edge ML - Project 4\n"
        << "Scalar vs NEON vs SDOT Benchmark\n\n";


    std::cout
        << "Target             : Native ARM64\n";

    std::cout
        << "Elements / dot     : "
        << element_count
        << '\n';

    std::cout
        << "Iterations / trial : "
        << iterations
        << '\n';

    std::cout
        << "Trials             : 7\n";

    std::cout
        << "SDOT supported     : "
        << (sdot_supported ? "YES" : "NO")
        << "\n\n";


    if (!sdot_supported)
    {
        std::cerr
            << "SDOT unavailable on this system.\n";

        return 1;
    }


    // --------------------------------------------------------
    // Verify correctness BEFORE benchmarking.
    // --------------------------------------------------------

    const std::int32_t scalar_result =
        dot_scalar_q7(
            input.data(),
            weights.data(),
            element_count);


    const std::int32_t neon_result =
        dot_neon_q7(
            input.data(),
            weights.data(),
            element_count);


    const std::int32_t sdot_result =
        dot_sdot_q7(
            input.data(),
            weights.data(),
            element_count);


    std::cout
        << "Scalar result      : "
        << scalar_result
        << '\n';

    std::cout
        << "NEON result        : "
        << neon_result
        << '\n';

    std::cout
        << "SDOT result        : "
        << sdot_result
        << '\n';


    if (scalar_result != neon_result ||
        scalar_result != sdot_result)
    {
        std::cerr
            << "Verification       : FAIL\n";

        return 1;
    }


    std::cout
        << "Verification       : PASS\n\n";


    // Warm all three implementations before collecting trials.
    for (int i = 0; i < 1000; ++i)
    {
        g_sink +=
            dot_scalar_q7(
                input.data(),
                weights.data(),
                element_count);

        g_sink +=
            dot_neon_q7(
                input.data(),
                weights.data(),
                element_count);

        g_sink +=
            dot_sdot_q7(
                input.data(),
                weights.data(),
                element_count);
    }


    const double scalar_ns =
        median_of_seven(
            dot_scalar_q7,
            input.data(),
            weights.data(),
            element_count,
            iterations);


    const double neon_ns =
        median_of_seven(
            dot_neon_q7,
            input.data(),
            weights.data(),
            element_count,
            iterations);


    const double sdot_ns =
        median_of_seven(
            dot_sdot_q7,
            input.data(),
            weights.data(),
            element_count,
            iterations);


    const double neon_speedup =
        scalar_ns / neon_ns;


    const double sdot_speedup =
        scalar_ns / sdot_ns;


    const double sdot_vs_neon =
        neon_ns / sdot_ns;


    std::cout
        << std::fixed
        << std::setprecision(3);


    std::cout
        << "-------------------------------------------------\n";

    std::cout
        << "Implementation       ns/dot       ns/element\n";

    std::cout
        << "-------------------------------------------------\n";


    std::cout
        << "Scalar MADD       "
        << std::setw(10)
        << scalar_ns
        << "      "
        << scalar_ns / element_count
        << '\n';


    std::cout
        << "NEON SMULL       "
        << std::setw(10)
        << neon_ns
        << "      "
        << neon_ns / element_count
        << '\n';


    std::cout
        << "NEON SDOT        "
        << std::setw(10)
        << sdot_ns
        << "      "
        << sdot_ns / element_count
        << '\n';


    std::cout
        << "-------------------------------------------------\n\n";


    std::cout
        << "General NEON vs Scalar : "
        << neon_speedup
        << "x\n";


    std::cout
        << "SDOT vs Scalar         : "
        << sdot_speedup
        << "x\n";


    std::cout
        << "SDOT vs General NEON   : "
        << sdot_vs_neon
        << "x\n";


    return 0;
}