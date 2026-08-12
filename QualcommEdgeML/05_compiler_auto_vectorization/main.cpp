#include "kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>


#if !defined(_M_ARM64)
#error This experiment requires native ARM64.
#endif


using DotFunction =
std::int32_t(*)(
    const std::int8_t*,
    const std::int8_t*,
    std::size_t);


volatile std::int64_t g_sink = 0;


double benchmark_ns_per_dot(
    DotFunction function,
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count,
    std::size_t iterations)
{
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


    for (std::size_t i = 0;
        i < element_count;
        ++i)
    {
        input[i] =
            static_cast<std::int8_t>(
                static_cast<int>(
                    (i * 17) % 127)
                - 63);


        weights[i] =
            static_cast<std::int8_t>(
                static_cast<int>(
                    (i * 29) % 127)
                - 63);
    }


    std::cout
        << "Qualcomm ARM64 Edge ML - Project 5\n"
        << "Compiler Auto-Vectorization\n\n";


    std::cout
        << "Elements / dot     : "
        << element_count
        << '\n';

    std::cout
        << "Iterations / trial : "
        << iterations
        << '\n';

    std::cout
        << "Trials             : 7\n\n";


    const std::int32_t scalar_result =
        dot_forced_scalar_q7(
            input.data(),
            weights.data(),
            element_count);


    const std::int32_t auto_v80_result =
        dot_auto_v80_q7(
            input.data(),
            weights.data(),
            element_count);


    const std::int32_t auto_v84_result =
        dot_auto_v84_q7(
            input.data(),
            weights.data(),
            element_count);


    std::cout
        << "Forced scalar result : "
        << scalar_result
        << '\n';

    std::cout
        << "Auto Armv8.0 result   : "
        << auto_v80_result
        << '\n';

    std::cout
        << "Auto Armv8.4 result   : "
        << auto_v84_result
        << '\n';


    if (scalar_result != auto_v80_result ||
        scalar_result != auto_v84_result)
    {
        std::cerr
            << "\nVerification : FAIL\n";

        return 1;
    }


    std::cout
        << "\nVerification : PASS\n\n";


    const double scalar_ns =
        median_of_seven(
            dot_forced_scalar_q7,
            input.data(),
            weights.data(),
            element_count,
            iterations);


    const double auto_v80_ns =
        median_of_seven(
            dot_auto_v80_q7,
            input.data(),
            weights.data(),
            element_count,
            iterations);


    const double auto_v84_ns =
        median_of_seven(
            dot_auto_v84_q7,
            input.data(),
            weights.data(),
            element_count,
            iterations);


    std::cout
        << std::fixed
        << std::setprecision(3);


    std::cout
        << "-----------------------------------------------\n";

    std::cout
        << "Implementation        ns/dot      ns/element\n";

    std::cout
        << "-----------------------------------------------\n";


    std::cout
        << "Forced scalar     "
        << std::setw(10)
        << scalar_ns
        << "   "
        << scalar_ns / element_count
        << '\n';


    std::cout
        << "Auto Armv8.0      "
        << std::setw(10)
        << auto_v80_ns
        << "   "
        << auto_v80_ns / element_count
        << '\n';


    std::cout
        << "Auto Armv8.4      "
        << std::setw(10)
        << auto_v84_ns
        << "   "
        << auto_v84_ns / element_count
        << '\n';


    std::cout
        << "-----------------------------------------------\n\n";


    std::cout
        << "Auto v8.0 vs Scalar : "
        << scalar_ns / auto_v80_ns
        << "x\n";


    std::cout
        << "Auto v8.4 vs Scalar : "
        << scalar_ns / auto_v84_ns
        << "x\n";


    std::cout
        << "Auto v8.4 vs v8.0   : "
        << auto_v80_ns / auto_v84_ns
        << "x\n";


    return 0;
}