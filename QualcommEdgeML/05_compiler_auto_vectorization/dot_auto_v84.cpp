#include "kernels.h"

#if !defined(_M_ARM64)
#error This experiment requires native ARM64.
#endif

#define NOINLINE __declspec(noinline)


NOINLINE std::int32_t dot_auto_v84_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count)
{
    std::int32_t accumulator = 0;


    for (std::size_t i = 0;
        i < count;
        ++i)
    {
        accumulator +=
            static_cast<std::int32_t>(input[i]) *
            static_cast<std::int32_t>(weights[i]);
    }


    return accumulator;
}