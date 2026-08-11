#include "kernels.h"

#include <arm64_neon.h>

#if !defined(_M_ARM64)
#error This experiment requires native ARM64.
#endif

#define NOINLINE __declspec(noinline)


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