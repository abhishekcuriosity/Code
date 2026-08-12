#pragma once

#include <cstddef>
#include <cstdint>


std::int32_t dot_forced_scalar_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);


std::int32_t dot_auto_v80_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);


std::int32_t dot_auto_v84_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);
