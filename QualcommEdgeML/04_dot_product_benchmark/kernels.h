#pragma once

#include <cstddef>
#include <cstdint>

std::int32_t dot_scalar_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);

std::int32_t dot_neon_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);

std::int32_t dot_sdot_q7(
    const std::int8_t* input,
    const std::int8_t* weights,
    std::size_t count);