#pragma once

#include <cstddef>

namespace attention::details {
    static inline constexpr std::size_t kDefaultRank = 3;

    static constexpr auto kMsgIndexOutOfRange = "Index out of range";
    static constexpr auto kMsgBatchIndexOutOfRange = "Batch index out of range";
    static constexpr auto kMsgMatMulInnerDimMismatch = "Inner dimensions must match (K-dimension mismatch)";
    static constexpr auto kMsgMatMulOutputDimMismatch = "Output tensor dimensions mismatch";
}