#pragma once

namespace attention::details {
    static inline constexpr std::size_t kDefaultRank = 3;

    static inline constexpr std::size_t kAlignment = 128;
    constexpr std::size_t kBlockSize = 128;

    static constexpr auto kMsgIndexOutOfRange = "Index out of range";
    static constexpr auto kMsgBatchIndexOutOfRange = "Batch index out of range";
    static constexpr auto kMsgMatMulInnerDimMismatch = "Inner dimensions must match (K-dimension mismatch)";
    static constexpr auto kMsgMatMulOutputDimMismatch = "Output tensor dimensions mismatch";
    static constexpr auto kMsgMatMulAliasingDetected = "Input and Result tensors share the same memory";

}