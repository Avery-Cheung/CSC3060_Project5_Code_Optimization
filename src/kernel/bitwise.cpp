#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result,
                 std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5A;
    constexpr std::uint8_t kMaskHi = 0xC3;

    constexpr std::uint64_t maskLo64 =
        0x5A5A5A5A5A5A5A5AULL;

    constexpr std::uint64_t maskHi64 =
        0xC3C3C3C3C3C3C3C3ULL;

    const std::size_t n =
        std::min({result.size(), a.size(), b.size()});

    if (n == 0) {
        return;
    }

    const auto* pa =
        reinterpret_cast<const std::uint8_t*>(a.data());

    const auto* pb =
        reinterpret_cast<const std::uint8_t*>(b.data());

    auto* pr =
        reinterpret_cast<std::uint8_t*>(result.data());

    constexpr std::size_t chunk = sizeof(std::uint64_t);

    const std::size_t limit =
        (n / chunk) * chunk;

    // Process 8 bytes at a time
    for (std::size_t i = 0; i < limit; i += chunk) {

        std::uint64_t va;
        std::uint64_t vb;

        std::memcpy(&va, pa + i, chunk);
        std::memcpy(&vb, pb + i, chunk);

        const std::uint64_t shared = va & vb;
        const std::uint64_t either = va | vb;
        const std::uint64_t diff = va ^ vb;

        const std::uint64_t mixed0 =
            (diff & maskLo64) |
            (~shared & ~maskLo64);

        const std::uint64_t mixed1 =
            ((either ^ maskHi64) &
            (shared | ~maskHi64)) ^ diff;

        const std::uint64_t res =
            mixed0 ^ mixed1;

        std::memcpy(pr + i, &res, chunk);
    }

    // Remaining bytes
    for (std::size_t i = limit; i < n; ++i) {

        const auto ua =
            static_cast<std::uint8_t>(pa[i]);

        const auto ub =
            static_cast<std::uint8_t>(pb[i]);

        const auto shared =
            static_cast<std::uint8_t>(ua & ub);

        const auto either =
            static_cast<std::uint8_t>(ua | ub);

        const auto diff =
            static_cast<std::uint8_t>(ua ^ ub);

        const auto mixed0 =
            static_cast<std::uint8_t>(
                (diff & kMaskLo) |
                (~shared & ~kMaskLo));

        const auto mixed1 =
            static_cast<std::uint8_t>(
                ((either ^ kMaskHi) &
                (shared | ~kMaskHi)) ^ diff);

        pr[i] =
            static_cast<std::uint8_t>(
                mixed0 ^ mixed1);
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
