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
static inline std::uint64_t process_chunk(std::uint64_t va, std::uint64_t vb,
                                           std::uint64_t mLo, std::uint64_t mHi) {
    std::uint64_t shared = va & vb;
    std::uint64_t either = va | vb;
    std::uint64_t diff   = va ^ vb;
    std::uint64_t mixed0 = (diff & mLo) | (~shared & ~mLo);
    std::uint64_t mixed1 = ((either ^ mHi) & (shared | ~mHi)) ^ diff;
    return mixed0 ^ mixed1;
}

void stu_bitwise(std::span<std::int8_t> result,
                 std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    if (n == 0) return;

    const std::uint8_t* pa = reinterpret_cast<const std::uint8_t*>(a.data());
    const std::uint8_t* pb = reinterpret_cast<const std::uint8_t*>(b.data());
    std::uint8_t* pr = reinterpret_cast<std::uint8_t*>(result.data());

    constexpr std::uint64_t mLo = 0x5A5A5A5A5A5A5A5AULL;
    constexpr std::uint64_t mHi = 0xC3C3C3C3C3C3C3C3ULL;
    constexpr std::size_t CHUNK = sizeof(std::uint64_t);
    const std::size_t limit = (n / CHUNK) * CHUNK;

    std::size_t i = 0;

    // ── 8x unrolled: process 64 bytes per iteration ─────────────────
    for (; i + 8 * CHUNK <= limit; i += 8 * CHUNK) {
        std::uint64_t va0, vb0, va1, vb1, va2, vb2, va3, vb3;
        std::uint64_t va4, vb4, va5, vb5, va6, vb6, va7, vb7;

        __builtin_memcpy(&va0, pa + i + 0*CHUNK, CHUNK);
        __builtin_memcpy(&vb0, pb + i + 0*CHUNK, CHUNK);
        __builtin_memcpy(&va1, pa + i + 1*CHUNK, CHUNK);
        __builtin_memcpy(&vb1, pb + i + 1*CHUNK, CHUNK);
        __builtin_memcpy(&va2, pa + i + 2*CHUNK, CHUNK);
        __builtin_memcpy(&vb2, pb + i + 2*CHUNK, CHUNK);
        __builtin_memcpy(&va3, pa + i + 3*CHUNK, CHUNK);
        __builtin_memcpy(&vb3, pb + i + 3*CHUNK, CHUNK);
        __builtin_memcpy(&va4, pa + i + 4*CHUNK, CHUNK);
        __builtin_memcpy(&vb4, pb + i + 4*CHUNK, CHUNK);
        __builtin_memcpy(&va5, pa + i + 5*CHUNK, CHUNK);
        __builtin_memcpy(&vb5, pb + i + 5*CHUNK, CHUNK);
        __builtin_memcpy(&va6, pa + i + 6*CHUNK, CHUNK);
        __builtin_memcpy(&vb6, pb + i + 6*CHUNK, CHUNK);
        __builtin_memcpy(&va7, pa + i + 7*CHUNK, CHUNK);
        __builtin_memcpy(&vb7, pb + i + 7*CHUNK, CHUNK);

        std::uint64_t r0 = process_chunk(va0, vb0, mLo, mHi);
        std::uint64_t r1 = process_chunk(va1, vb1, mLo, mHi);
        std::uint64_t r2 = process_chunk(va2, vb2, mLo, mHi);
        std::uint64_t r3 = process_chunk(va3, vb3, mLo, mHi);
        std::uint64_t r4 = process_chunk(va4, vb4, mLo, mHi);
        std::uint64_t r5 = process_chunk(va5, vb5, mLo, mHi);
        std::uint64_t r6 = process_chunk(va6, vb6, mLo, mHi);
        std::uint64_t r7 = process_chunk(va7, vb7, mLo, mHi);

        __builtin_memcpy(pr + i + 0*CHUNK, &r0, CHUNK);
        __builtin_memcpy(pr + i + 1*CHUNK, &r1, CHUNK);
        __builtin_memcpy(pr + i + 2*CHUNK, &r2, CHUNK);
        __builtin_memcpy(pr + i + 3*CHUNK, &r3, CHUNK);
        __builtin_memcpy(pr + i + 4*CHUNK, &r4, CHUNK);
        __builtin_memcpy(pr + i + 5*CHUNK, &r5, CHUNK);
        __builtin_memcpy(pr + i + 6*CHUNK, &r6, CHUNK);
        __builtin_memcpy(pr + i + 7*CHUNK, &r7, CHUNK);
    }
    // ── Tail chunks ────────────────────────────────────────────────
    for (; i < limit; i += CHUNK) {
        std::uint64_t va, vb;
        __builtin_memcpy(&va, pa + i, CHUNK);
        __builtin_memcpy(&vb, pb + i, CHUNK);
        std::uint64_t r = process_chunk(va, vb, mLo, mHi);
        __builtin_memcpy(pr + i, &r, CHUNK);
    }
    // ── Tail bytes ─────────────────────────────────────────────────
    for (; i < n; ++i) {
        std::uint8_t ua = pa[i];
        std::uint8_t ub = pb[i];
        std::uint8_t shared = ua & ub;
        std::uint8_t either = ua | ub;
        std::uint8_t diff   = ua ^ ub;
        std::uint8_t m0 = (diff & 0x5Au) | (~shared & ~0x5Au);
        std::uint8_t m1 = ((either ^ 0xC3u) & (shared | ~0xC3u)) ^ diff;
        pr[i] = m0 ^ m1;
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
