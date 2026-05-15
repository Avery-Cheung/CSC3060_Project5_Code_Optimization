#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <thread>

void initialize_matmul(matmul_args& args, int n, uint32_t seed) {
    if (n <= 0) {
        throw std::invalid_argument("initialize_matmul: n must be positive.");
    }

    args.n = n;
    args.epsilon = 1e-3;

    const size_t elem_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    args.A.resize(elem_count);
    args.B.resize(elem_count);
    args.C.assign(elem_count, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elem_count; ++i) {
        args.A[i] = dist(rng);
        args.B[i] = dist(rng);
    }
}

void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {

    constexpr int TILE_W = 128;

    std::fill(C.begin(), C.end(), 0.0f);

    const float* __restrict matA = A.data();
    const float* __restrict matB = B.data();
    float* __restrict matC = C.data();

    auto kernel = [&](int start, int stop) {

        for (int r = start; r < stop; ++r) {

            float* dst = matC + static_cast<std::size_t>(r) * n;
            const float* srcA =
                matA + static_cast<std::size_t>(r) * n;

            for (int jb = 0; jb < n; jb += TILE_W) {

                const int limit =
                    (jb + TILE_W < n) ? (jb + TILE_W) : n;

                for (int kk = 0; kk < n; ++kk) {

                    const float coeff = srcA[kk];
                    const float* srcB =
                        matB + static_cast<std::size_t>(kk) * n + jb;

                    float* out = dst + jb;

                    int width = limit - jb;

                    // 8-way unroll
                    while (width >= 8) {

                        out[0] += coeff * srcB[0];
                        out[1] += coeff * srcB[1];
                        out[2] += coeff * srcB[2];
                        out[3] += coeff * srcB[3];

                        out[4] += coeff * srcB[4];
                        out[5] += coeff * srcB[5];
                        out[6] += coeff * srcB[6];
                        out[7] += coeff * srcB[7];

                        out += 8;
                        srcB += 8;
                        width -= 8;
                    }

                    // tail
                    while (width > 0) {
                        *out += coeff * (*srcB);
                        ++out;
                        ++srcB;
                        --width;
                    }
                }
            }
        }
    };

    unsigned concurrency = std::thread::hardware_concurrency();

    if (concurrency == 0) {
        concurrency = 8;
    }

    const int workers =
        std::max(1u, std::min(16u, concurrency));

    if (workers == 1 || n < 128) {
        kernel(0, n);
        return;
    }

    std::vector<std::thread> jobs;

    const int stride =
        (n + workers - 1) / workers;

    int begin = 0;

    for (int id = 0; id + 1 < workers; ++id) {

        int end = std::min(begin + stride, n);

        if (begin >= end) {
            break;
        }

        jobs.emplace_back(kernel, begin, end);

        begin = end;
    }

    kernel(begin, n);

    for (auto& t : jobs) {
        t.join();
    }
}

void naive_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    naive_matmul(args.C, args.A, args.B, args.n);
}

void stu_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    stu_matmul(args.C, args.A, args.B, args.n);
}

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<matmul_args*>(stu_ctx);
    auto& ref_args = *static_cast<matmul_args*>(ref_ctx);

    if (stu_args.C.size() != ref_args.C.size()) {
        debug_log("\tDEBUG: matmul size mismatch: stu={} ref={}\n",
                  stu_args.C.size(),
                  ref_args.C.size());
        return false;
    }

    const double eps = ref_args.epsilon;
    const int n = ref_args.n;
    double max_rel = 0.0;
    size_t worst_idx = 0;

    for (size_t i = 0; i < ref_args.C.size(); ++i) {
        const double r = static_cast<double>(ref_args.C[i]);
        const double s = static_cast<double>(stu_args.C[i]);
        const double diff = std::abs(s - r);
        const double rel = (std::abs(r) > 1e-9) ? diff / std::abs(r) : diff;

        if (rel > max_rel) {
            max_rel = rel;
            worst_idx = i;
        }

        if (rel > eps) {
            const size_t row = (n > 0) ? (i / static_cast<size_t>(n)) : 0;
            const size_t col = (n > 0) ? (i % static_cast<size_t>(n)) : 0;
            debug_log("\tDEBUG: matmul fail at index {} (row={}, col={}): ref={} stu={} rel={} eps={}\n",
                      i,
                      row,
                      col,
                      ref_args.C[i],
                      stu_args.C[i],
                      rel,
                      eps);
            return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}