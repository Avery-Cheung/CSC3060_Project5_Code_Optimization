#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

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
    if (n <= 0) return;

    const int N = n;
    const size_t size = (size_t)N * (size_t)N;

    // local transpose buffer
    std::vector<float> BT(size);
    for (int i = 0; i < N; ++i) {
        const float* __restrict b_row = &B[(size_t)i * N];
        for (int j = 0; j < N; ++j) {
            BT[(size_t)j * N + i] = b_row[j];
        }
    }

    std::fill(C.begin(), C.end(), 0.0f);

    // Blocking and unrolling parameters tuned for general CPUs
    constexpr int BLOCK = 32;
    for (int ii = 0; ii < N; ii += BLOCK) {
        for (int jj = 0; jj < N; jj += BLOCK) {
            for (int kk = 0; kk < N; kk += BLOCK) {
                int i_max = std::min(ii + BLOCK, N);
                int j_max = std::min(jj + BLOCK, N);
                int k_max = std::min(kk + BLOCK, N);

                for (int i = ii; i < i_max; ++i) {
                    const float* __restrict a_row = &A[(size_t)i * N];
                    float* __restrict c_row = &C[(size_t)i * N];

                    int j = jj;
                    // process j in chunks of 4 for better ILP
                    for (; j + 3 < j_max; j += 4) {
                        const float* __restrict b0 = &BT[(size_t)(j + 0) * N];
                        const float* __restrict b1 = &BT[(size_t)(j + 1) * N];
                        const float* __restrict b2 = &BT[(size_t)(j + 2) * N];
                        const float* __restrict b3 = &BT[(size_t)(j + 3) * N];

                        float s0 = c_row[j + 0];
                        float s1 = c_row[j + 1];
                        float s2 = c_row[j + 2];
                        float s3 = c_row[j + 3];

                        int k = kk;
                        int k_unroll_end = k_max - ((k_max - k) % 4);
                        for (; k < k_unroll_end; k += 4) {
                            float a0 = a_row[k];
                            s0 += a0 * b0[k]; s1 += a0 * b1[k]; s2 += a0 * b2[k]; s3 += a0 * b3[k];
                            float a1 = a_row[k + 1];
                            s0 += a1 * b0[k + 1]; s1 += a1 * b1[k + 1]; s2 += a1 * b2[k + 1]; s3 += a1 * b3[k + 1];
                            float a2 = a_row[k + 2];
                            s0 += a2 * b0[k + 2]; s1 += a2 * b1[k + 2]; s2 += a2 * b2[k + 2]; s3 += a2 * b3[k + 2];
                            float a3 = a_row[k + 3];
                            s0 += a3 * b0[k + 3]; s1 += a3 * b1[k + 3]; s2 += a3 * b2[k + 3]; s3 += a3 * b3[k + 3];
                        }
                        for (; k < k_max; ++k) {
                            float a = a_row[k];
                            s0 += a * b0[k]; s1 += a * b1[k]; s2 += a * b2[k]; s3 += a * b3[k];
                        }

                        c_row[j + 0] = s0;
                        c_row[j + 1] = s1;
                        c_row[j + 2] = s2;
                        c_row[j + 3] = s3;
                    }

                    // tail for remaining j
                    for (; j < j_max; ++j) {
                        const float* __restrict b_row = &BT[(size_t)j * N];
                        float sum = c_row[j];
                        int k = kk;
                        int k_unroll_end = k_max - ((k_max - k) % 4);
                        for (; k < k_unroll_end; k += 4) {
                            sum += a_row[k] * b_row[k];
                            sum += a_row[k + 1] * b_row[k + 1];
                            sum += a_row[k + 2] * b_row[k + 2];
                            sum += a_row[k + 3] * b_row[k + 3];
                        }
                        for (; k < k_max; ++k) {
                            sum += a_row[k] * b_row[k];
                        }
                        c_row[j] = sum;
                    }
                }
            }
        }
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
