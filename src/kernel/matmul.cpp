#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

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

    const int N = n;
    const size_t SIZE = (size_t)N * N;

    std::vector<float> BT(SIZE);

    // transpose B
    for (int i = 0; i < N; ++i) {
        const float* __restrict b_row = &B[(size_t)i * N];

        for (int j = 0; j < N; ++j) {
            BT[(size_t)j * N + i] = b_row[j];
        }
    }

    std::fill(C.begin(), C.end(), 0.0f);

    constexpr int BLOCK = 64;

    for (int ii = 0; ii < N; ii += BLOCK) {

        for (int jj = 0; jj < N; jj += BLOCK) {

            for (int kk = 0; kk < N; kk += BLOCK) {

                const int i_max = std::min(ii + BLOCK, N);
                const int j_max = std::min(jj + BLOCK, N);
                const int k_max = std::min(kk + BLOCK, N);

                for (int i = ii; i < i_max; ++i) {

                    const float* __restrict a_row =
                        &A[(size_t)i * N];

                    float* __restrict c_row =
                        &C[(size_t)i * N];

                    for (int j = jj; j < j_max; ++j) {

                        const float* __restrict b_row =
                            &BT[(size_t)j * N];

                        __m256 vsum = _mm256_setzero_ps();

                        int k = kk;

                        // AVX2 vectorized loop
                        for (; k + 8 <= k_max; k += 8) {

                            __m256 va =
                                _mm256_loadu_ps(a_row + k);

                            __m256 vb =
                                _mm256_loadu_ps(b_row + k);

                            vsum =
                                _mm256_fmadd_ps(va, vb, vsum);
                        }

                        // horizontal sum
                        alignas(32) float temp[8];

                        _mm256_store_ps(temp, vsum);

                        float sum =
                            temp[0] + temp[1] +
                            temp[2] + temp[3] +
                            temp[4] + temp[5] +
                            temp[6] + temp[7];

                        // tail case
                        for (; k < k_max; ++k) {
                            sum += a_row[k] * b_row[k];
                        }

                        c_row[j] += sum;
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
