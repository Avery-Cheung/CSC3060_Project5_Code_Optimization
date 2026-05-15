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

    // ================================
    // 1. 初始化输出矩阵
    // ================================
    std::fill(C.begin(), C.end(), 0.0f);

    // 直接拿底层指针：减少 vector bounds / function call 开销
    const float* matA = A.data();
    const float* matB = B.data();
    float* matC = C.data();

    // ================================
    // 2. 分块大小（cache blocking 核心参数）
    // ================================
    constexpr int TILE = 64;

    // ================================
    // 3. 三重分块循环（tile traversal）
    //    外层控制大块位置
    // ================================
    for (int bi = 0; bi < n; bi += TILE) {
        for (int bk = 0; bk < n; bk += TILE) {
            for (int bj = 0; bj < n; bj += TILE) {

                const int i_max = std::min(bi + TILE, n);
                const int k_max = std::min(bk + TILE, n);
                const int j_max = std::min(bj + TILE, n);

                // ================================
                // 4. tile 内部计算
                // ================================
                for (int i = bi; i < i_max; ++i) {

                    const int rowA = i * n;
                    const int rowC = i * n;

                    for (int k = bk; k < k_max; ++k) {

                        // cache-friendly：A 的一个标量
                        const float a_val = matA[rowA + k];

                        // B 和 C 在 j 维度上是连续访问
                        const int rowB = k * n;

                        float* c_ptr = matC + rowC;

                        // ================================
                        // 5. 最内层向量化友好循环
                        // ================================
                        int j = bj;

                        // 手动 unroll（轻量优化）
                        for (; j + 3 < j_max; j += 4) {
                            c_ptr[j]     += a_val * matB[rowB + j];
                            c_ptr[j + 1] += a_val * matB[rowB + j + 1];
                            c_ptr[j + 2] += a_val * matB[rowB + j + 2];
                            c_ptr[j + 3] += a_val * matB[rowB + j + 3];
                        }

                        // tail case
                        for (; j < j_max; ++j) {
                            c_ptr[j] += a_val * matB[rowB + j];
                        }
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
