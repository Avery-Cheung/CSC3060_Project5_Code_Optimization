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

    constexpr int COL_TILE = 128;

    // 初始化输出矩阵
    std::fill(C.begin(), C.end(), 0.0f);

    const float* __restrict lhs = A.data();
    const float* __restrict rhs = B.data();
    float* __restrict out = C.data();

    // 每个线程处理一段连续行
    auto compute_rows = [&](int begin_row, int end_row) {

        for (int row = begin_row; row < end_row; ++row) {

            float* __restrict out_row =
                out + static_cast<std::size_t>(row) * n;

            const float* __restrict lhs_row =
                lhs + static_cast<std::size_t>(row) * n;

            // 按列分块，提升 cache locality
            for (int col_block = 0; col_block < n; col_block += COL_TILE) {

                const int col_limit =
                    std::min(col_block + COL_TILE, n);

                for (int k = 0; k < n; ++k) {

                    const float scale = lhs_row[k];

                    const float* __restrict rhs_row =
                        rhs + static_cast<std::size_t>(k) * n;

                    int col = col_block;

                    // 手动展开内层循环
                    for (; col + 7 < col_limit; col += 8) {

                        out_row[col]     += scale * rhs_row[col];
                        out_row[col + 1] += scale * rhs_row[col + 1];
                        out_row[col + 2] += scale * rhs_row[col + 2];
                        out_row[col + 3] += scale * rhs_row[col + 3];

                        out_row[col + 4] += scale * rhs_row[col + 4];
                        out_row[col + 5] += scale * rhs_row[col + 5];
                        out_row[col + 6] += scale * rhs_row[col + 6];
                        out_row[col + 7] += scale * rhs_row[col + 7];
                    }

                    // 处理剩余元素
                    for (; col < col_limit; ++col) {
                        out_row[col] += scale * rhs_row[col];
                    }
                }
            }
        }
    };

    // 根据硬件线程数决定并行规模
    const unsigned cpu_threads = std::thread::hardware_concurrency();

    const int worker_count =
        std::min<int>(
            16,
            std::max<int>(
                1,
                cpu_threads == 0 ? 8 : static_cast<int>(cpu_threads)));

    // 小矩阵直接单线程
    if (worker_count <= 1 || n < 128) {
        compute_rows(0, n);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(worker_count - 1));

    const int chunk =
        (n + worker_count - 1) / worker_count;

    int current_row = 0;

    for (int t = 1; t < worker_count; ++t) {

        const int next_row =
            std::min(n, current_row + chunk);

        if (current_row >= next_row) {
            break;
        }

        pool.emplace_back(
            compute_rows,
            current_row,
            next_row);

        current_row = next_row;
    }

    // 主线程处理剩余部分
    if (current_row < n) {
        compute_rows(current_row, n);
    }

    // 等待所有线程结束
    for (auto& th : pool) {
        th.join();
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