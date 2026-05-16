#include "grff.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <random>

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,fma")

#include <immintrin.h>


void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation (A Simplified Gated Residual Feature Fusion (GRFF))
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    
    // Intermediate buffers
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    // Stage 1: Gate
    for (size_t i = 0; i < n; ++i) 
        G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);

    // Stage 2: Update A (Residual)
    for (size_t i = 0; i < n; ++i) 
        A_prime[i] = args.a_features[i] + G[i];

    // Stage 3: Global Feature Scaling
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_a += A_prime[i];
    }
    float avg_a = sum_a / static_cast<float>(n);

    // Stage 4: Update A (Smooth)
    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) {
        Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    }

    // Stage 5: Update B (Suppression)
    for (size_t i = 0; i < n; ++i) 
        B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;

    // Stage 6: Context Integration 
    for (size_t i = 0; i < n; ++i) 
        C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));

    // Stage 7: Hidden Interaction
    for (size_t i = 0; i < n; ++i) 
        H[i] = Smooth_A[i] * C_prime[i];

    // Stage 8: Normalization
    for (size_t i = 0; i < n; ++i) 
        E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));

    // Stage 9: Final Output (ReLU)
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
void stu_grff(grff_args& args) {
    const size_t len = args.a_features.size();
    if (len == 0) return;

    float* __restrict__ scratch_ptr = args.scratch.data();
    const float* __restrict__ feat_a = args.a_features.data();
    const float* __restrict__ feat_b = args.b_features.data();
    const float* __restrict__ feat_c = args.c_features.data();
    float* __restrict__ result_ptr = args.f_output.data();

    float accum = 0.0f;

    #pragma omp parallel
    {
        float local_accum = 0.0f;
        __m256 v_local_accum = _mm256_setzero_ps();

        const __m256 ones   = _mm256_set1_ps(1.0f);
        const __m256 halves = _mm256_set1_ps(0.5f);
        const __m256 twos   = _mm256_set1_ps(2.0f);
        const __m256 zeroes = _mm256_setzero_ps();
        // AVX2 用于求绝对值的位掩码 (0x7FFFFFFF)
        const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        // =====================================
        // Stage 1: Gate + Residual Fusion
        // =====================================
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < len; i += 16) {
            if (i + 15 < len) {
                __m256 a0 = _mm256_loadu_ps(feat_a + i);
                __m256 a1 = _mm256_loadu_ps(feat_a + i + 8);
                __m256 b0 = _mm256_loadu_ps(feat_b + i);
                __m256 b1 = _mm256_loadu_ps(feat_b + i + 8);

                __m256 p0 = _mm256_mul_ps(a0, b0);
                __m256 p1 = _mm256_mul_ps(a1, b1);

                __m256 abs_p0 = _mm256_and_ps(p0, sign_mask);
                __m256 abs_p1 = _mm256_and_ps(p1, sign_mask);

                __m256 den0 = _mm256_add_ps(ones, abs_p0);
                __m256 den1 = _mm256_add_ps(ones, abs_p1);

                // AVX2 rcp 近似 + 1次 Newton-Raphson 迭代代替极慢的除法
                __m256 y0_0 = _mm256_rcp_ps(den0);
                __m256 y0_1 = _mm256_rcp_ps(den1);

                __m256 inv0 = _mm256_mul_ps(y0_0, _mm256_fnmadd_ps(den0, y0_0, twos));
                __m256 inv1 = _mm256_mul_ps(y0_1, _mm256_fnmadd_ps(den1, y0_1, twos));

                __m256 g0 = _mm256_mul_ps(halves, _mm256_fmadd_ps(p0, inv0, ones));
                __m256 g1 = _mm256_mul_ps(halves, _mm256_fmadd_ps(p1, inv1, ones));

                __m256 f0 = _mm256_add_ps(a0, g0);
                __m256 f1 = _mm256_add_ps(a1, g1);

                v_local_accum = _mm256_add_ps(v_local_accum, _mm256_add_ps(f0, f1));

                _mm256_storeu_ps(scratch_ptr + i, f0);
                _mm256_storeu_ps(scratch_ptr + i + 8, f1);
            } else {
                for (size_t j = i; j < len; ++j) {
                    float p = feat_a[j] * feat_b[j];
                    float g = 0.5f * (p / (1.0f + std::abs(p)) + 1.0f);
                    float f = feat_a[j] + g;
                    scratch_ptr[j] = f;
                    local_accum += f;
                }
                break;
            }
        }

        // 局部累加汇总
        float temp[8];
        _mm256_storeu_ps(temp, v_local_accum);
        for(int k = 0; k < 8; ++k) local_accum += temp[k];

        #pragma omp atomic
        accum += local_accum;

        #pragma omp barrier

        float global_mean = accum / static_cast<float>(len);
        __m256 v_global_mean = _mm256_set1_ps(global_mean);

        // 串行执行边界条件
        #pragma omp single
        {
            float smoothed = scratch_ptr[0];
            float recovered_gate = smoothed - feat_a[0];
            float magnitude = std::abs(smoothed);
            float denominator = 1.0f + magnitude;
            float contextual = feat_c[0] + smoothed / denominator;
            float suppressed = feat_b[0] * (1.0f - recovered_gate) * global_mean;
            float interacted = smoothed * contextual;
            float normalized = (interacted + suppressed) / denominator;
            result_ptr[0] = std::max(contextual - normalized, 0.0f);
        }

        // =====================================
        // Stage 2: Smoothing + Reconstruction
        // =====================================
        #pragma omp for schedule(static)
        for (size_t i = 1; i < len; i += 16) {
            if (i + 15 < len) {
                __m256 curr0 = _mm256_loadu_ps(scratch_ptr + i);
                __m256 curr1 = _mm256_loadu_ps(scratch_ptr + i + 8);

                // 通过错位读取来实现滑窗，避免了标量计算
                __m256 prev0 = _mm256_loadu_ps(scratch_ptr + i - 1);
                __m256 prev1 = _mm256_loadu_ps(scratch_ptr + i + 7);

                __m256 s0 = _mm256_mul_ps(_mm256_add_ps(curr0, prev0), halves);
                __m256 s1 = _mm256_mul_ps(_mm256_add_ps(curr1, prev1), halves);

                __m256 a0 = _mm256_loadu_ps(feat_a + i);
                __m256 a1 = _mm256_loadu_ps(feat_a + i + 8);

                __m256 rec_g0 = _mm256_sub_ps(curr0, a0);
                __m256 rec_g1 = _mm256_sub_ps(curr1, a1);

                __m256 abs_s0 = _mm256_and_ps(s0, sign_mask);
                __m256 abs_s1 = _mm256_and_ps(s1, sign_mask);

                __m256 den0 = _mm256_add_ps(ones, abs_s0);
                __m256 den1 = _mm256_add_ps(ones, abs_s1);

                __m256 y0_0 = _mm256_rcp_ps(den0);
                __m256 y0_1 = _mm256_rcp_ps(den1);

                __m256 inv0 = _mm256_mul_ps(y0_0, _mm256_fnmadd_ps(den0, y0_0, twos));
                __m256 inv1 = _mm256_mul_ps(y0_1, _mm256_fnmadd_ps(den1, y0_1, twos));

                __m256 c0 = _mm256_loadu_ps(feat_c + i);
                __m256 c1 = _mm256_loadu_ps(feat_c + i + 8);

                __m256 ctx0 = _mm256_fmadd_ps(s0, inv0, c0);
                __m256 ctx1 = _mm256_fmadd_ps(s1, inv1, c1);

                __m256 b0 = _mm256_loadu_ps(feat_b + i);
                __m256 b1 = _mm256_loadu_ps(feat_b + i + 8);

                __m256 omg0 = _mm256_sub_ps(ones, rec_g0);
                __m256 omg1 = _mm256_sub_ps(ones, rec_g1);

                __m256 supp0 = _mm256_mul_ps(_mm256_mul_ps(b0, v_global_mean), omg0);
                __m256 supp1 = _mm256_mul_ps(_mm256_mul_ps(b1, v_global_mean), omg1);

                __m256 int0 = _mm256_mul_ps(s0, ctx0);
                __m256 int1 = _mm256_mul_ps(s1, ctx1);

                __m256 norm0 = _mm256_mul_ps(_mm256_add_ps(int0, supp0), inv0);
                __m256 norm1 = _mm256_mul_ps(_mm256_add_ps(int1, supp1), inv1);

                __m256 out0 = _mm256_max_ps(_mm256_sub_ps(ctx0, norm0), zeroes);
                __m256 out1 = _mm256_max_ps(_mm256_sub_ps(ctx1, norm1), zeroes);

                _mm256_storeu_ps(result_ptr + i, out0);
                _mm256_storeu_ps(result_ptr + i + 8, out1);
            } else {
                for (size_t j = i; j < len; ++j) {
                    float curr = scratch_ptr[j];
                    float prev = scratch_ptr[j-1];
                    float smoothed = 0.5f * (curr + prev);
                    float recovered_gate = curr - feat_a[j];
                    float magnitude = std::abs(smoothed);
                    float denominator = 1.0f + magnitude;
                    float contextual = feat_c[j] + smoothed / denominator;
                    float suppressed = feat_b[j] * (1.0f - recovered_gate) * global_mean;
                    float interacted = smoothed * contextual;
                    float normalized = (interacted + suppressed) / denominator;
                    result_ptr[j] = std::max(contextual - normalized, 0.0f);
                }
                break;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            return false;
        }
    }
    return true;
}
