#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);
    args->scratch.resize(size * 2);

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
    const size_t n = args.a_features.size();
    if (n == 0) {
        return;
    }

    args.f_output.resize(n);
    args.scratch.resize(n * 2);

    const float* A = args.a_features.data();
    const float* B = args.b_features.data();
    const float* C = args.c_features.data();
    float* F = args.f_output.data();

    float* one_minus_G = args.scratch.data();
    float* A_prime = one_minus_G + n;

    float sum_a0 = 0.0f;
    float sum_a1 = 0.0f;
    float sum_a2 = 0.0f;
    float sum_a3 = 0.0f;

    size_t i = 0;
    const size_t limit = n & ~static_cast<size_t>(3);
    for (; i < limit; i += 4) {
        const float a0 = A[i];
        const float b0 = B[i];
        const float prod0 = a0 * b0;
        const float abs0 = (prod0 < 0.0f) ? -prod0 : prod0;
        const float inv0 = 1.0f / (1.0f + abs0);
        const float g0 = 0.5f * (prod0 * inv0 + 1.0f);
        one_minus_G[i] = 1.0f - g0;
        const float a_p0 = a0 + g0;
        A_prime[i] = a_p0;
        sum_a0 += a_p0;

        const float a1 = A[i + 1];
        const float b1 = B[i + 1];
        const float prod1 = a1 * b1;
        const float abs1 = (prod1 < 0.0f) ? -prod1 : prod1;
        const float inv1 = 1.0f / (1.0f + abs1);
        const float g1 = 0.5f * (prod1 * inv1 + 1.0f);
        one_minus_G[i + 1] = 1.0f - g1;
        const float a_p1 = a1 + g1;
        A_prime[i + 1] = a_p1;
        sum_a1 += a_p1;

        const float a2 = A[i + 2];
        const float b2 = B[i + 2];
        const float prod2 = a2 * b2;
        const float abs2 = (prod2 < 0.0f) ? -prod2 : prod2;
        const float inv2 = 1.0f / (1.0f + abs2);
        const float g2 = 0.5f * (prod2 * inv2 + 1.0f);
        one_minus_G[i + 2] = 1.0f - g2;
        const float a_p2 = a2 + g2;
        A_prime[i + 2] = a_p2;
        sum_a2 += a_p2;

        const float a3 = A[i + 3];
        const float b3 = B[i + 3];
        const float prod3 = a3 * b3;
        const float abs3 = (prod3 < 0.0f) ? -prod3 : prod3;
        const float inv3 = 1.0f / (1.0f + abs3);
        const float g3 = 0.5f * (prod3 * inv3 + 1.0f);
        one_minus_G[i + 3] = 1.0f - g3;
        const float a_p3 = a3 + g3;
        A_prime[i + 3] = a_p3;
        sum_a3 += a_p3;
    }
    for (; i < n; ++i) {
        const float a = A[i];
        const float b = B[i];
        const float prod = a * b;
        const float abs_prod = (prod < 0.0f) ? -prod : prod;
        const float inv = 1.0f / (1.0f + abs_prod);
        const float g = 0.5f * (prod * inv + 1.0f);
        one_minus_G[i] = 1.0f - g;
        const float a_p = a + g;
        A_prime[i] = a_p;
        sum_a0 += a_p;
    }

    const float avg_a = (sum_a0 + sum_a1 + sum_a2 + sum_a3) / static_cast<float>(n);

    float prev_a = A_prime[0];
    float smooth = prev_a;
    float inv_d0 = 1.0f / (1.0f + ((smooth < 0.0f) ? -smooth : smooth));
    float cprime = C[0] + smooth * inv_d0;
    float result = cprime - ((smooth * cprime + B[0] * one_minus_G[0] * avg_a) * inv_d0);
    F[0] = (result > 0.0f) ? result : 0.0f;

    i = 1;
    for (; i + 3 < n; i += 4) {
        const float ap0 = A_prime[i];
        const float smooth0 = 0.5f * (ap0 + prev_a);
        const float inv_d1 = 1.0f / (1.0f + ((smooth0 < 0.0f) ? -smooth0 : smooth0));
        const float cprime0 = C[i] + smooth0 * inv_d1;
        const float result0 = cprime0 - ((smooth0 * cprime0 + B[i] * one_minus_G[i] * avg_a) * inv_d1);
        F[i] = (result0 > 0.0f) ? result0 : 0.0f;
        prev_a = ap0;

        const float ap1 = A_prime[i + 1];
        const float smooth1 = 0.5f * (ap1 + prev_a);
        const float inv_d2 = 1.0f / (1.0f + ((smooth1 < 0.0f) ? -smooth1 : smooth1));
        const float cprime1 = C[i + 1] + smooth1 * inv_d2;
        const float result1 = cprime1 - ((smooth1 * cprime1 + B[i + 1] * one_minus_G[i + 1] * avg_a) * inv_d2);
        F[i + 1] = (result1 > 0.0f) ? result1 : 0.0f;
        prev_a = ap1;

        const float ap2 = A_prime[i + 2];
        const float smooth2 = 0.5f * (ap2 + prev_a);
        const float inv_d3 = 1.0f / (1.0f + ((smooth2 < 0.0f) ? -smooth2 : smooth2));
        const float cprime2 = C[i + 2] + smooth2 * inv_d3;
        const float result2 = cprime2 - ((smooth2 * cprime2 + B[i + 2] * one_minus_G[i + 2] * avg_a) * inv_d3);
        F[i + 2] = (result2 > 0.0f) ? result2 : 0.0f;
        prev_a = ap2;

        const float ap3 = A_prime[i + 3];
        const float smooth3 = 0.5f * (ap3 + prev_a);
        const float inv_d4 = 1.0f / (1.0f + ((smooth3 < 0.0f) ? -smooth3 : smooth3));
        const float cprime3 = C[i + 3] + smooth3 * inv_d4;
        const float result3 = cprime3 - ((smooth3 * cprime3 + B[i + 3] * one_minus_G[i + 3] * avg_a) * inv_d4);
        F[i + 3] = (result3 > 0.0f) ? result3 : 0.0f;
        prev_a = ap3;
    }
    for (; i < n; ++i) {
        const float ap = A_prime[i];
        const float smooth_i = 0.5f * (ap + prev_a);
        const float inv_d = 1.0f / (1.0f + ((smooth_i < 0.0f) ? -smooth_i : smooth_i));
        const float cprime_i = C[i] + smooth_i * inv_d;
        const float result_i = cprime_i - ((smooth_i * cprime_i + B[i] * one_minus_G[i] * avg_a) * inv_d);
        F[i] = (result_i > 0.0f) ? result_i : 0.0f;
        prev_a = ap;
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
