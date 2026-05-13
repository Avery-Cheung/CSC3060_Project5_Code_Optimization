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

    float* G = args.scratch.data();
    float* smooth = G + n;

    float sum_a = 0.0f;
    size_t i = 0;

    const size_t unroll_count = n / 4;
    for (size_t block = 0; block < unroll_count; ++block) {
        const size_t base = block * 4;
        for (size_t lane = 0; lane < 4; ++lane) {
            const size_t idx = base + lane;
            const float a = A[idx];
            const float b = B[idx];
            const float prod = a * b;
            const float g = 0.5f * (prod / (1.0f + std::fabs(prod)) + 1.0f);
            G[idx] = g;
            const float a_p = a + g;
            smooth[idx] = a_p;
            sum_a += a_p;
        }
        i += 4;
    }
    for (; i < n; ++i) {
        const float a = A[i];
        const float b = B[i];
        const float prod = a * b;
        const float g = 0.5f * (prod / (1.0f + std::fabs(prod)) + 1.0f);
        G[i] = g;
        const float a_p = a + g;
        smooth[i] = a_p;
        sum_a += a_p;
    }

    const float avg_a = sum_a / static_cast<float>(n);

    for (size_t idx = n; idx-- > 1;) {
        smooth[idx] = 0.5f * (smooth[idx] + smooth[idx - 1]);
    }

    float* const out_ptr = F;

    i = 0;
    for (; i + 3 < n; i += 4) {
        const float smooth0 = smooth[i];
        const float g0 = G[i];
        const float c0 = C[i];
        const float b0 = B[i];
        const float denom0 = 1.0f + std::fabs(smooth0);
        const float cprime0 = c0 + smooth0 / denom0;
        const float bprime0 = b0 * (1.0f - g0) * avg_a;
        const float result0 = cprime0 - ((smooth0 * cprime0 + bprime0) / denom0);
        out_ptr[i] = (result0 > 0.0f) ? result0 : 0.0f;

        const float smooth1 = smooth[i + 1];
        const float g1 = G[i + 1];
        const float c1 = C[i + 1];
        const float b1 = B[i + 1];
        const float denom1 = 1.0f + std::fabs(smooth1);
        const float cprime1 = c1 + smooth1 / denom1;
        const float bprime1 = b1 * (1.0f - g1) * avg_a;
        const float result1 = cprime1 - ((smooth1 * cprime1 + bprime1) / denom1);
        out_ptr[i + 1] = (result1 > 0.0f) ? result1 : 0.0f;

        const float smooth2 = smooth[i + 2];
        const float g2 = G[i + 2];
        const float c2 = C[i + 2];
        const float b2 = B[i + 2];
        const float denom2 = 1.0f + std::fabs(smooth2);
        const float cprime2 = c2 + smooth2 / denom2;
        const float bprime2 = b2 * (1.0f - g2) * avg_a;
        const float result2 = cprime2 - ((smooth2 * cprime2 + bprime2) / denom2);
        out_ptr[i + 2] = (result2 > 0.0f) ? result2 : 0.0f;

        const float smooth3 = smooth[i + 3];
        const float g3 = G[i + 3];
        const float c3 = C[i + 3];
        const float b3 = B[i + 3];
        const float denom3 = 1.0f + std::fabs(smooth3);
        const float cprime3 = c3 + smooth3 / denom3;
        const float bprime3 = b3 * (1.0f - g3) * avg_a;
        const float result3 = cprime3 - ((smooth3 * cprime3 + bprime3) / denom3);
        out_ptr[i + 3] = (result3 > 0.0f) ? result3 : 0.0f;
    }
    for (; i < n; ++i) {
        const float smooth_i = smooth[i];
        const float g_i = G[i];
        const float c_i = C[i];
        const float b_i = B[i];
        const float denom_i = 1.0f + std::fabs(smooth_i);
        const float cprime_i = c_i + smooth_i / denom_i;
        const float bprime_i = b_i * (1.0f - g_i) * avg_a;
        const float result_i = cprime_i - ((smooth_i * cprime_i + bprime_i) / denom_i);
        out_ptr[i] = (result_i > 0.0f) ? result_i : 0.0f;
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
