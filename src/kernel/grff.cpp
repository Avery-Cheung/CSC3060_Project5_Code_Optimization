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
    if (n == 0) return;

    args.f_output.resize(n);
    args.scratch.resize(n * 2);

    const float* __restrict A = args.a_features.data();
    const float* __restrict B = args.b_features.data();
    const float* __restrict C = args.c_features.data();
    float* __restrict F = args.f_output.data();
    float* __restrict B_omg = args.scratch.data();       // B * (1-G)
    float* __restrict A_prime = B_omg + n;                // later overwritten by smooth

    // ── Pass 1: B*(1-G), A_prime, and sum of A_prime ──────────────────────
    // G = 0.5*(prod*inv + 1),  1-G = 0.5*(1 - prod*inv)
    // Let half = 0.5*prod*inv, then  G = half + 0.5,  1-G = 0.5 - half
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    size_t i = 0;
    const size_t limit = n & ~static_cast<size_t>(3);

    for (; i < limit; i += 4) {
<<<<<<< HEAD
        float a0 = A[i], b0 = B[i];
        float prod0 = a0 * b0;
        float inv0 = 1.0f / (1.0f + std::fabs(prod0));
        float half0 = 0.5f * prod0 * inv0;
        B_omg[i] = b0 * (0.5f - half0);
        A_prime[i] = a0 + 0.5f + half0;
        sum0 += A_prime[i];

        float a1 = A[i + 1], b1 = B[i + 1];
        float prod1 = a1 * b1;
        float inv1 = 1.0f / (1.0f + std::fabs(prod1));
        float half1 = 0.5f * prod1 * inv1;
        B_omg[i + 1] = b1 * (0.5f - half1);
        A_prime[i + 1] = a1 + 0.5f + half1;
        sum1 += A_prime[i + 1];

        float a2 = A[i + 2], b2 = B[i + 2];
        float prod2 = a2 * b2;
        float inv2 = 1.0f / (1.0f + std::fabs(prod2));
        float half2 = 0.5f * prod2 * inv2;
        B_omg[i + 2] = b2 * (0.5f - half2);
        A_prime[i + 2] = a2 + 0.5f + half2;
        sum2 += A_prime[i + 2];

        float a3 = A[i + 3], b3 = B[i + 3];
        float prod3 = a3 * b3;
        float inv3 = 1.0f / (1.0f + std::fabs(prod3));
        float half3 = 0.5f * prod3 * inv3;
        B_omg[i + 3] = b3 * (0.5f - half3);
        A_prime[i + 3] = a3 + 0.5f + half3;
        sum3 += A_prime[i + 3];
=======
        {   // elem 0
            float a0 = A[i], b0 = B[i];
            float prod0 = a0 * b0;
            float inv0 = 1.0f / (1.0f + std::fabs(prod0));
            float half0 = 0.5f * prod0 * inv0;
            B_omg[i] = b0 * (0.5f - half0);
            float ap0 = a0 + 0.5f + half0;
            A_prime[i] = ap0;
            sum0 += ap0;
        }
        {   // elem 1
            float a1 = A[i + 1], b1 = B[i + 1];
            float prod1 = a1 * b1;
            float inv1 = 1.0f / (1.0f + std::fabs(prod1));
            float half1 = 0.5f * prod1 * inv1;
            B_omg[i + 1] = b1 * (0.5f - half1);
            float ap1 = a1 + 0.5f + half1;
            A_prime[i + 1] = ap1;
            sum1 += ap1;
        }
        {   // elem 2
            float a2 = A[i + 2], b2 = B[i + 2];
            float prod2 = a2 * b2;
            float inv2 = 1.0f / (1.0f + std::fabs(prod2));
            float half2 = 0.5f * prod2 * inv2;
            B_omg[i + 2] = b2 * (0.5f - half2);
            float ap2 = a2 + 0.5f + half2;
            A_prime[i + 2] = ap2;
            sum2 += ap2;
        }
        {   // elem 3
            float a3 = A[i + 3], b3 = B[i + 3];
            float prod3 = a3 * b3;
            float inv3 = 1.0f / (1.0f + std::fabs(prod3));
            float half3 = 0.5f * prod3 * inv3;
            B_omg[i + 3] = b3 * (0.5f - half3);
            float ap3 = a3 + 0.5f + half3;
            A_prime[i + 3] = ap3;
            sum3 += ap3;
        }
>>>>>>> parent of 9a62dd4 (2 functions bug-fix attempt)
    }
    for (; i < n; ++i) {
        float a = A[i], b = B[i];
        float prod = a * b;
        float inv = 1.0f / (1.0f + std::fabs(prod));
        float half = 0.5f * prod * inv;
        B_omg[i] = b * (0.5f - half);
        A_prime[i] = a + 0.5f + half;
        sum0 += A_prime[i];
    }

    const float avg_a = (sum0 + sum1 + sum2 + sum3) / static_cast<float>(n);

<<<<<<< HEAD
    // ── Pass 2: Fused smooth + output (4x unrolled, no pragmas) ───────────
    // smooth[0] = A_prime[0]
    // smooth[i] = 0.5*(A_prime[i] + A_prime[i-1])  for i > 0
    // F[i] = max((C[i] + sigmoid_s)*(1 - sigmoid_s) - B_omg[i]*avg_a*inv_s, 0)

    // i = 0
    {
        float s = A_prime[0];
        float inv_s = 1.0f / (1.0f + std::fabs(s));
        float sig_s = s * inv_s;
        float result = (C[0] + sig_s) * (1.0f - sig_s) - B_omg[0] * avg_a * inv_s;
        F[0] = (result > 0.0f) ? result : 0.0f;
    }

    // i >= 1
    float prev_ap = A_prime[0];
    i = 1;
    for (; i + 3 < n; i += 4) {
        float ap0 = A_prime[i];
        float ap1 = A_prime[i + 1];
        float ap2 = A_prime[i + 2];
        float ap3 = A_prime[i + 3];

        float s0 = 0.5f * (ap0 + prev_ap);
        float inv0 = 1.0f / (1.0f + std::fabs(s0));
        float sig0 = s0 * inv0;
        float res0 = (C[i] + sig0) * (1.0f - sig0) - B_omg[i] * avg_a * inv0;
        F[i] = (res0 > 0.0f) ? res0 : 0.0f;

        float s1 = 0.5f * (ap1 + ap0);
        float inv1 = 1.0f / (1.0f + std::fabs(s1));
        float sig1 = s1 * inv1;
        float res1 = (C[i + 1] + sig1) * (1.0f - sig1) - B_omg[i + 1] * avg_a * inv1;
        F[i + 1] = (res1 > 0.0f) ? res1 : 0.0f;

        float s2 = 0.5f * (ap2 + ap1);
        float inv2 = 1.0f / (1.0f + std::fabs(s2));
        float sig2 = s2 * inv2;
        float res2 = (C[i + 2] + sig2) * (1.0f - sig2) - B_omg[i + 2] * avg_a * inv2;
        F[i + 2] = (res2 > 0.0f) ? res2 : 0.0f;

        float s3 = 0.5f * (ap3 + ap2);
        float inv3 = 1.0f / (1.0f + std::fabs(s3));
        float sig3 = s3 * inv3;
        float res3 = (C[i + 3] + sig3) * (1.0f - sig3) - B_omg[i + 3] * avg_a * inv3;
        F[i + 3] = (res3 > 0.0f) ? res3 : 0.0f;

        prev_ap = ap3;
    }
    for (; i < n; ++i) {
        float ap = A_prime[i];
        float s = 0.5f * (ap + prev_ap);
=======
    // ── Pass 2a: In-place smooth (reverse traversal, fully vectorizable) ──
    // smooth[0] = A_prime[0]   (unchanged)
    // smooth[i] = 0.5 * (A_prime[i] + A_prime[i-1])  for i > 0
    // Going backwards, smooth[i-1] is still the original A_prime[i-1].
    float* __restrict smooth = A_prime;
    #pragma clang loop vectorize(enable)
    for (i = n - 1; i >= 1; --i) {
        smooth[i] = 0.5f * (smooth[i] + smooth[i - 1]);
    }

    // ── Pass 2b: Fused output (fully vectorizable) ────────────────────────
    // C_prime = C + sigmoid_s        where sigmoid_s = s / (1+|s|)
    // F = max(C_prime - (s*C_prime + B*(1-G)*avg_a)/(1+|s|), 0)
    //   = max((C + sigmoid_s)*(1 - sigmoid_s) - B_omg*avg_a*inv_s, 0)
    #pragma clang loop vectorize(enable)
    for (i = 0; i < n; ++i) {
        float s = smooth[i];
>>>>>>> parent of 9a62dd4 (2 functions bug-fix attempt)
        float inv_s = 1.0f / (1.0f + std::fabs(s));
        float sig_s = s * inv_s;
        float result = (C[i] + sig_s) * (1.0f - sig_s) - B_omg[i] * avg_a * inv_s;
        F[i] = (result > 0.0f) ? result : 0.0f;
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
