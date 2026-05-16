#include "grff.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
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
    args.scratch.resize(n);

<<<<<<< Updated upstream
    const float* __restrict A =
        args.a_features.data();

    const float* __restrict B =
        args.b_features.data();

    const float* __restrict C =
        args.c_features.data();

    float* __restrict F =
        args.f_output.data();

    float* __restrict G =
        args.scratch.data();

    // =================================================
    // Pass 1:
    // compute G + accumulate sum
    // =================================================

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    size_t i = 0;

    #pragma GCC ivdep
    #pragma GCC unroll 8
    for (; i + 4 <= n; i += 4) {

        float p0 = A[i]     * B[i];
        float p1 = A[i + 1] * B[i + 1];
        float p2 = A[i + 2] * B[i + 2];
        float p3 = A[i + 3] * B[i + 3];

        float g0 =
            0.5f *
            (
                p0 / (1.0f + std::fabs(p0))
                + 1.0f
            );

        float g1 =
            0.5f *
            (
                p1 / (1.0f + std::fabs(p1))
                + 1.0f
            );

        float g2 =
            0.5f *
            (
                p2 / (1.0f + std::fabs(p2))
                + 1.0f
            );

        float g3 =
            0.5f *
            (
                p3 / (1.0f + std::fabs(p3))
                + 1.0f
            );

        G[i]     = g0;
        G[i + 1] = g1;
        G[i + 2] = g2;
        G[i + 3] = g3;

        sum0 += A[i]     + g0;
        sum1 += A[i + 1] + g1;
        sum2 += A[i + 2] + g2;
        sum3 += A[i + 3] + g3;
    }

    float sum_a =
        sum0 + sum1 + sum2 + sum3;

    for (; i < n; ++i) {

        float p = A[i] * B[i];

        float g =
            0.5f *
            (
                p / (1.0f + std::fabs(p))
                + 1.0f
            );

        G[i] = g;

        sum_a += A[i] + g;
    }

    const float avg_a =
        sum_a / static_cast<float>(n);

    // =================================================
    // Pass 2:
    // fused pipeline
    // =================================================

    float prev_ap =
        A[0] + G[0];
=======
    const float* __restrict A = args.a_features.data();
    const float* __restrict B = args.b_features.data();
    const float* __restrict C = args.c_features.data();
    float* __restrict F = args.f_output.data();
    float* __restrict G = args.scratch.data();

    // ── Pass 1: G[i] = 0.5f * (prod/(1+|prod|) + 1) via AVX2 ──────────
    const __m256 vone       = _mm256_set1_ps(1.0f);
    const __m256 vhalf      = _mm256_set1_ps(0.5f);
    const __m256 vsign_mask = _mm256_set1_ps(-0.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va       = _mm256_loadu_ps(A + i);
        __m256 vb       = _mm256_loadu_ps(B + i);
        __m256 prod     = _mm256_mul_ps(va, vb);
        __m256 abs_prod = _mm256_andnot_ps(vsign_mask, prod);
        __m256 denom    = _mm256_add_ps(vone, abs_prod);
        __m256 frac     = _mm256_div_ps(prod, denom);
        __m256 g        = _mm256_mul_ps(vhalf, _mm256_add_ps(frac, vone));
        _mm256_storeu_ps(G + i, g);
    }
    for (; i < n; ++i) {
        float prod = A[i] * B[i];
        G[i] = 0.5f * (prod / (1.0f + std::fabs(prod)) + 1.0f);
    }

    // ── Sum: scalar serial (matching naive Stage 3 exactly) ────────────
    float sum_a = 0.0f;
    for (i = 0; i < n; ++i) {
        sum_a += A[i] + G[i];
    }
    const float avg_a = sum_a / static_cast<float>(n);
>>>>>>> Stashed changes

    // ── Pass 2: Fused smooth + Stages 5-9, 4x unrolled ─────────────────
    // i = 0
    {
<<<<<<< Updated upstream
        float s = prev_ap;

        float inv =
            1.0f /
            (1.0f + std::fabs(s));

        float cp =
            C[0] + s * inv;

        float bp =
            B[0] *
            (1.0f - G[0]) *
            avg_a;

        float r =
            cp -
            ((s * cp + bp) * inv);

        F[0] =
            (r > 0.0f)
            ? r
            : 0.0f;
=======
        float s = A[0] + G[0];
        float inv = 1.0f / (1.0f + std::fabs(s));
        float bp = B[0] * (1.0f - G[0]) * avg_a;
        float cp = C[0] + s * inv;
        float e = (s * cp + bp) * inv;
        float r = cp - e;
        F[0] = (r > 0.0f) ? r : 0.0f;
>>>>>>> Stashed changes
    }

    // i >= 1, 4x unrolled
    float prev_ap = A[0] + G[0];
    i = 1;
    for (; i + 3 < n; i += 4) {
        float ap0 = A[i] + G[i];
        float s0 = 0.5f * (ap0 + prev_ap);
        float inv0 = 1.0f / (1.0f + std::fabs(s0));
        float bp0 = B[i] * (1.0f - G[i]) * avg_a;
        float cp0 = C[i] + s0 * inv0;
        float e0 = (s0 * cp0 + bp0) * inv0;
        float r0 = cp0 - e0;
        F[i] = (r0 > 0.0f) ? r0 : 0.0f;

<<<<<<< Updated upstream
    // =================================================
    // software-pipelined main loop
    // =================================================

    #pragma GCC ivdep
    #pragma GCC unroll 8
    for (; i + 4 <= n; i += 4) {

        // -----------------------------
        // lane 0
        // -----------------------------

        float ap0 =
            A[i] + G[i];

        float s0 =
            0.5f *
            (ap0 + prev_ap);

        prev_ap = ap0;

        // -----------------------------
        // lane 1
        // -----------------------------

        float ap1 =
            A[i + 1] + G[i + 1];

        float s1 =
            0.5f *
            (ap1 + prev_ap);

        prev_ap = ap1;

        // -----------------------------
        // lane 2
        // -----------------------------

        float ap2 =
            A[i + 2] + G[i + 2];

        float s2 =
            0.5f *
            (ap2 + prev_ap);

        prev_ap = ap2;

        // -----------------------------
        // lane 3
        // -----------------------------

        float ap3 =
            A[i + 3] + G[i + 3];

        float s3 =
            0.5f *
            (ap3 + prev_ap);

        prev_ap = ap3;

        // -----------------------------
        // reciprocal terms
        // -----------------------------

        float inv0 =
            1.0f /
            (1.0f + std::fabs(s0));

        float inv1 =
            1.0f /
            (1.0f + std::fabs(s1));

        float inv2 =
            1.0f /
            (1.0f + std::fabs(s2));

        float inv3 =
            1.0f /
            (1.0f + std::fabs(s3));

        // -----------------------------
        // c_prime
        // -----------------------------

        float cp0 =
            C[i] + s0 * inv0;

        float cp1 =
            C[i + 1] + s1 * inv1;

        float cp2 =
            C[i + 2] + s2 * inv2;

        float cp3 =
            C[i + 3] + s3 * inv3;

        // -----------------------------
        // b_prime
        // -----------------------------

        float bp0 =
            B[i] *
            (1.0f - G[i]) *
            avg_a;

        float bp1 =
            B[i + 1] *
            (1.0f - G[i + 1]) *
            avg_a;

        float bp2 =
            B[i + 2] *
            (1.0f - G[i + 2]) *
            avg_a;

        float bp3 =
            B[i + 3] *
            (1.0f - G[i + 3]) *
            avg_a;

        // -----------------------------
        // final
        // -----------------------------

        float r0 =
            cp0 -
            ((s0 * cp0 + bp0) * inv0);

        float r1 =
            cp1 -
            ((s1 * cp1 + bp1) * inv1);

        float r2 =
            cp2 -
            ((s2 * cp2 + bp2) * inv2);

        float r3 =
            cp3 -
            ((s3 * cp3 + bp3) * inv3);

        F[i] =
            (r0 > 0.0f)
            ? r0
            : 0.0f;

        F[i + 1] =
            (r1 > 0.0f)
            ? r1
            : 0.0f;

        F[i + 2] =
            (r2 > 0.0f)
            ? r2
            : 0.0f;

        F[i + 3] =
            (r3 > 0.0f)
            ? r3
            : 0.0f;
=======
        float ap1 = A[i+1] + G[i+1];
        float s1 = 0.5f * (ap1 + ap0);
        float inv1 = 1.0f / (1.0f + std::fabs(s1));
        float bp1 = B[i+1] * (1.0f - G[i+1]) * avg_a;
        float cp1 = C[i+1] + s1 * inv1;
        float e1 = (s1 * cp1 + bp1) * inv1;
        float r1 = cp1 - e1;
        F[i+1] = (r1 > 0.0f) ? r1 : 0.0f;

        float ap2 = A[i+2] + G[i+2];
        float s2 = 0.5f * (ap2 + ap1);
        float inv2 = 1.0f / (1.0f + std::fabs(s2));
        float bp2 = B[i+2] * (1.0f - G[i+2]) * avg_a;
        float cp2 = C[i+2] + s2 * inv2;
        float e2 = (s2 * cp2 + bp2) * inv2;
        float r2 = cp2 - e2;
        F[i+2] = (r2 > 0.0f) ? r2 : 0.0f;

        float ap3 = A[i+3] + G[i+3];
        float s3 = 0.5f * (ap3 + ap2);
        float inv3 = 1.0f / (1.0f + std::fabs(s3));
        float bp3 = B[i+3] * (1.0f - G[i+3]) * avg_a;
        float cp3 = C[i+3] + s3 * inv3;
        float e3 = (s3 * cp3 + bp3) * inv3;
        float r3 = cp3 - e3;
        F[i+3] = (r3 > 0.0f) ? r3 : 0.0f;

        prev_ap = ap3;
>>>>>>> Stashed changes
    }
    for (; i < n; ++i) {
        float ap = A[i] + G[i];
        float s = 0.5f * (ap + prev_ap);
        float inv = 1.0f / (1.0f + std::fabs(s));
        float bp = B[i] * (1.0f - G[i]) * avg_a;
        float cp = C[i] + s * inv;
        float e = (s * cp + bp) * inv;
        float r = cp - e;
        F[i] = (r > 0.0f) ? r : 0.0f;
        prev_ap = ap;
<<<<<<< Updated upstream

        float inv =
            1.0f /
            (1.0f + std::fabs(s));

        float cp =
            C[i] + s * inv;

        float bp =
            B[i] *
            (1.0f - G[i]) *
            avg_a;

        float r =
            cp -
            ((s * cp + bp) * inv);

        F[i] =
            (r > 0.0f)
            ? r
            : 0.0f;
=======
>>>>>>> Stashed changes
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
