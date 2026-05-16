#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <immintrin.h>

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

    const float* __restrict A = args.a_features.data();
    const float* __restrict B = args.b_features.data();
    const float* __restrict C = args.c_features.data();

    float* __restrict F = args.f_output.data();
    float* __restrict G = args.scratch.data();

    constexpr size_t STEP = 8;

    const __m256 vone  = _mm256_set1_ps(1.0f);
    const __m256 vhalf = _mm256_set1_ps(0.5f);

    // -------------------------------------------------
    // Pass 1
    // Compute G + sum
    // -------------------------------------------------

    __m256 vsum = _mm256_setzero_ps();

    size_t i = 0;

    for (; i + STEP <= n; i += STEP) {

        __m256 va = _mm256_loadu_ps(A + i);
        __m256 vb = _mm256_loadu_ps(B + i);

        __m256 prod =
            _mm256_mul_ps(va, vb);

        // abs(x)
        __m256 abs_prod =
            _mm256_andnot_ps(
                _mm256_set1_ps(-0.0f),
                prod
            );

        __m256 denom =
            _mm256_add_ps(vone, abs_prod);

        __m256 frac =
            _mm256_div_ps(prod, denom);

        __m256 g =
            _mm256_mul_ps(
                vhalf,
                _mm256_add_ps(frac, vone)
            );

        _mm256_storeu_ps(G + i, g);

        vsum =
            _mm256_add_ps(
                vsum,
                _mm256_add_ps(va, g)
            );
    }

    alignas(32) float buf[8];
    _mm256_store_ps(buf, vsum);

    float sum_a =
        buf[0] + buf[1] + buf[2] + buf[3] +
        buf[4] + buf[5] + buf[6] + buf[7];

    for (; i < n; ++i) {

        float prod = A[i] * B[i];

        float g =
            0.5f *
            (
                prod /
                (1.0f + std::fabs(prod))
                + 1.0f
            );

        G[i] = g;

        sum_a += A[i] + g;
    }

    const float avg_a =
        sum_a / static_cast<float>(n);

    // -------------------------------------------------
    // Pass 2
    // -------------------------------------------------

    float prev_ap = A[0] + G[0];

    {
        float s = prev_ap;

        float inv =
            1.0f /
            (1.0f + std::fabs(s));

        float b_prime =
            B[0] *
            (1.0f - G[0]) *
            avg_a;

        float c_prime =
            C[0] + s * inv;

        float e =
            (s * c_prime + b_prime)
            * inv;

        float r = c_prime - e;

        F[0] =
            (r > 0.0f)
            ? r
            : 0.0f;
    }

    i = 1;

    // -------------------------------------------------
    // heavily unrolled scalar pipeline
    // -------------------------------------------------

    for (; i + 4 <= n; i += 4) {

        #pragma GCC unroll 4

        for (int j = 0; j < 4; ++j) {

            size_t idx = i + j;

            float ap =
                A[idx] + G[idx];

            float s =
                0.5f *
                (ap + prev_ap);

            prev_ap = ap;

            float inv =
                1.0f /
                (1.0f + std::fabs(s));

            float b_prime =
                B[idx] *
                (1.0f - G[idx]) *
                avg_a;

            float c_prime =
                C[idx] +
                s * inv;

            float e =
                (s * c_prime + b_prime)
                * inv;

            float r =
                c_prime - e;

            F[idx] =
                (r > 0.0f)
                ? r
                : 0.0f;
        }
    }

    for (; i < n; ++i) {

        float ap =
            A[i] + G[i];

        float s =
            0.5f *
            (ap + prev_ap);

        prev_ap = ap;

        float inv =
            1.0f /
            (1.0f + std::fabs(s));

        float b_prime =
            B[i] *
            (1.0f - G[i]) *
            avg_a;

        float c_prime =
            C[i] +
            s * inv;

        float e =
            (s * c_prime + b_prime)
            * inv;

        float r =
            c_prime - e;

        F[i] =
            (r > 0.0f)
            ? r
            : 0.0f;
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
