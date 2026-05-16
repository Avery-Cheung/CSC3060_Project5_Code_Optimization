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

    const size_t len = args.a_features.size();

    if (len == 0) {
        return;
    }

    static thread_local std::vector<float> fused_a;
    fused_a.resize(len);

    const float* __restrict__ feat_a =
        args.a_features.data();

    const float* __restrict__ feat_b =
        args.b_features.data();

    const float* __restrict__ feat_c =
        args.c_features.data();

    float* __restrict__ output =
        args.f_output.data();

    float* __restrict__ cache =
        fused_a.data();

    // =========================================
    // stage 1:
    // build fused feature + accumulate mean
    // =========================================

    float accum = 0.0f;

    for (size_t idx = 0; idx < len; ++idx) {

        const float mul =
            feat_a[idx] * feat_b[idx];

        const float gate =
            0.5f *
            (
                mul / (1.0f + std::abs(mul))
                + 1.0f
            );

        const float merged =
            feat_a[idx] + gate;

        cache[idx] = merged;

        accum += merged;
    }

    const float global_scale =
        accum / static_cast<float>(len);

    // =========================================
    // stage 2:
    // reconstruction pipeline
    // =========================================

    float prev_feature =
        cache[0];

    {
        const float smooth_feature =
            prev_feature;

        const float recovered_gate =
            smooth_feature - feat_a[0];

        const float damp =
            1.0f + std::abs(smooth_feature);

        const float context =
            feat_c[0] +
            smooth_feature / damp;

        const float suppress =
            feat_b[0] *
            (1.0f - recovered_gate) *
            global_scale;

        const float hidden =
            smooth_feature * context;

        const float normalized =
            (hidden + suppress) / damp;

        const float final_value =
            context - normalized;

        output[0] =
            std::max(final_value, 0.0f);
    }

    for (size_t idx = 1; idx < len; ++idx) {

        const float current_feature =
            cache[idx];

        const float blended =
            0.5f *
            (
                current_feature +
                prev_feature
            );

        prev_feature = current_feature;

        const float recovered_gate =
            current_feature - feat_a[idx];

        const float denom =
            1.0f + std::abs(blended);

        const float context =
            feat_c[idx] +
            blended / denom;

        const float suppress =
            feat_b[idx] *
            (1.0f - recovered_gate) *
            global_scale;

        const float interaction =
            blended * context;

        const float normalized =
            (interaction + suppress) / denom;

        const float result =
            context - normalized;

        output[idx] =
            std::max(result, 0.0f);
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
