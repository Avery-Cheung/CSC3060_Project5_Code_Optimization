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

    const auto& A = args.a_features;
    const auto& B = args.b_features;
    const auto& C = args.c_features;
    auto& F = args.f_output;

    // Ensure output size is correct
    if (F.size() != n) {
        F.resize(n);
    }

    // ---------------------------------------------------------------------
    // Temporary buffers
    //
    // We only keep the intermediates that are actually needed later:
    //   G        : gate values
    //   A_prime  : updated A
    //   Smooth_A : smoothed A
    //   B_prime  : updated B
    //   C_prime  : updated C
    //
    // H and E are not stored because they are used only once.
    // ---------------------------------------------------------------------
    std::vector<float> G(n);
    std::vector<float> A_prime(n);
    std::vector<float> Smooth_A(n);
    std::vector<float> B_prime(n);
    std::vector<float> C_prime(n);

    // ---------------------------------------------------------------------
    // Pass 1:
    // Compute:
    //   G[i]
    //   A_prime[i]
    // and simultaneously accumulate sum(A_prime)
    // ---------------------------------------------------------------------
    float sum_a = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        const float a = A[i];
        const float b = B[i];

        const float prod = a * b;
        const float g =
            0.5f * (prod / (1.0f + std::fabs(prod)) + 1.0f);

        G[i] = g;

        const float a_p = a + g;
        A_prime[i] = a_p;

        sum_a += a_p;
    }

    const float avg_a = sum_a / static_cast<float>(n);

    // ---------------------------------------------------------------------
    // Pass 2:
    // Compute Smooth_A
    // ---------------------------------------------------------------------
    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) {
        Smooth_A[i] = 0.5f * (A_prime[i] + A_prime[i - 1]);
    }

    // ---------------------------------------------------------------------
    // Pass 3:
    // Compute:
    //   B_prime[i]
    //   C_prime[i]
    // ---------------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
        const float g = G[i];
        const float smooth = Smooth_A[i];

        B_prime[i] = B[i] * (1.0f - g) * avg_a;

        C_prime[i] =
            C[i] + smooth / (1.0f + std::fabs(smooth));
    }

    // ---------------------------------------------------------------------
    // Pass 4:
    // Final output
    //
    // result = C_prime - E
    // E = (H + B_prime)/(1 + |Smooth_A|)
    // H = Smooth_A * C_prime
    //
    // We compute everything directly without storing H or E.
    // ---------------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
        const float smooth = Smooth_A[i];
        const float c_p = C_prime[i];
        const float b_p = B_prime[i];

        const float denom = 1.0f + std::fabs(smooth);
        const float h = smooth * c_p;
        const float e = (h + b_p) / denom;

        const float result = c_p - e;

        // ReLU
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
