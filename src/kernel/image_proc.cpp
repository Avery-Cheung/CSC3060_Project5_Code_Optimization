#include "image_proc.h"
#include <cmath>
#include <algorithm>
#include <random>

void initialize_image_proc(image_proc_args *args, size_t w, size_t h, uint64_t seed) {
    args->width = w;
    args->height = h;
    size_t n = w * h;
    
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->r_channel.assign(n, 0.0f);
    args->g_channel.assign(n, 0.0f);
    args->b_channel.assign(n, 0.0f);
    args->output.assign(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        args->r_channel[i] = dist(gen);
        args->g_channel[i] = dist(gen);
        args->b_channel[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Functions
// You should not modify these functions
// -------------------------------------------------------------------------

// Image Process A
__attribute__((noinline)) 
float apply_gain(float v) { return v * 1.05f; }

__attribute__((noinline)) 
float apply_shift(float v) { return v + 0.02f; }

__attribute__((noinline)) 
float apply_limit(float v) { return (v > 1.0f) ? 1.0f : v; }

__attribute__((noinline)) 
float color_correct(float v) {
    return apply_limit(apply_shift(apply_gain(v)));
}

// Image Process B
__attribute__((noinline)) 
float compute_gray(float r, float g, float b) {
    return (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
}

__attribute__((noinline)) 
float enhance_contrast(float gray){
    // imadjust logic: mapping 0.05-0.95 to 0.0-1.0
    float adjusted = std::clamp((gray - 0.05f) / 0.90f, 0.0f, 1.0f);
    
    // Sigmoidal S-Curve for "punchy" contrast
    return adjusted * adjusted * (3.0f - 2.0f * adjusted);
}

__attribute__((noinline)) 
float calculate_gain(float intensity) {
    float g1 = intensity * 0.5f;
    float g2 = g1 * g1 + 0.1f;
    float g3 = std::sqrt(g2);
    return (g3 > 1.0f) ? (1.0f / g3) : (g3 * 0.95f);
}

__attribute__((noinline)) 
float hdr_compress(float val) {
    float gain = calculate_gain(val * 1.2f);
    float result = val * gain;
    return result / (1.0f + result);
}

__attribute__((noinline)) 
float complex_mask_logic(float gray, float r, float g, float b, float thresh) {
    const float p0=0.11f, p1=0.22f, p2=0.33f, p3=0.44f, p4=0.55f;
    const float p5=0.66f, p6=0.77f, p7=0.88f, p8=0.99f, p9=1.01f;
    
    float mask = 0.0f;
    if (gray > thresh) {
        mask = (r * p0) + (g * p1) - (b * p2) + p9;
        if (mask > 0.8f) mask *= p3;
        else mask += p4;
    } else {
        mask = (r * p5) - (g * p6) + (b * p7) - p8;
        if (mask < 0.2f) mask += p1;
        else mask *= p2;
    }

    float noise = std::sin(gray * p0) * std::cos(r * p1);
    float final_val = (mask * 0.7f) + (noise * 0.3f);
    
    return std::clamp(final_val, 0.0f, 1.0f);
}

__attribute__((noinline)) 
float importance_weight(float val) {
    static const float lut[] = {0.0f, 0.3f, 1.0f, 0.3f, 0.0f};
    float scaled = val * 4.0f;
    int idx = std::clamp(static_cast<int>(scaled), 0, 4);
    float weight = scaled - static_cast<float>(idx);
    
    if (idx < 4) return lut[idx] * (1.0f - weight) + lut[idx + 1] * weight;
    return lut[4];
}

__attribute__((noinline)) 
float fast_activate(float val) {
    return val / (1.0f + std::abs(val));
}
// -------------------------------------------------------------------------
// End of Functions
// -------------------------------------------------------------------------


// -------------------------------------------------------------------------
// Naive Implementation: Image Processing
// -------------------------------------------------------------------------
void naive_image_proc(image_proc_args& args) {
    const size_t w = args.width;
    const size_t h = args.height;
    float* __restrict__ out = args.output.data();
    const float* __restrict__ r = args.r_channel.data();
    const float* __restrict__ g = args.g_channel.data();
    const float* __restrict__ b = args.b_channel.data();
    const float threshold = args.threshold;

    for (size_t y = 0; y < h; ++y)  {
        for (size_t x = 0; x < w; ++x){
            size_t i = y * w + x;

            // Stage 1: Update RGB Value
            float r_val = color_correct(r[i]);
            float g_val = color_correct(g[i]);
            float b_val = color_correct(b[i]);

            // Stage 2: Luminance Extraction
            float gray = compute_gray(r_val, g_val, b_val);

            // Stage 3: Contrast Enhancement
            float grayEnhance = enhance_contrast(gray);

            // Stage 4: HDR Compression
            float compress_val = hdr_compress(grayEnhance);

            // Stage 5: Masking
            float mask = complex_mask_logic(compress_val, r_val, g_val, b_val, threshold);

            // Stage 6: Importance Weighting
            float weight = importance_weight(mask);

            // Output
            out[i] = std::clamp(compress_val * weight, 0.0f, 1.0f);

        }
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
static inline float clamp01f(float v) {
    return (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
}

static inline float importance_weight_f(float val) {
    static const float lut[5] = {0.0f, 0.3f, 1.0f, 0.3f, 0.0f};
    float scaled = val * 4.0f;
    int idx = static_cast<int>(scaled);
    if (idx < 0) {
        idx = 0;
    } else if (idx > 4) {
        idx = 4;
    }
    if (idx < 4) {
        const float frac = scaled - static_cast<float>(idx);
        return lut[idx] * (1.0f - frac) + lut[idx + 1] * frac;
    }
    return lut[4];
}

static inline float mask_logic_f(float gray, float r_val, float g_val,
                                 float b_val, float threshold) {
    const float p0 = 0.11f;
    const float p1 = 0.22f;
    const float p2 = 0.33f;
    const float p3 = 0.44f;
    const float p4 = 0.55f;
    const float p5 = 0.66f;
    const float p6 = 0.77f;
    const float p7 = 0.88f;
    const float p8 = 0.99f;
    const float p9 = 1.01f;

    const float mask_true = (r_val * p0) + (g_val * p1) - (b_val * p2) + p9;
    const float mask_true_adj = (mask_true > 0.8f) ? (mask_true * p3)
                                                : (mask_true + p4);

    const float mask_false = (r_val * p5) - (g_val * p6) + (b_val * p7) - p8;
    const float mask_false_adj = (mask_false < 0.2f) ? (mask_false + p1)
                                                   : (mask_false * p2);

    return (gray > threshold) ? mask_true_adj : mask_false_adj;
}

static inline float color_correct_f(float v) {
    v = v * 1.05f + 0.02f;
    return clamp01f(v);
}

static inline float encode_color(float v) {
    // Contrast + HDR compression pipeline
    const float adjusted = clamp01f((v - 0.05f) * 1.1111111f);
    const float gray_enhanced = adjusted * adjusted * (3.0f - 2.0f * adjusted);
    const float intensity = gray_enhanced * 1.2f;
    const float g3 = std::sqrt(intensity * intensity * 0.25f + 0.1f);
    const float gain = (g3 > 1.0f) ? (1.0f / g3) : (g3 * 0.95f);
    const float compressed = gray_enhanced * gain;
    return compressed / (1.0f + compressed);
}

void stu_image_proc(image_proc_args& args) {
    const size_t n = args.width * args.height;
    if (args.output.size() != n) {
        args.output.resize(n);
    }

    const float* __restrict__ rptr = args.r_channel.data();
    const float* __restrict__ gptr = args.g_channel.data();
    const float* __restrict__ bptr = args.b_channel.data();
    float* __restrict__ outptr = args.output.data();
    const float threshold = args.threshold;

    const float gray_r = 0.299f;
    const float gray_g = 0.587f;
    const float gray_b = 0.114f;
    const float inv_scale = 1.1111111f;
    const float p_sin = 0.11f;
    const float p_cos = 0.22f;
    const size_t limit = n & ~static_cast<size_t>(3);

    size_t i = 0;
    for (; i < limit; i += 4) {
        float r0 = color_correct_f(rptr[i]);
        float g0 = color_correct_f(gptr[i]);
        float b0 = color_correct_f(bptr[i]);
        float gray0 = (r0 * gray_r) + (g0 * gray_g) + (b0 * gray_b);
        float compressed0 = encode_color(gray0);
        float mask0 = mask_logic_f(gray0, r0, g0, b0, threshold);
        float final0 = clamp01f((mask0 * 0.7f) + (std::sinf(gray0 * p_sin) * std::cosf(r0 * p_cos) * 0.3f));
        outptr[i] = clamp01f(compressed0 * importance_weight_f(final0));

        float r1 = color_correct_f(rptr[i + 1]);
        float g1 = color_correct_f(gptr[i + 1]);
        float b1 = color_correct_f(bptr[i + 1]);
        float gray1 = (r1 * gray_r) + (g1 * gray_g) + (b1 * gray_b);
        float compressed1 = encode_color(gray1);
        float mask1 = mask_logic_f(gray1, r1, g1, b1, threshold);
        float final1 = clamp01f((mask1 * 0.7f) + (std::sinf(gray1 * p_sin) * std::cosf(r1 * p_cos) * 0.3f));
        outptr[i + 1] = clamp01f(compressed1 * importance_weight_f(final1));

        float r2 = color_correct_f(rptr[i + 2]);
        float g2 = color_correct_f(gptr[i + 2]);
        float b2 = color_correct_f(bptr[i + 2]);
        float gray2 = (r2 * gray_r) + (g2 * gray_g) + (b2 * gray_b);
        float compressed2 = encode_color(gray2);
        float mask2 = mask_logic_f(gray2, r2, g2, b2, threshold);
        float final2 = clamp01f((mask2 * 0.7f) + (std::sinf(gray2 * p_sin) * std::cosf(r2 * p_cos) * 0.3f));
        outptr[i + 2] = clamp01f(compressed2 * importance_weight_f(final2));

        float r3 = color_correct_f(rptr[i + 3]);
        float g3 = color_correct_f(gptr[i + 3]);
        float b3 = color_correct_f(bptr[i + 3]);
        float gray3 = (r3 * gray_r) + (g3 * gray_g) + (b3 * gray_b);
        float compressed3 = encode_color(gray3);
        float mask3 = mask_logic_f(gray3, r3, g3, b3, threshold);
        float final3 = clamp01f((mask3 * 0.7f) + (std::sinf(gray3 * p_sin) * std::cosf(r3 * p_cos) * 0.3f));
        outptr[i + 3] = clamp01f(compressed3 * importance_weight_f(final3));
    }

    for (; i < n; ++i) {
        float r0 = color_correct_f(rptr[i]);
        float g0 = color_correct_f(gptr[i]);
        float b0 = color_correct_f(bptr[i]);
        float gray0 = (r0 * gray_r) + (g0 * gray_g) + (b0 * gray_b);
        float compressed0 = encode_color(gray0);
        float mask0 = mask_logic_f(gray0, r0, g0, b0, threshold);
        float final0 = clamp01f((mask0 * 0.7f) + (std::sinf(gray0 * p_sin) * std::cosf(r0 * p_cos) * 0.3f));
        outptr[i] = clamp01f(compressed0 * importance_weight_f(final0));
    }
}


// -------------------------------------------------------------------------
// Wrappers and Utilities
// -------------------------------------------------------------------------
void naive_image_proc_wrapper(void *ctx) {
    auto &args = *static_cast<image_proc_args *>(ctx);
    naive_image_proc(args);
}

void stu_image_proc_wrapper(void *ctx) {
    auto &args = *static_cast<image_proc_args *>(ctx);
    stu_image_proc(args);
}

bool image_proc_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu = *static_cast<image_proc_args *>(stu_ctx);
    auto &ref = *static_cast<image_proc_args *>(ref_ctx);

    if (stu.output.size() != ref.output.size()) {
        debug_log("DEBUG: image_proc size mismatch: stu={} ref={}\n",
                  stu.output.size(),
                  ref.output.size());
        return false;
    }

    for (size_t i = 0; i < ref.output.size(); ++i) {
        const double err =
            std::abs(static_cast<double>(stu.output[i]) -
                     static_cast<double>(ref.output[i]));
        if (err > 1e-4) {
            debug_log("DEBUG: image_proc fail at {}: ref={} stu={} err={}\n",
                      i,
                      ref.output[i],
                      stu.output[i],
                      err);
            return false;
        }
    }
    debug_log("DEBUG: image_proc_check passed. size={}\n", ref.output.size());
    return true;
}

