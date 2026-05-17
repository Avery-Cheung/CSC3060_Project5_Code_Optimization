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
    return std::fmin(std::fmax(v, 0.0f), 1.0f);
}

static inline float importance_weight_f(float val) {
    static const float lut[6] = {0.0f, 0.3f, 1.0f, 0.3f, 0.0f, 0.0f};
    float scaled = val * 4.0f;
    int idx = static_cast<int>(scaled);
    float frac = scaled - static_cast<float>(idx);
    return lut[idx] * (1.0f - frac) + lut[idx + 1] * frac;
}

void stu_image_proc(image_proc_args& args) {
    const size_t n = args.width * args.height;
    if (args.output.size() != n) args.output.resize(n);

    const float* __restrict rptr = args.r_channel.data();
    const float* __restrict gptr = args.g_channel.data();
    const float* __restrict bptr = args.b_channel.data();
    float* __restrict outptr = args.output.data();
    const float threshold = args.threshold;

    const float gray_r = 0.299f, gray_g = 0.587f, gray_b = 0.114f;
    const float inv_contrast = 1.1111111f;          // 1 / 0.9
    const float p_sin = 0.11f, p_cos = 0.22f;
    const float sin_c3 = 0.16666667f;               // 1/6 for Taylor sin
    const size_t limit = n & ~static_cast<size_t>(3);

    size_t i = 0;
    for (; i < limit; i += 4) {
        // ── Pixel 0 ──
        float r0 = clamp01f(rptr[i] * 1.05f + 0.02f);
        float g0 = clamp01f(gptr[i] * 1.05f + 0.02f);
        float b0 = clamp01f(bptr[i] * 1.05f + 0.02f);
        float gray0 = r0 * gray_r + g0 * gray_g + b0 * gray_b;
        float ge0 = clamp01f((gray0 - 0.05f) * inv_contrast);
        ge0 = ge0 * ge0 * (3.0f - 2.0f * ge0);
        float gain0 = 0.95f * std::sqrt(0.36f * ge0 * ge0 + 0.1f);
        float comp0 = ge0 * gain0;
        float hdr0 = comp0 / (1.0f + comp0);
        // Mask: naive passes compress_val (hdr) as "gray" to complex_mask_logic
        float mask0;
        if (hdr0 > threshold) {
            mask0 = r0 * 0.11f + g0 * 0.22f - b0 * 0.33f + 1.01f;
            mask0 = (mask0 > 0.8f) ? (mask0 * 0.44f) : (mask0 + 0.55f);
        } else {
            mask0 = r0 * 0.66f - g0 * 0.77f + b0 * 0.88f - 0.99f;
            mask0 = (mask0 < 0.2f) ? (mask0 + 0.22f) : (mask0 * 0.33f);
        }
        // Noise: sin(hdr*0.11)*cos(r*0.22) — naive uses compress_val for sin arg
        float sx0 = hdr0 * p_sin, cx0 = r0 * p_cos;
        float sin0 = sx0 * (1.0f - sx0 * sx0 * sin_c3);
        float cos0 = 1.0f - cx0 * cx0 * 0.5f;
        float final0 = clamp01f(mask0 * 0.7f + sin0 * cos0 * 0.3f);
        outptr[i] = clamp01f(hdr0 * importance_weight_f(final0));

        // ── Pixel 1 ──
        float r1 = clamp01f(rptr[i + 1] * 1.05f + 0.02f);
        float g1 = clamp01f(gptr[i + 1] * 1.05f + 0.02f);
        float b1 = clamp01f(bptr[i + 1] * 1.05f + 0.02f);
        float gray1 = r1 * gray_r + g1 * gray_g + b1 * gray_b;
        float ge1 = clamp01f((gray1 - 0.05f) * inv_contrast);
        ge1 = ge1 * ge1 * (3.0f - 2.0f * ge1);
        float gain1 = 0.95f * std::sqrt(0.36f * ge1 * ge1 + 0.1f);
        float comp1 = ge1 * gain1;
        float hdr1 = comp1 / (1.0f + comp1);
        float mask1;
        if (hdr1 > threshold) {
            mask1 = r1 * 0.11f + g1 * 0.22f - b1 * 0.33f + 1.01f;
            mask1 = (mask1 > 0.8f) ? (mask1 * 0.44f) : (mask1 + 0.55f);
        } else {
            mask1 = r1 * 0.66f - g1 * 0.77f + b1 * 0.88f - 0.99f;
            mask1 = (mask1 < 0.2f) ? (mask1 + 0.22f) : (mask1 * 0.33f);
        }
        float sx1 = hdr1 * p_sin, cx1 = r1 * p_cos;
        float sin1 = sx1 * (1.0f - sx1 * sx1 * sin_c3);
        float cos1 = 1.0f - cx1 * cx1 * 0.5f;
        float final1 = clamp01f(mask1 * 0.7f + sin1 * cos1 * 0.3f);
        outptr[i + 1] = clamp01f(hdr1 * importance_weight_f(final1));

        // ── Pixel 2 ──
        float r2 = clamp01f(rptr[i + 2] * 1.05f + 0.02f);
        float g2 = clamp01f(gptr[i + 2] * 1.05f + 0.02f);
        float b2 = clamp01f(bptr[i + 2] * 1.05f + 0.02f);
        float gray2 = r2 * gray_r + g2 * gray_g + b2 * gray_b;
        float ge2 = clamp01f((gray2 - 0.05f) * inv_contrast);
        ge2 = ge2 * ge2 * (3.0f - 2.0f * ge2);
        float gain2 = 0.95f * std::sqrt(0.36f * ge2 * ge2 + 0.1f);
        float comp2 = ge2 * gain2;
        float hdr2 = comp2 / (1.0f + comp2);
        float mask2;
        if (hdr2 > threshold) {
            mask2 = r2 * 0.11f + g2 * 0.22f - b2 * 0.33f + 1.01f;
            mask2 = (mask2 > 0.8f) ? (mask2 * 0.44f) : (mask2 + 0.55f);
        } else {
            mask2 = r2 * 0.66f - g2 * 0.77f + b2 * 0.88f - 0.99f;
            mask2 = (mask2 < 0.2f) ? (mask2 + 0.22f) : (mask2 * 0.33f);
        }
        float sx2 = hdr2 * p_sin, cx2 = r2 * p_cos;
        float sin2 = sx2 * (1.0f - sx2 * sx2 * sin_c3);
        float cos2 = 1.0f - cx2 * cx2 * 0.5f;
        float final2 = clamp01f(mask2 * 0.7f + sin2 * cos2 * 0.3f);
        outptr[i + 2] = clamp01f(hdr2 * importance_weight_f(final2));

        // ── Pixel 3 ──
        float r3 = clamp01f(rptr[i + 3] * 1.05f + 0.02f);
        float g3 = clamp01f(gptr[i + 3] * 1.05f + 0.02f);
        float b3 = clamp01f(bptr[i + 3] * 1.05f + 0.02f);
        float gray3 = r3 * gray_r + g3 * gray_g + b3 * gray_b;
        float ge3 = clamp01f((gray3 - 0.05f) * inv_contrast);
        ge3 = ge3 * ge3 * (3.0f - 2.0f * ge3);
        float gain3 = 0.95f * std::sqrt(0.36f * ge3 * ge3 + 0.1f);
        float comp3 = ge3 * gain3;
        float hdr3 = comp3 / (1.0f + comp3);
        float mask3;
        if (hdr3 > threshold) {
            mask3 = r3 * 0.11f + g3 * 0.22f - b3 * 0.33f + 1.01f;
            mask3 = (mask3 > 0.8f) ? (mask3 * 0.44f) : (mask3 + 0.55f);
        } else {
            mask3 = r3 * 0.66f - g3 * 0.77f + b3 * 0.88f - 0.99f;
            mask3 = (mask3 < 0.2f) ? (mask3 + 0.22f) : (mask3 * 0.33f);
        }
        float sx3 = hdr3 * p_sin, cx3 = r3 * p_cos;
        float sin3 = sx3 * (1.0f - sx3 * sx3 * sin_c3);
        float cos3 = 1.0f - cx3 * cx3 * 0.5f;
        float final3 = clamp01f(mask3 * 0.7f + sin3 * cos3 * 0.3f);
        outptr[i + 3] = clamp01f(hdr3 * importance_weight_f(final3));
    }
    for (; i < n; ++i) {
        float r0 = clamp01f(rptr[i] * 1.05f + 0.02f);
        float g0 = clamp01f(gptr[i] * 1.05f + 0.02f);
        float b0 = clamp01f(bptr[i] * 1.05f + 0.02f);
        float gray0 = r0 * gray_r + g0 * gray_g + b0 * gray_b;
        float ge0 = clamp01f((gray0 - 0.05f) * inv_contrast);
        ge0 = ge0 * ge0 * (3.0f - 2.0f * ge0);
        float gain0 = 0.95f * std::sqrt(0.36f * ge0 * ge0 + 0.1f);
        float comp0 = ge0 * gain0;
        float hdr0 = comp0 / (1.0f + comp0);
        float mask0;
        if (hdr0 > threshold) {
            mask0 = r0 * 0.11f + g0 * 0.22f - b0 * 0.33f + 1.01f;
            mask0 = (mask0 > 0.8f) ? (mask0 * 0.44f) : (mask0 + 0.55f);
        } else {
            mask0 = r0 * 0.66f - g0 * 0.77f + b0 * 0.88f - 0.99f;
            mask0 = (mask0 < 0.2f) ? (mask0 + 0.22f) : (mask0 * 0.33f);
        }
        float sx0 = hdr0 * p_sin, cx0 = r0 * p_cos;
        float sin0 = sx0 * (1.0f - sx0 * sx0 * sin_c3);
        float cos0 = 1.0f - cx0 * cx0 * 0.5f;
        float final0 = clamp01f(mask0 * 0.7f + sin0 * cos0 * 0.3f);
        outptr[i] = clamp01f(hdr0 * importance_weight_f(final0));
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

