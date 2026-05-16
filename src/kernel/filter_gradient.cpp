#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

// Convert from SoA (Structure of Arrays) to AoS (Array of Structures)
// This is called outside of the timing loop
void convert_soa_to_aos(std::vector<Pixel>& aos, const data_struct& data) {
    aos.resize(data.a.size());
    for (std::size_t i = 0; i < data.a.size(); ++i) {
        aos[i].a = data.a[i];
        aos[i].b = data.b[i];
        aos[i].c = data.c[i];
        aos[i].d = data.d[i];
        aos[i].e = data.e[i];
        aos[i].f = data.f[i];
        aos[i].g = data.g[i];
        aos[i].h = data.h[i];
        aos[i].i = data.i[i];
    }
}

void stu_filter_gradient(float& out, const std::vector<Pixel>& aos,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    const Pixel* __restrict base = aos.data();

    double total0 = 0.0, total1 = 0.0, total2 = 0.0, total3 = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const Pixel* __restrict top_row = base + (y - 1) * W;
        const Pixel* __restrict mid_row = top_row + W;
        const Pixel* __restrict bot_row = mid_row + W;

        std::size_t x = 1;
        // ── 4x unrolled: 18 struct reads for 4 pixels ──────────────────
        for (; x + 4 < W; x += 4) {
            // Pre-load 6 columns × 3 rows = 18 pixels
            const Pixel* __restrict t0 = top_row + x - 1;  // col -1
            const Pixel* __restrict t1 = top_row + x;      // col  0
            const Pixel* __restrict t2 = top_row + x + 1;  // col +1
            const Pixel* __restrict t3 = top_row + x + 2;  // col +2
            const Pixel* __restrict t4 = top_row + x + 3;  // col +3
            const Pixel* __restrict t5 = top_row + x + 4;  // col +4

            const Pixel* __restrict m0 = mid_row + x - 1;
            const Pixel* __restrict m1 = mid_row + x;
            const Pixel* __restrict m2 = mid_row + x + 1;
            const Pixel* __restrict m3 = mid_row + x + 2;
            const Pixel* __restrict m4 = mid_row + x + 3;
            const Pixel* __restrict m5 = mid_row + x + 4;

            const Pixel* __restrict b0 = bot_row + x - 1;
            const Pixel* __restrict b1 = bot_row + x;
            const Pixel* __restrict b2 = bot_row + x + 1;
            const Pixel* __restrict b3 = bot_row + x + 2;
            const Pixel* __restrict b4 = bot_row + x + 3;
            const Pixel* __restrict b5 = bot_row + x + 4;

            // ── Pixel 0 (center at x) ──
            {
                float sa = t0->a+t1->a+t2->a + m0->a+m1->a+m2->a + b0->a+b1->a+b2->a;
                float sb = t0->b+t1->b+t2->b + m0->b+m1->b+m2->b + b0->b+b1->b+b2->b;
                float sc = t0->c+t1->c+t2->c + m0->c+m1->c+m2->c + b0->c+b1->c+b2->c;
                float p1 = (sa*inv9)*(sb*inv9) + sc*inv9;
                float sdx = -t0->d+t2->d -2.0f*m0->d+2.0f*m2->d -b0->d+b2->d;
                float sex = -t0->e+t2->e -2.0f*m0->e+2.0f*m2->e -b0->e+b2->e;
                float sfx = -t0->f+t2->f -2.0f*m0->f+2.0f*m2->f -b0->f+b2->f;
                float p2 = sdx*sex + sfx;
                float sgy = -t0->g-2.0f*t1->g-t2->g + b0->g+2.0f*b1->g+b2->g;
                float shy = -t0->h-2.0f*t1->h-t2->h + b0->h+2.0f*b1->h+b2->h;
                float siy = -t0->i-2.0f*t1->i-t2->i + b0->i+2.0f*b1->i+b2->i;
                float p3 = sgy*shy + siy;
                total0 += p1 + p2 + p3;
            }
            // ── Pixel 1 (center at x+1) ──
            {
                float sa = t1->a+t2->a+t3->a + m1->a+m2->a+m3->a + b1->a+b2->a+b3->a;
                float sb = t1->b+t2->b+t3->b + m1->b+m2->b+m3->b + b1->b+b2->b+b3->b;
                float sc = t1->c+t2->c+t3->c + m1->c+m2->c+m3->c + b1->c+b2->c+b3->c;
                float p1 = (sa*inv9)*(sb*inv9) + sc*inv9;
                float sdx = -t1->d+t3->d -2.0f*m1->d+2.0f*m3->d -b1->d+b3->d;
                float sex = -t1->e+t3->e -2.0f*m1->e+2.0f*m3->e -b1->e+b3->e;
                float sfx = -t1->f+t3->f -2.0f*m1->f+2.0f*m3->f -b1->f+b3->f;
                float p2 = sdx*sex + sfx;
                float sgy = -t1->g-2.0f*t2->g-t3->g + b1->g+2.0f*b2->g+b3->g;
                float shy = -t1->h-2.0f*t2->h-t3->h + b1->h+2.0f*b2->h+b3->h;
                float siy = -t1->i-2.0f*t2->i-t3->i + b1->i+2.0f*b2->i+b3->i;
                float p3 = sgy*shy + siy;
                total1 += p1 + p2 + p3;
            }
            // ── Pixel 2 (center at x+2) ──
            {
                float sa = t2->a+t3->a+t4->a + m2->a+m3->a+m4->a + b2->a+b3->a+b4->a;
                float sb = t2->b+t3->b+t4->b + m2->b+m3->b+m4->b + b2->b+b3->b+b4->b;
                float sc = t2->c+t3->c+t4->c + m2->c+m3->c+m4->c + b2->c+b3->c+b4->c;
                float p1 = (sa*inv9)*(sb*inv9) + sc*inv9;
                float sdx = -t2->d+t4->d -2.0f*m2->d+2.0f*m4->d -b2->d+b4->d;
                float sex = -t2->e+t4->e -2.0f*m2->e+2.0f*m4->e -b2->e+b4->e;
                float sfx = -t2->f+t4->f -2.0f*m2->f+2.0f*m4->f -b2->f+b4->f;
                float p2 = sdx*sex + sfx;
                float sgy = -t2->g-2.0f*t3->g-t4->g + b2->g+2.0f*b3->g+b4->g;
                float shy = -t2->h-2.0f*t3->h-t4->h + b2->h+2.0f*b3->h+b4->h;
                float siy = -t2->i-2.0f*t3->i-t4->i + b2->i+2.0f*b3->i+b4->i;
                float p3 = sgy*shy + siy;
                total2 += p1 + p2 + p3;
            }
            // ── Pixel 3 (center at x+3) ──
            {
                float sa = t3->a+t4->a+t5->a + m3->a+m4->a+m5->a + b3->a+b4->a+b5->a;
                float sb = t3->b+t4->b+t5->b + m3->b+m4->b+m5->b + b3->b+b4->b+b5->b;
                float sc = t3->c+t4->c+t5->c + m3->c+m4->c+m5->c + b3->c+b4->c+b5->c;
                float p1 = (sa*inv9)*(sb*inv9) + sc*inv9;
                float sdx = -t3->d+t5->d -2.0f*m3->d+2.0f*m5->d -b3->d+b5->d;
                float sex = -t3->e+t5->e -2.0f*m3->e+2.0f*m5->e -b3->e+b5->e;
                float sfx = -t3->f+t5->f -2.0f*m3->f+2.0f*m5->f -b3->f+b5->f;
                float p2 = sdx*sex + sfx;
                float sgy = -t3->g-2.0f*t4->g-t5->g + b3->g+2.0f*b4->g+b5->g;
                float shy = -t3->h-2.0f*t4->h-t5->h + b3->h+2.0f*b4->h+b5->h;
                float siy = -t3->i-2.0f*t4->i-t5->i + b3->i+2.0f*b4->i+b5->i;
                float p3 = sgy*shy + siy;
                total3 += p1 + p2 + p3;
            }
        }
        // ── Tail: remaining 1-3 pixels ──────────────────────────────────
        for (; x + 1 < W; ++x) {
            const Pixel* t0 = top_row + x - 1;
            const Pixel* t1 = top_row + x;
            const Pixel* t2 = top_row + x + 1;
            const Pixel* m0 = mid_row + x - 1;
            const Pixel* m1 = mid_row + x;
            const Pixel* m2 = mid_row + x + 1;
            const Pixel* b0 = bot_row + x - 1;
            const Pixel* b1 = bot_row + x;
            const Pixel* b2 = bot_row + x + 1;

            float sa = t0->a+t1->a+t2->a + m0->a+m1->a+m2->a + b0->a+b1->a+b2->a;
            float sb = t0->b+t1->b+t2->b + m0->b+m1->b+m2->b + b0->b+b1->b+b2->b;
            float sc = t0->c+t1->c+t2->c + m0->c+m1->c+m2->c + b0->c+b1->c+b2->c;
            float p1 = (sa*inv9)*(sb*inv9) + sc*inv9;

            float sdx = -t0->d+t2->d -2.0f*m0->d+2.0f*m2->d -b0->d+b2->d;
            float sex = -t0->e+t2->e -2.0f*m0->e+2.0f*m2->e -b0->e+b2->e;
            float sfx = -t0->f+t2->f -2.0f*m0->f+2.0f*m2->f -b0->f+b2->f;
            float p2 = sdx*sex + sfx;

            float sgy = -t0->g-2.0f*t1->g-t2->g + b0->g+2.0f*b1->g+b2->g;
            float shy = -t0->h-2.0f*t1->h-t2->h + b0->h+2.0f*b1->h+b2->h;
            float siy = -t0->i-2.0f*t1->i-t2->i + b0->i+2.0f*b1->i+b2->i;
            float p3 = sgy*shy + siy;

            total0 += p1 + p2 + p3;
        }
    }

    out = static_cast<float>(total0 + total1 + total2 + total3);
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.aos_data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
