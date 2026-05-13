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

    const Pixel* base = aos.data();
    double total = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const Pixel* row_m1 = base + (y - 1) * W;
        const Pixel* row_0 = row_m1 + W;
        const Pixel* row_p1 = row_0 + W;

        const Pixel* top = row_m1;
        const Pixel* mid = row_0;
        const Pixel* bot = row_p1;

        for (std::size_t x = 1; x + 1 < W; ++x) {
            const float sum_a = top[0].a + top[1].a + top[2].a +
                                mid[0].a + mid[1].a + mid[2].a +
                                bot[0].a + bot[1].a + bot[2].a;
            const float sum_b = top[0].b + top[1].b + top[2].b +
                                mid[0].b + mid[1].b + mid[2].b +
                                bot[0].b + bot[1].b + bot[2].b;
            const float sum_c = top[0].c + top[1].c + top[2].c +
                                mid[0].c + mid[1].c + mid[2].c +
                                bot[0].c + bot[1].c + bot[2].c;

            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const float sobel_dx =
                -top[0].d + top[2].d
                -2.0f * mid[0].d + 2.0f * mid[2].d
                -bot[0].d + bot[2].d;
            const float sobel_ex =
                -top[0].e + top[2].e
                -2.0f * mid[0].e + 2.0f * mid[2].e
                -bot[0].e + bot[2].e;
            const float sobel_fx =
                -top[0].f + top[2].f
                -2.0f * mid[0].f + 2.0f * mid[2].f
                -bot[0].f + bot[2].f;
            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -top[0].g - 2.0f * top[1].g - top[2].g
                + bot[0].g + 2.0f * bot[1].g + bot[2].g;
            const float sobel_hy =
                -top[0].h - 2.0f * top[1].h - top[2].h
                + bot[0].h + 2.0f * bot[1].h + bot[2].h;
            const float sobel_iy =
                -top[0].i - 2.0f * top[1].i - top[2].i
                + bot[0].i + 2.0f * bot[1].i + bot[2].i;
            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;

            ++top;
            ++mid;
            ++bot;
        }
    }

    out = total;
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
