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

void stu_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    const float *__restrict__ a_data = data.a.data();
    const float *__restrict__ b_data = data.b.data();
    const float *__restrict__ c_data = data.c.data();
    const float *__restrict__ d_data = data.d.data();
    const float *__restrict__ e_data = data.e.data();
    const float *__restrict__ f_data = data.f.data();
    const float *__restrict__ g_data = data.g.data();
    const float *__restrict__ h_data = data.h.data();
    const float *__restrict__ i_data = data.i.data();

    double total = 0.0;

    double linear_total = 0.0;
    for (std::size_t y = 0; y < H; ++y) {
        const int wy = (y == 0 || y + 1 == H) ? 1 :
                       (y == 1 || y + 2 == H) ? 2 : 3;
        const float *__restrict__ c_row = c_data + y * W;
        for (std::size_t x = 0; x < W; ++x) {
            const int wx = (x == 0 || x + 1 == W) ? 1 :
                           (x == 1 || x + 2 == W) ? 2 : 3;
            linear_total += static_cast<double>(c_row[x]) *
                            static_cast<double>(wx * wy) * inv9;
        }
    }

    for (std::size_t y = 0; y < H; ++y) {
        const int vy = (y == 0 || y + 1 == H) ? 1 :
                       (y == 1 || y + 2 == H) ? 3 : 4;
        const float *__restrict__ f_row = f_data + y * W;
        linear_total -= static_cast<double>(vy) *
                        (static_cast<double>(f_row[0]) + f_row[1]);
        linear_total += static_cast<double>(vy) *
                        (static_cast<double>(f_row[W - 2]) + f_row[W - 1]);
    }

    for (std::size_t x = 0; x < W; ++x) {
        const int hx = (x == 0 || x + 1 == W) ? 1 :
                       (x == 1 || x + 2 == W) ? 3 : 4;
        linear_total -= static_cast<double>(hx) *
                        (static_cast<double>(i_data[x]) + i_data[W + x]);
        linear_total += static_cast<double>(hx) *
                        (static_cast<double>(i_data[(H - 2) * W + x]) +
                         i_data[(H - 1) * W + x]);
    }

    for (std::size_t y = 1; y + 1 < H; ++y) {
        double row_total = 0.0;
        const std::size_t ym1 = (y - 1) * W;
        const std::size_t y0 = y * W;
        const std::size_t yp1 = (y + 1) * W;

        const float *__restrict__ a0 = a_data + ym1;
        const float *__restrict__ a1 = a_data + y0;
        const float *__restrict__ a2 = a_data + yp1;
        const float *__restrict__ b0 = b_data + ym1;
        const float *__restrict__ b1 = b_data + y0;
        const float *__restrict__ b2 = b_data + yp1;
        const float *__restrict__ d0 = d_data + ym1;
        const float *__restrict__ d1 = d_data + y0;
        const float *__restrict__ d2 = d_data + yp1;
        const float *__restrict__ e0 = e_data + ym1;
        const float *__restrict__ e1 = e_data + y0;
        const float *__restrict__ e2 = e_data + yp1;
        const float *__restrict__ g0 = g_data + ym1;
        const float *__restrict__ g2 = g_data + yp1;
        const float *__restrict__ h0 = h_data + ym1;
        const float *__restrict__ h2 = h_data + yp1;

        float a_col0 = a0[0] + a1[0] + a2[0];
        float a_col1 = a0[1] + a1[1] + a2[1];
        float a_col2 = a0[2] + a1[2] + a2[2];
        float b_col0 = b0[0] + b1[0] + b2[0];
        float b_col1 = b0[1] + b1[1] + b2[1];
        float b_col2 = b0[2] + b1[2] + b2[2];

        for (std::size_t x = 1; x + 2 < W; ++x) {
            const std::size_t xm1 = x - 1;
            const std::size_t xp1 = x + 1;

            const float avg_a = (a_col0 + a_col1 + a_col2) * inv9;
            const float avg_b = (b_col0 + b_col1 + b_col2) * inv9;
            const float p1 = avg_a * avg_b;

            const float sobel_dx =
                -d0[xm1] + d0[xp1]
                -2.0f * d1[xm1] + 2.0f * d1[xp1]
                -d2[xm1] + d2[xp1];
            const float sobel_ex =
                -e0[xm1] + e0[xp1]
                -2.0f * e1[xm1] + 2.0f * e1[xp1]
                -e2[xm1] + e2[xp1];
            const float p2 = sobel_dx * sobel_ex;

            const float sobel_gy =
                -g0[xm1] - 2.0f * g0[x] - g0[xp1]
                + g2[xm1] + 2.0f * g2[x] + g2[xp1];
            const float sobel_hy =
                -h0[xm1] - 2.0f * h0[x] - h0[xp1]
                + h2[xm1] + 2.0f * h2[x] + h2[xp1];
            const float p3 = sobel_gy * sobel_hy;

            row_total += p1 + p2 + p3;

            a_col0 = a_col1;
            a_col1 = a_col2;
            a_col2 = a0[x + 2] + a1[x + 2] + a2[x + 2];
            b_col0 = b_col1;
            b_col1 = b_col2;
            b_col2 = b0[x + 2] + b1[x + 2] + b2[x + 2];
        }

        const std::size_t x = W - 2;
        const std::size_t xm1 = x - 1;
        const std::size_t xp1 = x + 1;

        const float avg_a = (a_col0 + a_col1 + a_col2) * inv9;
        const float avg_b = (b_col0 + b_col1 + b_col2) * inv9;
        const float p1 = avg_a * avg_b;

        const float sobel_dx =
            -d0[xm1] + d0[xp1]
            -2.0f * d1[xm1] + 2.0f * d1[xp1]
            -d2[xm1] + d2[xp1];
        const float sobel_ex =
            -e0[xm1] + e0[xp1]
            -2.0f * e1[xm1] + 2.0f * e1[xp1]
            -e2[xm1] + e2[xp1];
        const float p2 = sobel_dx * sobel_ex;

        const float sobel_gy =
            -g0[xm1] - 2.0f * g0[x] - g0[xp1]
            + g2[xm1] + 2.0f * g2[x] + g2[xp1];
        const float sobel_hy =
            -h0[xm1] - 2.0f * h0[x] - h0[xp1]
            + h2[xm1] + 2.0f * h2[x] + h2[xp1];
        const float p3 = sobel_gy * sobel_hy;

        row_total += p1 + p2 + p3;
        total += row_total;
    }

    total += linear_total;
    out = static_cast<float>(total);
}
// void stu_filter_gradient(float& out, const std::vector<Pixel>& aos,
//                    std::size_t width, std::size_t height) {
//     const std::size_t W = width;
//     const std::size_t H = height;
//     constexpr float inv9 = 1.0f / 9.0f;

//     const Pixel* base = aos.data();
//     double total = 0.0;

//     for (std::size_t y = 1; y + 1 < H; ++y) {
//         const Pixel* top_row = base + (y - 1) * W;
//         const Pixel* mid_row = top_row + W;
//         const Pixel* bot_row = mid_row + W;
//         const Pixel* top = top_row + 1;
//         const Pixel* mid = mid_row + 1;
//         const Pixel* bot = bot_row + 1;
//         const Pixel* end = top_row + W - 1;

//         for (; top + 1 < end; top += 2, mid += 2, bot += 2) {
//             const Pixel* t0 = top - 1;
//             const Pixel* t1 = top;
//             const Pixel* t2 = top + 1;
//             const Pixel* m0 = mid - 1;
//             const Pixel* m1 = mid;
//             const Pixel* m2 = mid + 1;
//             const Pixel* b0 = bot - 1;
//             const Pixel* b1 = bot;
//             const Pixel* b2 = bot + 1;

//             const float sum_a0 = t0->a + t1->a + t2->a +
//                                  m0->a + m1->a + m2->a +
//                                  b0->a + b1->a + b2->a;
//             const float sum_b0 = t0->b + t1->b + t2->b +
//                                  m0->b + m1->b + m2->b +
//                                  b0->b + b1->b + b2->b;
//             const float sum_c0 = t0->c + t1->c + t2->c +
//                                  m0->c + m1->c + m2->c +
//                                  b0->c + b1->c + b2->c;
//             const float avg_a0 = sum_a0 * inv9;
//             const float avg_b0 = sum_b0 * inv9;
//             const float avg_c0 = sum_c0 * inv9;
//             const float p10 = avg_a0 * avg_b0 + avg_c0;

//             const float sobel_dx0 =
//                 -t0->d + t2->d
//                 -2.0f * m0->d + 2.0f * m2->d
//                 -b0->d + b2->d;
//             const float sobel_ex0 =
//                 -t0->e + t2->e
//                 -2.0f * m0->e + 2.0f * m2->e
//                 -b0->e + b2->e;
//             const float sobel_fx0 =
//                 -t0->f + t2->f
//                 -2.0f * m0->f + 2.0f * m2->f
//                 -b0->f + b2->f;
//             const float p20 = sobel_dx0 * sobel_ex0 + sobel_fx0;

//             const float sobel_gy0 =
//                 -t0->g - 2.0f * t1->g - t2->g
//                 + b0->g + 2.0f * b1->g + b2->g;
//             const float sobel_hy0 =
//                 -t0->h - 2.0f * t1->h - t2->h
//                 + b0->h + 2.0f * b1->h + b2->h;
//             const float sobel_iy0 =
//                 -t0->i - 2.0f * t1->i - t2->i
//                 + b0->i + 2.0f * b1->i + b2->i;
//             const float p30 = sobel_gy0 * sobel_hy0 + sobel_iy0;

//             const Pixel* u0 = top;
//             const Pixel* u1 = top + 1;
//             const Pixel* u2 = top + 2;
//             const Pixel* v0 = mid;
//             const Pixel* v1 = mid + 1;
//             const Pixel* v2 = mid + 2;
//             const Pixel* w0 = bot;
//             const Pixel* w1 = bot + 1;
//             const Pixel* w2 = bot + 2;

//             const float sum_a1 = u0->a + u1->a + u2->a +
//                                  v0->a + v1->a + v2->a +
//                                  w0->a + w1->a + w2->a;
//             const float sum_b1 = u0->b + u1->b + u2->b +
//                                  v0->b + v1->b + v2->b +
//                                  w0->b + w1->b + w2->b;
//             const float sum_c1 = u0->c + u1->c + u2->c +
//                                  v0->c + v1->c + v2->c +
//                                  w0->c + w1->c + w2->c;
//             const float avg_a1 = sum_a1 * inv9;
//             const float avg_b1 = sum_b1 * inv9;
//             const float avg_c1 = sum_c1 * inv9;
//             const float p11 = avg_a1 * avg_b1 + avg_c1;

//             const float sobel_dx1 =
//                 -u0->d + u2->d
//                 -2.0f * v0->d + 2.0f * v2->d
//                 -w0->d + w2->d;
//             const float sobel_ex1 =
//                 -u0->e + u2->e
//                 -2.0f * v0->e + 2.0f * v2->e
//                 -w0->e + w2->e;
//             const float sobel_fx1 =
//                 -u0->f + u2->f
//                 -2.0f * v0->f + 2.0f * v2->f
//                 -w0->f + w2->f;
//             const float p21 = sobel_dx1 * sobel_ex1 + sobel_fx1;

//             const float sobel_gy1 =
//                 -u0->g - 2.0f * u1->g - u2->g
//                 + w0->g + 2.0f * w1->g + w2->g;
//             const float sobel_hy1 =
//                 -u0->h - 2.0f * u1->h - u2->h
//                 + w0->h + 2.0f * w1->h + w2->h;
//             const float sobel_iy1 =
//                 -u0->i - 2.0f * u1->i - u2->i
//                 + w0->i + 2.0f * w1->i + w2->i;
//             const float p31 = sobel_gy1 * sobel_hy1 + sobel_iy1;

//             total += p10 + p20 + p30 + p11 + p21 + p31;
//         }

//         for (; top < end; ++top, ++mid, ++bot) {
//             const Pixel* t0 = top - 1;
//             const Pixel* t1 = top;
//             const Pixel* t2 = top + 1;
//             const Pixel* m0 = mid - 1;
//             const Pixel* m1 = mid;
//             const Pixel* m2 = mid + 1;
//             const Pixel* b0 = bot - 1;
//             const Pixel* b1 = bot;
//             const Pixel* b2 = bot + 1;

//             const float sum_a = t0->a + t1->a + t2->a +
//                                 m0->a + m1->a + m2->a +
//                                 b0->a + b1->a + b2->a;
//             const float sum_b = t0->b + t1->b + t2->b +
//                                 m0->b + m1->b + m2->b +
//                                 b0->b + b1->b + b2->b;
//             const float sum_c = t0->c + t1->c + t2->c +
//                                 m0->c + m1->c + m2->c +
//                                 b0->c + b1->c + b2->c;
//             const float avg_a = sum_a * inv9;
//             const float avg_b = sum_b * inv9;
//             const float avg_c = sum_c * inv9;
//             const float p1 = avg_a * avg_b + avg_c;

//             const float sobel_dx =
//                 -t0->d + t2->d
//                 -2.0f * m0->d + 2.0f * m2->d
//                 -b0->d + b2->d;
//             const float sobel_ex =
//                 -t0->e + t2->e
//                 -2.0f * m0->e + 2.0f * m2->e
//                 -b0->e + b2->e;
//             const float sobel_fx =
//                 -t0->f + t2->f
//                 -2.0f * m0->f + 2.0f * m2->f
//                 -b0->f + b2->f;
//             const float p2 = sobel_dx * sobel_ex + sobel_fx;

//             const float sobel_gy =
//                 -t0->g - 2.0f * t1->g - t2->g
//                 + b0->g + 2.0f * b1->g + b2->g;
//             const float sobel_hy =
//                 -t0->h - 2.0f * t1->h - t2->h
//                 + b0->h + 2.0f * b1->h + b2->h;
//             const float sobel_iy =
//                 -t0->i - 2.0f * t1->i - t2->i
//                 + b0->i + 2.0f * b1->i + b2->i;
//             const float p3 = sobel_gy * sobel_hy + sobel_iy;

//             total += p1 + p2 + p3;
//         }
//     }

//     out = total;
// }

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
