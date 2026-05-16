#include "blackscholes.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <cmath>

#define inv_sqrt_2xPI 0.39894228040143270286
#define p_val 0.2316419
#define coefficient_a1 0.319381530
#define coefficient_a2 -0.356563782
#define coefficient_a3 1.781477937
#define coefficient_a4 -1.821255978
#define coefficient_a5 1.330274429

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n,
                             std::uint32_t seed) {
    args.call_option_price.assign(n, 0.0f);
    args.put_option_price.assign(n, 0.0f);
    args.epsilon = 5e-3;

    args.spot_price.resize(n);
    args.strike.resize(n);
    args.rate.resize(n);
    args.volatility.resize(n);
    args.time.resize(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> spot_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> strike_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> rate_dist(0.0275f, 0.1f);
    std::uniform_real_distribution<float> vol_dist(0.05f, 0.6f);
    std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        args.spot_price[i] = spot_dist(rng);
        args.strike[i] = strike_dist(rng);
        args.rate[i] = rate_dist(rng);
        args.volatility[i] = vol_dist(rng);
        args.time[i] = time_dist(rng);
    }
}

void CNDF(float &InputX, float &OutputX) {
    int sign = 0;
    float x = InputX;

    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX = std::exp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

static inline void naive_BlkSchls_one(float &CallOptionPrice,
                                      float &PutOptionPrice, float spotPrice,
                                      float strike, float rate,
                                      float volatility, float time) {
    const float xSqrtTime = std::sqrt(time);
    const float xLogTerm = std::log(spotPrice / strike);
    const float xPowerTerm = 0.5f * volatility * volatility;

    float xD1 = (rate + xPowerTerm) * time + xLogTerm;
    const float xDen = volatility * xSqrtTime;
    xD1 = xD1 / xDen;
    const float xD2 = xD1 - xDen;

    float d1 = xD1;
    float d2 = xD2;
    float NofXd1 = 0.0f;
    float NofXd2 = 0.0f;

    CNDF(d1, NofXd1);
    CNDF(d2, NofXd2);

    const float FutureValueX = strike * std::exp(-(rate) * (time));
    CallOptionPrice = (spotPrice * NofXd1) - (FutureValueX * NofXd2);

    const float NegNofXd1 = 1.0f - NofXd1;
    const float NegNofXd2 = 1.0f - NofXd2;
    PutOptionPrice = (FutureValueX * NegNofXd2) - (spotPrice * NegNofXd1);
}

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time) {
    size_t n = spotPrice.size();
    for (size_t i = 0; i < n; ++i) {
        naive_BlkSchls_one(CallOptionPrice[i],
                           PutOptionPrice[i],
                           spotPrice[i],
                           strike[i],
                           rate[i],
                           volatility[i],
                           time[i]);
    }
}
static inline void fast_CNDF(float x, float& out) {
    const float ax = std::fabs(x);

    const float xNPrimeofX =
        std::exp(-0.5f * ax * ax) * inv_sqrt_2xPI;

    const float k =
        1.0f / (1.0f + p_val * ax);

    const float k2 = k * k;
    const float k3 = k2 * k;
    const float k4 = k3 * k;
    const float k5 = k4 * k;

    float local =
        k * coefficient_a1 +
        k2 * coefficient_a2 +
        k3 * coefficient_a3 +
        k4 * coefficient_a4 +
        k5 * coefficient_a5;

    local = 1.0f - local * xNPrimeofX;

    out = (x < 0.0f)
        ? (1.0f - local)
        : local;
}
static inline void bs_cndf(float d, float& Nd, float inv_s2pi,
                             float p, float a1, float a2,
                             float a3, float a4, float a5) {
    float L = fabsf(d);
    float K = 1.0f / (1.0f + p * L);
    float poly = K * (a1 + K * (a2 + K * (a3 + K * (a4 + K * a5))));
    float n_prime = expf(-0.5f * d * d) * inv_s2pi;
    float res = poly * n_prime;
    Nd = (d < 0.0f) ? res : 1.0f - res;
}

void stu_BlkSchls(
    std::vector<float>& CallOptionPrice,
    std::vector<float>& PutOptionPrice,
    const std::vector<float>& spotPrice,
    const std::vector<float>& strike,
    const std::vector<float>& rate,
    const std::vector<float>& volatility,
    const std::vector<float>& time)
{
    const size_t n = spotPrice.size();

    const float* __restrict s = spotPrice.data();
    const float* __restrict k = strike.data();
    const float* __restrict r = rate.data();
    const float* __restrict v = volatility.data();
    const float* __restrict t = time.data();
    float* __restrict call = CallOptionPrice.data();
    float* __restrict put  = PutOptionPrice.data();

    const float inv_s2pi = 0.3989422804f;
    const float p_cnd    = 0.2316419f;
    const float a1       = 0.319381530f;
    const float a2       = -0.356563782f;
    const float a3       = 1.781477937f;
    const float a4       = -1.821255978f;
    const float a5       = 1.330274429f;

    const size_t limit = n & ~static_cast<size_t>(3);
    size_t i = 0;

    for (; i < limit; i += 4) {
        // ── Load 4 elements ──
        float s0 = s[i], s1 = s[i+1], s2 = s[i+2], s3 = s[i+3];
        float k0 = k[i], k1 = k[i+1], k2 = k[i+2], k3 = k[i+3];
        float r0 = r[i], r1 = r[i+1], r2 = r[i+2], r3 = r[i+3];
        float v0 = v[i], v1 = v[i+1], v2 = v[i+2], v3 = v[i+3];
        float t0 = t[i], t1 = t[i+1], t2 = t[i+2], t3 = t[i+3];

        // ── Elem 0 ──
        {
            float sqT = sqrtf(t0);
            float v_sqT = v0 * sqT;
            float d1 = ((r0 + 0.5f * v0 * v0) * t0 + logf(s0 / k0)) / v_sqT;
            float d2 = d1 - v_sqT;
            float Nd1, Nd2;
            bs_cndf(d1, Nd1, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            bs_cndf(d2, Nd2, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            float fv = k0 * expf(-r0 * t0);
            float cv = s0 * Nd1 - fv * Nd2;
            call[i] = cv;
            put[i]  = cv - s0 + fv;
        }
        // ── Elem 1 ──
        {
            float sqT = sqrtf(t1);
            float v_sqT = v1 * sqT;
            float d1 = ((r1 + 0.5f * v1 * v1) * t1 + logf(s1 / k1)) / v_sqT;
            float d2 = d1 - v_sqT;
            float Nd1, Nd2;
            bs_cndf(d1, Nd1, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            bs_cndf(d2, Nd2, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            float fv = k1 * expf(-r1 * t1);
            float cv = s1 * Nd1 - fv * Nd2;
            call[i+1] = cv;
            put[i+1]  = cv - s1 + fv;
        }
        // ── Elem 2 ──
        {
            float sqT = sqrtf(t2);
            float v_sqT = v2 * sqT;
            float d1 = ((r2 + 0.5f * v2 * v2) * t2 + logf(s2 / k2)) / v_sqT;
            float d2 = d1 - v_sqT;
            float Nd1, Nd2;
            bs_cndf(d1, Nd1, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            bs_cndf(d2, Nd2, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            float fv = k2 * expf(-r2 * t2);
            float cv = s2 * Nd1 - fv * Nd2;
            call[i+2] = cv;
            put[i+2]  = cv - s2 + fv;
        }
        // ── Elem 3 ──
        {
            float sqT = sqrtf(t3);
            float v_sqT = v3 * sqT;
            float d1 = ((r3 + 0.5f * v3 * v3) * t3 + logf(s3 / k3)) / v_sqT;
            float d2 = d1 - v_sqT;
            float Nd1, Nd2;
            bs_cndf(d1, Nd1, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            bs_cndf(d2, Nd2, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
            float fv = k3 * expf(-r3 * t3);
            float cv = s3 * Nd1 - fv * Nd2;
            call[i+3] = cv;
            put[i+3]  = cv - s3 + fv;
        }
    }
    for (; i < n; ++i) {
        float sqT = sqrtf(t[i]);
        float v_sqT = v[i] * sqT;
        float d1 = ((r[i] + 0.5f * v[i] * v[i]) * t[i] + logf(s[i] / k[i])) / v_sqT;
        float d2 = d1 - v_sqT;
        float Nd1, Nd2;
        bs_cndf(d1, Nd1, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
        bs_cndf(d2, Nd2, inv_s2pi, p_cnd, a1, a2, a3, a4, a5);
        float fv = k[i] * expf(-r[i] * t[i]);
        float cv = s[i] * Nd1 - fv * Nd2;
        call[i] = cv;
        put[i]  = cv - s[i] + fv;
    }
}

void naive_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    naive_BlkSchls(args.call_option_price,
                   args.put_option_price,
                   args.spot_price,
                   args.strike,
                   args.rate,
                   args.volatility,
                   args.time);
}

void stu_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    stu_BlkSchls(args.call_option_price,
                 args.put_option_price,
                 args.spot_price,
                 args.strike,
                 args.rate,
                 args.volatility,
                 args.time);
}

bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<blackscholes_args *>(stu_ctx);
    auto &ref_args = *static_cast<blackscholes_args *>(ref_ctx);
    const double eps = ref_args.epsilon; // relative tolerance

    if (ref_args.call_option_price.size() != stu_args.call_option_price.size() ||
        ref_args.put_option_price.size() != stu_args.put_option_price.size())
        return false;

    const double atol = 1e-5; // absolute tolerance for near-zero prices
    const size_t n = ref_args.call_option_price.size();
    double max_rel = 0.0, max_abs = 0.0;
    size_t max_idx = 0;
    const char *max_leg = "call";

    for (size_t i = 0; i < n; ++i) {
        const double rc = static_cast<double>(ref_args.call_option_price[i]);
        const double rp = static_cast<double>(ref_args.put_option_price[i]);
        const double sc = static_cast<double>(stu_args.call_option_price[i]);
        const double sp = static_cast<double>(stu_args.put_option_price[i]);

        const double err_c = std::abs(rc - sc);
        const double err_p = std::abs(rp - sp);
        const double rel_c = (err_c - atol) / std::abs(rc);
        const double rel_p = (err_p - atol) / std::abs(rp);

        const bool call_ok = err_c <= (atol + eps * std::abs(rc));
        const bool put_ok = err_p <= (atol + eps * std::abs(rp));

        if (rel_c > max_rel) {
            max_abs = err_c;
            max_rel = rel_c;
            max_idx = i;
            max_leg = "call";
        }
        if (rel_p > max_rel) {
            max_abs = err_p;
            max_rel = rel_p;
            max_idx = i;
            max_leg = "put";
        }

        if (!call_ok || !put_ok) {
            debug_log("\tDEBUG: fail idx={} | call ref={} stu={} err={} thr={} | put ref={} stu={} err={} thr={}\n",
                      i,
                      rc,
                      sc,
                      err_c,
                      (atol + eps * std::abs(rc)),
                      rp,
                      sp,
                      err_p,
                      (atol + eps * std::abs(rp)));
            return false;
        }
    }
    debug_log("\tBlkSchls_check passed: n={}, max_rel_err={}, max_abs_err={} at idx={} ({})\n",
              n,
              max_rel,
              max_abs,
              max_idx,
              max_leg);

    return true;
}
