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

    // 使用 restrict 关键字帮助编译器优化，避免别名检查
    const float* __restrict s = spotPrice.data();
    const float* __restrict k = strike.data();
    const float* __restrict r = rate.data();
    const float* __restrict v = volatility.data();
    const float* __restrict t = time.data();

    float* __restrict call = CallOptionPrice.data();
    float* __restrict put  = PutOptionPrice.data();

    // 常量定义，确保使用单精度浮点数
    const float inv_s2pi = 0.3989422804f; 
    const float p        = 0.2316419f;
    const float a1       = 0.319381530f;
    const float a2       = -0.356563782f;
    const float a3       = 1.781477937f;
    const float a4       = -1.821255978f;
    const float a5       = 1.330274429f;

    // 启用编译器矢量化
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        float si = s[i];
        float ki = k[i];
        float ri = r[i];
        float vi = v[i];
        float ti = t[i];

        // 1. 基础参数计算
        float sqrtT = sqrtf(ti);
        float v_sqrtT = vi * sqrtT;
        float inv_v_sqrtT = 1.0f / v_sqrtT;
        
        float logTerm = logf(si / ki);
        float powerTerm = 0.5f * vi * vi;
        
        // d1 = (log(s/k) + (r + v^2/2)*t) / (v*sqrt(t))
        float d1 = ( (ri + powerTerm) * ti + logTerm ) * inv_v_sqrtT;
        float d2 = d1 - v_sqrtT;

        // 2. 高效 CNDF 计算 (针对 d1 和 d2)
        // 计算 Nd1
        float L1 = fabsf(d1);
        float K1 = 1.0f / (1.0f + p * L1);
        // 使用 Horner 方案减少乘法
        float poly1 = K1 * (a1 + K1 * (a2 + K1 * (a3 + K1 * (a4 + K1 * a5))));
        float n_prime1 = expf(-0.5f * d1 * d1) * inv_s2pi;
        float res1 = poly1 * n_prime1;
        float Nd1 = (d1 < 0.0f) ? res1 : 1.0f - res1;

        // 计算 Nd2
        float L2 = fabsf(d2);
        float K2 = 1.0f / (1.0f + p * L2);
        float poly2 = K2 * (a1 + K2 * (a2 + K2 * (a3 + K2 * (a4 + K2 * a5))));
        float n_prime2 = expf(-0.5f * d2 * d2) * inv_s2pi;
        float res2 = poly2 * n_prime2;
        float Nd2 = (d2 < 0.0f) ? res2 : 1.0f - res2;

        // 3. 计算最终价格
        float expRT = expf(-ri * ti);
        float futureValue = ki * expRT;

        float c_val = (si * Nd1) - (futureValue * Nd2);
        
        // 使用 Put-Call Parity (期权平价公式): Put = Call - Spot + Strike * exp(-rt)
        // 这不仅快，而且在数学上是等价的
        float p_val = c_val - si + futureValue;

        call[i] = c_val;
        put[i]  = p_val;
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
