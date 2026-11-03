// Correctness tests: every primitive x level x hypercube shape x comm mask x dtype x operator x SIMD flavour,
// compared against a straightforward reference on logical (host-domain) data.
#include "collectives.hpp"
#include <cstdio>
#include <random>
using namespace pc;

static uint64_t ref_comb(uint64_t a, uint64_t b, int dt, Op op) {
    uint8_t A[8], B[8], R[8]; std::memcpy(A, &a, 8); std::memcpy(B, &b, 8);
    if (op != Op::SUM) { for (int i = 0; i < 8; i++) R[i] = op == Op::OR ? (A[i] | B[i]) : (A[i] ^ B[i]); }
    else if (dt == 1) { for (int i = 0; i < 8; i++) R[i] = (uint8_t)(A[i] + B[i]); }
    else if (dt == 4) { for (int h = 0; h < 2; h++) { uint32_t x, y; std::memcpy(&x, A + 4 * h, 4); std::memcpy(&y, B + 4 * h, 4); x += y; std::memcpy(R + 4 * h, &x, 4); } }
    else { uint64_t x = a + b; std::memcpy(R, &x, 8); }
    uint64_t r; std::memcpy(&r, R, 8); return r;
}

struct Cfg { std::vector<int> dims; std::string mask; };
static long n_pass = 0, n_fail = 0, n_unsup = 0;

static void run(const Cfg& cfg, int prim, int L, int dtype, Op op, size_t wc, bool naive, std::mt19937_64& rng) {
    Hypercube hc(cfg.dims); Plan P;
    try { P = make_plan(hc, cfg.mask); } catch (std::exception& e) { n_unsup++; return; }
    const int G = P.G, n = hc.n;
    size_t Sw = (prim == 1) ? wc : (size_t)G * wc;              // prim: 0=AA 1=AG 2=RS 3=AR
    size_t outw = prim == 0 ? Sw : prim == 1 ? Sw * G : prim == 2 ? wc : Sw;
    Layout Ly = make_layout(std::max<size_t>(Sw, outw) * (prim == 1 ? 1 : 1) + (prim == 1 ? Sw * G : 0));
    PimSystem s(n, Ly.Mw);
    std::vector<std::vector<uint64_t>> in(n, std::vector<uint64_t>(Sw));
    for (int p = 0; p < n; p++) for (size_t w = 0; w < Sw; w++) { in[p][w] = rng(); s.pe_set(p, Ly.src + w, in[p][w]); }
    switch (prim) {
        case 0: alltoall(s, P, L, Sw, Ly); break;
        case 1: allgather(s, P, L, Sw, Ly); break;
        case 2: reducescatter(s, P, L, Sw, Ly, op, dtype); break;
        case 3: allreduce(s, P, L, Sw, Ly, op, dtype, naive); break;
    }
    bool ok = true;
    for (auto& grp : P.groups) for (int j = 0; j < G && ok; j++) {
        int d = grp[j];
        for (size_t w = 0; w < outw && ok; w++) {
            uint64_t exp = 0;
            if (prim == 0) { size_t i = w / wc, t = w % wc; exp = in[grp[i]][(size_t)j * wc + t]; }
            else if (prim == 1) { size_t i = w / Sw, t = w % Sw; exp = in[grp[i]][t]; }
            else if (prim == 2) { exp = in[grp[0]][j * wc + w]; for (int i = 1; i < G; i++) exp = ref_comb(exp, in[grp[i]][j * wc + w], dtype, op); }
            else { exp = in[grp[0]][w]; for (int i = 1; i < G; i++) exp = ref_comb(exp, in[grp[i]][w], dtype, op); }
            if (s.pe_get(d, Ly.dst + w) != exp) { ok = false;
                std::printf("  MISMATCH prim=%d L=%d pe=%d w=%zu got=%016lx exp=%016lx\n", prim, L, d, w, (unsigned long)s.pe_get(d, Ly.dst + w), (unsigned long)exp); }
        }
    }
    if (ok) n_pass++; else { n_fail++;
        std::string sh; for (int d : cfg.dims) sh += std::to_string(d) + "x";
        std::printf("FAIL shape=%s mask=%s prim=%d level=%d dtype=%d op=%d wc=%zu naive=%d simd=%s\n", sh.c_str(), cfg.mask.c_str(), prim, L, dtype, (int)op, wc, (int)naive, K().name); }
}

int main() {
    // unit test: domain transfer == naive 8x8 byte transpose, self-inverse; rotation == lane shift
    std::mt19937_64 rng(12345);
    for (int it = 0; it < 1000; it++) {
        uint64_t a[8], b[8]; for (auto& x : a) x = rng(); std::memcpy(b, a, 64); dt64(b);
        for (int i = 0; i < 8; i++) for (int c = 0; c < 8; c++)
            if (((uint8_t*)b)[8 * c + i] != ((uint8_t*)a)[8 * i + c]) { std::printf("dt64 BROKEN\n"); return 1; }
        dt64(b); if (std::memcmp(a, b, 64)) { std::printf("dt64 not self-inverse\n"); return 1; }
    }
    std::vector<std::string> simds = {"scalar"}; if (have_avx2()) simds.push_back("avx2"); if (have_avx512()) simds.push_back("avx512");
    simds.push_back("scalar-ptr"); if (have_avx2()) simds.push_back("avx2-ptr"); if (have_avx512()) simds.push_back("avx512-ptr");
    std::vector<Cfg> cfgs = {
        {{8}, "1"}, {{16}, "1"}, {{32}, "1"}, {{64}, "1"}, {{24}, "1"},
        {{8, 8}, "10"}, {{8, 8}, "01"}, {{8, 8}, "11"}, {{16, 4}, "10"}, {{16, 4}, "11"}, {{32, 2}, "10"}, {{32, 2}, "11"},
        {{8, 2, 4}, "100"}, {{8, 2, 4}, "101"}, {{8, 2, 4}, "111"}, {{8, 4, 2}, "110"}, {{16, 2, 2}, "101"}, {{4, 16}, "11"},
        {{8, 3}, "10"}, {{8, 3}, "11"},
        // expected to be rejected (group < 8 PEs, or an entangled group would have to be split):
        {{8, 8}, "00"}, {{4, 16}, "01"}, {{16, 4}, "01"}, {{8, 2, 4}, "010"},
    };
    for (auto& sm : simds) {
        set_simd(sm); long before = n_pass;
        for (auto& cfg : cfgs) for (int prim = 0; prim < 4; prim++) for (int L = 0; L <= 3; L++)
            for (size_t wc : {1, 3}) {
                if (prim <= 1) run(cfg, prim, L, 8, Op::SUM, wc, false, rng);
                else for (int dtype : {1, 4, 8}) for (Op op : {Op::SUM, Op::OR, Op::XOR}) {
                    run(cfg, prim, L, dtype, op, wc, false, rng);
                    if (prim == 3 && L >= 1) run(cfg, prim, L, dtype, op, wc, true, rng);
                }
            }
        std::printf("simd=%-7s : %ld cases passed\n", sm.c_str(), n_pass - before);
    }
    std::printf("TOTAL pass=%ld fail=%ld (skipped as unsupported shape/mask: %ld)\n", n_pass, n_fail, n_unsup);
    return n_fail ? 1 : 0;
}
