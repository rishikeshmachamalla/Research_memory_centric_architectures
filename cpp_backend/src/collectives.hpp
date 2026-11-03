// collectives.hpp -- AlltoAll / AllGather / ReduceScatter / AllReduce on the simulated PIM system.
//
// Implementation "levels" (the ablation of the paper, Fig. 16):
//   0 = Baseline : PEs -> host memory (domain transfer) -> global rearrangement/reduction in host memory
//                  -> domain transfer -> PEs.                         (paper Fig. 3a / 7a)
//   1 = +PR      : PE-assisted reordering. PEs locally pre/post-reorder their data so the host only moves
//                  whole 64 B blocks (cache friendly) but still stages them in host memory.
//   2 = +IM      : In-register modulation. Blocks are streamed through "registers" (stack) - no host staging.
//   3 = +CM      : Cross-domain modulation. Where no arithmetic is needed, the domain transfer is dropped and
//                  the lane shift is a byte-level rotate in the raw domain.
#pragma once
#include "pimsim.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <tuple>
#include <cassert>

namespace pc {

// ---------------------------------------------------------------------------------------------
// Plan: lane-consistent routing of one hypercube/communication-mask pair.
// For every PE p and every group member j, the word destined to member j travels from lane
// (p&7) of EG e to lane (dst&7) of EG D.  k = (dst chip - src chip) mod 8 is the lane rotation.
// PEs sort their chunks by (D,k) so that after the PE-side reorder every 64 B block of an EG has
// ONE destination EG and ONE rotation (this is what makes block moves possible).
// ---------------------------------------------------------------------------------------------
struct Key { int D, k; bool operator==(const Key& o) const { return D == o.D && k == o.k; } };

struct Plan {
    int n, neg, G, ngroups;
    std::vector<std::vector<int>> groups;               // [g][member] -> PE
    std::vector<int> idx, gid;                          // member index / group id of each PE
    std::vector<std::vector<int>> pre;                  // [pe][pos]  -> chunk placed at position pos
    std::vector<std::vector<Key>> keys;                 // [eg][pos]  -> (dst EG, rotation)
    std::vector<std::vector<int>> inc_u;                // [eg][pos]  -> arrival slot u at the destination EG
    std::vector<std::vector<std::pair<int,int>>> incoming; // [D][u]  -> (src eg, pos)
    std::vector<std::vector<int>> post_src;             // [pe][u]    -> member index of the sender of slot u
};

inline Plan make_plan(const Hypercube& hc, const std::string& mask) {
    Plan P; P.n = hc.n; P.neg = P.n / 8; P.groups = hc.groups(mask);
    P.G = (int)P.groups[0].size(); P.ngroups = (int)P.groups.size();
    if (P.G < 8) throw std::runtime_error("communication group must contain at least 8 PEs");
    P.idx.assign(P.n, 0); P.gid.assign(P.n, 0);
    for (int g = 0; g < P.ngroups; g++) for (int i = 0; i < P.G; i++) { P.gid[P.groups[g][i]] = g; P.idx[P.groups[g][i]] = i; }
    P.pre.assign(P.n, {}); P.keys.assign(P.neg, {});
    for (int p = 0; p < P.n; p++) {
        int e = p >> 3, c = p & 7;
        std::vector<std::tuple<int,int,int>> v;
        for (int j = 0; j < P.G; j++) { int d = P.groups[P.gid[p]][j]; v.emplace_back(d >> 3, ((d & 7) - c) & 7, j); }
        std::sort(v.begin(), v.end());
        std::vector<Key> kp;
        for (auto& t : v) { P.pre[p].push_back(std::get<2>(t)); kp.push_back({std::get<0>(t), std::get<1>(t)}); }
        if (c == 0) P.keys[e] = kp;
        else if (!(kp == P.keys[e])) throw std::runtime_error("shape/mask is not lane-consistent (an entangled group would have to be split)");
    }
    P.inc_u.assign(P.neg, std::vector<int>(P.G)); P.incoming.assign(P.neg, {});
    for (int e = 0; e < P.neg; e++) for (int pos = 0; pos < P.G; pos++) {
        int D = P.keys[e][pos].D; P.inc_u[e][pos] = (int)P.incoming[D].size(); P.incoming[D].push_back({e, pos});
    }
    for (int D = 0; D < P.neg; D++) if ((int)P.incoming[D].size() != P.G) throw std::runtime_error("unbalanced routing plan");
    P.post_src.assign(P.n, std::vector<int>(P.G));
    for (int d = 0; d < P.n; d++) {
        int D = d >> 3, c = d & 7; std::vector<char> seen(P.G, 0);
        for (int u = 0; u < P.G; u++) {
            auto [e, pos] = P.incoming[D][u]; int src = e * 8 + ((c - P.keys[e][pos].k) & 7);
            if (P.gid[src] != P.gid[d] || seen[P.idx[src]]) throw std::runtime_error("inconsistent plan");
            seen[P.idx[src]] = 1; P.post_src[d][u] = P.idx[src];
        }
    }
    return P;
}

// MRAM layout used by all primitives: [src | dst | buf (pre-reordered data) + staging]
struct Layout { size_t src, dst, buf, Mw; };
inline Layout make_layout(size_t max_words) { size_t X = max_words + 8; return {0, X, 2 * X, 4 * X}; }

// ------------------------------ PE-side kernels (simulated) ------------------------------
inline void pe_pre(PimSystem& s, const Plan& P, size_t src, size_t buf, size_t wc) {
    Timer t;
    for (int p = 0; p < P.n; p++) for (int pos = 0; pos < P.G; pos++) {
        size_t j = P.pre[p][pos];
        for (size_t x = 0; x < wc; x++) s.pe_copy(p, src + j * wc + x, buf + (size_t)pos * wc + x);
    }
    s.st.pe_s += t.s();
}
inline void pe_post(PimSystem& s, const Plan& P, size_t stage, size_t dst, size_t cw) {
    Timer t;
    for (int d = 0; d < P.n; d++) for (int u = 0; u < P.G; u++) {
        size_t i = P.post_src[d][u];
        for (size_t x = 0; x < cw; x++) s.pe_copy(d, stage + (size_t)u * cw + x, dst + i * cw + x);
    }
    s.st.pe_s += t.s();
}

// ------------------------------ host-side routing (non-arithmetic) ------------------------------
// Moves, for every source EG e, position pos and t<tc, the block at word (pos*ss + t) of the source
// (bus at src_w, or the host buffer hsrc) to destination EG D = keys[e][pos].D, word (u*ds + t) at dst_w,
// shifting lanes by k.
#define PC_ROUTE_ARGS PimSystem& s, const Plan& P, int L, size_t src_w, const uint64_t* hsrc, size_t hslots, size_t tc, size_t ss, size_t dst_w, size_t ds
#define PC_ROUTE_CALL s, P, L, src_w, hsrc, hslots, tc, ss, dst_w, ds
template <class KP> static inline __attribute__((always_inline)) void route_t(PC_ROUTE_ARGS) {
    Timer tm; const int G = P.G, neg = P.neg;
    auto load = [&](int e, size_t slot, uint64_t* b) {
        if (hsrc) std::memcpy(b, hsrc + ((size_t)e * hslots + slot) * 8, 64); else s.bus_rd(e, src_w + slot, b);
    };
    if (L == 3) {                                   // cross-domain modulation: no DT at all
        for (int e = 0; e < neg; e++) for (int pos = 0; pos < G; pos++) {
            const Key& key = P.keys[e][pos]; size_t u = P.inc_u[e][pos];
            for (size_t t = 0; t < tc; t++) { uint64_t b[8]; load(e, pos * ss + t, b); KP::rot(b, key.k); s.bus_wr(key.D, dst_w + u * ds + t, b); }
        }
    } else if (L == 2) {                            // in-register modulation: DT, word shift, DT
        for (int e = 0; e < neg; e++) for (int pos = 0; pos < G; pos++) {
            const Key& key = P.keys[e][pos]; size_t u = P.inc_u[e][pos];
            for (size_t t = 0; t < tc; t++) {
                uint64_t b[8]; load(e, pos * ss + t, b); dt64(b); perm_lanes(b, key.k); dt64(b); s.st.dt += 2;
                s.bus_wr(key.D, dst_w + u * ds + t, b);
            }
        }
    } else {                                        // PE-assisted reordering only: stage in host memory
        std::vector<uint64_t> H((size_t)neg * G * tc * 8), O((size_t)neg * G * tc * 8);
        for (int e = 0; e < neg; e++) for (int pos = 0; pos < G; pos++) for (size_t t = 0; t < tc; t++) {
            uint64_t* h = &H[(((size_t)e * G + pos) * tc + t) * 8]; load(e, pos * ss + t, h); dt64(h); s.st.dt++; s.st.host_mem_blk++;
        }
        for (int e = 0; e < neg; e++) for (int pos = 0; pos < G; pos++) {
            const Key& key = P.keys[e][pos]; size_t u = P.inc_u[e][pos];
            for (size_t t = 0; t < tc; t++) {
                uint64_t* o = &O[(((size_t)key.D * G + u) * tc + t) * 8];
                std::memcpy(o, &H[(((size_t)e * G + pos) * tc + t) * 8], 64); perm_lanes(o, key.k); s.st.host_mem_blk++;
            }
        }
        for (int D = 0; D < neg; D++) for (int u = 0; u < G; u++) for (size_t t = 0; t < tc; t++) {
            uint64_t b[8]; std::memcpy(b, &O[(((size_t)D * G + u) * tc + t) * 8], 64); dt64(b); s.st.dt++; s.bus_wr(D, dst_w + (size_t)u * ds + t, b);
        }
    }
    s.st.host_s += tm.s();
}
// The loop is instantiated once per ISA and compiled FOR that ISA, so staging copies use full-width moves and the kernel inlines.
__attribute__((target("avx2"))) inline void route_avx2_(PC_ROUTE_ARGS) { route_t<Avx2KP>(PC_ROUTE_CALL); }
__attribute__((target("avx512f,avx512bw"))) inline void route_avx512_(PC_ROUTE_ARGS) { route_t<Avx512KP>(PC_ROUTE_CALL); }
inline void route(PC_ROUTE_ARGS) {
    switch (K().mode) {
        case Mode::Avx2: route_avx2_(PC_ROUTE_CALL); break;
        case Mode::Avx512: route_avx512_(PC_ROUTE_CALL); break;
        case Mode::Ptr: route_t<PtrKP>(PC_ROUTE_CALL); break;
        default: route_t<ScalarKP>(PC_ROUTE_CALL);
    }
}

// ------------------------------ host-side routing (reduction) ------------------------------
// For every destination EG D and t<wc: reduce the blocks arriving from all members.  The reduced block
// (raw layout) goes to the bus at dst_w+t, or - if hout != nullptr - into a host buffer (used by the fused AllReduce).
#define PC_RED_ARGS PimSystem& s, const Plan& P, int L, Op op, int dtype, size_t src_w, size_t wc, uint64_t* hout, size_t dst_w
#define PC_RED_CALL s, P, L, op, dtype, src_w, wc, hout, dst_w
template <class KP> static inline __attribute__((always_inline)) void reduce_route_t(PC_RED_ARGS) {
    Timer tm; const int G = P.G, neg = P.neg;
    // Cross-domain modulation is possible for reductions only if the operator does not care about the byte layout:
    // bitwise ops always, 8-bit sums (one element per byte lane). 32/64-bit sums need the domain transfer.
    const bool needs_dt = !(L == 3 && (op != Op::SUM || dtype == 1));
    auto emit = [&](int D, size_t t, uint64_t* acc) {
        if (needs_dt) { dt64(acc); s.st.dt++; }
        if (hout) std::memcpy(hout + ((size_t)D * wc + t) * 8, acc, 64); else s.bus_wr(D, dst_w + t, acc);
    };
    if (L >= 2) {
        for (int D = 0; D < neg; D++) for (size_t t = 0; t < wc; t++) {
            uint64_t acc[8] = {0}, b[8];
            for (int u = 0; u < G; u++) {
                auto [e, pos] = P.incoming[D][u];
                s.bus_rd(e, src_w + (size_t)pos * wc + t, b); KP::rot(b, P.keys[e][pos].k);
                if (needs_dt) { dt64(b); s.st.dt++; }
                if (u == 0) std::memcpy(acc, b, 64); else KP::comb(acc, b, dtype, op);
            }
            emit(D, t, acc);
        }
    } else {                                        // L==1: every partial block is staged in host memory first
        std::vector<uint64_t> H((size_t)neg * G * wc * 8);
        for (int D = 0; D < neg; D++) for (int u = 0; u < G; u++) for (size_t t = 0; t < wc; t++) {
            auto [e, pos] = P.incoming[D][u]; uint64_t* h = &H[(((size_t)D * G + u) * wc + t) * 8];
            s.bus_rd(e, src_w + (size_t)pos * wc + t, h); KP::rot(h, P.keys[e][pos].k); dt64(h); s.st.dt++; s.st.host_mem_blk++;
        }
        for (int D = 0; D < neg; D++) for (size_t t = 0; t < wc; t++) {
            uint64_t acc[8] = {0}; std::memcpy(acc, &H[((size_t)D * G * wc + t) * 8], 64);
            for (int u = 1; u < G; u++) KP::comb(acc, &H[(((size_t)D * G + u) * wc + t) * 8], dtype, op);
            s.st.host_mem_blk += G; emit(D, t, acc);
        }
    }
    s.st.host_s += tm.s();
}
__attribute__((target("avx2"))) inline void reduce_route_avx2_(PC_RED_ARGS) { reduce_route_t<Avx2KP>(PC_RED_CALL); }
__attribute__((target("avx512f,avx512bw"))) inline void reduce_route_avx512_(PC_RED_ARGS) { reduce_route_t<Avx512KP>(PC_RED_CALL); }
inline void reduce_route(PC_RED_ARGS) {
    switch (K().mode) {
        case Mode::Avx2: reduce_route_avx2_(PC_RED_CALL); break;
        case Mode::Avx512: reduce_route_avx512_(PC_RED_CALL); break;
        case Mode::Ptr: reduce_route_t<PtrKP>(PC_RED_CALL); break;
        default: reduce_route_t<ScalarKP>(PC_RED_CALL);
    }
}

// ------------------------------ baseline (level 0) ------------------------------
inline void to_host(PimSystem& s, size_t src_w, size_t Sw, std::vector<uint64_t>& H) {
    H.assign((size_t)s.n * Sw, 0);
    for (int e = 0; e < s.neg; e++) for (size_t w = 0; w < Sw; w++) {
        uint64_t b[8]; s.bus_rd(e, src_w + w, b); dt64(b); s.st.dt++;
        for (int c = 0; c < 8; c++) H[((size_t)e * 8 + c) * Sw + w] = b[c];
    }
    s.st.host_mem_blk += (uint64_t)s.neg * Sw;
}
inline void from_host(PimSystem& s, size_t dst_w, size_t Sw, const std::vector<uint64_t>& O) {
    for (int e = 0; e < s.neg; e++) for (size_t w = 0; w < Sw; w++) {
        uint64_t b[8]; for (int c = 0; c < 8; c++) b[c] = O[((size_t)e * 8 + c) * Sw + w];
        dt64(b); s.st.dt++; s.bus_wr(e, dst_w + w, b);
    }
    s.st.host_mem_blk += (uint64_t)s.neg * Sw;
}

// ------------------------------ public primitives ------------------------------
// Sw = 64-bit words per PE of INPUT data (for AllGather the input; output is G*Sw words).
inline void alltoall(PimSystem& s, const Plan& P, int L, size_t Sw, const Layout& Ly) {
    if (Sw % P.G) throw std::runtime_error("AlltoAll: data per PE must be a multiple of group size * 8 bytes");
    size_t wc = Sw / P.G;
    if (L == 0) {
        Timer tm; std::vector<uint64_t> H, O((size_t)s.n * Sw); to_host(s, Ly.src, Sw, H);
        for (auto& grp : P.groups) for (int i = 0; i < P.G; i++) for (int j = 0; j < P.G; j++)
            for (size_t t = 0; t < wc; t++) O[(size_t)grp[j] * Sw + i * wc + t] = H[(size_t)grp[i] * Sw + j * wc + t];
        from_host(s, Ly.dst, Sw, O); s.st.host_s += tm.s();
    } else {
        pe_pre(s, P, Ly.src, Ly.buf, wc);
        route(s, P, L, Ly.buf, nullptr, 0, wc, wc, Ly.buf + Sw, wc);
        pe_post(s, P, Ly.buf + Sw, Ly.dst, wc);
    }
}
inline void allgather(PimSystem& s, const Plan& P, int L, size_t Sw, const Layout& Ly) {
    if (L == 0) {
        Timer tm; std::vector<uint64_t> H, O((size_t)s.n * P.G * Sw); to_host(s, Ly.src, Sw, H);
        for (auto& grp : P.groups) for (int i = 0; i < P.G; i++) for (int j = 0; j < P.G; j++)
            for (size_t t = 0; t < Sw; t++) O[(size_t)grp[j] * P.G * Sw + i * Sw + t] = H[(size_t)grp[i] * Sw + t];
        from_host(s, Ly.dst, P.G * Sw, O); s.st.host_s += tm.s();
    } else {
        route(s, P, L, Ly.src, nullptr, 0, Sw, 0, Ly.buf, Sw);
        pe_post(s, P, Ly.buf, Ly.dst, Sw);
    }
}
inline void reducescatter(PimSystem& s, const Plan& P, int L, size_t Sw, const Layout& Ly, Op op, int dtype) {
    if (Sw % P.G) throw std::runtime_error("ReduceScatter: data per PE must be a multiple of group size * 8 bytes");
    size_t wc = Sw / P.G;
    if (L == 0) {
        Timer tm; std::vector<uint64_t> H, O((size_t)s.n * wc); to_host(s, Ly.src, Sw, H);
        for (auto& grp : P.groups) for (int j = 0; j < P.G; j++) for (size_t t = 0; t < wc; t++) {
            uint64_t a = H[(size_t)grp[0] * Sw + j * wc + t];
            for (int i = 1; i < P.G; i++) a = comb_word(a, H[(size_t)grp[i] * Sw + j * wc + t], dtype, op);
            O[(size_t)grp[j] * wc + t] = a;
        }
        from_host(s, Ly.dst, wc, O); s.st.host_s += tm.s();
    } else {
        pe_pre(s, P, Ly.src, Ly.buf, wc);
        reduce_route(s, P, L, op, dtype, Ly.buf, wc, nullptr, Ly.dst);
    }
}
// naive=true : ReduceScatter followed by AllGather through PE memory (what generic libraries do).
// naive=false: PID-Comm's fused AllReduce - the reduced blocks never go back to the PEs before being broadcast.
inline void allreduce(PimSystem& s, const Plan& P, int L, size_t Sw, const Layout& Ly, Op op, int dtype, bool naive = false) {
    if (Sw % P.G) throw std::runtime_error("AllReduce: data per PE must be a multiple of group size * 8 bytes");
    size_t wc = Sw / P.G;
    if (L == 0) {
        Timer tm; std::vector<uint64_t> H, O((size_t)s.n * Sw); to_host(s, Ly.src, Sw, H);
        std::vector<uint64_t> red(Sw);
        for (auto& grp : P.groups) {
            for (size_t t = 0; t < Sw; t++) {
                uint64_t a = H[(size_t)grp[0] * Sw + t];
                for (int i = 1; i < P.G; i++) a = comb_word(a, H[(size_t)grp[i] * Sw + t], dtype, op);
                red[t] = a;
            }
            for (int j = 0; j < P.G; j++) std::memcpy(&O[(size_t)grp[j] * Sw], red.data(), Sw * 8);
        }
        from_host(s, Ly.dst, Sw, O); s.st.host_s += tm.s();
    } else if (naive) {
        reducescatter(s, P, L, Sw, Ly, op, dtype);
        Layout L2 = Ly; L2.src = Ly.dst;               // AllGather reads the chunk RS left in dst, writes the full result to dst
        allgather(s, P, L, wc, L2);
    } else {
        pe_pre(s, P, Ly.src, Ly.buf, wc);
        std::vector<uint64_t> R((size_t)s.neg * wc * 8);
        reduce_route(s, P, L, op, dtype, Ly.buf, wc, R.data(), 0);
        route(s, P, L, 0, R.data(), wc, wc, 0, Ly.buf + Sw, wc);
        pe_post(s, P, Ly.buf + Sw, Ly.dst, wc);
    }
}

} // namespace pc
