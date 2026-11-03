// pimsim.hpp -- software model of a UPMEM-style PIM-DIMM system + virtual hypercube.
//
// Memory layout model (mirrors Fig. 1 of the PID-Comm paper):
//   * PEs are grouped in "entangled groups" (EG) of 8 (one PE per chip).
//   * A DDR4 burst to an EG is 64 B; raw byte [8*i + c] belongs to PE c (byte i of its 64-bit word).
//   * We therefore store every EG's memory directly in this raw (bus) layout.
//     - host bus access  = 64 B memcpy                (bus_rd / bus_wr)
//     - PE-local access  = byte gather/scatter        (pe_get / pe_set / pe_copy)
//   * "Domain transfer" is the 8x8 byte transpose that turns a raw block into 8 usable words.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>

namespace pc {

struct Stats {
    uint64_t bus_rd = 0, bus_wr = 0;   // 64 B bursts over the (simulated) memory bus
    uint64_t dt = 0;                   // 8x8 domain transfers performed by the host
    uint64_t host_mem_blk = 0;         // 64 B blocks staged through host memory
    uint64_t pe_words = 0;             // 64-bit words moved by PE-side reorder kernels
    double host_s = 0, pe_s = 0;       // wall time of host-side work / (simulated) PE-side work
    void reset() { *this = Stats(); }
};

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double s() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

struct PimSystem {
    int n, neg; size_t Mw;                      // #PEs, #entangled groups, MRAM words per PE
    std::vector<std::vector<uint8_t>> raw;      // [eg] -> Mw*64 bytes in bus layout
    Stats st;

    PimSystem(int n_, size_t mram_words) : n(n_), neg(n_ / 8), Mw(mram_words) {
        if (n % 8) throw std::runtime_error("number of PEs must be a multiple of 8");
        raw.assign(neg, std::vector<uint8_t>(Mw * 64, 0));
    }
    // ---- PE-side access (what a PE kernel sees) ----
    uint64_t pe_get(int pe, size_t w) const {
        const uint8_t* p = &raw[pe >> 3][w * 64]; int c = pe & 7; uint64_t v = 0;
        for (int i = 0; i < 8; i++) v |= (uint64_t)p[8 * i + c] << (8 * i);
        return v;
    }
    void pe_set(int pe, size_t w, uint64_t v) {
        uint8_t* p = &raw[pe >> 3][w * 64]; int c = pe & 7;
        for (int i = 0; i < 8; i++) p[8 * i + c] = (uint8_t)(v >> (8 * i));
    }
    void pe_copy(int pe, size_t ws, size_t wd) {
        uint8_t* b = raw[pe >> 3].data(); int c = pe & 7;
        for (int i = 0; i < 8; i++) b[wd * 64 + 8 * i + c] = b[ws * 64 + 8 * i + c];
        st.pe_words++;
    }
    // ---- host-side access over the memory bus (64 B bursts, raw layout) ----
    inline void bus_rd(int eg, size_t w, uint64_t* out) { std::memcpy(out, &raw[eg][w * 64], 64); st.bus_rd++; }
    inline void bus_wr(int eg, size_t w, const uint64_t* in) { std::memcpy(&raw[eg][w * 64], in, 64); st.bus_wr++; }
};

// ---------------- virtual hypercube ----------------
struct Hypercube {
    std::vector<int> dims; int n = 1;
    explicit Hypercube(std::vector<int> d) : dims(std::move(d)) {
        for (size_t i = 0; i < dims.size(); i++) {
            n *= dims[i];
            if (i + 1 < dims.size() && (dims[i] & (dims[i] - 1)))
                throw std::runtime_error("all but the last hypercube dimension must be a power of two");
        }
        if (n % 8) throw std::runtime_error("hypercube size must be a multiple of 8 PEs");
    }
    // mask like "101": '1' = axis takes part in the communication group (x is the first character).
    // groups[g][i] = PE id of member i of group g; members are ordered with the first selected axis fastest.
    std::vector<std::vector<int>> groups(const std::string& mask) const {
        if (mask.size() != dims.size()) throw std::runtime_error("mask length must equal hypercube dimensionality");
        int G = 1, NG = 1;
        for (size_t a = 0; a < dims.size(); a++) (mask[a] == '1' ? G : NG) *= dims[a];
        std::vector<std::vector<int>> g(NG, std::vector<int>(G));
        for (int id = 0; id < n; id++) {
            int rem = id, gi = 0, gs = 1, mi = 0, ms = 1;
            for (size_t a = 0; a < dims.size(); a++) {
                int c = rem % dims[a]; rem /= dims[a];
                if (mask[a] == '1') { mi += c * ms; ms *= dims[a]; } else { gi += c * gs; gs *= dims[a]; }
            }
            g[gi][mi] = id;
        }
        return g;
    }
};

} // namespace pc
