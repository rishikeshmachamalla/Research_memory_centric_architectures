// kernels.hpp -- host-side data-modulation kernels (scalar / AVX2 / AVX-512).
// Data model: a "block" is 64 B = 8 x uint64_t = one DDR4 burst to one entangled group.
//   raw (PIM-domain) block : raw byte [8*i + c] = byte i of PE c's 64-bit word
//   host-domain block      : lane c (uint64_t) = PE c's 64-bit word
//   dt64()                 : raw <-> host (8x8 byte transpose, self-inverse)
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <immintrin.h>

namespace pc {

enum class Op { SUM, OR, XOR };

// ---- domain transfer: 8x8 byte transpose (portable SWAR, no SIMD needed) ----
static inline void dt64(uint64_t* b) {
    uint64_t t;
    for (int i : {0, 1, 2, 3}) { t = ((b[i] >> 32) ^ b[i + 4]) & 0x00000000FFFFFFFFull; b[i + 4] ^= t; b[i] ^= t << 32; }
    for (int i : {0, 1, 4, 5}) { t = ((b[i] >> 16) ^ b[i + 2]) & 0x0000FFFF0000FFFFull; b[i + 2] ^= t; b[i] ^= t << 16; }
    for (int i : {0, 2, 4, 6}) { t = ((b[i] >> 8)  ^ b[i + 1]) & 0x00FF00FF00FF00FFull; b[i + 1] ^= t; b[i] ^= t << 8;  }
}

// host-domain lane rotation: new lane (c+k)%8 = old lane c
static inline void perm_lanes(uint64_t* b, int k) {
    uint64_t t[8]; std::memcpy(t, b, 64);
    for (int c = 0; c < 8; c++) b[(c + k) & 7] = t[c];
}

// scalar reference for one 64-bit word (used by the baseline path)
static inline uint64_t comb_word(uint64_t a, uint64_t b, int dtype, Op op) {
    if (op == Op::OR)  return a | b;
    if (op == Op::XOR) return a ^ b;
    if (dtype == 8) return a + b;
    if (dtype == 4) { uint32_t x[2], y[2]; std::memcpy(x, &a, 8); std::memcpy(y, &b, 8); x[0] += y[0]; x[1] += y[1]; uint64_t r; std::memcpy(&r, x, 8); return r; }
    uint8_t x[8], y[8]; std::memcpy(x, &a, 8); std::memcpy(y, &b, 8);
    for (int i = 0; i < 8; i++) x[i] = (uint8_t)(x[i] + y[i]);
    uint64_t r; std::memcpy(&r, x, 8); return r;
}

// ---------------- scalar kernels ----------------
// raw-domain rotation: rotate every 64-bit raw lane left by 8k bits == shift PE lanes by k
inline void rot_scalar(uint64_t* b, int k) {
    int r = 8 * k;
    if (r == 0) return;
    for (int i = 0; i < 8; i++) b[i] = (b[i] << r) | (b[i] >> (64 - r));
}
inline void comb_scalar(uint64_t* acc, const uint64_t* x, int dtype, Op op) {
    if (op == Op::OR)  { for (int i = 0; i < 8; i++) acc[i] |= x[i]; return; }
    if (op == Op::XOR) { for (int i = 0; i < 8; i++) acc[i] ^= x[i]; return; }
    if (dtype == 8) { for (int i = 0; i < 8; i++) acc[i] += x[i]; return; }
    if (dtype == 4) { uint32_t* a = (uint32_t*)acc; const uint32_t* y = (const uint32_t*)x; for (int i = 0; i < 16; i++) a[i] += y[i]; return; }
    uint8_t* a = (uint8_t*)acc; const uint8_t* y = (const uint8_t*)x; for (int i = 0; i < 64; i++) a[i] = (uint8_t)(a[i] + y[i]);
}

// ---------------- AVX2 kernels (what a CPU without AVX-512 can do) ----------------
__attribute__((target("avx2")))
inline void rot_avx2(uint64_t* b, int k) {
    __m128i s = _mm_cvtsi32_si128(8 * k), s2 = _mm_cvtsi32_si128(64 - 8 * k);
    for (int h = 0; h < 2; h++) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(b + 4 * h));
        v = _mm256_or_si256(_mm256_sll_epi64(v, s), _mm256_srl_epi64(v, s2));   // count 64 -> 0, so k=0 is a no-op
        _mm256_storeu_si256((__m256i*)(b + 4 * h), v);
    }
}
__attribute__((target("avx2")))
inline void comb_avx2(uint64_t* acc, const uint64_t* x, int dtype, Op op) {
    for (int h = 0; h < 2; h++) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(acc + 4 * h));
        __m256i y = _mm256_loadu_si256((const __m256i*)(x + 4 * h));
        if (op == Op::OR) a = _mm256_or_si256(a, y);
        else if (op == Op::XOR) a = _mm256_xor_si256(a, y);
        else if (dtype == 8) a = _mm256_add_epi64(a, y);
        else if (dtype == 4) a = _mm256_add_epi32(a, y);
        else a = _mm256_add_epi8(a, y);
        _mm256_storeu_si256((__m256i*)(acc + 4 * h), a);
    }
}

// ---------------- AVX-512 kernels (what the paper uses; only runs on AVX-512 CPUs) ----------------
__attribute__((target("avx512f,avx512bw")))
inline void rot_avx512(uint64_t* b, int k) {
    __m512i v = _mm512_loadu_si512((const void*)b);
    v = _mm512_rolv_epi64(v, _mm512_set1_epi64(8 * k));
    _mm512_storeu_si512((void*)b, v);
}
__attribute__((target("avx512f,avx512bw")))
inline void comb_avx512(uint64_t* acc, const uint64_t* x, int dtype, Op op) {
    __m512i a = _mm512_loadu_si512((const void*)acc), y = _mm512_loadu_si512((const void*)x);
    if (op == Op::OR) a = _mm512_or_si512(a, y);
    else if (op == Op::XOR) a = _mm512_xor_si512(a, y);
    else if (dtype == 8) a = _mm512_add_epi64(a, y);
    else if (dtype == 4) a = _mm512_add_epi32(a, y);
    else a = _mm512_add_epi8(a, y);
    _mm512_storeu_si512((void*)acc, a);
}

// ---------------- runtime dispatch ----------------
// Mode::Scalar/Avx2/Avx512 : the whole host loop is compiled for that ISA and the kernels are inlined (default).
// Mode::Ptr                : kernels are reached through function pointers from baseline-ISA code (the original design).
enum class Mode { Scalar, Avx2, Avx512, Ptr };
struct Kernels {
    void (*rot)(uint64_t*, int);
    void (*comb)(uint64_t*, const uint64_t*, int, Op);
    const char* name;
    Mode mode;
};
inline Kernels& K() { static Kernels k{rot_scalar, comb_scalar, "scalar", Mode::Scalar}; return k; }

// kernel policies used by the templated host loops in collectives.hpp
struct ScalarKP { static inline void rot(uint64_t* b, int k) { rot_scalar(b, k); }
                  static inline void comb(uint64_t* a, const uint64_t* x, int d, Op o) { comb_scalar(a, x, d, o); } };
struct Avx2KP   { static inline void rot(uint64_t* b, int k) { rot_avx2(b, k); }
                  static inline void comb(uint64_t* a, const uint64_t* x, int d, Op o) { comb_avx2(a, x, d, o); } };
struct Avx512KP { static inline void rot(uint64_t* b, int k) { rot_avx512(b, k); }
                  static inline void comb(uint64_t* a, const uint64_t* x, int d, Op o) { comb_avx512(a, x, d, o); } };
struct PtrKP    { static inline void rot(uint64_t* b, int k) { K().rot(b, k); }
                  static inline void comb(uint64_t* a, const uint64_t* x, int d, Op o) { K().comb(a, x, d, o); } };
inline bool have_avx2()   { __builtin_cpu_init(); return __builtin_cpu_supports("avx2"); }
inline bool have_avx512() { __builtin_cpu_init(); return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"); }
// name: scalar | avx2 | avx512 | auto, optionally with suffix "-ptr" (function-pointer dispatch, the original design).
// Returns false if the CPU cannot run the request.
inline bool set_simd(const std::string& s) {
    Kernels& k = K();
    if (s == "auto") { if (have_avx512()) return set_simd("avx512"); if (have_avx2()) return set_simd("avx2"); return set_simd("scalar"); }
    if (s == "scalar")     { k = {rot_scalar, comb_scalar, "scalar", Mode::Scalar}; return true; }
    if (s == "scalar-ptr") { k = {rot_scalar, comb_scalar, "scalar-ptr", Mode::Ptr}; return true; }
    if (s == "avx2")       { if (!have_avx2())   return false; k = {rot_avx2, comb_avx2, "avx2", Mode::Avx2}; return true; }
    if (s == "avx2-ptr")   { if (!have_avx2())   return false; k = {rot_avx2, comb_avx2, "avx2-ptr", Mode::Ptr}; return true; }
    if (s == "avx512")     { if (!have_avx512()) return false; k = {rot_avx512, comb_avx512, "avx512", Mode::Avx512}; return true; }
    if (s == "avx512-ptr") { if (!have_avx512()) return false; k = {rot_avx512, comb_avx512, "avx512-ptr", Mode::Ptr}; return true; }
    return false;
}

} // namespace pc
