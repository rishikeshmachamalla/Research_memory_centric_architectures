// Same data path as the real route(): bus memcpy -> 64B stack block -> kernel -> bus memcpy.  Variants isolate the cause.
#include "kernels.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
using namespace pc;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> static double best(F f, size_t n) { double b = 1e9; for (int r = 0; r < 15; r++) { double t = now(); f(); b = std::min(b, now() - t); } return b / n * 1e9; }

static void staged(const uint64_t* src, uint64_t* dst, size_t nblk, size_t mask, void (*rot)(uint64_t*, int)) {
    for (size_t i = 0; i < nblk; i++) { uint64_t b[8]; std::memcpy(b, src + 8 * (i & mask), 64); rot(b, (int)(i & 7)); std::memcpy(dst + 8 * (i & mask), b, 64); }
}
__attribute__((target("avx2"))) static void fused_avx2(const uint64_t* src, uint64_t* dst, size_t nblk, size_t mask) {   // no staging copy: load, rotate, store
    for (size_t i = 0; i < nblk; i++) {
        __m128i s = _mm_cvtsi32_si128(8 * (int)(i & 7)), s2 = _mm_cvtsi32_si128(64 - 8 * (int)(i & 7));
        for (int h = 0; h < 2; h++) { __m256i v = _mm256_loadu_si256((const __m256i*)(src + 8 * (i & mask) + 4 * h));
            _mm256_storeu_si256((__m256i*)(dst + 8 * (i & mask) + 4 * h), _mm256_or_si256(_mm256_sll_epi64(v, s), _mm256_srl_epi64(v, s2))); }
    }
}
__attribute__((target("avx2"))) static void staged_avx2_wide(const uint64_t* src, uint64_t* dst, size_t nblk, size_t mask) {   // staging copy done with 32B moves, kernel inlined
    for (size_t i = 0; i < nblk; i++) { uint64_t b[8]; std::memcpy(b, src + 8 * (i & mask), 64); rot_avx2(b, (int)(i & 7)); std::memcpy(dst + 8 * (i & mask), b, 64); }
}
int main() {
    const size_t N = 2000000, mask = 4095;                      // 4096 blocks = 256 KB per array (cache resident, like a typical route)
    std::vector<uint64_t> a(8 * 4096, 1), b(8 * 4096, 0);
    std::printf("ns per block through the real data path (min of 15):\n");
    std::printf("  scalar kernel, staged (memcpy in/out)            : %.2f\n", best([&] { staged(a.data(), b.data(), N, mask, rot_scalar); }, N));
    std::printf("  AVX2 kernel,   staged (memcpy in/out)  [current] : %.2f\n", best([&] { staged(a.data(), b.data(), N, mask, rot_avx2); }, N));
    std::printf("  AVX2, staged but whole loop compiled for AVX2     : %.2f\n", best([&] { staged_avx2_wide(a.data(), b.data(), N, mask); }, N));
    std::printf("  AVX2, fused (no staging copy)                     : %.2f\n", best([&] { fused_avx2(a.data(), b.data(), N, mask); }, N));
}
