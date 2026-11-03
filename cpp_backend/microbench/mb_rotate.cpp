// Microbenchmark: cost of rotating one 64 B block, via function pointer vs direct inlined call, scalar vs AVX2.
#include "kernels.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
using namespace pc;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__attribute__((target("avx2"))) static void loop_direct_avx2(uint64_t* buf, size_t nblk) {      // whole loop compiled for AVX2: kernel inlines
    for (size_t i = 0; i < nblk; i++) rot_avx2(buf + 8 * (i & 1023), (int)(i & 7));
}
static void loop_direct_scalar(uint64_t* buf, size_t nblk) { for (size_t i = 0; i < nblk; i++) rot_scalar(buf + 8 * (i & 1023), (int)(i & 7)); }
static void loop_ptr(uint64_t* buf, size_t nblk, void (*f)(uint64_t*, int)) { for (size_t i = 0; i < nblk; i++) f(buf + 8 * (i & 1023), (int)(i & 7)); }

template <class F> static double best(F f, size_t n) { double b = 1e9; for (int r = 0; r < 15; r++) { double t = now(); f(); b = std::min(b, now() - t); } return b / n * 1e9; }
int main() {
    const size_t N = 4000000; std::vector<uint64_t> buf(8 * 1024, 0x0123456789abcdefull); uint64_t* p = buf.data();
    std::printf("ns per 64B block rotate (min of 15):\n");
    std::printf("  scalar, via function pointer : %.2f\n", best([&] { loop_ptr(p, N, rot_scalar); }, N));
    std::printf("  AVX2,   via function pointer : %.2f\n", best([&] { loop_ptr(p, N, rot_avx2); }, N));
    std::printf("  scalar, direct (inlinable)   : %.2f\n", best([&] { loop_direct_scalar(p, N); }, N));
    std::printf("  AVX2,   direct (inlined)     : %.2f\n", best([&] { loop_direct_avx2(p, N); }, N));
}
