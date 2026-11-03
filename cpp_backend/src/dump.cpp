// dump.cpp -- run ONE collective on caller-supplied data and write the result (used by the Python cross-check).
// Files are raw little-endian uint64, PE-major: in = n_pe * words, out = n_pe * out_words.
//   ./pcdump --shape 8x8 --mask 11 --prim aa|ag|rs|ar|arnaive --level 0..3 --dtype 1|4|8 --op sum|or|xor
//            --simd auto|scalar|avx2|avx512 --words W --in in.bin --out out.bin
// Exit codes: 0 ok, 2 usage/IO, 3 SIMD unsupported, 5 invalid configuration (message on stderr).
#include "collectives.hpp"
#include <cstdio>
#include <cstdlib>
#include <sstream>
using namespace pc;

int main(int argc, char** argv) {
    std::string shape = "8", mask = "1", prim = "aa", simd = "auto", opn = "sum", fin, fout; int level = 3, dtype = 8; size_t words = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return 2; }
        std::string v = argv[++i];
        if (a == "--shape") shape = v; else if (a == "--mask") mask = v; else if (a == "--prim") prim = v; else if (a == "--level") level = std::atoi(v.c_str());
        else if (a == "--dtype") dtype = std::atoi(v.c_str()); else if (a == "--op") opn = v; else if (a == "--simd") simd = v;
        else if (a == "--words") words = std::strtoull(v.c_str(), 0, 10); else if (a == "--in") fin = v; else if (a == "--out") fout = v;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (!set_simd(simd)) { std::fprintf(stderr, "SIMD '%s' not supported on this CPU\n", simd.c_str()); return 3; }
    if (level < 0 || level > 3 || (dtype != 1 && dtype != 4 && dtype != 8) || (opn != "sum" && opn != "or" && opn != "xor") ||
        (prim != "aa" && prim != "ag" && prim != "rs" && prim != "ar" && prim != "arnaive") || !words || fin.empty() || fout.empty()) { std::fprintf(stderr, "bad arguments\n"); return 2; }
    Op op = opn == "or" ? Op::OR : opn == "xor" ? Op::XOR : Op::SUM;
    try {
        std::vector<int> dims; { std::stringstream ss(shape); std::string t; while (std::getline(ss, t, 'x')) dims.push_back(std::atoi(t.c_str())); }
        Hypercube hc(dims); Plan P = make_plan(hc, mask); const int G = P.G, n = hc.n;
        int pi = prim == "aa" ? 0 : prim == "ag" ? 1 : prim == "rs" ? 2 : 3;
        if (pi != 1 && words % G) throw std::runtime_error("words per PE must be a multiple of the group size");
        size_t outw = pi == 0 ? words : pi == 1 ? words * G : pi == 2 ? words / G : words;
        std::vector<uint64_t> in((size_t)n * words);
        FILE* f = std::fopen(fin.c_str(), "rb"); if (!f) { std::perror("in"); return 2; }
        if (std::fread(in.data(), 8, in.size(), f) != in.size()) { std::fprintf(stderr, "input file too short (need %zu words)\n", in.size()); return 2; } std::fclose(f);
        Layout Ly = make_layout(std::max(words, outw) + (pi == 1 ? words * G : 0)); PimSystem s(n, Ly.Mw);
        for (int p = 0; p < n; p++) for (size_t w = 0; w < words; w++) s.pe_set(p, Ly.src + w, in[(size_t)p * words + w]);
        switch (pi) { case 0: alltoall(s, P, level, words, Ly); break; case 1: allgather(s, P, level, words, Ly); break;
                      case 2: reducescatter(s, P, level, words, Ly, op, dtype); break; default: allreduce(s, P, level, words, Ly, op, dtype, prim == "arnaive"); }
        std::vector<uint64_t> out((size_t)n * outw);
        for (int p = 0; p < n; p++) for (size_t w = 0; w < outw; w++) out[(size_t)p * outw + w] = s.pe_get(p, Ly.dst + w);
        f = std::fopen(fout.c_str(), "wb"); if (!f) { std::perror("out"); return 2; }
        std::fwrite(out.data(), 8, out.size(), f); std::fclose(f);
    } catch (std::exception& e) { std::fprintf(stderr, "invalid configuration: %s\n", e.what()); return 5; }
    return 0;
}
