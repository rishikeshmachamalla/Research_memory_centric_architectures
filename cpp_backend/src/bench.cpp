// bench.cpp -- measures the HOST-SIDE cost of each primitive at each implementation level (ablation).
// Output: CSV rows (one per primitive/size/level).  Throughput = (total data size) / (host-side time),
// following the paper's definition (larger side of the data, before reduction).
#include "collectives.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <algorithm>
using namespace pc;

static std::vector<std::string> split(const std::string& s, char d) { std::vector<std::string> v; std::stringstream ss(s); std::string x; while (std::getline(ss, x, d)) if (!x.empty()) v.push_back(x); return v; }

int main(int argc, char** argv) {
    std::string shape = "8x8", mask = "11", sizes = "1024,4096,16384,65536", prims = "aa,rs,ag,ar,arnaive", levels = "0,1,2,3";
    std::string simd = "auto", opn = "sum", csv, stat = "median"; int dtype = 4, reps = 7; bool header = true;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--shape") shape = nx(); else if (a == "--mask") mask = nx(); else if (a == "--sizes") sizes = nx();
        else if (a == "--prims") prims = nx(); else if (a == "--levels") levels = nx(); else if (a == "--simd") simd = nx();
        else if (a == "--dtype") dtype = std::atoi(nx().c_str()); else if (a == "--op") opn = nx();
        else if (a == "--reps") reps = std::atoi(nx().c_str()); else if (a == "--stat") stat = nx(); else if (a == "--csv") csv = nx(); else if (a == "--no-header") header = false;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (!set_simd(simd)) { std::fprintf(stderr, "SIMD '%s' not supported on this CPU\n", simd.c_str()); return 3; }
    Op op = opn == "or" ? Op::OR : opn == "xor" ? Op::XOR : Op::SUM;
    std::vector<int> dims; for (auto& t : split(shape, 'x')) dims.push_back(std::atoi(t.c_str()));
    Hypercube hc(dims); Plan P = make_plan(hc, mask);
    FILE* out = csv.empty() ? stdout : std::fopen(csv.c_str(), "a"); if (!out) { std::perror("csv"); return 1; }
    if (header) std::fprintf(out, "prim,shape,mask,n_pe,group,bytes_per_pe,dtype,op,simd,level,host_ms,pe_ms,gbps,bus_rd,bus_wr,dt,host_mem_blk,pe_words\n");
    std::mt19937_64 rng(7);
    for (auto& pn : split(prims, ',')) for (auto& sz : split(sizes, ',')) {
        size_t bytes = std::strtoull(sz.c_str(), nullptr, 10), Sw = bytes / 8;
        if (Sw == 0 || Sw % P.G) { std::fprintf(stderr, "skip %s size %zu: must be a multiple of group*8 = %d bytes\n", pn.c_str(), bytes, P.G * 8); continue; }
        int prim = pn == "aa" ? 0 : pn == "ag" ? 1 : pn == "rs" ? 2 : 3; bool naive = pn == "arnaive";
        size_t in_w = prim == 1 ? Sw / P.G : Sw;                 // AllGather: OUTPUT size is `bytes`
        Layout Ly = make_layout(prim == 1 ? Sw : Sw);
        PimSystem s(hc.n, Ly.Mw);
        for (int p = 0; p < hc.n; p++) for (size_t w = 0; w < in_w; w++) s.pe_set(p, Ly.src + w, rng());
        uint64_t ref_hash = 0; bool have_ref = false;
        for (auto& ls : split(levels, ',')) {
            int L = std::atoi(ls.c_str()); if (naive && L == 0) continue;
            std::vector<double> th, tp; Stats last;
            for (int r = 0; r < reps + 1; r++) {                  // first rep is a warm-up
                s.st.reset();
                switch (prim) { case 0: alltoall(s, P, L, in_w, Ly); break; case 1: allgather(s, P, L, in_w, Ly); break;
                                case 2: reducescatter(s, P, L, in_w, Ly, op, dtype); break; default: allreduce(s, P, L, in_w, Ly, op, dtype, naive); }
                if (r) { th.push_back(s.st.host_s); tp.push_back(s.st.pe_s); } last = s.st;
            }
            // sanity: all levels of one primitive must give the identical result
            uint64_t h = 1469598103934665603ull; size_t ow = prim == 2 ? Sw / P.G : Sw;
            for (int p = 0; p < hc.n; p += std::max(1, hc.n / 16)) for (size_t w = 0; w < ow; w += 7) h = (h ^ s.pe_get(p, Ly.dst + w)) * 1099511628211ull;
            if (!have_ref) { ref_hash = h; have_ref = true; } else if (h != ref_hash) { std::fprintf(stderr, "RESULT MISMATCH between levels (%s size %zu level %d)\n", pn.c_str(), bytes, L); return 4; }
            std::sort(th.begin(), th.end()); std::sort(tp.begin(), tp.end());
            double mh = stat == "min" ? th.front() : th[th.size() / 2], mp = stat == "min" ? tp.front() : tp[tp.size() / 2];
            double data = (double)hc.n * Sw * 8;                  // total bytes on the larger side
            std::fprintf(out, "%s,%s,%s,%d,%d,%zu,%d,%s,%s,%d,%.4f,%.4f,%.4f,%lu,%lu,%lu,%lu,%lu\n", pn.c_str(), shape.c_str(), mask.c_str(), hc.n, P.G, bytes, dtype, opn.c_str(),
                         K().name, L, mh * 1e3, mp * 1e3, data / mh / 1e9, (unsigned long)last.bus_rd, (unsigned long)last.bus_wr, (unsigned long)last.dt, (unsigned long)last.host_mem_blk, (unsigned long)last.pe_words);
            std::fflush(out);
        }
    }
    if (!csv.empty()) std::fclose(out);
    return 0;
}
