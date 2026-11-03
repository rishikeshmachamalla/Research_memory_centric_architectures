// bfs.cpp -- Breadth-first search on the simulated PIM system, communication = AllReduce(bitwise OR)
// (the structure the paper uses for BFS).  Edges are partitioned across PEs; every PE holds the whole
// frontier/visited bitmap.  Per iteration: PE kernel expands the frontier over its edges -> AllReduce(OR)
// -> PE kernel computes the new frontier.  The result is checked against a plain CPU BFS.
#include "collectives.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <queue>
using namespace pc;

int main(int argc, char** argv) {
    int n = 64, deg = 8, runs = 5; size_t V = 16384; std::string simd = "auto", csv, graph, stat = "median"; bool header = true; uint32_t src = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--n") n = std::atoi(nx().c_str()); else if (a == "--V") V = std::strtoull(nx().c_str(), 0, 10);
        else if (a == "--deg") deg = std::atoi(nx().c_str()); else if (a == "--simd") simd = nx();
        else if (a == "--runs") runs = std::atoi(nx().c_str()); else if (a == "--graph") graph = nx(); else if (a == "--stat") stat = nx(); else if (a == "--src") src = (uint32_t)std::atoi(nx().c_str()); else if (a == "--csv") csv = nx(); else if (a == "--no-header") header = false;
    }
    if (!set_simd(simd)) { std::fprintf(stderr, "SIMD not supported\n"); return 3; }
    Hypercube hc({n}); Plan P = make_plan(hc, "1"); const int G = P.G;
    if (V % (64 * G)) { std::fprintf(stderr, "V must be a multiple of 64*group = %d\n", 64 * G); return 2; }
    const size_t Wv = V / 64;                                       // bitmap words per PE (also AllReduce data size)
    // ---- graph: from file ("V E" then E lines "u v") or random (undirected); edges dealt round-robin to PEs ----
    std::mt19937_64 rng(2024); std::vector<std::vector<std::pair<uint32_t,uint32_t>>> edges(n); std::vector<std::vector<uint32_t>> adj(V);
    std::vector<std::pair<uint32_t,uint32_t>> el;
    if (!graph.empty()) {
        FILE* gf = std::fopen(graph.c_str(), "r"); size_t Vf = 0, Ef = 0;
        if (!gf || std::fscanf(gf, "%zu %zu", &Vf, &Ef) != 2 || Vf != V) { std::fprintf(stderr, "bad graph file or V mismatch\n"); return 2; }
        for (size_t e = 0; e < Ef; e++) { unsigned u, v; if (std::fscanf(gf, "%u %u", &u, &v) != 2 || u >= V || v >= V) { std::fprintf(stderr, "bad edge\n"); return 2; } el.push_back({u, v}); }
        std::fclose(gf);
    } else {
        for (size_t e = 0; e < V * deg / 2; e++) { uint32_t u = rng() % V, v = rng() % V; if (u != v) el.push_back({u, v}); }
    }
    size_t E = el.size();
    for (size_t e = 0; e < E; e++) { uint32_t u = el[e].first, v = el[e].second;
        edges[e % n].push_back({u, v}); edges[(e + 1) % n].push_back({v, u}); adj[u].push_back(v); adj[v].push_back(u); }
    std::vector<int> ref(V, -1); { std::queue<uint32_t> q; q.push(src); ref[src] = 0; while (!q.empty()) { auto u = q.front(); q.pop(); for (auto v : adj[u]) if (ref[v] < 0) { ref[v] = ref[u] + 1; q.push(v); } } }
    FILE* out = csv.empty() ? stdout : std::fopen(csv.c_str(), "a");
    if (header) std::fprintf(out, "app,n_pe,V,edges,level,iters,comm_host_ms,comm_pe_ms,bus_rd,bus_wr,dt,correct,reached,dist_sum\n");
    Layout Ly = make_layout(Wv);
    for (int L = 0; L <= 3; L++) {
        std::vector<double> times, pes; int iters = 0; bool correct = true; Stats tot;
        for (int run = 0; run < runs; run++) {
            PimSystem s(n, Ly.Mw);
            std::vector<std::vector<uint64_t>> vis(n, std::vector<uint64_t>(Wv, 0)), fr(n, std::vector<uint64_t>(Wv, 0));
            for (int p = 0; p < n; p++) { vis[p][src / 64] |= 1ull << (src % 64); fr[p][src / 64] |= 1ull << (src % 64); }
            std::vector<int> dist(V, -1); dist[src] = 0; double ch = 0, cp = 0; iters = 0; Stats acc;
            for (int level = 1;; level++) {
                for (int p = 0; p < n; p++) {                        // PE kernel 1: expand frontier over local edges
                    std::vector<uint64_t> nx(Wv, 0);
                    for (auto& e : edges[p]) if (fr[p][e.first / 64] >> (e.first % 64) & 1) nx[e.second / 64] |= 1ull << (e.second % 64);
                    for (size_t w = 0; w < Wv; w++) s.pe_set(p, Ly.src + w, nx[w]);
                }
                s.st.reset(); allreduce(s, P, L, Wv, Ly, Op::OR, 8, false);        // <-- the communication being studied
                ch += s.st.host_s; cp += s.st.pe_s; acc.bus_rd += s.st.bus_rd; acc.bus_wr += s.st.bus_wr; acc.dt += s.st.dt; iters++;
                bool any = false;
                for (int p = 0; p < n; p++) for (size_t w = 0; w < Wv; w++) {   // PE kernel 2: new frontier
                    uint64_t g = s.pe_get(p, Ly.dst + w), nf = g & ~vis[p][w]; vis[p][w] |= nf; fr[p][w] = nf;
                    if (p == 0 && nf) { any = true; for (int b = 0; b < 64; b++) if (nf >> b & 1) dist[w * 64 + b] = level; }
                }
                if (!any) break;
            }
            if (dist != ref) correct = false;
            times.push_back(ch); pes.push_back(cp); tot = acc;
        }
        std::sort(times.begin(), times.end()); std::sort(pes.begin(), pes.end());
        long reached = 0, dsum = 0; for (int d : ref) if (d >= 0) { reached++; dsum += d; }
        std::fprintf(out, "bfs,%d,%zu,%zu,%d,%d,%.4f,%.4f,%lu,%lu,%lu,%d,%ld,%ld\n", n, V, E, L, iters, (stat == "min" ? times.front() : times[times.size() / 2]) * 1e3, (stat == "min" ? pes.front() : pes[pes.size() / 2]) * 1e3,
                     (unsigned long)tot.bus_rd, (unsigned long)tot.bus_wr, (unsigned long)tot.dt, (int)correct, reached, dsum);
        std::fflush(out); if (!correct) { std::fprintf(stderr, "BFS RESULT WRONG at level %d\n", L); return 4; }
    }
    return 0;
}
