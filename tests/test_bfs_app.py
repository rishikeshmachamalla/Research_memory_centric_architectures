import numpy as np
import pytest

from pidcomm_sim import bfs_app as app
from pidcomm_sim.experiments import bfs_path, run_bfs


@pytest.mark.parametrize("n_pe", [8, 16, 32, 64])
@pytest.mark.parametrize("seed", [1, 2])
def test_pim_bfs_matches_plain_bfs_random_graphs(n_pe, seed):
    V = 64 * n_pe * 2
    edges = app.random_graph(V, deg=4, seed=seed)
    dist, iters = app.bfs_on_pim(V, edges, n_pe)
    assert (dist == app.reference_bfs(V, edges)).all()
    assert iters == dist.max() + 1                           # one round per BFS level, plus the final round that finds nothing


def test_deep_graph_needs_many_iterations():
    V = 512
    dist, iters = app.bfs_on_pim(V, app.path_graph(V), 8)
    assert (dist == np.arange(V)).all() and iters == V          # V-1 expansions + 1 empty final iteration


def test_disconnected_graph_leaves_unreachable_vertices():
    V = 512
    edges = np.array([[0, 1], [1, 2], [100, 101]])
    dist, _ = app.bfs_on_pim(V, edges, 8)
    assert dist[2] == 2 and dist[100] == -1 and dist[101] == -1
    assert app.summary(dist) == (3, 3)


def test_invalid_vertex_count_rejected():
    with pytest.raises(ValueError):
        app.bfs_on_pim(1000, app.random_graph(1000), 8)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.random(640) < 0.3
    assert (app._unpack(app._pack(m), 640) == m).all()


needs_bfs = pytest.mark.skipif(not bfs_path().exists(), reason="C++ bfs not built (make in cpp_backend)")


@needs_bfs
@pytest.mark.parametrize("n_pe,V", [(8, 1024), (16, 2048), (64, 4096)])
def test_cpp_bfs_agrees_with_python_on_identical_graph(tmp_path, n_pe, V):
    edges = app.random_graph(V, deg=6, seed=7)
    g = tmp_path / "g.txt"; app.write_graph(g, V, edges)
    reached, dsum = app.summary(app.reference_bfs(V, edges))
    rows = run_bfs(n_pe, V, runs=1, graph=g)
    assert [r["level"] for r in rows] == [0, 1, 2, 3]
    for r in rows:
        assert r["correct"] == 1 and (r["reached"], r["dist_sum"]) == (reached, dsum)
    _, iters = app.bfs_on_pim(V, edges, n_pe)
    assert all(r["iters"] == iters for r in rows)             # same number of communication rounds in both implementations


@needs_bfs
def test_cpp_bfs_on_path_graph(tmp_path):
    V = 512; g = tmp_path / "p.txt"; app.write_graph(g, V, app.path_graph(V))
    for r in run_bfs(8, V, runs=1, graph=g):
        assert r["correct"] == 1 and r["iters"] == V and r["reached"] == V
