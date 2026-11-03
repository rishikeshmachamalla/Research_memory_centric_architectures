import pytest

from pidcomm_sim.hypercube import Hypercube


@pytest.mark.parametrize(
    ("num_pes", "expected_dimensions"),
    [(8, 3), (16, 4), (32, 5), (64, 6)],
)
def test_hypercube_dimensions(num_pes, expected_dimensions):
    cube = Hypercube(num_pes)
    assert cube.dimensions == expected_dimensions


@pytest.mark.parametrize("num_pes", [8, 16, 32, 64])
def test_every_pe_has_correct_neighbors(num_pes):
    cube = Hypercube(num_pes)

    for rank in range(num_pes):
        neighbors = cube.neighbors(rank)

        assert len(neighbors) == cube.dimensions
        assert len(set(neighbors)) == cube.dimensions

        for other in neighbors:
            assert 0 <= other < num_pes
            assert rank in cube.neighbors(other)
            assert cube.hamming_distance(rank, other) == 1


def test_known_neighbors():
    cube = Hypercube(8)

    assert cube.neighbors(0) == (1, 2, 4)
    assert cube.neighbor(0, 1) == 2


def test_hamming_distance():
    cube = Hypercube(8)

    assert cube.hamming_distance(0, 7) == 3
    assert cube.hamming_distance(3, 7) == 1
    assert cube.hamming_distance(5, 5) == 0


@pytest.mark.parametrize("invalid_size", [0, -2, 3, 6, 12])
def test_invalid_hypercube_sizes(invalid_size):
    with pytest.raises(ValueError):
        Hypercube(invalid_size)


def test_invalid_rank():
    cube = Hypercube(8)

    with pytest.raises(ValueError):
        cube.neighbors(8)


def test_invalid_dimension():
    cube = Hypercube(8)

    with pytest.raises(ValueError):
        cube.neighbor(0, 3)
