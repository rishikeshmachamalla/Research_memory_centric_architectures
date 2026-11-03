"""Virtual hypercube topology for simulated processing elements."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypercube:
    """A hypercube containing a power-of-two number of PEs.

    PE IDs range from 0 to num_pes - 1.
    Dimension d connects rank r to rank r XOR (1 << d).
    """

    num_pes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_pes, int)
            or isinstance(self.num_pes, bool)
            or self.num_pes < 1
            or (self.num_pes & (self.num_pes - 1)) != 0
        ):
            raise ValueError("num_pes must be a positive power of two")

    @property
    def dimensions(self) -> int:
        """Return the number of hypercube dimensions."""
        return self.num_pes.bit_length() - 1

    def _check_rank(self, rank: int) -> None:
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < self.num_pes
        ):
            raise ValueError(f"rank must be between 0 and {self.num_pes - 1}")

    def neighbors(self, rank: int) -> tuple[int, ...]:
        """Return the neighboring PE IDs of a rank."""
        self._check_rank(rank)
        return tuple(
            rank ^ (1 << dimension)
            for dimension in range(self.dimensions)
        )

    def neighbor(self, rank: int, dimension: int) -> int:
        """Return the neighbor reached by flipping one dimension."""
        self._check_rank(rank)

        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or not 0 <= dimension < self.dimensions
        ):
            raise ValueError(
                f"dimension must be between 0 and {self.dimensions - 1}"
            )

        return rank ^ (1 << dimension)

    def hamming_distance(self, source: int, destination: int) -> int:
        """Count the dimensions in which two PE IDs differ."""
        self._check_rank(source)
        self._check_rank(destination)
        return (source ^ destination).bit_count()
