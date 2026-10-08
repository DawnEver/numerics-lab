"""Known immutable finite canonical CSR accumulator; no executable serialization."""

import itertools
import math
from dataclasses import dataclass

import scipy.sparse as sp

__all__ = ['CycleTangentCSR']


@dataclass(frozen=True)
class CycleTangentCSR:
    shape: tuple[int, int]
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    data: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not all(isinstance(value, tuple) for value in (self.shape, self.indptr, self.indices, self.data))
            or len(self.shape) != 2
            or any(type(value) is not int or value < 1 for value in self.shape)
            or self.shape[0] != self.shape[1]
        ):
            msg = 'EEC tangent CSR requires an explicit positive square shape'
            raise ValueError(msg)
        n = self.shape[0]
        if (
            len(self.indptr) != n + 1
            or len(self.indices) != len(self.data)
            or len(self.data) > n * n
            or any(type(value) is not int for value in self.indptr + self.indices)
            or self.indptr[0] != 0
            or self.indptr[-1] != len(self.data)
            or any(a > b or a < 0 for a, b in zip(self.indptr[:-1], self.indptr[1:], strict=True))
            or any(index < 0 or index >= n for index in self.indices)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value == 0
                for value in self.data
            )
        ):
            msg = 'EEC tangent CSR pointers, indices and finite data must be canonical'
            raise ValueError(msg)
        for left, right in zip(self.indptr[:-1], self.indptr[1:], strict=True):
            row = self.indices[left:right]
            if any(a >= b for a, b in itertools.pairwise(row)):
                msg = 'EEC tangent CSR row indices must be sorted and unique'
                raise ValueError(msg)

    def matrix(self):
        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)

    @classmethod
    def of(cls, matrix):
        matrix = sp.csr_matrix(matrix).copy()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return cls(
            tuple(matrix.shape),
            tuple(int(x) for x in matrix.indptr),
            tuple(int(x) for x in matrix.indices),
            tuple(float(x) for x in matrix.data),
        )

    def to_mapping(self):
        return {
            'shape': list(self.shape),
            'indptr': list(self.indptr),
            'indices': list(self.indices),
            'data': list(self.data),
        }

    @classmethod
    def from_mapping(cls, data):
        if (
            not isinstance(data, dict)
            or set(data) != {'shape', 'indptr', 'indices', 'data'}
            or any(not isinstance(value, list) for value in data.values())
        ):
            msg = 'EEC tangent CSR requires exactly its declared array fields'
            raise ValueError(msg)
        return cls(tuple(data['shape']), tuple(data['indptr']), tuple(data['indices']), tuple(data['data']))
