#!/usr/bin/env python3
"""Small helper for comparing reference and migrated numerical outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(f"{path} contains {len(keys)} arrays; expected one")
        return data[keys[0]]
    return np.loadtxt(path, delimiter=",")


def compare(expected: np.ndarray, actual: np.ndarray, rtol: float, atol: float) -> None:
    if expected.shape != actual.shape:
        raise AssertionError(f"shape mismatch: expected {expected.shape}, got {actual.shape}")
    if expected.dtype != actual.dtype:
        raise AssertionError(f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def self_test() -> None:
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    actual = expected + np.array([0.0, 1e-7, -1e-7], dtype=np.float32)
    compare(expected, actual, rtol=1e-5, atol=1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", nargs="?", type=Path)
    parser.add_argument("actual", nargs="?", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("self-test passed")
        return 0

    if args.expected is None or args.actual is None:
        parser.error("expected and actual paths are required unless --self-test is used")

    compare(load_array(args.expected), load_array(args.actual), args.rtol, args.atol)
    print("equivalence check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
