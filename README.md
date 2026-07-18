# sanos-superfast

Fast C++17 implied volatility surface calibration using the [SANOS methodology](https://arxiv.org/abs/2601.11209) and [volfi](https://github.com/wol-fi/volfi) for IV inversion.

## What it does

Calibrates a smooth, arbitrage-free implied volatility surface from option bid/ask quotes using the SANOS methodology, benchmarked against a generator.

## Performance

15 expiries, 900 total strikes, Clang 22.1.8, AVX2:

| Operation | Median | P95 |
|-----------|--------|-----|
| Warm recalibration (all expiries) | 494 μs | 554 μs |
| Cold calibration (new surface) | 1.57 ms | 2.09 ms |
| Tick update (1 strike changed) | 79 μs | 101 μs |
| Surface query (per point) | 4.4 μs | — |

## Build

```bash
# Clang (recommended):
cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_RC_COMPILER=llvm-rc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build

# MSVC:
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Run

```bash
./build/test_sanos     # unit tests
./build/bench_surface  # benchmark with synthetic SPX data
```
