# sanos-superfast

Fast C++17 implied volatility surface calibration using the [SANOS methodology](https://arxiv.org/abs/2601.11209) and [volfi](https://github.com/wol-fi/volfi) for IV inversion.

## What it does

Calibrates a smooth, arbitrage-free implied volatility surface from option bid/ask quotes using the SANOS methodology. Fits real (delayed) SPX option data fetched from Yahoo Finance via `scripts/fetch_spx_options.py`.

## Performance

20 expiries, 2298 total strikes (real SPX chain), Clang 22.1.8, AVX2:

| Operation | Median | P95 | Min |
|-----------|--------|-----|-----|
| Warm recalibration (all expiries) | 367 μs | 427 μs | 313 μs |
| Cold calibration (new surface) | 1.14 ms | 1.38 ms | 946 μs |
| Tick update (1 strike changed) | 66 μs | 75 μs | 7 μs |

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
./build/bench_surface  # benchmark
uv run scripts/fetch_spx_options.py  # fetch fresh SPX option chain
```
