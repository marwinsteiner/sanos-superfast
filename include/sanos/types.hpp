#pragma once
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>
#include <cstring>

namespace sanos {

static constexpr std::size_t CACHE_LINE = 64;

template <typename T, std::size_t Align = CACHE_LINE>
struct AlignedAllocator {
    using value_type = T;

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Align>; };

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p;
#ifdef _MSC_VER
        p = _aligned_malloc(n * sizeof(T), Align);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, Align, n * sizeof(T))) throw std::bad_alloc();
#endif
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Align>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Align>&) const noexcept { return false; }
};

template <typename T>
using AVec = std::vector<T, AlignedAllocator<T>>;

// Dense matrix stored column-major for BLAS-like access patterns.
// M[row, col] = data[col * rows + row]
struct DenseMat {
    AVec<double> data;
    int rows = 0;
    int cols = 0;

    DenseMat() = default;
    DenseMat(int r, int c) : data(r * c, 0.0), rows(r), cols(c) {}

    void resize(int r, int c) {
        rows = r;
        cols = c;
        data.assign(r * c, 0.0);
    }

    double& operator()(int r, int c) { return data[c * rows + r]; }
    double  operator()(int r, int c) const { return data[c * rows + r]; }

    double* col_ptr(int c) { return data.data() + c * rows; }
    const double* col_ptr(int c) const { return data.data() + c * rows; }
};

} // namespace sanos
