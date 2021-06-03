#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <utility>

/**
 * SYCL USM Deleter. The std::unique_ptr deleter cannot take arguments so the
 * only way to get it working is to pass a function like this one
 */
template<typename T>
struct sycl_deleter {
    sycl::queue q_;

    explicit sycl_deleter(sycl::queue q) : q_(std::move(q)) {}

    void operator()(T *ptr) const noexcept {
        if (ptr)
            sycl::free(ptr, q_);
    }
};

/**
 * Wrapper for a std::unique_ptr that calls the SYCL deleter (sycl::free).
 * Also holds the number of elements allocated.
 */
template<typename T>
struct sycl_unique {
    std::unique_ptr<T, sycl_deleter<T>> ptr_;
    size_t count_;

    [[nodiscard]] size_t size() const noexcept { return count_ * sizeof(T); }

    [[nodiscard]] size_t count() const noexcept { return count_; }

    [[nodiscard]] T *get() const noexcept { return ptr_.get(); }
};

/**
 * Builds a sycl_unique pointer
 */
template<typename T>
sycl_unique<T> make_sycl_unique(size_t count, sycl::queue &q) {
    return {std::unique_ptr<T, sycl_deleter<T>>(sycl::malloc_shared<T>(count, q), sycl_deleter<T>{q}), count};
}