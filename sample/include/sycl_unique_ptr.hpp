#pragma once
#include <sycl/sycl.hpp>
#include <memory>

template <typename T>
struct sycl_deleter
{
    sycl::queue q_;

    explicit sycl_deleter(const sycl::queue &q) : q_(q) {}

    void operator()(T *ptr) const noexcept
    {
        if (ptr)
            sycl::free(ptr, q_);
    }
};

template <typename T>
struct sycl_unique
{
    std::unique_ptr<T, sycl_deleter<T>> ptr_;
    size_t count_;

    [[nodiscard]] size_t size() const noexcept { return count_ * sizeof(T); }

    [[nodiscard]] size_t count() const noexcept { return count_; }

    [[nodiscard]] T *get() const noexcept { return ptr_.get(); }
};

template <typename T>
sycl_unique<T> make_sycl_unique(size_t count, sycl::queue &q)
{
    return {std::unique_ptr<T, sycl_deleter<T>>(sycl::malloc_shared<T>(count, q), sycl_deleter<T>{q}), count};
}