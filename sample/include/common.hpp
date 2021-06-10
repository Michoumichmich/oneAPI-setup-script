#pragma once

#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <type_traits>
#include <sycl_unique_ptr.hpp>

/**
 * CUDA Selector using SYCL 2020 backends
 */
class CUDADeviceSelector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
        return device.get_platform().get_backend() == sycl::backend::cuda ? 1 : -1;
    }
};

/**
 * Tries to get a CUDA device else returns the host device
 */
sycl::queue try_get_cuda_queue() {
    /**
     * SYCL exception handler
     * Create asynchronous exceptions handler to be attached to queue.
     * Not required; can provide helpful information in case the system isn’t correctly configured.
     */
    auto my_exception_handler = [](const sycl::exception_list &exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
            catch (std::exception const &e) {
                std::cout << "Caught asynchronous STL exception: " << e.what() << std::endl;
            }
        }
    };

    sycl::device my_device;
    try {
        my_device = sycl::device(CUDADeviceSelector());
    }
    catch (...) {
        my_device = sycl::device(sycl::host_selector());
        std::cout << "Warning: GPU device not found! Fall back on: " << my_device.get_info<sycl::info::device::name>()
                  << std::endl;
    }
    return sycl::queue(my_device, my_exception_handler);
}


/**
 * Fills a container/array with random numbers from positions first to last
 */
template<typename T, class ForwardIt>
void do_fill_rand(ForwardIt first, ForwardIt last) {
    static std::random_device dev;
    static std::mt19937 engine(dev());
    auto generator = [&]() {
        if constexpr (std::is_integral<T>::value) {
            static std::uniform_int_distribution<T> distribution;
            return distribution(engine);
        } else if constexpr (std::is_floating_point<T>::value) {
            static std::uniform_real_distribution<T> distribution;
            return distribution(engine);
        } else if constexpr (std::is_same_v<T, sycl::half>) {
            static std::uniform_real_distribution<float> distribution;
            return distribution(engine);
        }
    };
    std::generate(first, last, generator);
}

template<typename T>
void fill_rand(sycl_unique<T> &v) {
    do_fill_rand<T>(v.get(), v.get() + v.count());
}

template<typename T>
void fill_rand(std::vector<T> &v) {
    do_fill_rand<T>(v.begin(), v.end());
}
