#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <exception>
#include <iostream>

#include "chrono.hpp"
#include "sycl_unique_ptr.hpp"
#include "common.hpp"


int main(int argc, char *argv[])
{
    using T = float;
    size_t n_laps = 30;
    long int mat_size = (16384 + 4096); // Bound by your GPU's memory.
    T alpha = 1, beta = 0;
    // Create GPU device
    sycl::device my_device;
    try
    {
        my_device = sycl::device(sycl::gpu_selector());
    }
    catch (...)
    {
        my_device = sycl::device(sycl::host_selector());
        std::cout << "Warning: GPU device not found! Fall back on " << my_device.get_info<sycl::info::device::name>() << std::endl;
    }
    // Create asynchronous exceptions handler to be attached to queue.
    // Not required; can provide helpful information in case the system isn’t correctly configured.
    auto my_exception_handler = [](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const &e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e)
            {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
            catch (std::exception const &e)
            {
                std::cout << "Caught asynchronous STL exception: " << e.what() << std::endl;
            }
        }
    };
    // create execution queue on my gpu device with exception handler attached
    sycl::queue my_queue(my_device, my_exception_handler);

    std::cout << "Initalizing the matrices..." << std::endl;
    long n = mat_size, m = mat_size, k = mat_size, ldA = mat_size, ldB = mat_size, ldC = mat_size;
    auto A = make_sycl_unique<T>(mat_size * mat_size, my_queue);
    auto B = make_sycl_unique<T>(mat_size * mat_size, my_queue);
    auto C = make_sycl_unique<T>(mat_size * mat_size, my_queue);
    // fill_rand(A);
    // fill_rand(B);

    std::cout << "Running on:" << my_device.get_info<sycl::info::device::name>() << std::endl;
    Chrono c("computing + error handling");
    for (size_t i = 0; i < n_laps; i++)
    {
        std::cout << i << '/' << n_laps << '\n';
        // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous exceptions
        try
        {
            using oneapi::mkl::transpose;
            using oneapi::mkl::blas::column_major::gemm;
            // C <- alpha*OP(A)*OP(B) + beta*C
            gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A.get(), ldA, B.get(), ldB, beta, C.get(), ldC);
            //gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::cublas>{my_queue}, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A.get(), ldA, B.get(), ldB, beta, C.get(), ldC);
        }
        catch (sycl::exception const &e)
        {
            std::cout << "Caught synchronous SYCL exception during GEMM: " << e.what() << std::endl;
        }
        catch (std::exception const &e)
        {
            std::cout << "Caught synchronous STL exception during GEMM: " << e.what() << std::endl;
        }
        // ensure any asynchronous exceptions caught are handled before proceeding
        my_queue.wait_and_throw();
    }
    uint64_t operations_performed = n_laps * mat_size * mat_size * (2 * mat_size - 1);
    std::cout << "Gflops : " << operations_performed / 1000000000 / c.stop() << std::endl;

    return 0;
}