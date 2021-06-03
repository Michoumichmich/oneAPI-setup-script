#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <chrono.hpp>
#include <common.hpp>

int main(int argc, char *argv[]) {
    using T = float;
    size_t n_laps = 30;
    size_t mat_size = 16384; // Bound by your GPU's memory.

    if (argc > 1) {
        mat_size = std::strtoul(argv[1], nullptr, 10);
    }
    T alpha = 1, beta = 0; // gemm parameters

    sycl::device my_device = try_get_cuda_device();
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
    sycl::queue my_queue(my_device, my_exception_handler);

    std::cout << "Initalizing the matrices..." << std::endl;
    long n = mat_size, m = mat_size, k = mat_size, ldA = mat_size, ldB = mat_size, ldC = mat_size;
    auto A = make_sycl_unique<T>(mat_size * mat_size, my_queue);
    auto B = make_sycl_unique<T>(mat_size * mat_size, my_queue);
    auto C = make_sycl_unique<T>(mat_size * mat_size, my_queue);
    fill_rand(A);
    fill_rand(B);

    std::cout << "Running on:" << my_device.get_info<sycl::info::device::name>() << std::endl;
    Chrono c("computing + error handling");
    for (size_t i = 0; i < n_laps; i++) {
        std::cout << i << '/' << n_laps << '\n';
        try {
            using oneapi::mkl::transpose;
            using oneapi::mkl::blas::column_major::gemm;
            gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A.get(), ldA, B.get(), ldB, beta, C.get(), ldC);  // C <- alpha*OP(A)*OP(B) + beta*C
        }
        catch (sycl::exception const &e) {
            std::cout << "Caught synchronous SYCL exception during GEMM: " << e.what() << std::endl;
        }
        catch (std::exception const &e) {
            std::cout << "Caught synchronous STL exception during GEMM: " << e.what() << std::endl;
        }
        my_queue.wait_and_throw();
    }
    uint64_t operations_performed = n_laps * mat_size * mat_size * (2 * mat_size - 1);
    std::cout << "Gflops : " << operations_performed / 1000000000 / c.stop() << std::endl;

    return 0;
}