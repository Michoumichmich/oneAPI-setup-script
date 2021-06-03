cmake_minimum_required(VERSION 3.9)
project(Sycl_Tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -fno-common -Winit-self -Wuninitialized -Wmissing-declarations -Woverloaded-virtual ")
set(EXTRA_W_FLAGS "-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self \
 -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow \
-Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wundef -Wno-unused") #-Wnoexcept -Wold-style-cast -Wstrict-null-sentinel -switch-default -Wlogical-op
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-inline-namespace-reopened-noninline")

set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")


SET(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -sycl-std=2020 -fsycl-unnamed-lambda -Wno-unknown-cuda-version")
SET(CMAKE_EXE_LINKER_FLAGS "-lonemkl_blas_cublas -lonemkl")

set(CMAKE_CXX_FLAGS_RELEASE "${OPT_FLAGS} ${DISABLED_WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG " ${WARNING_FLAGS} ${EXTRA_W_FLAGS} ${DISABLED_WARNINGS} -g3 -Og")

include_directories(include/)
#include_directories(/home/michel/sycl_workspace/deploy/include)
#link_directories(/home/michel/sycl_workspace/deploy/lib)

add_executable(mkl_kernel mkl_matmult.cpp)
add_executable(mkl_kernel_usm mkl_matmult_usm.cpp)

