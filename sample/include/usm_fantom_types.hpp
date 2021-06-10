#pragma once
#include <sycl/sycl.hpp>


//#include <concepts>
//template <sycl::usm::alloc location>
//concept host_accessible = location == sycl::usm::alloc::host || location == sycl::usm::alloc::shared;

template<class T, sycl::usm::alloc Tag>
struct usm_ptr {
  usm_ptr(T* t) : val_(t) {}
  operator T*(){return val_;}
  const T* get() const { return val_; }private:
  T* val_;
};

