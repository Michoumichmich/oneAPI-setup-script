export DPCPP_HOME=~/sycl_workspace

cpu_ocl_version=2021.13.11.0.23
wget -nc https://github.com/intel/llvm/releases/download/2021-WW50/oclcpuexp-${cpu_ocl_version}_rel.tar.gz

mkdir -p /opt/intel/oclcpuexp_$cpu_ocl_version
tar -C /opt/intel/oclcpuexp_$cpu_ocl_version -zxvf oclcpuexp-${cpu_ocl_version}_rel.tar.gz

echo /opt/intel/oclcpuexp_$cpu_ocl_version/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd

grep -qxF /opt/intel/oclcpuexp_$cpu_ocl_version/x64 /etc/ld.so.conf.d/libintelopenclexp.conf || (echo /opt/intel/oclcpuexp_$cpu_ocl_version/x64 >> /etc/ld.so.conf.d/libintelopenclexp.conf ; ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf)
