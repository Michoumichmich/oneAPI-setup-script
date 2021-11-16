export DPCPP_HOME=~/sycl_workspace

ocl_fpga_emu_ver=2021.12.9.0.24
wget -nc https://github.com/intel/llvm/releases/download/2021-WW40/fpgaemu-${ocl_fpga_emu_ver}_rel.tar.gz

mkdir -p /opt/intel/oclfpgaemu_$ocl_fpga_emu_ver
tar -C /opt/intel/oclfpgaemu_$ocl_fpga_emu_ver -zxvf fpgaemu-${ocl_fpga_emu_ver}_rel.tar.gz

echo /opt/intel/oclfpgaemu_$ocl_fpga_emu_ver/x64/libintelocl_emu.so > /etc/OpenCL/vendors/intel_fpgaemu.icd

grep -qxF /opt/intel/oclfpgaemu_$ocl_fpga_emu_ver/x64 /etc/ld.so.conf.d/libintelopenclexp.conf || (echo /opt/intel/oclfpgaemu_$ocl_fpga_emu_ver/x64 >> /etc/ld.so.conf.d/libintelopenclexp.conf ; ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf)


