set -x
clang++ -O2 -gline-tables-only -fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda  *.cpp -I/opt/intel/oneapi/dpcpp-ct/latest/include -o BlackScholes.dpct.nvgpu.inlineattribute 

# Generating IR
#clang++ -S -emit-llvm -O2 -gline-tables-only -fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda -I/opt/intel/oneapi/dpcpp-ct/latest/include -c *.cpp -save-temps

