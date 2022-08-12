CXX=dpcpp

CXXFLAGS=-O3

all: BlackScholes.dpct

BlackScholes.dpct: BlackScholes_gold.cpp BlackScholes.dp.cpp BlackScholes_kernel_DP.dp.hpp
	$(CXX) $(CXXFLAGS) -o $@ BlackScholes.dp.cpp BlackScholes_gold.cpp

clean	:
	rm -f BlackScholes.dpct BlackScholes.dpct.nvgpu

