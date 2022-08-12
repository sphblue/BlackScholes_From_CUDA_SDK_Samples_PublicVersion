/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio> //@ JD
#include <chrono> //@
//@ #include <helper_functions.h>   // helper functions for string parsing
//@ #include <helper_cuda.h>        // helper functions CUDA error checking and initialization

#include "realtype.h"

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    real *h_CallResult,
    real *h_PutResult,
    real *h_StockPrice,
    real *h_OptionStrike,
    real *h_OptionYears,
    real Riskfree,
    real Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////

#ifndef DOUBLE_PRECISION
//#include "BlackScholes_kernel.cuh"
// manual porting of migrated kernel for quick check
#include "BlackScholes_kernel_floatversion_migrated_manual.dp.hpp"
//typedef float2 real2;
typedef sycl::float2 real2;
#else
#include "BlackScholes_kernel_DP.dp.hpp"
#include <cmath>

typedef sycl::double2 real2;
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random real in [low, high] range
////////////////////////////////////////////////////////////////////////////////
real Randreal(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
int OPT_N = 16384000/2; //4000000;
// int OPT_N_L2_V100 = 81920;
int  NUM_ITERATIONS = 1000;

const int OPT_SZ = OPT_N * sizeof(real);
const real RISKFREE = 0.02f;
const real VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )


int checkArguments(int argc, char **argv) {

	int total_options = 0;
	for (int i = 1; i < argc; ++i) {
		if (strstr(argv[i], "-?") || strstr(argv[i], "-help")) {
			std::cout << "Supported Input Arguments:" << std::endl;
			std::cout << "-i:10      --> Iteration count. Requirements: Must be > 0." << std::endl;
			std::cout << "               This example will run the Kernel 10 times." << std::endl;
			std::cout << "               Default is 1." << std::endl;
			std::cout << "-o:1048576 --> Option count. Requirements: 1024 minimum." << std::endl;
			std::cout << "               This example will calculate results for 1M options." << std::endl;
			std::cout << "               Default is 8M." << std::endl;
			std::cout << "********************************" << std::endl;
			return -1;
		}
		if (strstr(argv[i], "-i:")) {
			sscanf(argv[i] + 3, "%d", &NUM_ITERATIONS);
			if (NUM_ITERATIONS < 0)
			{
				std::cout << "ERROR Invalid Iteration count = " << argv[i] + 3 << std::endl;
				return -1;
			}
			continue;
		}
		else if (strstr(argv[i], "-o:")) {
			sscanf(argv[i] + 3, "%d", &total_options);
			if (total_options < 1024 )
			{
				std::cout << "ERROR Invalid Option count = " << argv[i] + 3 << std::endl;
				return -1;
			}
			OPT_N = total_options/2;
			continue;
		}
		else {
			std::cout << "ERROR Invalid input Argument: " << argv[i] << std::endl;
			std::cout << "      Use -? or -help" << std::endl;
			return -1;
		}

	}
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int ret = checkArguments(argc, argv);
    if(ret != 0)
    {
        exit (EXIT_FAILURE);
    }

    dpct::device_ext &dev_ct1 = dpct::get_current_device(); std::cout << "dev_info = " << dev_ct1.get_info<cl::sycl::info::device::name>() << std::endl;
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    real
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    real
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    std::chrono::high_resolution_clock::time_point start, stop;	//@ StopWatchInterface *hTimer = NULL;
    int i;

    //@ findCudaDevice(argc, (const char **)argv);

    //@ sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (real *)malloc(OPT_SZ);
    h_PutResultCPU  = (real *)malloc(OPT_SZ);
    h_CallResultGPU = (real *)malloc(OPT_SZ);
    h_PutResultGPU  = (real *)malloc(OPT_SZ);
    h_StockPrice    = (real *)malloc(OPT_SZ);
    h_OptionStrike  = (real *)malloc(OPT_SZ);
    h_OptionYears   = (real *)malloc(OPT_SZ);

    printf("...allocating GPU memory for options.\n");
    d_CallResult = (real *)sycl::malloc_device(
        OPT_SZ,
        q_ct1); //@ checkCudaErrors(cudaMalloc((void **)&d_CallResult, OPT_SZ));
    d_PutResult = (real *)sycl::malloc_device(
        OPT_SZ,
        q_ct1); //@ checkCudaErrors(cudaMalloc((void **)&d_PutResult, OPT_SZ));
    d_StockPrice = (real *)sycl::malloc_device(
        OPT_SZ,
        q_ct1); //@ checkCudaErrors(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
    d_OptionStrike = (real *)sycl::malloc_device(
        OPT_SZ, q_ct1); //@ checkCudaErrors(cudaMalloc((void **)&d_OptionStrike,
                        //OPT_SZ));
    d_OptionYears = (real *)sycl::malloc_device(
        OPT_SZ, q_ct1); //@ checkCudaErrors(cudaMalloc((void **)&d_OptionYears,
                        //OPT_SZ));

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = Randreal(5.0f, 30.0f);
        h_OptionStrike[i]  = Randreal(1.0f, 100.0f);
        h_OptionYears[i]   = Randreal(0.25f, 10.0f);
    }

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    q_ct1.memcpy(d_StockPrice, h_StockPrice, OPT_SZ)
        .wait(); //@ checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,
                 //OPT_SZ, cudaMemcpyHostToDevice));
    q_ct1.memcpy(d_OptionStrike, h_OptionStrike, OPT_SZ)
        .wait(); //@ checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,
                 //OPT_SZ, cudaMemcpyHostToDevice));
    q_ct1.memcpy(d_OptionYears, h_OptionYears, OPT_SZ)
        .wait(); //@ checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,
                 //OPT_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");


    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    dev_ct1
        .queues_wait_and_throw(); //@ checkCudaErrors(cudaDeviceSynchronize());
    //@ sdkResetTimer(&hTimer);
    start = std::chrono::high_resolution_clock::now();	//@ sdkStartTimer(&hTimer);

    for (i = 0; i < NUM_ITERATIONS; i++)
    {

#if 1
    q_ct1.submit([&](sycl::handler &cgh) {
      auto RISKFREE_ct5 = RISKFREE;
      auto VOLATILITY_ct6 = VOLATILITY;
      auto OPT_N_ct7 = OPT_N;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, DIV_UP((OPT_N / 2), 128)) *
                                sycl::range<3>(1, 1, 128),
                            sycl::range<3>(1, 1, 128)),
          [=](sycl::nd_item<3> item_ct1) {
            BlackScholesGPU((real2 *)d_CallResult, (real2 *)d_PutResult,
                            (real2 *)d_StockPrice, (real2 *)d_OptionStrike,
                            (real2 *)d_OptionYears, RISKFREE_ct5,
                            VOLATILITY_ct6, OPT_N_ct7, item_ct1);
          });
    });

#else
		BlackScholesGPU_SingleOpt<<<DIV_UP(OPT_N, 256), 256>>>(
			d_CallResult,
			d_PutResult,
			d_StockPrice,
			d_OptionStrike,
			d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
#endif
        //@ getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    dev_ct1
        .queues_wait_and_throw(); //@ checkCudaErrors(cudaDeviceSynchronize());
    stop = std::chrono::high_resolution_clock::now();	//@ sdkStopTimer(&hTimer);
    gpuTime = std::chrono::duration<double>(stop - start).count();	//@ sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(real)) / (1024*1024*1024)) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    q_ct1.memcpy(h_CallResultGPU, d_CallResult, OPT_SZ)
        .wait(); //@ checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult,
                 //OPT_SZ, cudaMemcpyDeviceToHost));
    q_ct1.memcpy(h_PutResultGPU, d_PutResult, OPT_SZ)
        .wait(); //@ checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,
                 //OPT_SZ, cudaMemcpyDeviceToHost));

    printf("Checking the results...\n");
    printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );

    printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    sycl::free(d_OptionYears,
               q_ct1); //@ checkCudaErrors(cudaFree(d_OptionYears));
    sycl::free(d_OptionStrike,
               q_ct1); //@ checkCudaErrors(cudaFree(d_OptionStrike));
    sycl::free(d_StockPrice,
               q_ct1);              //@ checkCudaErrors(cudaFree(d_StockPrice));
    sycl::free(d_PutResult, q_ct1); //@ checkCudaErrors(cudaFree(d_PutResult));
    sycl::free(d_CallResult,
               q_ct1); //@ checkCudaErrors(cudaFree(d_CallResult));

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    //@ sdkDeleteTimer(&hTimer);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    dev_ct1.reset();

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
