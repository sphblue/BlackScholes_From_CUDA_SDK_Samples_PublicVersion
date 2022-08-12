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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "realtype.h"

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

#ifdef USE_FAST_MATH
    float K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
#else
    float K = 1.0f / (1.0f + 0.2316419f * sycl::fabs(d));

    float cnd = RSQRT2PI * sycl::exp(-0.5f * d * d) *
                (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
#endif

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

#ifdef USE_FAST_MATH
    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
#else
    sqrtT = 1.0F / sycl::rsqrt(T);
    d1 = (sycl::log(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
#endif
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
#ifdef USE_FAST_MATH
    expRT = __expf(- R * T);
#else
    expRT = sycl::exp(-R * T);
#endif

    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////

void BlackScholesGPU(sycl::float2 *__restrict d_CallResult,
                     sycl::float2 *__restrict d_PutResult,
                     sycl::float2 *__restrict d_StockPrice,
                     sycl::float2 *__restrict d_OptionStrike,
                     sycl::float2 *__restrict d_OptionYears, float Riskfree,
                     float Volatility, int optN, sycl::nd_item<3> item_ct1)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    const int opt = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

     // Calculating 2 options per thread to increase ILP (instruction level parallelism)
    if (opt < (optN / 2))
    {
        float callResult1, callResult2;
        float putResult1, putResult2;
        BlackScholesBodyGPU(callResult1, putResult1, d_StockPrice[opt].x(),
                            d_OptionStrike[opt].x(), d_OptionYears[opt].x(),
                            Riskfree, Volatility);
        BlackScholesBodyGPU(callResult2, putResult2, d_StockPrice[opt].y(),
                            d_OptionStrike[opt].y(), d_OptionYears[opt].y(),
                            Riskfree, Volatility);
        d_CallResult[opt] = sycl::float2(callResult1, callResult2);
        d_PutResult[opt] = sycl::float2(putResult1, putResult2);
         }
}

////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
void BlackScholesGPU_SingleOpt(
    float * __restrict d_CallResult,
    float * __restrict d_PutResult,
    float * __restrict d_StockPrice,
    float * __restrict d_OptionStrike,
    float * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
,
    sycl::nd_item<3> item_ct1)
{
    int opt = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
        if (opt < optN)
	{
		float callResult1;
		float putResult1;
		BlackScholesBodyGPU(callResult1, putResult1, 
			d_StockPrice[opt], d_OptionStrike[opt], d_OptionYears[opt], Riskfree, Volatility);
		
		d_CallResult[opt] = callResult1;
		d_PutResult[opt] = putResult1;
	}
}

