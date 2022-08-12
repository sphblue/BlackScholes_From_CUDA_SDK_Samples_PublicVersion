#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
inline double cndGPU(double d)
{
    const double       A1 = 0.31938153f;
    const double       A2 = -0.356563782f;
    const double       A3 = 1.781477937f;
    const double       A4 = -1.821255978f;
    const double       A5 = 1.330274429f;
    const double RSQRT2PI = 0.39894228040143267793994605993438f;

#if 0 //ORIG CODE
    double
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    double
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;
#endif

#ifdef USE_FAST_MATH
    double K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));
     
    double
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

#else
    double K = 1.0 / (1.0 + 0.2316419 * sycl::fabs(d));

    double cnd = RSQRT2PI * sycl::exp(-0.5 * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;
#endif

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
inline void BlackScholesBodyGPU(
    double &CallResult,
    double &PutResult,
    double S, //Stock price
    double X, //Option strike
    double T, //Option years
    double R, //Riskless rate
    double V  //Volatility rate
)
{
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;

#ifdef USE_FAST_MATH
    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
#else
    sqrtT = 1.0 / sycl::rsqrt(T);
    d1 = (sycl::log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
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


#ifdef USE_FAST_MATH
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
#else
    PutResult  = X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1);
#endif
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////

inline sycl::double2 make_sycl_double2(double x, double y)
{  
  sycl::double2 t = {x, y}; return t;  
}

#if 1
__attribute__((always_inline))
#endif
void BlackScholesGPU(sycl::double2 *__restrict d_CallResult,
                     sycl::double2 *__restrict d_PutResult,
                     sycl::double2 *__restrict d_StockPrice,
                     sycl::double2 *__restrict d_OptionStrike,
                     sycl::double2 *__restrict d_OptionYears, double Riskfree,
                     double Volatility, int optN, sycl::nd_item<3> item_ct1)
{
    int opt = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

        // Calculating 2 options per thread to increase ILP (instruction level parallelism)
	if (opt < (optN / 2))
	{
		double callResult1, callResult2;
		double putResult1, putResult2;
                BlackScholesBodyGPU(
                    callResult1, putResult1, d_StockPrice[opt].x(),
                    d_OptionStrike[opt].x(), d_OptionYears[opt].x(), Riskfree,
                    Volatility);

                BlackScholesBodyGPU(
                    callResult2, putResult2, d_StockPrice[opt].y(),
                    d_OptionStrike[opt].y(), d_OptionYears[opt].y(), Riskfree,
                    Volatility);

                d_CallResult[opt] = sycl::double2(callResult1, callResult2);
                d_PutResult[opt] = sycl::double2(putResult1, putResult2);
    }
}

////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
void BlackScholesGPU_SingleOpt(
    double * __restrict d_CallResult,
    double * __restrict d_PutResult,
    double * __restrict d_StockPrice,
    double * __restrict d_OptionStrike,
    double * __restrict d_OptionYears,
    double Riskfree,
    double Volatility,
    int optN
,
    sycl::nd_item<3> item_ct1)
{
    int opt = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
        if (opt < optN)
	{
		double callResult1;
		double putResult1;
		BlackScholesBodyGPU(callResult1, putResult1, 
			d_StockPrice[opt], d_OptionStrike[opt], d_OptionYears[opt], Riskfree, Volatility);
		
		d_CallResult[opt] = callResult1;
		d_PutResult[opt] = putResult1;
	}
}
