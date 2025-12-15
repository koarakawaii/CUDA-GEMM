/* Example code from https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
   For more detail, see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-matrix-functions
   assert(...) in the kernel seems to degrade the performance, so disable it when -DPERFORMANCE is on
 */

#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include <assert.h>
#include <mma.h>
#include <omp.h>

using namespace nvcuda;

#define CUDA_CHECK_ERROR( Call )   CUDA_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )
#define CUBLAS_CHECK_ERROR( Call ) CUBLAS_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )

// Input type must be float, even though you specify precision::tf32 for frament; TF32 precision affects how tensor cores internally quantize FP32 inputs. So you still feed FP32 arrays; Tensor Cores handle the rounding
#define T_ELEM_IN_FRAG  wmma::precision::tf32
#define T_ELEM_IN       float
#define T_ELEM_OUT      float
#define CUBLAS_ELEM_IN  CUDA_R_32F
#define CUBLAS_ELEM_OUT CUDA_R_32F
#define TEST_IDX     2097152
#define WARPSIZE     32
#define THREADTILE   8
#define THREADTILE_X 8
#define THREADTILE_Y 16
#define SKEW_MINE    8

//// for RTX5090
#define BM_Warptiling 128
#define BK_Warptiling 32
#define BN_Warptiling 128
#define WM_Warptiling 64
#define WN_Warptiling 64
#define TM_Warptiling 8
#define TN_Warptiling 4
#define WMITER        2
#define WNITER        2
const int WARP_TILING_MAX_NUM_THREADS = 128;
////

typedef float4 copy_batch_t;

//// for RTX3080Ti
//#define BM_Warptiling 64
//#define BK_Warptiling 16
//#define BN_Warptiling 128
//#define WM_Warptiling 32
//#define WN_Warptiling 64
//#define TM_Warptiling 4
//#define TN_Warptiling 4
//#define WMITER        2
//#define WNITER        2
//const int WARP_TILING_MAX_NUM_THREADS = 128;
////

// The only dimensions currently supported by WMMA
const int WMMA_M              = 16;
const int WMMA_N              = 16;
const int WMMA_K              = 8;
const int warpK_stride_shared = 4; // use shared memory to cache matrix A and B; warpK_stride_shared is the sum of the number of WMMA_M*WMMA_K blocks for matrix A + WMMA_K*WMMA_N blocks for matrix B that can be cached by shared memory

const long  seed              = 152897564;
const float alpha             = 1.0f;
const float beta              = 1.0f;
const int   copy_batch_factor = sizeof(copy_batch_t)/sizeof(T_ELEM_IN);
#ifdef TIMING_FLAG
const float ms_to_sec         = 1.0e3;
#endif


/* General CUDA error handling */
inline bool CUDA_Check_Error( cudaError Return, const char *File, const int Line, const char *Func )
{
   if ( Return != cudaSuccess )
   {
      printf("CUDA ERROR : %s at %s : Line: %d ; Function: %s !!\n", cudaGetErrorString( Return ), File, Line, Func);
      return false;
   }
   else
      return true;
}


/* cuBLAS error handling */
const char* cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS: No errors";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED: cuBLAS not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE: Invalid value was passed";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH: Architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR: Memory mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED: Execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR: Internal cuBLAS error";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED: Operation not supported";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR: Licensing error";
        default:                             return "Unknown cuBLAS error";
    }
}
inline bool CUBLAS_Check_Error( cublasStatus_t Return, const char *File, const int Line, const char *Func )
{
   if ( Return != CUBLAS_STATUS_SUCCESS )
   {
      printf("CUBLAS ERROR : %s at %s : Line: %d ; Function: %s !!\n", cublasGetErrorString( Return ), File, Line, Func);
      return false;
   }
   else
      return true;
}


#ifdef PERFORMANCE
__global__ void __launch_bounds__(WARP_TILING_MAX_NUM_THREADS)
#else
__global__ void
#endif
                gemm_kernel(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                            int M, int N, int K,
                            T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    const int lda         = K;
    const int ldb         = N;
    const int ldc         = N;

    extern __shared__ T_ELEM_IN shared_buffer[];

    T_ELEM_IN *a_shared_buffer = &(shared_buffer[0]);
    T_ELEM_IN *b_shared_buffer = &(shared_buffer[(warpK_stride_shared*WMMA_K)*blockDim.y]);

    int array_idx_y = threadIdx.y + blockDim.y * blockIdx.y;
    while ( array_idx_y < blockDim.y * (int)((M + blockDim.y - 1) / blockDim.y) )
    {
        int array_idx_x = threadIdx.x + blockDim.x * blockIdx.x;
        while ( array_idx_x < blockDim.x * (int)((N + blockDim.x - 1) / blockDim.x ) )
        {
            int aRow       = array_idx_y;
            int bCol       = array_idx_x;
            T_ELEM_OUT acc = (T_ELEM_OUT)0.0;

            for (int i=0; i<K; i+=(warpK_stride_shared*WMMA_K))
            {
                __syncthreads();
                int thread_idx_x = threadIdx.x;
                while (thread_idx_x < (warpK_stride_shared*WMMA_K))
                {
                    int aCol                                                                   = i + thread_idx_x;
                    if ( (aRow < M) && (aCol < K) )
                        a_shared_buffer[thread_idx_x+threadIdx.y*(warpK_stride_shared*WMMA_K)] = a[aCol + aRow*lda];
                    else
                        a_shared_buffer[thread_idx_x+threadIdx.y*(warpK_stride_shared*WMMA_K)] = 0.0;
                    thread_idx_x                                                              += blockDim.x;
                }
                int thread_idx_y  = threadIdx.y;
                while (thread_idx_y < (warpK_stride_shared*WMMA_K))
                {
                    int bRow                                                 = i + thread_idx_y;
                    if ( (bRow < K) && (bCol < N) )
                        b_shared_buffer[threadIdx.x+thread_idx_y*blockDim.x] = b[bCol + bRow*ldb];
                    else
                        b_shared_buffer[threadIdx.x+thread_idx_y*blockDim.x] = 0.0;
                    thread_idx_y                                            += blockDim.y;
                }
                __syncthreads();

                if ( (array_idx_y < M) && (array_idx_x < N) )
                {
                    #pragma unroll
                    for (int j=0; j<(warpK_stride_shared*WMMA_K); ++j)
                    {
                        if ( (i + j) < K )
                        {
                            acc += (T_ELEM_OUT)a_shared_buffer[j + threadIdx.y*(warpK_stride_shared*WMMA_K)] * (T_ELEM_OUT)b_shared_buffer[threadIdx.x + j*blockDim.x];
                            //acc += (T_ELEM_OUT)a[(i+j) + aRow*lda] * (T_ELEM_OUT)b[bCol + (i+j)*ldb];
                        }
                        else
                            break;
                    }
                }
            }

            if ( (array_idx_y < M) && (array_idx_x < N) )
            {
                c[bCol + aRow*ldc] = alpha * acc + beta* c[bCol + aRow*ldc];
            }

            array_idx_x += blockDim.x * gridDim.x;
        }
        array_idx_y += blockDim.y * gridDim.y;
    }
}


#ifdef PERFORMANCE
__global__ void __launch_bounds__(WARP_TILING_MAX_NUM_THREADS)
#else
__global__ void
#endif
                gemm_kernel_1DTiling(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                                     int M, int N, int K, int BM,
                                     T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    const int lda         = K;
    const int ldb         = N;
    const int ldc         = N;

    extern __shared__ T_ELEM_IN shared_buffer[];

    T_ELEM_IN *a_shared_buffer = &(shared_buffer[0]);
    T_ELEM_IN *b_shared_buffer = &(shared_buffer[BM*(warpK_stride_shared*WMMA_K)]);

    int threadtile_idx_y = (THREADTILE*threadIdx.y) + BM*blockIdx.y; // BM = THREADTILE*blockDim.y
    while ( threadtile_idx_y < (BM * (int)((M + BM - 1) / BM)) )
    {
        int threadtile_idx_x = threadIdx.x + blockDim.x * blockIdx.x;
        while ( threadtile_idx_x < blockDim.x * (int)((N + blockDim.x - 1) / blockDim.x ) )
        {
            int aRow0                  = threadtile_idx_y;
            int bCol                   = threadtile_idx_x;
            T_ELEM_OUT acc[THREADTILE] = {(T_ELEM_OUT)0.0};

            for (int i=0; i<K; i+=(warpK_stride_shared*WMMA_K))
            {
                __syncthreads();
                int thread_idx_x = threadIdx.x;
                while (thread_idx_x < (warpK_stride_shared*WMMA_K))
                {
                    int aCol                                                                                              = i + thread_idx_x;
                    #pragma unroll
                    for (int inner_idx=0; inner_idx<THREADTILE; ++inner_idx)
                    {
                        int aRow                                                                                          = aRow0 + inner_idx;
                        if ( (aRow < M) && (aCol < K) )
                            a_shared_buffer[thread_idx_x+(THREADTILE*threadIdx.y+inner_idx)*(warpK_stride_shared*WMMA_K)] = a[aCol + aRow*lda];
                        else
                            a_shared_buffer[thread_idx_x+(THREADTILE*threadIdx.y+inner_idx)*(warpK_stride_shared*WMMA_K)] = 0.0;
                    }
                    thread_idx_x                                                                                         += blockDim.x;
                }
                int thread_idx_y  = threadIdx.y;
                while (thread_idx_y < (warpK_stride_shared*WMMA_K))
                {
                    int bRow                                                 = i + thread_idx_y;
                    if ( (bRow < K) && (bCol < N) )
                        b_shared_buffer[threadIdx.x+thread_idx_y*blockDim.x] = b[bCol + bRow*ldb];
                    else
                        b_shared_buffer[threadIdx.x+thread_idx_y*blockDim.x] = 0.0;
                    thread_idx_y                                            += blockDim.y;
                }
                __syncthreads();

                #pragma unroll
                for (int j=0; j<(warpK_stride_shared*WMMA_K); ++j)
                {
                    if ( (i + j) < K )
                    {
                        T_ELEM_IN b_temp    = b_shared_buffer[threadIdx.x + j*blockDim.x];
                        #pragma unroll
                        for (int inner_idx=0; inner_idx<THREADTILE; ++inner_idx)
                        {
                            int aRow_shared = (threadIdx.y*THREADTILE) + inner_idx;
                            acc[inner_idx] += (T_ELEM_OUT)a_shared_buffer[j + aRow_shared*(warpK_stride_shared*WMMA_K)] * (T_ELEM_OUT)b_temp;
                            //int aRow            = aRow0 + inner_idx;
                            //if ( (aRow < M) && (bCol < N) )
                                //acc[inner_idx] += (T_ELEM_OUT)a[(i+j) + aRow*lda] * (T_ELEM_OUT)b[bCol + (i+j)*ldb];
                        }
                    }
                    else
                        break;
                }
            }

            #pragma unroll
            for (int inner_idx=0; inner_idx<THREADTILE; ++inner_idx)
            {
                int aRow               = aRow0 + inner_idx;
                if ( (aRow < M) && (bCol < N) )
                    c[bCol + aRow*ldc] = alpha*acc[inner_idx] + beta*c[bCol + aRow*ldc];
            }

            threadtile_idx_x += blockDim.x * gridDim.x;
        }
        threadtile_idx_y += BM * gridDim.y;
    }
}


#ifdef PERFORMANCE
__global__ void __launch_bounds__(WARP_TILING_MAX_NUM_THREADS)
#else
__global__ void
#endif
                gemm_kernel_2DTiling(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                                     int M,  int N, int K, int BM, int BN,
                                     T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    const int lda           = K;
    const int ldb           = N;
    const int ldc           = N;
    //const int thread_idx    = threadIdx.x + blockDim.x * threadIdx.y;
    //const int total_threads = blockDim.x * blockDim.y;

    extern __shared__ T_ELEM_IN shared_buffer[];

    T_ELEM_IN *a_shared_buffer = &(shared_buffer[0]);
    T_ELEM_IN *b_shared_buffer = &(shared_buffer[BM*(warpK_stride_shared*WMMA_K)]);
    T_ELEM_IN  a_col_cache[THREADTILE_Y]; // cache the column of A for a given coloumn index (i+j) when doing accumulation
    T_ELEM_IN  b_row_cache[THREADTILE_X]; // cache the row of B for a given row index (i+j) when doing accumulation

    int threadtile_idx_y = BM*blockIdx.y;     // BM = THREADTILE_Y*blockDim.y
    while ( threadtile_idx_y < (BM * (int)((M + BM - 1) / BM)) )
    {
        // in order to realize memory coalescing for threads in the same wrap, we use threadIdx.x+blockDim.x*inner_idx instead of THREADTILE_X*threadIdx.x+inner_idx for mapping along row direction for b
        int threadtile_idx_x = BN*blockIdx.x; // BN = THREADTILE_X*blockDim.x
        while ( threadtile_idx_x < (BN * (int)((N + BN - 1) / BN)) )
        {
            int aRow0                                 = threadtile_idx_y;
            int bCol0                                 = threadtile_idx_x;
            T_ELEM_OUT acc[THREADTILE_X*THREADTILE_Y] = {(T_ELEM_OUT)0.0};

            for (int i=0; i<K; i+=(warpK_stride_shared*WMMA_K))
            {
                // older version, but faster than new version for this kernel
                __syncthreads();
                int thread_idx_x = threadIdx.x;
                while (thread_idx_x < (warpK_stride_shared*WMMA_K))
                {
                    int aCol                                                                                                = i + thread_idx_x;
                    #pragma unroll
                    for (int inner_idx=0; inner_idx<THREADTILE_Y; ++inner_idx)
                    {
                        int aRow                                                                                            = aRow0 + (THREADTILE_Y*threadIdx.y) + inner_idx;
                        if ( (aRow < M) && (aCol < K) )
                            a_shared_buffer[thread_idx_x+(THREADTILE_Y*threadIdx.y+inner_idx)*(warpK_stride_shared*WMMA_K)] = a[aCol + aRow*lda];
                        else
                            a_shared_buffer[thread_idx_x+(THREADTILE_Y*threadIdx.y+inner_idx)*(warpK_stride_shared*WMMA_K)] = 0.0;
                    }
                    thread_idx_x                                                                                           += blockDim.x;
                }
                int thread_idx_y  = threadIdx.y;
                while (thread_idx_y < (warpK_stride_shared*WMMA_K))
                {
                    int bRow                                                                      = i + thread_idx_y;
                    #pragma unroll
                    for (int inner_idx=0; inner_idx<THREADTILE_X; ++inner_idx)
                    {
                        int bCol                                                                  = bCol0 + threadIdx.x + blockDim.x*inner_idx;
                        // in order to realize memory coalescing for threads in the same wrap, we use threadIdx.x+blockDim.x*inner_idx instead of THREADTILE_X*threadIdx.x+inner_idx for mapping along row direction for b
                        if ( (bRow < K) && (bCol < N) )
                            b_shared_buffer[(threadIdx.x+blockDim.x*inner_idx)+thread_idx_y*(BN)] = b[bCol + bRow*ldb];
                        else
                            b_shared_buffer[(threadIdx.x+blockDim.x*inner_idx)+thread_idx_y*(BN)] = 0.0;
                    }
                    thread_idx_y                                                                 += blockDim.y;
                }
                __syncthreads();
                //

                //__syncthreads();
                ////// !!!!only use this when BM = BN !!!!!
                //////assert (BN == BM);
                ////int idx_copy = thread_idx*copy_batch_factor;
                ////int idx_copy_min      = (BM > BN) ? BN*(warpK_stride_shared*WMMA_K) : BM*(warpK_stride_shared*WMMA_K);
                ////int aCol_shared, aRow_shared, aCol, aRow;
                ////int bCol_shared, bRow_shared, bCol, bRow;
                ////#pragma unroll
                ////for (; idx_copy<idx_copy_min; idx_copy+=copy_batch_factor*total_threads)
                ////{
                ////    aCol_shared = idx_copy % (warpK_stride_shared*WMMA_K);
                ////    aCol        = i + aCol_shared;
                ////    aRow_shared = idx_copy / (warpK_stride_shared*WMMA_K);
                ////    aRow        = aRow0 + aRow_shared;
                ////    if ( (aRow < M) && (aCol < K) )
                ////        *((copy_batch_t*)&(a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K)])) = *((copy_batch_t*)&(a[aCol + aRow*lda]));
                ////    bCol_shared = idx_copy % BN;
                ////    bCol        = bCol0 + bCol_shared;
                ////    bRow_shared = idx_copy / BN;
                ////    bRow        = i + bRow_shared;
                ////    if ( (bRow < K) && (bCol < N) )
                ////        *((copy_batch_t*)&(b_shared_buffer[bCol_shared+bRow_shared*(BN)]))                         = *((copy_batch_t*)&(b[bCol + bRow*ldb]));
                ////}
                //////

                //// ~~~~Use this for general size of (BM, BN); use 1st half of the threads to copy matrix a, and 2nd half of the threads to copy matrix b~~~~
                //if (thread_idx < (total_threads)/2)
                //{
                //    int aCol_shared, aRow_shared, aCol, aRow;
                //    #pragma unroll
                //    for (int idx_copy = thread_idx*copy_batch_factor; idx_copy<BM*(warpK_stride_shared*WMMA_K); idx_copy+=copy_batch_factor*total_threads/2)
                //    {
                //        aCol_shared = idx_copy % (warpK_stride_shared*WMMA_K);
                //        aCol        = i + aCol_shared;
                //        aRow_shared = idx_copy / (warpK_stride_shared*WMMA_K);
                //        aRow        = aRow0 + aRow_shared;
                //        if ( (aRow < M) && (aCol < K) )
                //            *((copy_batch_t*)&(a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K)])) = *((copy_batch_t*)&(a[aCol + aRow*lda]));
                //    }
                //}
                //else
                //{
                //    int bCol_shared, bRow_shared, bCol, bRow;
                //    #pragma unroll
                //    for (int idx_copy = (thread_idx-total_threads/2)*copy_batch_factor; idx_copy<(warpK_stride_shared*WMMA_K)*BN; idx_copy+=copy_batch_factor*total_threads/2)
                //    {
                //        bCol_shared = idx_copy % BN;
                //        bCol        = bCol0 + bCol_shared;
                //        bRow_shared = idx_copy / BN;
                //        bRow        = i + bRow_shared;
                //        if ( (bRow < K) && (bCol < N) )
                //            *((copy_batch_t*)&(b_shared_buffer[bCol_shared+bRow_shared*(BN)])) = *((copy_batch_t*)&(b[bCol + bRow*ldb]));
                //    }
                //}
                ////
                //__syncthreads();
                //

                #pragma unroll
                for (int j=0; j<(warpK_stride_shared*WMMA_K); ++j)
                {
                    if ( (i + j) < K )
                    {
                        #pragma unroll
                        for (int inner_idx=0; inner_idx<THREADTILE_Y; ++inner_idx)
                        {
                            int aRow_shared = (threadIdx.y*THREADTILE_Y) + inner_idx;
                            a_col_cache[inner_idx] = a_shared_buffer[j + aRow_shared*(warpK_stride_shared*WMMA_K)];
                            //int aRow               = aRow0 + aRow_shared;
                            //a_col_cache[inner_idx] = a[(i+j) + aRow*lda];
                        }
                        #pragma unroll
                        for (int inner_idx=0; inner_idx<THREADTILE_X; ++inner_idx)
                        {
                            int bCol_shared        = threadIdx.x + (blockDim.x*inner_idx);
                            b_row_cache[inner_idx] = b_shared_buffer[bCol_shared + j*(BN)];
                            //int bCol               = bCol0 + bCol_shared;
                            //b_row_cache[inner_idx] = b[bCol + (i+j)*ldb];
                        }
                        #pragma unroll
                        for (int inner_idx_M=0; inner_idx_M<THREADTILE_Y; ++inner_idx_M)
                        {
                            #pragma unroll
                            for (int inner_idx_N=0; inner_idx_N<THREADTILE_X; ++inner_idx_N)
                            {
                                acc[inner_idx_M*THREADTILE_X + inner_idx_N]
                                               += (T_ELEM_OUT)a_col_cache[inner_idx_M] * (T_ELEM_OUT)b_row_cache[inner_idx_N];
                                //int aRow            = aRow0 + (THREADTILE_Y*threadIdx.y) + inner_idx_M;
                                //int bCol            = bCol0 + threadIdx.x + blockDim.x*inner_idx_N;
                                //if ( (aRow < M) && (bCol < N) )
                                //    acc[inner_idx_M*THREADTILE_X + inner_idx_N] += (T_ELEM_OUT)a[(i+j) + aRow*lda] * (T_ELEM_OUT)b[bCol + (i+j)*ldb];
                            }
                        }
                    }
                    else
                        break;
                }
            }

            #pragma unroll
            for (int inner_idx_M=0; inner_idx_M<THREADTILE_Y; ++inner_idx_M)
            {
                #pragma unroll
                for (int inner_idx_N=0; inner_idx_N<THREADTILE_X; ++inner_idx_N)
                {
                    int aRow               = aRow0 + (THREADTILE_Y*threadIdx.y) + inner_idx_M;
                    int bCol               = bCol0 + threadIdx.x + blockDim.x*inner_idx_N;
                    if ( (aRow < M) && (bCol < N) )
                        c[bCol + aRow*ldc] = alpha*acc[inner_idx_M*THREADTILE_X + inner_idx_N] + beta*c[bCol + aRow*ldc];
                }
            }

            threadtile_idx_x += BN * gridDim.x;
        }
        threadtile_idx_y += BM * gridDim.y;
    }
}


/* taken from Kernel 10: Warptiling of https://siboehm.com/articles/22/CUDA-MMM , with some little bug fix */
#ifdef PERFORMANCE
__global__ void __launch_bounds__(WARP_TILING_MAX_NUM_THREADS)
#else
__global__ void
#endif
                gemmWarptiling(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                               int M, int N, int K,
                               T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN_Warptiling / WN_Warptiling);
    const uint warpRow = warpIdx / (BN_Warptiling / WN_Warptiling);

    // size of the warp subtile
    constexpr uint WSUBM = WM_Warptiling / WMITER;
    constexpr uint WSUBN = WN_Warptiling / WNITER;

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN_Warptiling);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN_Warptiling);

    // allocate space for the current blocktile in SMEM
    extern __shared__ T_ELEM_IN shared_buffer[];
    T_ELEM_IN *a_shared_buffer = &(shared_buffer[0]);
    T_ELEM_IN *b_shared_buffer = &(shared_buffer[BM_Warptiling*BK_Warptiling]);

    // Move blocktile to beginning of a's row and b's column
    a += cRow * BM_Warptiling * K;
    b += cCol * BN_Warptiling;
    // Move C_ptr to warp's output tile
    c += (cRow * BM_Warptiling + warpRow * WM_Warptiling) * N + cCol * BN_Warptiling + warpCol * WN_Warptiling;

    // allocate thread-local cache for results in registerfile
    T_ELEM_OUT threadResults[WMITER * TM_Warptiling * WNITER * TN_Warptiling] = {0.0};
    // we cache into registers on the warptile level
    T_ELEM_IN regM[WMITER * TM_Warptiling] = {0.0};
    T_ELEM_IN regN[WNITER * TN_Warptiling] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK_Warptiling)
    {
        int thread_idx = threadIdx.x + blockDim.x*threadIdx.y;
        int aRow0      = blockIdx.y*BM_Warptiling;
        int bCol0      = blockIdx.x*BN_Warptiling;

        #pragma unroll
        for (int inner_idx=thread_idx; inner_idx<BM_Warptiling*BK_Warptiling; inner_idx+=blockDim.x*blockDim.y)
        {
            int thread_idx_x                                                            = inner_idx % BK_Warptiling;
            int thread_idx_y                                                            = inner_idx / BK_Warptiling;
            int aCol                                                                    = bkIdx + thread_idx_x;
            int aRow                                                                    = aRow0 + thread_idx_y;
            if ( (aRow < M) && (aCol < K) )
                a_shared_buffer[thread_idx_x+thread_idx_y*BK_Warptiling] = a[thread_idx_x + thread_idx_y*K];
                //a_shared_buffer[thread_idx_y+thread_idx_x*BM_Warptiling] = a[thread_idx_x + thread_idx_y*lda];
            else
                a_shared_buffer[thread_idx_x+thread_idx_y*BK_Warptiling] = 0.0;
                //a_shared_buffer[thread_idx_y+thread_idx_x*BM_Warptiling] = 0.0;
        }
        #pragma unroll
        for (int inner_idx=thread_idx; inner_idx<BK_Warptiling*BN_Warptiling; inner_idx+=blockDim.x*blockDim.y)
        {
            int thread_idx_x                                  = inner_idx % BN_Warptiling;
            int thread_idx_y                                  = inner_idx / BN_Warptiling;
            int bCol                                          = bCol0 + thread_idx_x;
            int bRow                                          = bkIdx + thread_idx_y;
            if ( (bRow < K) && (bCol < N) )
                b_shared_buffer[thread_idx_x+thread_idx_y*BN_Warptiling] = b[thread_idx_x + thread_idx_y*N];
            else
                b_shared_buffer[thread_idx_x+thread_idx_y*BN_Warptiling] = 0.0;
        }
        __syncthreads();


        for (uint dotIdx = 0; dotIdx < BK_Warptiling; ++dotIdx)
        {
          // populate registers for whole warptile
          for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
          {
              for (uint i = 0; i < TM_Warptiling; ++i)
              {
                  //regM[wSubRowIdx * TM_Warptiling + i] =
                  //    a_shared_buffer[(dotIdx * BM_Warptiling) + warpRow * WM_Warptiling + wSubRowIdx * WSUBM +
                  //                    threadRowInWarp * TM_Warptiling + i];
                  regM[wSubRowIdx * TM_Warptiling + i] =
                      a_shared_buffer[(warpRow * WM_Warptiling + wSubRowIdx * WSUBM + threadRowInWarp * TM_Warptiling + i) * BK_Warptiling + dotIdx];
              }
          }
          for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
          {
              for (uint i = 0; i < TN_Warptiling; ++i)
              {
                  regN[wSubColIdx * TN_Warptiling + i] =
                      b_shared_buffer[(dotIdx * BN_Warptiling) + warpCol * WN_Warptiling + wSubColIdx * WSUBN + threadColInWarp * TN_Warptiling + i];
              }
          }

          // execute warptile matmul
          for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
          {
              for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
              {
                  // calculate per-thread results
                  for (uint resIdxM = 0; resIdxM < TM_Warptiling; ++resIdxM)
                  {
                      for (uint resIdxN = 0; resIdxN < TN_Warptiling; ++resIdxN)
                      {
                          threadResults[(wSubRowIdx * TM_Warptiling + resIdxM) * (WNITER * TN_Warptiling) +
                                        (wSubColIdx * TN_Warptiling) + resIdxN] +=
                              (T_ELEM_OUT)regM[wSubRowIdx * TM_Warptiling + resIdxM] *
                              (T_ELEM_OUT)regN[wSubColIdx * TN_Warptiling + resIdxN];
                      }
                  }
              }
          }
      }
      a += BK_Warptiling;     // move BK_Warptiling columns to right
      b += BK_Warptiling * N; // move BK_Warptiling rows down
      __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
        {
            // move C pointer to current warp subtile
            T_ELEM_OUT *c_interim = c + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM_Warptiling; resIdxM += 1)
            {
                //for (uint resIdxN = 0; resIdxN < TN_Warptiling; resIdxN += 4)
                for (uint resIdxN = 0; resIdxN < TN_Warptiling; resIdxN+=1)
                {
                    const int i = (wSubRowIdx * TM_Warptiling + resIdxM) * (WNITER * TN_Warptiling) +
                                  wSubColIdx * TN_Warptiling + resIdxN;
                    c_interim[(threadRowInWarp * TM_Warptiling + resIdxM) * N + threadColInWarp * TN_Warptiling + resIdxN]
                            = alpha*threadResults[i] + beta * c_interim[(threadRowInWarp * TM_Warptiling + resIdxM) * N + threadColInWarp * TN_Warptiling + resIdxN];
                }
            }
        }
    }
}


/* WMMA GPU kernel, modified based on this example: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/cudaTensorCoreGemm
   For tf32 to float precision conversion, we must manually do it via inline function __float_to_tf32. Please refer to: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#alternate-floating-point . */
#define WARP_REPEAT_X 4
#define WARP_REPEAT_Y 2
__global__ void wmma_kernel(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                            int M, int N, int K,
                            int BM, int BN, int wpB_x, int wpB_y,
                            T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    // Leading dimensions. Packed with no transpositions.
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

#ifndef PERFORMANCE
    if ( lda%4 != 0 ) // lda must be multiple of 4 for data type float
    {
        assert(0);
    }
    if ( ldb%4 != 0 ) // ldb must be multiple of 4 for data type float
    {
        assert(0);
    }
    if ( ldc%4 != 0 ) // ldc must be multiple of 4 for data type float
    {
        assert(0);
    }
#endif

    const int thread_idx        = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_idx          = thread_idx / WARPSIZE;
    //const int THREADTILE_WMMA_Y = BM / blockDim.y;
    //const int THREADTILE_WMMA_X = BN / blockDim.x;
    const int total_threads    = blockDim.x * blockDim.y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, T_ELEM_IN_FRAG, wmma::row_major> a_frag[WARP_REPEAT_Y];
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, T_ELEM_IN_FRAG, wmma::row_major> b_frag[WARP_REPEAT_X];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T_ELEM_OUT> acc_frag[WARP_REPEAT_Y][WARP_REPEAT_X];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T_ELEM_OUT> c_frag;

    // Declare shared memory
    extern __shared__ T_ELEM_IN shared_buffer[];

    T_ELEM_IN *a_shared_buffer = &(shared_buffer[0]);
    T_ELEM_IN *b_shared_buffer = &(shared_buffer[BM*(warpK_stride_shared*WMMA_K+SKEW_MINE)]);

    int threadtile_idx_y = BM*blockIdx.y;     // BM = THREADTILE_WMMA_Y*blockDim.y
    while ( threadtile_idx_y < (BM * (int)((M + BM - 1) / BM)) )
    {
        int threadtile_idx_x = BN*blockIdx.x; // BN = THREADTILE_WMMA_X*blockDim.x
        while ( threadtile_idx_x < (BN * (int)((N + BN - 1) / BN)) )
        {
            #pragma unroll
            for (int idx_warp_y=0; idx_warp_y<WARP_REPEAT_Y; ++idx_warp_y)
            {
                #pragma unroll
                for (int idx_warp_x=0; idx_warp_x<WARP_REPEAT_X; ++idx_warp_x)
                {
                    wmma::fill_fragment(acc_frag[idx_warp_y][idx_warp_x], (T_ELEM_OUT)(0.0));
                }
            }

            int aRow0 = threadtile_idx_y;
            int bCol0 = threadtile_idx_x;

            for (int i=0; i<K; i+=(warpK_stride_shared*WMMA_K))
            {
                //// older version for global to shared memory copy, slower
                //__syncthreads();
                //int thread_idx_x = threadIdx.x;
                //while (thread_idx_x < (warpK_stride_shared*WMMA_K))
                //{
                //    int aCol                                                                                 = i + thread_idx_x;
                //    #pragma unroll
                //    for (int inner_idx=0; inner_idx<THREADTILE_WMMA_Y; ++inner_idx)
                //    {
                //        int aRow_shared                                                                      = THREADTILE_WMMA_Y*threadIdx.y + inner_idx;
                //        int aRow                                                                             = aRow0 + aRow_shared;
                //        if ( (aRow < M) && (aCol < K) )
                //            a_shared_buffer[thread_idx_x+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)] = wmma::__float_to_tf32(a[aCol + aRow*lda]);
                //        else
                //            a_shared_buffer[thread_idx_x+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)] = wmma::__float_to_tf32(0.0);
                //    }
                //    thread_idx_x                                                                            += blockDim.x;
                //}
                //int thread_idx_y  = threadIdx.y;
                //while (thread_idx_y < (warpK_stride_shared*WMMA_K))
                //{
                //    int bRow                                                         = i + thread_idx_y;
                //    #pragma unroll
                //    for (int inner_idx=0; inner_idx<THREADTILE_WMMA_X; ++inner_idx)
                //    {
                //        int bCol_shared                                              = threadIdx.x + blockDim.x*inner_idx;
                //        int bCol                                                     = bCol0 + bCol_shared;
                //        // in order to realize memory coalescing for threads in the same wrap, we use threadIdx.x+blockDim.x*inner_idx instead of THREADTILE_WMMA_X*threadIdx.x+inner_idx for mapping along row direction for b
                //        if ( (bRow < K) && (bCol < N) )
                //            b_shared_buffer[bCol_shared+thread_idx_y*(BN+SKEW_MINE)] = wmma::__float_to_tf32(b[bCol + bRow*ldb]);
                //        else
                //            b_shared_buffer[bCol_shared+thread_idx_y*(BN+SKEW_MINE)] = wmma::__float_to_tf32(0.0);
                //    }
                //    thread_idx_y                                                    += blockDim.y;
                //}
                //__syncthreads();
                ////

                __syncthreads();
                //// !!!!only use this when BM = BN !!!!!
                ////assert (BN == BM);
                //int idx_copy = thread_idx*copy_batch_factor;
                //int idx_copy_min      = (BM > BN) ? BN*(warpK_stride_shared*WMMA_K) : BM*(warpK_stride_shared*WMMA_K);
                //int aCol_shared, aRow_shared, aCol, aRow;
                //int bCol_shared, bRow_shared, bCol, bRow;
                //#pragma unroll
                //for (; idx_copy<idx_copy_min; idx_copy+=copy_batch_factor*total_threads)
                //{
                //    aCol_shared = idx_copy % (warpK_stride_shared*WMMA_K);
                //    aCol        = i + aCol_shared;
                //    aRow_shared = idx_copy / (warpK_stride_shared*WMMA_K);
                //    aRow        = aRow0 + aRow_shared;
                //    if ( (aRow < M) && (aCol < K) )
                //    {
                //        *((copy_batch_t*)&(a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)])) = *((copy_batch_t*)&(a[aCol + aRow*lda]));
                //        #pragma unroll
                //        for (int idx_element=0; idx_element<4; ++idx_element)
                //        {
                //            a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)+idx_element]
                //                = wmma::__float_to_tf32(a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)+idx_element]);
                //        }
                //    }
                //    bCol_shared = idx_copy % BN;
                //    bCol        = bCol0 + bCol_shared;
                //    bRow_shared = idx_copy / BN;
                //    bRow        = i + bRow_shared;
                //    if ( (bRow < K) && (bCol < N) )
                //    {
                //        *((copy_batch_t*)&(b_shared_buffer[bCol_shared+bRow_shared*(BN+SKEW_MINE)])) = *((copy_batch_t*)&(b[bCol + bRow*ldb]));
                //        #pragma unroll
                //        for (int idx_element=0; idx_element<4; ++idx_element)
                //        {
                //            b_shared_buffer[bCol_shared+bRow_shared*(BN+SKEW_MINE)+idx_element]
                //                = wmma::__float_to_tf32(b_shared_buffer[bCol_shared+bRow_shared*(BN+SKEW_MINE)+idx_element]);
                //        }
                //    }
                //}
                ////

                // ~~~~Use this for general size of (BM, BN); use 1st half of the threads to copy matrix a, and 2nd half of the threads to copy matrix b~~~~
                if (thread_idx < (total_threads)/2)
                {
                    int aCol_shared, aRow_shared, aCol, aRow;
                    #pragma unroll
                    for (int idx_copy = thread_idx*copy_batch_factor; idx_copy<BM*(warpK_stride_shared*WMMA_K); idx_copy+=copy_batch_factor*total_threads/2)
                    {
                        aCol_shared = idx_copy % (warpK_stride_shared*WMMA_K);
                        aCol        = i + aCol_shared;
                        aRow_shared = idx_copy / (warpK_stride_shared*WMMA_K);
                        aRow        = aRow0 + aRow_shared;
                        if ( (aRow < M) && (aCol < K) )
                        {
                            *((copy_batch_t*)&(a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)])) = *((copy_batch_t*)&(a[aCol + aRow*lda]));
                            #pragma unroll
                            for (int idx_element=0; idx_element<4; ++idx_element)
                            {
                                a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)+idx_element]
                                    = wmma::__float_to_tf32(a_shared_buffer[aCol_shared+aRow_shared*(warpK_stride_shared*WMMA_K+SKEW_MINE)+idx_element]);
                            }
                        }
                    }
                }
                else
                {
                    int bCol_shared, bRow_shared, bCol, bRow;
                    #pragma unroll
                    for (int idx_copy = (thread_idx-total_threads/2)*copy_batch_factor; idx_copy<(warpK_stride_shared*WMMA_K)*BN; idx_copy+=copy_batch_factor*total_threads/2)
                    {
                        bCol_shared = idx_copy % BN;
                        bCol        = bCol0 + bCol_shared;
                        bRow_shared = idx_copy / BN;
                        bRow        = i + bRow_shared;
                        if ( (bRow < K) && (bCol < N) )
                        {
                            *((copy_batch_t*)&(b_shared_buffer[bCol_shared+bRow_shared*(BN+SKEW_MINE)])) = *((copy_batch_t*)&(b[bCol + bRow*ldb]));
                            #pragma unroll
                            for (int idx_element=0; idx_element<4; ++idx_element)
                            {
                                b_shared_buffer[bCol_shared+bRow_shared*(BN+SKEW_MINE) + idx_element]
                                    = wmma::__float_to_tf32(b_shared_buffer[bCol_shared+bRow_shared*(BN+SKEW_MINE)+idx_element]);
                            }
                        }
                    }
                }
                //
                __syncthreads();
                //

                #pragma unroll
                for (int j=0; j<(warpK_stride_shared*WMMA_K); j+=WMMA_K)
                {
                    if ( (i + j) < K )
                    {
                        #pragma unroll
                        for (int idx_warp_y=0; idx_warp_y<WARP_REPEAT_Y; ++idx_warp_y)
                        {
                            wmma::load_matrix_sync(a_frag[idx_warp_y], a_shared_buffer + j + (warp_idx/wpB_x+idx_warp_y*wpB_y)*WMMA_M*(warpK_stride_shared*WMMA_K+SKEW_MINE), warpK_stride_shared*WMMA_K+SKEW_MINE);
                            #pragma unroll
                            for (int idx_warp_x=0; idx_warp_x<WARP_REPEAT_X; ++idx_warp_x)
                            {
                                if (idx_warp_y == 0)
                                {
                                    wmma::load_matrix_sync(b_frag[idx_warp_x], b_shared_buffer + (warp_idx%wpB_x+idx_warp_x*wpB_x)*WMMA_N + j*(BN+SKEW_MINE), BN+SKEW_MINE);
                                }
                                // Perform the matrix multiplication
                                wmma::mma_sync(acc_frag[idx_warp_y][idx_warp_x], a_frag[idx_warp_y], b_frag[idx_warp_x], acc_frag[idx_warp_y][idx_warp_x]);
                            }
                        }
                    }
                    else
                        break;
                }
            }

            #pragma unroll
            for (int idx_warp_y=0; idx_warp_y<WARP_REPEAT_Y; ++idx_warp_y)
            {
                #pragma unroll
                for (int idx_warp_x=0; idx_warp_x<WARP_REPEAT_X; ++idx_warp_x)
                {
                    int cRow = aRow0 + (warp_idx/wpB_x + idx_warp_y*wpB_y)*WMMA_M;
                    int cCol = bCol0 + (warp_idx%wpB_x + idx_warp_x*wpB_x)*WMMA_N;
                    if (cRow < M && cCol < N)
                    {
                        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

                        for(int i=0; i < c_frag.num_elements; i++)
                        {
                            c_frag.x[i] = alpha * acc_frag[idx_warp_y][idx_warp_x].x[i] + beta * c_frag.x[i];
                        }

                        // Store the output
                        wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
                    }
                }
            }

            threadtile_idx_x += BN * gridDim.x;
        }
        threadtile_idx_y += BM * gridDim.y;
    }
}
//// older version, slower
//{
//    // Leading dimensions. Packed with no transpositions.
//    const int lda = K;
//    const int ldb = N;
//    const int ldc = N;
//
//#ifndef PERFORMANCE
//    if ( lda%4 != 0 ) // lda must be multiple of 4 for data type float
//    {
//        assert(0);
//    }
//    if ( ldb%4 != 0 ) // ldb must be multiple of 4 for data type float
//    {
//        assert(0);
//    }
//    if ( ldc%4 != 0 ) // ldc must be multiple of 4 for data type float
//    {
//        assert(0);
//    }
//#endif
//
//    const int  warpSize_x    = (blockDim.x < WARPSIZE) ? blockDim.x : WARPSIZE;
//    const int  warpSize_y    = WARPSIZE / warpSize_x;
//    const long total_warp_x  = (N/WMMA_N);
//    const long total_warp_y  = (M/WMMA_M);
//    const long array_idx_x   = blockIdx.x * blockDim.x + threadIdx.x;
//    const long array_idx_y   = blockIdx.y * blockDim.y + threadIdx.y;
//    const int  warp_stride_x = (blockDim.x * gridDim.x) / warpSize_x;
//    const int  warp_stride_y = (blockDim.y * gridDim.y) / warpSize_y;
//
//    // Declare the fragments
//    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, T_ELEM_IN_FRAG, wmma::row_major> a_frag;
//    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, T_ELEM_IN_FRAG, wmma::row_major> b_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T_ELEM_OUT> acc_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T_ELEM_OUT> c_frag;
//
//    // Declare shared memory
//    extern __shared__ T_ELEM_IN shared_buffer[];
//
//    T_ELEM_IN *a_shared_buffer = &(shared_buffer[0]);
//    T_ELEM_IN *b_shared_buffer = &(shared_buffer[(WMMA_M*wpB_y)*(warpK_stride_shared*WMMA_K)]);
//
//    // Tile using a 2D grid
//    int warpM  = array_idx_y / warpSize_y;
//    int warpM0 = blockIdx.y * blockDim.y / warpSize_y;
//    while (warpM < total_warp_y)
//    {
//        int warpN  = array_idx_x / warpSize_x;
//        int warpN0 = blockIdx.x * blockDim.x / warpSize_x;
//        while (warpN < total_warp_x)
//        {
//            wmma::fill_fragment(acc_frag, (T_ELEM_OUT)(0.0));
//
//            int aRow     = warpM * WMMA_M;
//            int bCol     = warpN * WMMA_N;
//            for (int i=0; i<K; i+=(warpK_stride_shared*WMMA_K))
//            {
//                int aCol = i;
//                int bRow = i;
//
//                __syncthreads();
//                int thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
//                while ( thread_idx < (WMMA_M*wpB_y)*(warpK_stride_shared*WMMA_K) )
//                {
//                   int aRow_cache = warpM0 * WMMA_M + thread_idx/(warpK_stride_shared*WMMA_K);
//                   int aCol_cache = aCol + thread_idx%(warpK_stride_shared*WMMA_K);
//
//                   if ( (aRow_cache < M) && (aCol_cache < K) )
//                       a_shared_buffer[thread_idx] = wmma::__float_to_tf32(a[ aCol_cache + aRow_cache * lda ]);
//                   else
//                       a_shared_buffer[thread_idx] = wmma::__float_to_tf32(0.0);
//                   thread_idx                     += blockDim.x * blockDim.y;
//                }
//
//                thread_idx     = threadIdx.x + blockDim.x * threadIdx.y;
//                while ( thread_idx < (WMMA_N*wpB_x)*(warpK_stride_shared*WMMA_K) )
//                {
//                   int bRow_cache = bRow + thread_idx/(WMMA_N*wpB_x);
//                   int bCol_cache = warpN0 * WMMA_N + thread_idx%(WMMA_N*wpB_x);
//
//                   if ( (bRow_cache < K) && (bCol_cache < N) )
//                       b_shared_buffer[thread_idx] = wmma::__float_to_tf32(b[ bCol_cache + bRow_cache * ldb ]);
//                   else
//                       b_shared_buffer[thread_idx] = wmma::__float_to_tf32(0.0);
//                   thread_idx                     += blockDim.x * blockDim.y;
//                }
//                __syncthreads();
//
//                // Bounds checking
//                if ( (aRow < M) && (bCol < N) )
//                {
//                    #pragma unroll
//                    // Loop over the K-dimension
//                    for (int j=0; j<(warpK_stride_shared*WMMA_K); j+=WMMA_K)
//                    {
//
//                        if ( (i + j) < K )
//                        {
//                            aCol       = i + j;
//                            bRow       = i + j;
//                            thread_idx   = threadIdx.x + blockDim.x * threadIdx.y;
//                            int warp_idx = thread_idx / WARPSIZE;
//#ifndef PERFORMANCE
//                            //if ( j + (warp_idx/wpB_x)*WMMA_M*(warpK_stride_shared*WMMA_K) >= (WMMA_M*wpB_y)*(warpK_stride_shared*WMMA_K) )
//                            //{
//                            //    assert(0);
//                            //}
//                            //if ( j + (warp_idx%wpB_x)*WMMA_N*(warpK_stride_shared*WMMA_K) >= (WMMA_N*wpB_x)*(warpK_stride_shared*WMMA_K) )
//                            //{
//                            //    assert(0);
//                            //}
//#endif
//                            // Load the inputs
//                            wmma::load_matrix_sync(a_frag, a_shared_buffer + j + (warp_idx/wpB_x)*WMMA_M*(warpK_stride_shared*WMMA_K), warpK_stride_shared*WMMA_K);
//                            wmma::load_matrix_sync(b_frag, b_shared_buffer + (warp_idx%wpB_x)*WMMA_N + j*(WMMA_N*wpB_x), WMMA_N*wpB_x);
//                            // Perform the matrix multiplication
//                            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//                        }
//                        else
//                            break;
//                    }
//                }
//            }
//
//            // Load in current value of c, scale by beta, and add to result scaled by alpha
//            int cRow = warpM * WMMA_M;
//            int cCol = warpN * WMMA_N;
//
//            if (cRow < M && cCol < N)
//            {
//                wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);
//
//                for(int i=0; i < c_frag.num_elements; i++)
//                {
//                    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
//                }
//
//                // Store the output
//                wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
//            }
//            warpN  += warp_stride_x;
//            warpN0 += warp_stride_x;
//        }
//        warpM  += warp_stride_y;
//        warpM0 += warp_stride_y;
//    }
//}


/* Main function */
int main(int argc, char* argv[])
{
    int   M, K, N;
    int   tpB_x, tpB_y;
    int   bpG_x, bpG_y;
    int   bpG_x_GEMM, bpG_y_GEMM;
    int   bpG_x_WMMA, bpG_y_WMMA;
    int   N_thread;

    T_ELEM_IN  *matrix_a_device, *matrix_b_device;
    T_ELEM_OUT *matrix_c_device;
    std::vector<T_ELEM_IN>  matrix_a_host, matrix_b_host, matrix_b_host_col_major;
    std::vector<T_ELEM_OUT> matrix_c_host, matrix_c_buffer, matrix_c_host_backup; // when beta != 0, matrix_c_host will be updated by the new answer, so we make a backup matrix_c_host_backup, and used matrix_c_host to save the new answer

    printf("Test matrix-matrix multiplication by Warp Matrix Multiply Accumulate(WMMA).\n");
    printf("First matrix has dimension MXK and second matrix has dimension KXN .\n");

    printf("Enter the value for M:\n");
    scanf("%d", &M);
    if ( M <= 0 )
    {
        printf("Error!! M(%d) must be positive!! Exit!!\n", M);
        return EXIT_FAILURE;
    }
    if ( M % WMMA_M != 0 )
    {
        printf("Error!! M(%d) must be multiple of WMMA_M(%d)!! Exit!!\n", M, WMMA_M);
        return EXIT_FAILURE;
    }

    printf("Enter the value for K:\n");
    scanf("%d", &K);
    if ( K <= 0 )
    {
        printf("Error!! K(%d) must be positive!! Exit!!\n", K);
        return EXIT_FAILURE;
    }
    if ( K % WMMA_K != 0 )
    {
        printf("Error!! K(%d) must be multiple of WMMA_K(%d)!! Exit!!\n", K, WMMA_K);
        return EXIT_FAILURE;
    }

    printf("Enter the value for N:\n");
    scanf("%d", &N);
    if ( N <= 0 )
    {
        printf("Error!! N(%d) must be positive!! Exit!!\n", N);
        return EXIT_FAILURE;
    }
    if ( N % WMMA_N != 0 )
    {
        printf("Error!! N(%d) must be multiple of WMMA_N(%d)!! Exit!!\n", N, WMMA_N);
        return EXIT_FAILURE;
    }

    printf("Matrix demensions are: M = %d , K = %d , N = %d .\n", M, K, N);
    printf("---------------------------------------------------------------\n");

    printf("Set 2D threads per block (tpB_x, tpB_y) .\n");
    printf("Enter the value for tpB_x:\n");
    scanf("%d", &tpB_x);
    if ( tpB_x <= 0 )
    {
        printf("Error!! tpB_x(%d) must be positive!! Exit!!\n", tpB_x);
        return EXIT_FAILURE;
    }
    if ( (tpB_x % WARPSIZE != 0) && (WARPSIZE % tpB_x != 0) )
    {
        printf("Error!! tpB_x(%d) must be multiple of warp size(%d) or warp size(%d) must be multiple of tpB_x(%d)!! Exit!!\n", tpB_x, WARPSIZE, WARPSIZE, tpB_x);
        return EXIT_FAILURE;
    }

    printf("Enter the value for tpB_y:\n");
    scanf("%d", &tpB_y);
    if ( tpB_y <= 0 )
    {
        printf("Error!! tpB_y(%d) must be positive!! Exit!!\n", tpB_y);
        return EXIT_FAILURE;
    }
    if ( (tpB_x * tpB_y) < WARPSIZE )
    {
        printf("Error!! total threads per block(%d) must >= warp size(%d)!! Exit!!\n", tpB_x * tpB_y, WARPSIZE);
        return EXIT_FAILURE;
    }

    printf("2D threads per grid is: (%d,%d) .\n", tpB_x, tpB_y);
    printf("---------------------------------------------------------------\n");

    printf("Set 2D blocks per grid (bpG_x, bpG_y) .\n");
    printf("Enter the value for bpG_x: (<=0 will set bpG_x automatically such that:\n                            bpG_x = (N+tpB_x-1)/tpb_x for hand-carved GEMM\n                            N/WMMA_N = (tpB_x*bpG_x)/warpSize_x) for WMMA)\n");
    scanf("%d", &bpG_x);
    if ( bpG_x <= 0 )
    {
        printf("bpG_x will be automatically.\n");
    }

    printf("Enter the value for bpG_y: (<=0 will set bpG_y automatically such that:\n                            bpG_y = (M+tpG_y-1)/tpb_y for hand-carved GEMM\n                            M/WMMA_M = (tpB_y*bpG_y)/warpSize_y) for WMMA)\n");
    scanf("%d", &bpG_y);
    if ( bpG_y <= 0 )
    {
        printf("bpG_y will be automatically.\n");
    }

    printf("---------------------------------------------------------------\n");

    printf("Set number of OpenMP thread(s) for CPU computation.\n");
    printf("Enter number of thread(s): (<=0 will use the maixmum number of threads)\n");
    scanf("%d", &N_thread);
    if ( N_thread <= 0 )
    {
        N_thread = omp_get_max_threads();
    }

    printf("%d thread(s) will be use for CPU computation .\n", N_thread);
    printf("---------------------------------------------------------------\n");
    std::vector<std::mt19937>                    generator(N_thread);
    std::vector<std::normal_distribution<float>> norm_dist(N_thread);

    for (int idx_thread=0; idx_thread<N_thread; ++idx_thread)
    {
        generator[idx_thread] = std::mt19937(seed + (long)(idx_thread));
        norm_dist[idx_thread] = std::normal_distribution<float>(0.0f, 1.0f);
    }

    /* memory allocation in each matrices:
        matrix_a: row major
        matrix_b: row major
        matrix_c: row major
    */

    matrix_a_host.resize((size_t)M*(size_t)K);
    matrix_b_host.resize((size_t)K*(size_t)N);
    matrix_b_host_col_major.resize((size_t)K*(size_t)N);
    matrix_c_host.resize((size_t)M*(size_t)N);
    matrix_c_buffer.resize((size_t)M*(size_t)N);
    matrix_c_host_backup.resize((size_t)M*(size_t)N);

    omp_set_num_threads(N_thread);
    dim3 tpB(tpB_x, tpB_y);
    cublasHandle_t handle;
    CUDA_CHECK_ERROR(cudaSetDevice(0));

#ifdef TIMING_FLAG
    float       elapsed_time;
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&t_start));
    CUDA_CHECK_ERROR(cudaEventCreate(&t_stop));
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    // initialization
    #pragma omp parallel for shared(matrix_a_host)
    for (long idx_array=0L; idx_array<(long)M*(long)K; ++idx_array)
    {
        int idx_thread           = omp_get_thread_num();
        matrix_a_host[idx_array] = norm_dist[idx_thread](generator[idx_thread]);
    }
    // notice that matrix_b_host_col_major is in colum major (to accelerate OpenMP computation), while matrix_b_host is in row major
    #pragma omp parallel for shared(matrix_b_host, matrix_b_host_col_major)
    for (long idx_array=0L; idx_array<(long)K*(long)N; ++idx_array)
    {
        int idx_thread                               = omp_get_thread_num();
        long idx_array_col_major                     = (long)K*(idx_array%(long)N) + (idx_array/(long)N);
        matrix_b_host_col_major[idx_array_col_major] = norm_dist[idx_thread](generator[idx_thread]);
        matrix_b_host[idx_array]                     = matrix_b_host_col_major[idx_array_col_major];
    }
    #pragma omp parallel for shared(matrix_c_host, matrix_c_host_backup)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        int idx_thread                  = omp_get_thread_num();
        matrix_c_host[idx_array]        = norm_dist[idx_thread](generator[idx_thread]);
        matrix_c_host_backup[idx_array] = matrix_c_host[idx_array];
    }
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Initialization time = %.4f s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* CPU OpenMP computation *////
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    // start CPU computation
    float sum;
    long  idx_array_a, idx_array_b, idx_array_c;
    #pragma omp parallel for collapse(2) private(sum, idx_array_a, idx_array_b, idx_array_c) shared(matrix_a_host, matrix_b_host_col_major, matrix_c_host)
    for (int idx_row_c=0; idx_row_c<M; ++idx_row_c)
    {
        for (int idx_col_c=0; idx_col_c<N; ++idx_col_c)
        {
            sum                        = 0.0f;
            idx_array_c                = (long)N * (long)idx_row_c + idx_col_c;
            #pragma omp simd
            for (int idx_accum=0; idx_accum<K; ++idx_accum)
            {
                idx_array_a            = (long)K * (long)idx_row_c + idx_accum;
                idx_array_b            = (long)K * (long)idx_col_c + idx_accum;
                sum                   += matrix_a_host[idx_array_a] * matrix_b_host_col_major[idx_array_b];
            }
            matrix_c_host[idx_array_c] = alpha * sum + beta * matrix_c_host[idx_array_c];
        }
    }
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("OpenMP matrix-matrix multiplication time = %.4f s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    //if ((long)TEST_IDX >= (long)M*(long)N)
    //{
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_host[TEST_IDX] << std::endl;
    //}

    ////* End *////

    ////* GPU memory operations*////
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matrix_a_device, (size_t)M*(size_t)K*sizeof(T_ELEM_IN)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matrix_b_device, (size_t)K*(size_t)N*sizeof(T_ELEM_IN)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT)));

    CUDA_CHECK_ERROR(cudaMemcpy(matrix_a_device, matrix_a_host.data(),        (size_t)M*(size_t)K*sizeof(T_ELEM_IN),  cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_b_device, matrix_b_host.data(),        (size_t)K*(size_t)N*sizeof(T_ELEM_IN),  cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory allocation, initialization and H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

    ////* End *////

    ////* GPU hand-carved GEMM computation *////
    if (bpG_x <= 0)
    {
        bpG_x_GEMM = (N + tpB_x - 1) / tpB_x;
    }
    else
        bpG_x_GEMM = bpG_x;
    if (bpG_y <= 0)
    {
        bpG_y_GEMM = (M + tpB_y - 1) / tpB_y;
    }
    else
        bpG_y_GEMM = bpG_y;
    dim3 bpG_gemm(bpG_x_GEMM, bpG_y_GEMM);
    printf("2D blocks per grid is: (%d,%d) for hand-carved GEMM.\n", bpG_x_GEMM, bpG_y_GEMM);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    cudaDeviceProp deviceProp;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    size_t shmem_size_max = deviceProp.sharedMemPerBlockOptin;
    size_t shmem_size     = ( tpB_y*(warpK_stride_shared * WMMA_K) + tpB_x*(warpK_stride_shared * WMMA_K) ) *sizeof(T_ELEM_IN);
    if ( shmem_size > shmem_size_max )
    {
        printf("Shared memory size (%lu) > Maximum shared Memory size (%lu) !! Exit!!\n", shmem_size, shmem_size_max);
        return EXIT_FAILURE;
    }
    CUDA_CHECK_ERROR(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_max));
    gemm_kernel<<<bpG_gemm, tpB, shmem_size, 0>>>(matrix_a_device, matrix_b_device, matrix_c_device,
                                                  M, N, K,
                                                  (T_ELEM_OUT)alpha, (T_ELEM_OUT)beta);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("CUDA hand-carved matrix-matrix multiplication time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    //if ((long)TEST_IDX >= (long)M*(long)N)
    //{
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_host[TEST_IDX] << std::endl;
    //}

    double diff = 0.0;
    double norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("Hand-carved GEMM relative L2 error between matrix_c_buffer and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

    ////* GPU hand-carved GEMM_1DTiling computation *////
    const int BM_1DTiling = THREADTILE * tpB_y;
    if (bpG_x <= 0)
    {
        bpG_x_GEMM = (N + tpB_x - 1) / tpB_x;
    }
    else
        bpG_x_GEMM = bpG_x;
    if (bpG_y <= 0)
    {
        bpG_y_GEMM = (M + BM_1DTiling - 1) / BM_1DTiling;
    }
    else
        bpG_y_GEMM = bpG_y;
    dim3 bpG_gemm_1DTiling(bpG_x_GEMM, bpG_y_GEMM);
    printf("2D blocks per grid is: (%d,%d) for hand-carved GEMM_1DTiling.\n", bpG_x_GEMM, bpG_y_GEMM);
    printf("THREADTILE = %d\n", THREADTILE);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    shmem_size = ( BM_1DTiling*(warpK_stride_shared * WMMA_K) + tpB_x*(warpK_stride_shared * WMMA_K) ) *sizeof(T_ELEM_IN);
    if ( shmem_size > shmem_size_max )
    {
        printf("Shared memory size (%lu) > Maximum shared Memory size (%lu) !! Exit!!\n", shmem_size, shmem_size_max);
        return EXIT_FAILURE;
    }
    CUDA_CHECK_ERROR(cudaFuncSetAttribute(gemm_kernel_1DTiling, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_max));
    gemm_kernel_1DTiling<<<bpG_gemm_1DTiling, tpB, shmem_size, 0>>>(matrix_a_device, matrix_b_device, matrix_c_device,
                                                                    M, N, K, BM_1DTiling,
                                                                    (T_ELEM_OUT)alpha, (T_ELEM_OUT)beta);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("CUDA hand-carved matrix-matrix multiplication with 1D tiling time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    //if ((long)TEST_IDX >= (long)M*(long)N)
    //{
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_host[TEST_IDX] << std::endl;
    //}

    diff = 0.0;
    norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("Hand-carved GEMM_1DTiling relative L2 error between matrix_c_buffer and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

    ////* GPU hand-carved GEMM_2DTiling computation *////
    const int BN_2DTiling = THREADTILE_X * tpB_x;
    const int BM_2DTiling = THREADTILE_Y * tpB_y;
    if (bpG_x <= 0)
    {
        bpG_x_GEMM = (N + BN_2DTiling - 1) / BN_2DTiling;
    }
    else
        bpG_x_GEMM = bpG_x;
    if (bpG_y <= 0)
    {
        bpG_y_GEMM = (M + BM_2DTiling - 1) / BM_2DTiling;
    }
    else
        bpG_y_GEMM = bpG_y;
    dim3 bpG_gemm_2DTiling(bpG_x_GEMM, bpG_y_GEMM);
    printf("2D blocks per grid is: (%d,%d) for hand-carved GEMM_2DTiling.\n", bpG_x_GEMM, bpG_y_GEMM);
    printf("THREADTILE_X = %d , THREADTILE_Y = %d\n", THREADTILE_X, THREADTILE_Y);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    shmem_size = ( BM_2DTiling*(warpK_stride_shared * WMMA_K) + BN_2DTiling*(warpK_stride_shared * WMMA_K) ) *sizeof(T_ELEM_IN);
    if ( shmem_size > shmem_size_max )
    {
        printf("Shared memory size (%lu) > Maximum shared Memory size (%lu) !! Exit!!\n", shmem_size, shmem_size_max);
        return EXIT_FAILURE;
    }
    CUDA_CHECK_ERROR(cudaFuncSetAttribute(gemm_kernel_2DTiling, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_max));
    gemm_kernel_2DTiling<<<bpG_gemm_2DTiling, tpB, shmem_size, 0>>>(matrix_a_device, matrix_b_device, matrix_c_device,
                                                                    M, N, K, BM_2DTiling, BN_2DTiling,
                                                                    (T_ELEM_OUT)alpha, (T_ELEM_OUT)beta);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("CUDA hand-carved matrix-matrix multiplication with 2D tiling time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    //if ((long)TEST_IDX >= (long)M*(long)N)
    //{
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_host[TEST_IDX] << std::endl;
    //}

    diff = 0.0;
    norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("Hand-carved GEMM_2DTiling relative L2 error between matrix_c_buffer and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

    ////* GPU hand-carved GEMM_WarpTiling computation *////
    if ( M % BM_Warptiling != 0 ) { printf("M (%d) must be divisible by BM_Warptiling (%d) !! Exit!!\n", M, BM_Warptiling); return EXIT_FAILURE; }
    if ( K % BK_Warptiling != 0 ) { printf("K (%d) must be divisible by BK_Warptiling (%d) !! Exit!!\n", K, BK_Warptiling); return EXIT_FAILURE; }
    if ( N % BN_Warptiling != 0 ) { printf("N (%d) must be divisible by BN_Warptiling (%d) !! Exit!!\n", N, BN_Warptiling); return EXIT_FAILURE; }
    if (WARP_TILING_MAX_NUM_THREADS / WARPSIZE != (BM_Warptiling/WM_Warptiling) * (BN_Warptiling/WN_Warptiling))
    {
        printf("WARP_TILING_MAX_NUM_THREADS / WARPSIZE (%d) != (BM_Warptiling/WM_Warptiling) * (BN_Warptiling/WN_Warptiling) (%d) !! Exit!!\n", WARP_TILING_MAX_NUM_THREADS / WARPSIZE, (BM_Warptiling/WM_Warptiling) * (BN_Warptiling/WN_Warptiling));
        return EXIT_FAILURE;
    }
    if (TM_Warptiling * TN_Warptiling * WARPSIZE != (WM_Warptiling/WMITER) * (WN_Warptiling/WNITER))
    {
        printf("TM_Warptiling * TN_Warptiling * WARPSIZE (%d) != (WM_Warptiling/WMITER) * (WN_Warptiling/WNITER) (%d) !! Exit!!\n", TM_Warptiling * TN_Warptiling * WARPSIZE, (WM_Warptiling/WMITER) * (WN_Warptiling/WNITER));
        return EXIT_FAILURE;
    }
    bpG_x_GEMM = (N + BN_Warptiling - 1) / BN_Warptiling;
    bpG_y_GEMM = (M + BM_Warptiling - 1) / BM_Warptiling;
    dim3 bpG_gemm_WarpTiling(bpG_x_GEMM, bpG_y_GEMM);
    dim3 tpB_gemm_WarpTiling(WARP_TILING_MAX_NUM_THREADS, 1);
    printf("2D blocks per grid is: (%d,%d) for hand-carved GEMM_WarpTiling.\n", bpG_x_GEMM, bpG_y_GEMM);
    printf("BM_Warptiling = %d , BK_Warptiling = %d , BN_Warptiling = %d\n", BM_Warptiling, BK_Warptiling, BN_Warptiling);
    printf("WM_Warptiling = %d , WN_Warptiling = %d\n", WM_Warptiling, WN_Warptiling);
    printf("TM_Warptiling = %d , TN_Warptiling = %d\n", TM_Warptiling, TN_Warptiling);
    printf("WMITER = %d , WNITER = %d\n", WMITER, WNITER);
    printf("WARP_TILING_MAX_NUM_THREADS = %d\n", WARP_TILING_MAX_NUM_THREADS);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    shmem_size = ( BM_Warptiling*BK_Warptiling + BK_Warptiling*BN_Warptiling ) *sizeof(T_ELEM_IN);
    if ( shmem_size > shmem_size_max )
    {
        printf("Shared memory size (%lu) > Maximum shared Memory size (%lu) !! Exit!!\n", shmem_size, shmem_size_max);
        return EXIT_FAILURE;
    }
    CUDA_CHECK_ERROR(cudaFuncSetAttribute(gemmWarptiling, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_max));
    gemmWarptiling<<<bpG_gemm_WarpTiling, tpB_gemm_WarpTiling, shmem_size, 0>>>(matrix_a_device, matrix_b_device, matrix_c_device,
                                                                                M, N, K,
                                                                                (T_ELEM_OUT)alpha, (T_ELEM_OUT)beta);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("CUDA hand-carved matrix-matrix multiplication with warp-tiling time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    //if ((long)TEST_IDX >= (long)M*(long)N)
    //{
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_host[TEST_IDX] << std::endl;
    //}

    diff = 0.0;
    norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("Hand-carved GEMM_WarpTiling relative L2 error between matrix_c_buffer and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

    ////* GPU WMMA computation *////
    //const int warpSize_x = (tpB_x < WARPSIZE) ? tpB_x : WARPSIZE;
    const int warpSize_x = std::min(std::min(tpB_x, WARPSIZE), WMMA_N); // to constraint THREADTILE_WMMA_Y >= 1 when doing WMMA, such that each threads in a warp can handle at least one matrix element along column direction
    const int warpSize_y = WARPSIZE / warpSize_x;
    if ( warpSize_x > tpB_x )
    {
        printf("warpSize_x (%d) cannot be greater than tpB_x (%d), otherwise the threads along column-direction will be insufficient!! Exit!!\n", warpSize_x, tpB_x);
        return EXIT_FAILURE;
    }
    if ( warpSize_y > tpB_y )
    {
        printf("warpSize_y (%d) cannot be greater than tpB_y (%d), otherwise the threads along row-direction will be insufficient!! Exit!!\n", warpSize_y, tpB_y);
        return EXIT_FAILURE;
    }

    int wpB_x   = (tpB_x + warpSize_x - 1)/warpSize_x;
    int wpB_y   = (tpB_y + warpSize_y - 1)/warpSize_y;
    int BM_WMMA = WARP_REPEAT_Y * WMMA_M * wpB_y;
    int BN_WMMA = WARP_REPEAT_X * WMMA_N * wpB_x;
    printf("Warp size is (%d,%d) for WMMA.\n", warpSize_x, warpSize_y);
    printf("Warp per block is (%d,%d) for WMMA.\n", wpB_x, wpB_y);
    printf("BM_WMMA = %d ; BN_WMMA = %d\n", BM_WMMA, BN_WMMA);
    if ( BM_WMMA%tpB_y != 0 )
    {
        printf("BM_WMMA (%d) must be must be divisible by tpB_y (%d) !! Exit!!\n", BM_WMMA, tpB_y);
        return EXIT_FAILURE;
    }
    if ( BN_WMMA%tpB_x != 0 )
    {
        printf("BN_WMMA (%d) must be must be divisible by tpB_x (%d) !! Exit!!\n", BN_WMMA, tpB_x);
        return EXIT_FAILURE;
    };
    if (bpG_x <= 0)
    {
        bpG_x_WMMA = (N/WMMA_N * warpSize_x + tpB_x - 1) / tpB_x;
    }
    else
        bpG_x_WMMA = bpG_x;
    if (bpG_y <= 0)
    {
        bpG_y_WMMA = (M/WMMA_M * warpSize_y + tpB_y - 1) / tpB_y;
    }
    else
        bpG_y_WMMA = bpG_y;
    if ( ((warpK_stride_shared*WMMA_K)*(BM_WMMA/tpB_y)) % copy_batch_factor != 0 )
        {
            printf("BK_WMMA*THREADTILE_WMMA_Y (%d) must be must be divisible by copy_batch_factor (%d) for vectorization!! Exit!!\n", (warpK_stride_shared*WMMA_K)*(BM_WMMA/tpB_y), copy_batch_factor);
            return EXIT_FAILURE;
        };
    if ( ((warpK_stride_shared*WMMA_K)*(BN_WMMA/tpB_x)) % copy_batch_factor != 0 )
    {
        printf("BK_WMMA*THREADTILE_WMMA_X (%d) must be must be divisible by copy_batch_factor (%d) for vectorization!! Exit!!\n", (warpK_stride_shared*WMMA_K)*(BN_WMMA/tpB_x), copy_batch_factor);
        return EXIT_FAILURE;
    };
    dim3 bpG_WMMA(bpG_x_WMMA, bpG_y_WMMA);
    printf("2D blocks per grid is: (%d,%d) for WMMA.\n", bpG_x_WMMA, bpG_y_WMMA);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    shmem_size = (BM_WMMA*(warpK_stride_shared*WMMA_K+SKEW_MINE) + (warpK_stride_shared*WMMA_K)*(BN_WMMA+SKEW_MINE)) * sizeof(T_ELEM_IN);
    if ( shmem_size > shmem_size_max )
    {
        printf("Shared memory size (%lu) > Maximum shared Memory size (%lu) !! Exit!!\n", shmem_size, shmem_size_max);
        return EXIT_FAILURE;
    }
    CUDA_CHECK_ERROR(cudaFuncSetAttribute(wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_max));
    wmma_kernel<<<bpG_WMMA, tpB, shmem_size, 0>>>(matrix_a_device, matrix_b_device, matrix_c_device,
                                                  M, N, K,
                                                  BM_WMMA, BN_WMMA, wpB_x, wpB_y,
                                                  (T_ELEM_OUT)alpha, (T_ELEM_OUT)beta);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("CUDA Warp matrix-matrix multiplication time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    //if ((long)TEST_IDX >= (long)M*(long)N)
    //{
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_host[TEST_IDX] << std::endl;
    //}

    diff = 0.0;
    norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("WMMA relative L2 error between matrix_c_host and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

    ////* GPU cuBLAS computation *////
    /* Since cuBLAS use column major convention, instead of computing C = A*B + C, where A, B and C are all in row major, we compute C^T = B^T*A^T + C^T for cuBLAS, so:
        * when A^T is represented in cuBLAS convention as KXM column major manner, the memory packing will be identical to our design (row major MXK matrix)
        * when B^T is represented in cuBLAS convention as NXK column major manner, the memory packing will be identical to our design (row major MXK matrix)
        * when C^T is represented in cuBLAS convention as NXM column major manner, the memory packing will be identical to our design (row major MXN matrix)
    */
    CUBLAS_CHECK_ERROR(cublasCreate(&handle));
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // K, N, lda, ldb, ldc and are all multiples of 4 for float precision in our case for GemmEx:
    if ( K%4 != 0 )
    {
        assert(0);
    }
    if ( N%4 != 0 )
    {
        assert(0);
    }
    if ( lda%4 != 0 )
    {
        assert(0);
    }
    if ( ldb%4 != 0 )
    {
        assert(0);
    }
    if ( ldc%4 != 0 )
    {
        assert(0);
    }

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    // Set the math mode to allow cuBLAS to use Tensor Cores:
    CUBLAS_CHECK_ERROR(cublasSetMathMode(handle, cublasMath_t(CUBLAS_TF32_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));

    // Invoke the GEMM, ensuring k, n(since now our B is the A in the example), ldb, lda, and ldc are all multiples of 4 for float precision, i.e. 16 bytes
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasOperation_t trans_a = CUBLAS_OP_N;
    CUBLAS_CHECK_ERROR(cublasGemmEx(handle, trans_b, trans_a,
                                    N,               M,               K,   &alpha,
                                    matrix_b_device, CUBLAS_ELEM_IN,  ldb,
                                    matrix_a_device, CUBLAS_ELEM_IN,  lda,
                                    &beta,
                                    matrix_c_device, CUBLAS_ELEM_OUT, ldc,
                                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUBLAS_CHECK_ERROR(cublasDestroy(handle));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("cuBLAS GemmEx time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    diff = 0.0;
    norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("cuBLAS GemmEx relative L2 error between matrix_c_host and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////


    ////* GPU cuBLAS computation *////
    /* Since cuBLAS use column major convention, instead of computing C = A*B + C, where A, B and C are all in row major, we compute C^T = B^T*A^T + C^T for cuBLAS, so:
        * when A^T is represented in cuBLAS convention as KXM column major manner, the memory packing will be identical to our design (row major MXK matrix)
        * when B^T is represented in cuBLAS convention as NXK column major manner, the memory packing will be identical to our design (row major MXK matrix)
        * when C^T is represented in cuBLAS convention as NXM column major manner, the memory packing will be identical to our design (row major MXN matrix)
    */
    CUBLAS_CHECK_ERROR(cublasCreate(&handle));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    // reset matrix_a_device, matrix_b_device back to float precision
    CUDA_CHECK_ERROR(cudaFree(matrix_a_device));
    CUDA_CHECK_ERROR(cudaFree(matrix_b_device));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matrix_a_device,                     (size_t)M*(size_t)K*sizeof(T_ELEM_IN)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matrix_b_device,                     (size_t)K*(size_t)N*sizeof(T_ELEM_IN)));
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_a_device, matrix_a_host.data(),        (size_t)M*(size_t)K*sizeof(T_ELEM_IN),  cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_b_device, matrix_b_host.data(),        (size_t)K*(size_t)N*sizeof(T_ELEM_IN),  cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_device, matrix_c_host_backup.data(), (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyHostToDevice));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory allocation, initialization and H2D copy time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    // Set the math mode to disallow cuBLAS to use Tensor Cores:
    CUBLAS_CHECK_ERROR(cublasSetMathMode(handle, cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));

    trans_b = CUBLAS_OP_N;
    trans_a = CUBLAS_OP_N;
    CUBLAS_CHECK_ERROR(cublasSgemm(handle, trans_b, trans_a,
                                   N,               M,               K,   &alpha,
                                   matrix_b_device, ldb,
                                   matrix_a_device, lda,
                                   &beta,
                                   matrix_c_device, ldc));

    CUBLAS_CHECK_ERROR(cublasDestroy(handle));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("cuBLAS Sgemm time = %.4f s\n", elapsed_time/ms_to_sec);
#endif

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_buffer.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("GPU memory D2H copy time = %.4e s\n", elapsed_time/ms_to_sec);
#endif

    diff = 0.0;
    norm = 0.0;
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_start,0));
#endif
    #pragma omp parallel for shared(matrix_c_host, matrix_c_buffer) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_host[idx_array]-matrix_c_buffer[idx_array]), 2.0);
        norm += pow((double)(matrix_c_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("cuBLAS Sgemm relative L2 error between matrix_c_host and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    // free GPU memory
    CUDA_CHECK_ERROR(cudaFree(matrix_a_device));
    CUDA_CHECK_ERROR(cudaFree(matrix_b_device));
    CUDA_CHECK_ERROR(cudaFree(matrix_c_device));

#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventDestroy(t_start));
    CUDA_CHECK_ERROR(cudaEventDestroy(t_stop));
#endif

    CUDA_CHECK_ERROR(cudaDeviceReset());

    return EXIT_SUCCESS;
}
