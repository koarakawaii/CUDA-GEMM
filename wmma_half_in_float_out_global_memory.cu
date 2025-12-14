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

// Available data format with corresponding WMMA_* are shown in this table: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes
#define T_ELEM_IN  half
#define T_ELEM_OUT float
#define CUBLAS_ELEM_IN  CUDA_R_16F
#define CUBLAS_ELEM_OUT CUDA_R_32F
#define TEST_IDX   2097152
#define WARPSIZE   32

// The only dimensions currently supported by WMMA
const int WMMA_M       = 16;
const int WMMA_N       = 16;
const int WMMA_K       = 16;
const int warpK_stride = 1; // use shared memory to cache matrix A and B; warpK_stride is the sum of the number of WMMA_M*WMMA_K blocks for matrix A + WMMA_K*WMMA_N blocks for matrix B that can be cached by shared memory

const long seed        = 152897564;
const float alpha      = 1.0f;
const float beta       = 1.0f;
#ifdef TIMING_FLAG
const float ms_to_sec  = 1.0e3;
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


__global__ void gemm_kernel(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                            int M, int N, int K,
                            T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    const int lda         = K;
    const int ldb         = N;
    const int ldc         = N;

    int array_idx_y = threadIdx.y + blockDim.y * blockIdx.y;
    while ( array_idx_y < M )
    {
        int array_idx_x = threadIdx.x + blockDim.x * blockIdx.x;
        while ( array_idx_x < N )
        {
            int aRow       = array_idx_y;
            int bCol       = array_idx_x;
            T_ELEM_OUT acc = (T_ELEM_OUT)0.0;

            for (int i=0; i<K; i+=(warpK_stride*WMMA_K))
            {
                #pragma unroll
                for (int j=0; j<(warpK_stride*WMMA_K); ++j)
                {
                    if ( (i + j) < K )
                    {
                        acc += (T_ELEM_OUT)a[(i+j) + aRow*lda] * (T_ELEM_OUT)b[bCol + (i+j)*ldb];
                    }
                    else
                        break;
                }
            }

            c[bCol + aRow*ldc] = alpha * acc + beta* c[bCol + aRow*ldc];
            array_idx_x += blockDim.x * gridDim.x;
        }
        array_idx_y += blockDim.y * gridDim.y;
    }
}


/* WMMA GPU kernel */
__global__ void wmma_kernel(T_ELEM_IN *a, T_ELEM_IN *b, T_ELEM_OUT *c,
                            int M, int N, int K,
                            T_ELEM_OUT alpha, T_ELEM_OUT beta)
{
    // Leading dimensions. Packed with no transpositions.
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

#ifndef PERFORMANCE
    if ( lda%8 != 0 ) // lda must be multiple of 8 for data type half
    {
        assert(0);
    }
    if ( ldb%8 != 0 ) // ldb must be multiple of 8 for data type half
    {
        assert(0);
    }
    if ( ldc%4 != 0 ) // ldc must be multiple of 4 for data type float
    {
        assert(0);
    }
#endif

    const int  warpSize_x    = (blockDim.x < warpSize) ? blockDim.x : warpSize;
    const int  warpSize_y    = warpSize / warpSize_x;
    const long total_warp_x  = (N/WMMA_N);
    const long total_warp_y  = (M/WMMA_M);
    const long array_idx_x   = blockIdx.x * blockDim.x + threadIdx.x;
    const long array_idx_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int  warp_stride_x = (blockDim.x * gridDim.x) / warpSize_x;
    const int  warp_stride_y = (blockDim.y * gridDim.y) / warpSize_y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, T_ELEM_IN, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, T_ELEM_IN, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T_ELEM_OUT> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T_ELEM_OUT> c_frag;

    // Tile using a 2D grid
    int warpM  = array_idx_y / warpSize_y;
    while (warpM < total_warp_y)
    {
        int warpN  = array_idx_x / warpSize_x;
        while (warpN < total_warp_x)
        {
            wmma::fill_fragment(acc_frag, (T_ELEM_OUT)(0.0));

            int aRow     = warpM * WMMA_M;
            int bCol     = warpN * WMMA_N;
            for (int i=0; i<K; i+=(warpK_stride*WMMA_K))
            {
                int aCol = i;
                int bRow = i;

                // Bounds checking
                if ( (aRow < M) && (bCol < N) )
                {
                    #pragma unroll
                    // Loop over the K-dimension
                    for (int j=0; j<(warpK_stride*WMMA_K); j+=WMMA_K)
                    {
                        aCol       = i + j;
                        bRow       = i + j;

                        if ( aCol < K )
                        {
                            // Load the inputs
                            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
                            wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

                            // Perform the matrix multiplication
                            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                        }
                        else
                            break;
                    }
                }
            }

            // Load in current value of c, scale by beta, and add to result scaled by alpha
            int cRow = warpM * WMMA_M;
            int cCol = warpN * WMMA_N;

            if (cRow < M && cCol < N)
            {
                wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

                for(int i=0; i < c_frag.num_elements; i++)
                {
                    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                    //if (threadIdx.x == 0 && threadIdx.y == 0)
                    //    printf("(%d,%d) c_frag.x[%d] = %.4f\n", warpM, warpN, i, c_frag.x[i]);
                }

                // Store the output
                wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
            }
            warpN  += warp_stride_x;
        }
        warpM  += warp_stride_y;
    }
}


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
    std::vector<float>      matrix_a_float_host, matrix_b_float_host, matrix_c_float_host;
    std::vector<T_ELEM_IN>  matrix_a_host, matrix_b_host;
    std::vector<T_ELEM_OUT> matrix_c_host, matrix_c_host_backup; // when beta != 0, matrix_c_host will be updated by the new answer, so we make a backup matrix_c_host_backup, and used matrix_c_host to save the new answer

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

    const int warpSize_x = (tpB_x < WARPSIZE) ? tpB_x : WARPSIZE;
    const int warpSize_y = WARPSIZE / warpSize_x;

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

    matrix_a_float_host.resize((size_t)M*(size_t)K);
    matrix_b_float_host.resize((size_t)K*(size_t)N);
    matrix_c_float_host.resize((size_t)M*(size_t)N);
    matrix_a_host.resize((size_t)M*(size_t)K);
    matrix_b_host.resize((size_t)K*(size_t)N);
    matrix_c_host.resize((size_t)M*(size_t)N);
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
    #pragma omp parallel for shared(matrix_a_float_host, matrix_a_host)
    for (long idx_array=0L; idx_array<(long)M*(long)K; ++idx_array)
    {
        int idx_thread                 = omp_get_thread_num();
        matrix_a_float_host[idx_array] = norm_dist[idx_thread](generator[idx_thread]);
        matrix_a_host[idx_array]       = (T_ELEM_IN)(matrix_a_float_host[idx_array]);
    }
    // notice that matrix_b_float_host is in colum major (to accelerate OpenMP computation), while matrix_b_host is in row major
    #pragma omp parallel for shared(matrix_b_float_host, matrix_b_host)
    for (long idx_array=0L; idx_array<(long)K*(long)N; ++idx_array)
    {
        int idx_thread                           = omp_get_thread_num();
        long idx_array_col_major                 = (long)K*(idx_array%(long)N) + (idx_array/(long)N);
        matrix_b_float_host[idx_array_col_major] = norm_dist[idx_thread](generator[idx_thread]);
        matrix_b_host[idx_array]                 = (T_ELEM_IN)(matrix_b_float_host[idx_array_col_major]);
    }
    #pragma omp parallel for shared(matrix_c_float_host, matrix_c_host, matrix_c_host_backup)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        int idx_thread                  = omp_get_thread_num();
        matrix_c_float_host[idx_array]  = norm_dist[idx_thread](generator[idx_thread]);
        matrix_c_host[idx_array]        = (T_ELEM_OUT)(matrix_c_float_host[idx_array]);
        matrix_c_host_backup[idx_array] = (T_ELEM_OUT)(matrix_c_float_host[idx_array]);
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
    #pragma omp parallel for collapse(2) private(sum, idx_array_a, idx_array_b, idx_array_c) shared(matrix_a_float_host, matrix_b_float_host, matrix_c_float_host)
    for (int idx_row_c=0; idx_row_c<M; ++idx_row_c)
    {
        for (int idx_col_c=0; idx_col_c<N; ++idx_col_c)
        {
            sum                              = 0.0f;
            idx_array_c                      = (long)N * (long)idx_row_c + idx_col_c;
            #pragma omp simd
            for (int idx_accum=0; idx_accum<K; ++idx_accum)
            {
                idx_array_a                  = (long)K * (long)idx_row_c + idx_accum;
                idx_array_b                  = (long)K * (long)idx_col_c + idx_accum;
                sum                         += matrix_a_float_host[idx_array_a] * matrix_b_float_host[idx_array_b];
            }
            matrix_c_float_host[idx_array_c] = alpha * sum + beta * matrix_c_float_host[idx_array_c];
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
    //     std::cout << (long)M*(long)N - 1L << " " << matrix_c_float_host[(long)M*(long)N - 1L] << std::endl;
    //}
    //else
    //{
    //     std::cout << TEST_IDX << " " << matrix_c_float_host[TEST_IDX] << std::endl;
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
    gemm_kernel<<<bpG_gemm, tpB>>>(matrix_a_device, matrix_b_device, matrix_c_device,
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
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_host.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
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
    #pragma omp parallel for shared(matrix_c_float_host, matrix_c_host) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_float_host[idx_array]-matrix_c_host[idx_array]), 2.0);
        norm += pow((double)(matrix_c_float_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("Hand-carved GEMM relative L2 error between matrix_c_float_host and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

    ////* GPU WMMA computation *////
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
    wmma_kernel<<<bpG_WMMA, tpB>>>(matrix_a_device, matrix_b_device, matrix_c_device,
                                   M, N, K,
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
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_host.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
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
    #pragma omp parallel for shared(matrix_c_float_host, matrix_c_host) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_float_host[idx_array]-matrix_c_host[idx_array]), 2.0);
        norm += pow((double)(matrix_c_float_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("WMMA relative L2 error between matrix_c_float_host and matrix_c_host is %.4e .\n", diff/norm);
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

    // K, lda, ldb are all multiples of 8 for half precision, and N, ldc is a multiple of 4 for float precision in our case for GemmEx:
    if ( K%8 != 0 )
    {
        assert(0);
    }
    if ( N%4 != 0 )
    {
        assert(0);
    }
    if ( lda%8 != 0 )
    {
        assert(0);
    }
    if ( ldb%8 != 0 )
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

    // Invoke the GEMM, ensuring k, ldb, lda are all multiples of 8 for half precision, and n(since now our B is the A in the example), ldc are all multiples of 4 for float precision, i.e. 16 bytes
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasOperation_t trans_a = CUBLAS_OP_N;
    CUBLAS_CHECK_ERROR(cublasGemmEx(handle, trans_b, trans_a,
                                    N,               M,               K,   &alpha,
                                    matrix_b_device, CUBLAS_ELEM_IN,  ldb,
                                    matrix_a_device, CUBLAS_ELEM_IN,  lda,
                                    &beta,
                                    matrix_c_device, CUBLAS_ELEM_OUT, ldc,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

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
    CUDA_CHECK_ERROR(cudaMemcpy(matrix_c_host.data(), matrix_c_device, (size_t)M*(size_t)N*sizeof(T_ELEM_OUT), cudaMemcpyDeviceToHost));
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
    #pragma omp parallel for shared(matrix_c_float_host, matrix_c_host) reduction(+:diff) reduction(+:norm)
    for (long idx_array=0L; idx_array<(long)M*(long)N; ++idx_array)
    {
        diff += pow((double)(matrix_c_float_host[idx_array]-matrix_c_host[idx_array]), 2.0);
        norm += pow((double)(matrix_c_float_host[idx_array]), 2.0);
    }
    diff = sqrt(diff / ((double)M * (double)N));
    norm = sqrt(norm / ((double)M * (double)N));
    printf("cuBLAS GemmEx relative L2 error between matrix_c_float_host and matrix_c_host is %.4e .\n", diff/norm);
#ifdef TIMING_FLAG
    CUDA_CHECK_ERROR(cudaEventRecord(t_stop,0));
    CUDA_CHECK_ERROR(cudaEventSynchronize(t_stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, t_start, t_stop));
    printf("Compute relative L2 error time = %.4e s\n", elapsed_time/ms_to_sec);
    printf("***************************************************************\n");
#endif

    ////* End *////

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
