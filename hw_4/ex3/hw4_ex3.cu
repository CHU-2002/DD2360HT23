#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
  } while (0)

 
struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}
void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}





// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX,
    double alpha) {
  // Stencil from the finete difference discretization of the equation
  double stencil[] = { 1, -2, 1 };
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  // Insert a row of zeros at the beginning of the matrix
  ArowPtr[1] = ptr;
  // Fill the non zero entries of the matrix
  for (int i = 1; i < (dimX - 1); ++i) {
    // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
    for (int k = 0; k < 3; ++k) {
      // Set the value for A[i][i+k-1]
      A[ptr] = stencil[k];
      // Set the column index for A[i][i+k-1]
      AcolIndx[ptr++] = i + k - 1;
    }
    // Set the number of newly added elements
    ArowPtr[i + 1] = ptr;
  }
  // Insert a row of zeros at the end of the matrix
  ArowPtr[dimX] = ptr;
}

int main(int argc, char **argv) {
  
  cputimer_start();
  int device = 0;            // Device to be used
  int dimX;                  // Dimension of the metal rod
  int nsteps;                // Number of time steps to perform
  double alpha = 0.4;        // Diffusion coefficient
  double beta = 0.0;
  double* temp;              // Array to store the final time step
  double* A;                 // Sparse matrix A values in the CSR format
  int* ARowPtr;              // Sparse matrix A row pointers in the CSR format
  int* AColIndx;             // Sparse matrix A col values in the CSR format
  int nzv;                   // Number of non zero values in the sparse matrix
  double* tmp;               // Temporal array of dimX for computations
  size_t bufferSize = 0;     // Buffer size needed by some routines
  void* buffer = nullptr;    // Buffer used by some routines in the libraries
  int concurrentAccessQ;     // Check if concurrent access flag is set
  double zero = 0;           // Zero constant
  double one = 1;            // One constant
  double norm;               // Variable for norm values
  double error;              // Variable for storing the relative error
  double tempLeft = 200.;    // Left heat source applied to the rod
  double tempRight = 300.;   // Right heat source applied to the rod
  cublasHandle_t cublasHandle;      // cuBLAS handle
  cusparseHandle_t cusparseHandle;  // cuSPARSE handle
  cusparseMatDescr_t Adescriptor;   // Mat descriptor needed by cuSPARSE

  // Read the arguments from the command line
  dimX = atoi(argv[1]);
  nsteps = atoi(argv[2]);
  usePrefetch  = atoi(argv[3]);

  // Print input arguments
  printf("The X dimension of the grid is %d \n", dimX);
  printf("The number of time steps to perform is %d \n", nsteps);

  // Get if the cudaDevAttrConcurrentManagedAccess flag is set
  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

  // Calculate the number of non zero values in the sparse matrix. This number
  // is known from the structure of the sparse matrix
  nzv = 3 * dimX - 6;

  //@@ Insert the code to allocate the temp, tmp and the sparse matrix
  //@@ arrays using Unified Memory
  cudaMallocManaged(&temp, dimX * sizeof(double));
  cudaMallocManaged(&tmp, dimX * sizeof(double));
  cudaMallocManaged(&A, nzv * sizeof(double));
  cudaMallocManaged(&ARowPtr, (dimX + 1) * sizeof(int));
  cudaMallocManaged(&AColIndx, nzv * sizeof(int));



  // Check if concurrentAccessQ is non zero in order to prefetch memory
  if (concurrentAccessQ) {
    //@@ Insert code to prefetch in Unified Memory asynchronously to CPU
    if(usePrefetch != 0){
      
      cudaMemPrefetchAsync(temp, dimX * sizeof(double), cudaCpuDeviceId, 0);
      cudaMemPrefetchAsync(tmp, dimX * sizeof(double), cudaCpuDeviceId, 0);
      cudaMemPrefetchAsync(A, nzv * sizeof(double), cudaCpuDeviceId, 0);
      cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), cudaCpuDeviceId, 0);
      cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), cudaCpuDeviceId, 0);

    }

  }

  // Initialize the sparse matrix
  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);

  //Initiliaze the boundary conditions for the heat equation
  memset(temp, 0, sizeof(double) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;

  if (concurrentAccessQ) {
    //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
    
    if(usePrefetch != 0){
    cudaMemPrefetchAsync(temp, dimX * sizeof(double), device, 0);
    cudaMemPrefetchAsync(tmp, dimX * sizeof(double), device, 0);
    cudaMemPrefetchAsync(A, nzv * sizeof(double), device, 0);
    cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), device, 0);
    cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), device, 0);
    }
  }

  //@@ Insert code to create the cuBLAS handle
  cublasCreate(&cublasHandle);    

  //@@ Insert code to create the cuSPARSE handle
  cusparseCreate(&cusparseHandle);


  //@@ Insert code to set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);

  //@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
  cusparseCreateMatDescr(&Adescriptor);

  //@@ Insert code to call cusparse api to get the buffer size needed by the sparse matrix per
  //@@ vector (SMPV) CSR routine of cuSPARSE
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void* dBuffer = nullptr;

  // 创建稀疏矩阵描述符
  cusparseCreateCsr(&matA, dimX, dimX, ARowPtr[dimX], ARowPtr, AColIndx, A, 
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  // 创建输入和输出向量的描述符
  cusparseCreateDnVec(&vecX, dimX, temp, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, dimX, tmp, CUDA_R_64F);

  // 调用 cusparseSpMV_bufferSize 获取所需的缓冲区大小
  cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                          &alpha, matA, vecX, &beta, vecY, 
                          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);


  //@@ Insert code to allocate the buffer needed by cuSPARSE
  cudaMalloc(&dBuffer, bufferSize);


  // Perform the time step iterations
  for (int it = 0; it < nsteps; ++it) {
    //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
    //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
    //@@ tmp = 1 * A * temp + 0 * tmp
  
    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    //@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
    //@@ This calculation corresponds to: temp = alpha * tmp + temp
    cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1);

    //@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
    //@@ This calculation corresponds to: ||temp||
    cublasDnrm2(cublasHandle, dimX, temp, 1, &norm);

    // If the norm of A*temp is smaller than 10^-4 exit the loop
    if (norm < 1e-4)
      break;
  }

  // Calculate the exact solution using thrust
  thrust::device_ptr<double> thrustPtr(tmp);
  thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
      (tempRight - tempLeft) / (dimX - 1));

  // Calculate the relative approximation error:
  one = -1;
  //@@ Insert the code to call cublas api to compute the difference between the exact solution
  //@@ and the approximation
  //@@ This calculation corresponds to: tmp = -temp + tmp
  cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1);

  //@@ Insert the code to call cublas api to compute the norm of the absolute error
  //@@ This calculation corresponds to: || tmp ||
  cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm);

  error = norm;
  //@@ Insert the code to call cublas api to compute the norm of temp
  //@@ This calculation corresponds to: || temp ||
  cublasDnrm2(cublasHandle, dimX, temp, 1, &norm);

  // Calculate the relative error
  error = error / norm;
  printf("The relative error of the approximation is %f\n", error);

  //@@ Insert the code to destroy the mat descriptor
  cusparseDestroyMatDescr(Adescriptor);

  //@@ Insert the code to destroy the cuSPARSE handle
  cusparseDestroy(cusparseHandle);

  //@@ Insert the code to destroy the cuBLAS handle
  cublasDestroy(cublasHandle);


  //@@ Insert the code for deallocating memory
  cudaFree(temp);
  cudaFree(tmp);
  cudaFree(A);
  cudaFree(ARowPtr);
  cudaFree(AColIndx);

  
    if(usePrefetch == 0){
    cputimer_stop("Total Execuation TIme without Prefetch");

    }else{
      
    cputimer_stop("Total Execuation TIme with Prefetch");
    }
  return 0;
}
