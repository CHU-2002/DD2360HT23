#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define MatrixSize 4


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }


 
// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < numARows && col < numBColumns) {
      DataType sum = 0;
      for (int i = 0; i < numAColumns; ++i) {
          sum += A[row * numAColumns + i] * B[i * numBColumns + col];
      }
      C[row * numBColumns + col] = sum;
      
  }
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows = MatrixSize;            // number of rows in the matrix A
  int numAColumns = MatrixSize;         // number of columns in the matrix A
  int numBRows = numAColumns;    // number of rows in the matrix B
  int numBColumns = MatrixSize;         // number of columns in the matrix B
  int numCRows = numARows;
  int numCColumns = numBColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args


  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));


  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows * numAColumns; i++) {
      hostA[i] = rand() % 100 / 10.0;
  }

  for (int i = 0; i < numBRows * numBColumns; i++) {
      hostB[i] = rand() % 100 / 10.0;
  }

  for(int row = 0; row < numARows; row++ ){
    for(int col = 0; col < numBColumns; col ++){
        DataType sum = 0;
        for (int i = 0; i < numAColumns; ++i) {
            sum += hostA[row * numAColumns + i] * hostB[i * numBColumns + col];
        }
        resultRef[row * numBColumns + col] = sum;
    }
  }


  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));



  //@@ Insert code to below to Copy memory to the GPU here
  double Start1 = cpuSecond();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double End1 = cpuSecond() - Start1;
  printf("data copy from host to device : %f\n", End1);


  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(16, 16);
  dim3 dimGrid((numBColumns + dimBlock.x - 1) / dimBlock.x, (numARows + dimBlock.y - 1) / dimBlock.y);

  //@@ Launch the GPU Kernel here
  double Start2 = cpuSecond();
  gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double End2 = cpuSecond() - Start2;
  printf("CUDA Kernel : %f\n", End2);

  //@@ Copy the GPU memory back to the CPU here
  double Start3 = cpuSecond();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  double End3 = cpuSecond() - Start3;
  printf("data copy from device to host : %f\n", End3);


  //@@ Insert code below to compare the output with the reference

  for(int i = 0; i < (numARows * numBColumns); i++){
    printf("resultRef[%d] = %f\n",i, resultRef[i]);
    printf("hostC[%d] = %f\n",i, hostC[i]);
    if(resultRef[i] == hostC[i]){
      printf("[%d] output equals to reference\n\n",i);
    }
  }


  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
