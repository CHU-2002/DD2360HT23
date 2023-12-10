
#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define RAND_MAX 0x7fffffff
#define TPB 256

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    out[i] = in1[i] + in2[i];
    //printf("vecadd[%d] = %lf\n",i,out[i]);
  }
}

//@@ Insert code to implement timer start
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//@@ Insert code to implement timer stop

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  double iStart, iElaps;
  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("input format: ./a,out inputLength\n");
    return 1;
  }
  inputLength = atoi(argv[1]);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType*)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType*)malloc(inputLength * sizeof(DataType));
  resultRef = (DataType*)malloc(inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(NULL));
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = rand() / (DataType)RAND_MAX;
    hostInput2[i] = rand() / (DataType)RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  iStart =  cpuSecond();

  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  
  iElaps = cpuSecond() - iStart;
  printf("Copy memory to the GPU time elapsed %lf sec\n", iElaps);

  //@@ Initialize the 1D grid and block dimensions here
  dim3 block((TPB+inputLength - 1) / TPB);
  dim3 grid(TPB);

  //@@ Launch the GPU Kernel here
  iStart = cpuSecond();

  vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  
  iElaps = cpuSecond() - iStart;
  printf("GPU Kernel time elapsed %lf sec\n", iElaps);

  //@@ Copy the GPU memory back to the CPU here
  iStart = cpuSecond();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  iElaps = cpuSecond() - iStart;
  printf("Copy the GPU memory back to the CPU time elapsed %lf sec\n", iElaps);
  //@@ Insert code below to compare the output with the reference
  for(int i = 0; i < inputLength; i++) {
    if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
      printf("Result verification failed at element %d!\n", i);
      break;
    }
  }
  printf("Results correct!\n");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  
  return 0;
}
