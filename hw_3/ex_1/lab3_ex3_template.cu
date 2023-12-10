#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 1024
#define MUM_Block 1024
#define nStreams 4
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics

  int tid = threadIdx.x;
  int i = tid + blockDim.x * blockIdx.x;

  // Compute local histogram using atomicAdd
/*  while (i < num_elements) {
      atomicAdd(&bins[input[i]], 1);
      i += blockDim.x * gridDim.x;
      //bins[i] = bins[i] > 127 ? 127 : bins[i];
  }*/


  __shared__ unsigned int hist[NUM_BINS];


  // Initialize shared memory histogram
    if (threadIdx.x == 0)
      for(int i = 0; i < num_bins; i++)
        hist[i] = 0;
    __syncthreads();

  // Compute local histogram using atomicAdd
  while (i < num_elements) {
      atomicAdd(&hist[input[i]], 1);
      i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Accumulate local histograms into global histogram
  if(threadIdx.x == 0) {
    for(int i = 0; i < num_bins; i++) {
      atomicAdd(&bins[i], hist[i]);
    }
  }

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < num_bins) {
      bins[i] = bins[i] > 127 ? 127 : bins[i];
  }

}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void drawHistogram(unsigned int* values) {
  for(int i = 0; i < NUM_BINS; i++) {
    printf("%4d: ", i);
    for(int j = 0; j < values[i]; j++) {
      printf("\033[1;30m█\033[0m");
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  /*cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("%d\n",prop.maxThreadsPerBlock);*/
  
  double iStart, iElaps;
  //@@ Insert code below to read in inputLength from args
  if(argc != 2) {
    printf("Input format: ./a.out <inputLength>\n");
    exit(1);
  }
  inputLength = atoi(argv[1]);

  int streamSize = inputLength / nStreams;
  int streamBytes = streamSize * sizeof(int);
  cudaStream_t streams[nStreams];
  for(int i = 0;i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *) malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *) malloc(NUM_BINS * sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(time(NULL));
  for(int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() % NUM_BINS;
  }
  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int *) malloc(NUM_BINS * sizeof(unsigned int));
  memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));
  for(int i = 0; i < inputLength; i++) {
    resultRef[hostInput[i]]++;
    resultRef[hostInput[i]] = resultRef[hostInput[i]] > 127 ? 127 : resultRef[hostInput[i]];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **) &deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **) &deviceBins, NUM_BINS * sizeof(unsigned int));


  //@@ Insert code to Copy memory to the GPU here
/*  iStart =  cpuSecond();
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  iElaps = cpuSecond() - iStart;
  printf("GPU cudaMemcpy Time elapsed %f sec\n", iElaps);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  dim3 block(MUM_Block);
  dim3 grid(TPB);

  //@@ Launch the GPU Kernel here
  iStart =  cpuSecond();
  histogram_kernel<<<grid, block>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  iElaps = cpuSecond() - iStart;
  printf("GPU histogram_kernel  1 Time elapsed %f sec\n", iElaps);*/

  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  dim3 block(MUM_Block);
  dim3 grid(TPB);
  iStart =  cpuSecond();
  for(int i = 0; i < nStreams; i++) {
    int offset = i * streamSize;
        if(i == nStreams - 1) {
      streamSize += inputLength % nStreams;
      streamBytes = streamSize * sizeof(int);
    }
    cudaMemcpyAsync(&deviceInput[offset], &hostInput[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
    histogram_kernel<<<grid, block, 0, streams[i]>>>(&deviceInput[offset], deviceBins, streamSize, NUM_BINS);
  }
  iElaps = cpuSecond() - iStart;
  printf("GPU histogram_kernel 4 Time elapsed %f sec\n", iElaps);

  //@@ Initialize the second grid and block dimensions here
  dim3 block2((TPB+NUM_BINS - 1) / TPB);
  dim3 grid2(TPB);

  //@@ Launch the second GPU Kernel here
  iStart =  cpuSecond();
  convert_kernel<<<grid2, block2>>>(deviceBins, NUM_BINS);
  iElaps = cpuSecond() - iStart;
  printf("GPU convert_kernel Time elapsed %f sec\n", iElaps);

  //@@ Copy the GPU memory back to the CPU here
  iStart =  cpuSecond();
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  iElaps = cpuSecond() - iStart;
  printf("GPU cudaMemcpy Time elapsed %f sec\n", iElaps);

  //@@ Insert code below to compare the output with the reference
  for(int i = 0; i < NUM_BINS; i++) {
    if(hostBins[i] != resultRef[i]) {
      printf("Mismatch at %d: reference = %d, GPU = %d\n", i, resultRef[i], hostBins[i]);
      //exit(1);
    }
  }
  printf("Correct!\n");

  drawHistogram(hostBins);

  for(int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

