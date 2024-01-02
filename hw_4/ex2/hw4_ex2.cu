#include <stdio.h>
#include <sys/time.h>
#define DataType double
#define RAND_MAX 0x7fffffff
#define TPB 256
#define S_seg 2500
__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
	//@@ Insert code to implement vector addition here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		out[i] = in1[i] + in2[i];
	}
}
//@@ Insert code to implement timer start
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
	int numSegments = (inputLength + S_seg - 1) / S_seg;
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
	cudaHostAlloc(&deviceInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
	cudaHostAlloc(&deviceInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
	cudaHostAlloc(&deviceOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);


	cudaStream_t streams[4];
	for (int i = 0; i < 4; ++i) {
		cudaStreamCreate(&streams[i]);
	}
	//@@ Insert code to below to Copy memory to the GPU here
	iStart =  cpuSecond();
	for (int i = 0; i < numSegments; ++i) {
		int offset = i * S_seg;
		int size = min(S_seg, inputLength - offset);
		int streamIndex = i % 4;
		cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIndex]);
	}

		for (int i = 0; i < numSegments; ++i) {
		int offset = i * S_seg;
		int size = min(S_seg, inputLength - offset);
		int streamIndex = i % 4;
		cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, streams[streamIndex]);
	}


		for (int i = 0; i < numSegments; ++i) {
		int offset = i * S_seg;
		int size = min(S_seg, inputLength - offset);
		int streamIndex = i % 4;
		vecAdd<<<(size + TPB - 1) / TPB, TPB, 0, streams[streamIndex]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, size);
	}


		for (int i = 0; i < numSegments; ++i) {
		int offset = i * S_seg;
		int size = min(S_seg, inputLength - offset);
		int streamIndex = i % 4;
		cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, size * sizeof(DataType), cudaMemcpyDeviceToHost, streams[streamIndex]);
	}


  cudaDeviceSynchronize();

	iElaps = cpuSecond() - iStart;
	printf("Total time with streams: %f seconds\n", iElaps);
	for (int i = 0; i < inputLength; i++) {
		if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
			printf("Result verification failed at element %d!\n", i);
			break;
		}
	}
	printf("Results correct!\n");
	// Free Memory
	for (int i = 0; i < 4; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
	free(hostInput1);
	free(hostInput2);
	free(hostOutput);
	return 0;
}