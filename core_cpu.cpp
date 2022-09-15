#ifndef WARPRNA_ENABLE_CPU
#define WARPRNA_ENABLE_CPU
#endif
#define __WARPRNA_CPU

#include <stddef.h>

#define __forceinline__
#define __device__
#define __global__

struct _int3 {
    int x, y, z;
    _int3(int _x=1, int _y=1, int _z=1) : x(_x), y(_y), z(_z) {}
};

struct _uint3 {
    /*
    Like CUDA dim3.
    This type is an integer vector type based on uint3 that is used to specify dimensions.
    When defining a variable of type dim3, any component left unspecified is initialized to 1.
    */
    unsigned int x, y, z;
    _uint3(unsigned int _x=1, unsigned int _y=1, unsigned int _z=1) : x(_x), y(_y), z(_z) {}
};

template<typename T>
static void resetVec3(T& v) {
    v.x = v.y = v.z = 0;
}

#if __cplusplus <= 199711L
#define thread_local static
#endif

#define dim3 _int3
thread_local size_t _shared_size;
thread_local _uint3 _threadIdx;
thread_local _uint3 _blockIdx;
thread_local dim3 _blockDim;
thread_local dim3 _gridDim;
// We need those as macros to not infer with the CUDA versions if CUDA was also included.
#define threadIdx _threadIdx
#define blockIdx _blockIdx
#define blockDim _blockDim
#define gridDim _gridDim

struct _KernelLoop {
	_KernelLoop(dim3 dim_grid = 1, dim3 dim_block = 1) {
		resetVec3(gridDim); // numBlocks
		resetVec3(blockDim); // threadsPerBlock
		resetVec3(blockIdx);
		resetVec3(threadIdx);
		gridDim = dim_grid;
		blockDim = dim_block;
	}
	bool finished() {
		return blockIdx.z >= gridDim.z;
	}
	void next() {
		threadIdx.x++;
		if(threadIdx.x >= blockDim.x) {
		    threadIdx.x = 0;
			threadIdx.y++;
			if(threadIdx.y >= blockDim.y) {
				threadIdx.y = 0;
				threadIdx.z++;
				if(threadIdx.z >= blockDim.z) {
					threadIdx.z = 0;
					blockIdx.x++;
					if(blockIdx.x >= gridDim.x) {
						blockIdx.x = 0;
						blockIdx.y++;
						if(blockIdx.y >= gridDim.y) {
							blockIdx.y = 0;
							blockIdx.z++;
							// no further check here. finished() covers that
						}
					}
				}
			}
		}
	}
};

#define cudaGetLastError() (0)
#define cudaSuccess (0)

#define start_dev_kernel(kernel, dim_grid, dim_block, args) \
	{ for(_KernelLoop loop(dim_grid, dim_block); !loop.finished(); loop.next()) { cpu::kernel args; } }


// ignore atomic for now...
#define atomicAdd(x, v) (*(x) += (v))
#define __threadfence() (0)
#define __shfl_up_sync(mask, var, delta) (var)

#include "core.cu"

#undef __shfl_up_sync
#undef __threadfence
#undef atomicAdd
#undef dim3
#undef cudaGetLastError
#undef cudaSuccess
#undef __forceinline__
#undef __device__
#undef __global__
