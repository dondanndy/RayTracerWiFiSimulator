#include "cuda_runtime_api.h"
#include <vector>

namespace CudaHelpers {

	template<typename T>
	bool copy_vector_to_gpu(T* gpu_mem, const std::vector<T>& vec){
		cudaError_t err;
		err = cudaMalloc((void**)&gpu_mem, vec.size() * (size_t)sizeof(T));
		cudaMemcpy((void*)gpu_mem, (void*)vec.data(), vec.size() * (size_t)sizeof(T), cudaMemcpyHostToDevice);

		return err;
	}

	template<typename T>
	bool retrieve_vector_from_gpu(T* gpu_mem, const std::vector<T>& vec) {
		return cudaMemcpy((void*)vec.data(), gpu_mem, vec.size() * (size_t)sizeof(T), cudaMemcpyDeviceToHost);
	}
}

