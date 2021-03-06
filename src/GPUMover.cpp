#include "GPUMover.h"
#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"


void copyGrid(grid *cpu_grid, grid *gpu_grid, size_t size){

	cudaError_t code;



	FPfield *XN_flat;
	FPfield *YN_flat;
	FPfield *ZN_flat;
	code = cudaMalloc(&gpu_grid,sizeof(grid));
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC1" << std::endl;
	}
	code = cudaMemcpy(gpu_grid, cpu_grid, sizeof(grid), cudaMemcpyHostToDevice);
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC2" << std::endl;
	}

	code = cudaMalloc(&XN_flat, size * sizeof(FPfield));
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC3" << std::endl;
	}
	code = cudaMemcpy(XN_flat, cpu_grid->XN_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC4" << std::endl;
	}
	code = cudaMemcpy(&(gpu_grid->XN_flat), &XN_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC5" << std::endl;
	}

	code = cudaMalloc(&YN_flat, size * sizeof(FPfield));
	code = cudaMemcpy(YN_flat, cpu_grid->YN_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	code = cudaMemcpy(&(gpu_grid->YN_flat), &YN_flat, sizeof(FPfield), cudaMemcpyHostToDevice);

	code = cudaMalloc(&ZN_flat, size * sizeof(FPfield));
	code = cudaMemcpy(ZN_flat, cpu_grid->ZN_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	code = cudaMemcpy(&(gpu_grid->ZN_flat), &ZN_flat, sizeof(FPfield), cudaMemcpyHostToDevice);





}




void copyParam(parameters *param, parameters *gpu_param) {


	cudaMalloc(&gpu_param, sizeof(parameters));
	cudaMemcpy(gpu_param, param, sizeof(parameters), cudaMemcpyHostToDevice);


}

void copyEMfield(EMfield *emf, EMfield *gpu_emf, size_t size) {

	cudaError_t code;





	FPfield* Ex_flat;
	FPfield* Ey_flat;
	FPfield* Ez_flat;
	FPfield* Bxn_flat;
	FPfield* Byn_flat;
	FPfield* Bzn_flat;

	code = cudaMalloc(&gpu_emf, sizeof(EMfield));
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC1" << std::endl;
	}
	code = cudaMemcpy(gpu_emf, emf, sizeof(EMfield), cudaMemcpyHostToDevice);
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC2" << std::endl;
	}



	//Ex
	//malloc space for pointer(array)
	code = cudaMalloc(&Ex_flat, size * sizeof(FPfield));
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC3" << std::endl;
	}
	//fill array with values
	code = cudaMemcpy(Ex_flat, emf->Ex_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC4" << std::endl;
	}
	//make our gpu_emf->ex_flat point to the address of ex_flat we just created.
	code = cudaMemcpy(&(gpu_emf->Ex_flat), &Ex_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	if (code != cudaSuccess) {
		std::cout << "CUDA FAILED MALLOC5" << std::endl;
	}
	//ey
	code = cudaMalloc(&Ey_flat, size * sizeof(FPfield));
	code = cudaMemcpy(Ey_flat, emf->Ey_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	code = cudaMemcpy(&(gpu_emf->Ey_flat), &Ey_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//ez
	code = cudaMalloc(&Ez_flat, size * sizeof(FPfield));
	code = cudaMemcpy(Ez_flat, emf->Ez_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	code = cudaMemcpy(&(gpu_emf->Ez_flat), &Ez_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//bx
	code = cudaMalloc(&Bxn_flat, size * sizeof(FPfield));
	code = cudaMemcpy(Bxn_flat, emf->Bxn_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	code = cudaMemcpy(&(gpu_emf->Bxn_flat), &Bxn_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//by
	code = cudaMalloc(&Byn_flat, size * sizeof(FPfield));
	code = cudaMemcpy(Byn_flat, emf->Byn_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	code = cudaMemcpy(&(gpu_emf->Byn_flat), &Byn_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//bz
	cudaMalloc(&Bzn_flat, size * sizeof(FPfield));
	cudaMemcpy(Bzn_flat, emf->Bzn_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpu_emf->Bzn_flat), &Bzn_flat, sizeof(FPfield), cudaMemcpyHostToDevice);

	
	}