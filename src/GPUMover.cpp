#include "GPUMover.h"
#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"


void copyGrid(grid *cpu_grid, grid **gpu_grid, size_t size){

	//Copy over the grid to GPU
	cudaMalloc(gpu_grid, sizeof(grid));
	FPfield *XN_flat = cpu_grid->XN_flat;
	FPfield *YN_flat = cpu_grid->YN_flat;
	FPfield *ZN_flat = cpu_grid->ZN_flat;
	cudaMalloc(&(cpu_grid->XN_flat), size * sizeof(FPfield));
	cudaMalloc(&(cpu_grid->YN_flat), size * sizeof(FPfield));
	cudaMalloc(&(cpu_grid->ZN_flat), size * sizeof(FPfield));
	cudaMemcpy(cpu_grid->XN_flat, XN_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_grid->YN_flat, YN_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_grid->ZN_flat, ZN_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(*gpu_grid, cpu_grid, sizeof(grid), cudaMemcpyHostToDevice);
	cpu_grid->XN_flat = XN_flat;
	cpu_grid->YN_flat = YN_flat;
	cpu_grid->ZN_flat = ZN_flat;

}




void copyParam(parameters *param, parameters *gpu_param) {


	cudaMalloc(&gpu_param, sizeof(parameters));
	cudaMemcpy(gpu_param, param, sizeof(parameters), cudaMemcpyHostToDevice);


}

void copyEMfield(EMfield *emf, EMfield *gpu_emf, size_t size) {


	FPfield* Ex_flat;
	FPfield* Ey_flat;
	FPfield* Ez_flat;
	FPfield* Bxn_flat;
	FPfield* Byn_flat;
	FPfield* Bzn_flat;

	cudaMalloc(&gpu_emf, sizeof(EMfield));
	cudaMemcpy(gpu_emf, emf, size*sizeof(EMfield), cudaMemcpyHostToDevice);



	//Ex
	//malloc space for pointer(array)
	cudaMalloc(&Ex_flat, size * sizeof(FPfield));
	//fill array with values
	cudaMemcpy(Ex_flat, emf->Ex_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	//make our gpu_emf->ex_flat point to the address of ex_flat we just created.
	cudaMemcpy(&(gpu_emf->Ex_flat), &Ex_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//ey
	cudaMalloc(&Ey_flat, size * sizeof(FPfield));
	cudaMemcpy(Ey_flat, emf->Ey_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpu_emf->Ey_flat), &Ey_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//ez
	cudaMalloc(&Ez_flat, size * sizeof(FPfield));
	cudaMemcpy(Ez_flat, emf->Ez_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpu_emf->Ez_flat), &Ez_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//bx
	cudaMalloc(&Bxn_flat, size * sizeof(FPfield));
	cudaMemcpy(Bxn_flat, emf->Bxn_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpu_emf->Bxn_flat), &Bxn_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//by
	cudaMalloc(&Byn_flat, size * sizeof(FPfield));
	cudaMemcpy(Byn_flat, emf->Byn_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpu_emf->Byn_flat), &Byn_flat, sizeof(FPfield), cudaMemcpyHostToDevice);
	//bz
	cudaMalloc(&Bzn_flat, size * sizeof(FPfield));
	cudaMemcpy(Bzn_flat, emf->Bzn_flat, size * sizeof(FPfield), cudaMemcpyHostToDevice);
	cudaMemcpy(&(gpu_emf->Bzn_flat), &Bzn_flat, sizeof(FPfield), cudaMemcpyHostToDevice);

	
	}


void mallocParticles(particles *part,particles *gpu_particle, particles *particle_pointers, size_t size) {


	cudaMalloc(&gpu_particle, size * sizeof(particles));


	//Move each particle over
	for (int i = 0; i < size; i++) {
		int npmax = part[i].npmax;
		//start by copying info about particle over
		cudaMemcpy(gpu_particle[i], part[i], sizeof(particles), cudaMemcpyHostToDevice);




		//FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
		/** q must have precision of interpolated quantities: typically double. Not used in mover */
		//FPinterp* q;




	}


}

void copyParticlesToGPU(particles *part, particles *gpu_particle, particles *particle_pointers, size_t size) {

	long npmax;


	for (size_t i = 0; i < size; i++) {
		npmax = part[i].npmax;

		cudaMemcpy(gpu_particle[i].x, part[i].x, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_particle[i].y, part[i].y, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_particle[i].z, part[i].z, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_particle[i].u, part[i].u, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_particle[i].v, part[i].v, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_particle[i].w, part[i].w, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_particle[i].q, part[i].q, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

	}

}

void loadParticlesToCpu(particles *part, particles *gpu_particle, particles *particle_pointers, size_t size) {
	long npmax;

	for (size_t i = 0; i < size; i++) {
		npmax = part[i].npmax;
		cudaMemcpy(part[i].x, gpu_particle[i].x, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(part[i].y, gpu_particle[i].y, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(part[i].z, gpu_particle[i].z, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(part[i].u, gpu_particle[i].u, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(part[i].v, gpu_particle[i].v, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(part[i].w, gpu_particle[i].w, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
		cudaMemcpy(part[i].q, gpu_particle[i].q, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

	}


}

