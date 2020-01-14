#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}



__global__ void MOVER_KERNEL(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param){



	
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	printf("blockIdx.x:%d * blockDim.x:%d + threadIdx.x:%d = \n", blockIdx.x, blockDim.x, threadIdx.x);


	if(i >= 500){
		return;
	}
	printf("gridStuff:%d",grd->nyn);
	printf("gridStuff:%d",grd->XN_flat[2]);
	

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)];
                eta[0]  = part->y[i] - grd->YN_flat[get_idx(ix, iy-1, iz, grd->nyn, grd->nzn)];
                zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz-1, grd->nyn, grd->nzn)];
                xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[i];
                eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)]  - part->y[i];
                zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
							int index = get_idx(ix-ii,iy-jj,iz-kk,grd->nyn,grd->nzn);
                            Exl += weight[ii][jj][kk]*field->Ex_flat[index];
                            Eyl += weight[ii][jj][kk]*field->Ey_flat[index];
                            Ezl += weight[ii][jj][kk]*field->Ez_flat[index];
                            Bxl += weight[ii][jj][kk]*field->Bxn_flat[index];
                            Byl += weight[ii][jj][kk]*field->Byn_flat[index];
                            Bzl += weight[ii][jj][kk]*field->Bzn_flat[index];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                 
    } // end of one particle

	
}

__global__ void MOVER_KERNEL_BRUTEFORCE(FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, 
										int nyn, int nzn, double xStart, double yStart, double zStart, FPfield invdx, FPfield invdy, FPfield invdz, 
										double Lx, double Ly, double Lz, FPfield invVOL, FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat, FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat,
										 bool PERIODICX, bool PERIODICY, bool PERIODICZ, FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2, int NiterMover, int npmax)
{
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    int flat_idx = 0;
    
    if(i > npmax)
    {
        return;
    }

    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = x[i];
    yptilde = y[i];
    zptilde = z[i];

    // calculate the average velocity iteratively
    for(int innter=0; innter < NiterMover; innter++){
        
        // interpolation G-->P
        ix = 2 +  int((x[i] - xStart)*invdx);
        iy = 2 +  int((y[i] - yStart)*invdy);
        iz = 2 +  int((z[i] - zStart)*invdz);

        // calculate weights

        flat_idx = get_idx(ix-1, iy, iz, nyn, nzn);
        xi[0]   = x[i] - XN_flat[flat_idx];

        flat_idx = get_idx(ix, iy-1, iz, nyn, nzn);
        eta[0]  = y[i] - YN_flat[flat_idx];

        flat_idx = get_idx(ix, iy, iz-1, nyn, nzn);
        zeta[0] = z[i] - ZN_flat[flat_idx];

        flat_idx = get_idx(ix, iy, iz, nyn, nzn);
        xi[1]   = XN_flat[flat_idx] - x[i];
        eta[1]  = YN_flat[flat_idx] - y[i];
        zeta[1] = ZN_flat[flat_idx] - z[i];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;

        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){

                    flat_idx = get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn);
                    Exl += weight[ii][jj][kk]*Ex_flat[flat_idx];
                    Eyl += weight[ii][jj][kk]*Ey_flat[flat_idx];
                    Ezl += weight[ii][jj][kk]*Ez_flat[flat_idx];
                    Bxl += weight[ii][jj][kk]*Bxn_flat[flat_idx];
                    Byl += weight[ii][jj][kk]*Byn_flat[flat_idx];
                    Bzl += weight[ii][jj][kk]*Bzn_flat[flat_idx];
            
        } // end interpolation
        
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);

        // solve the position equation
        ut= u[i] + qomdt2*Exl;
        vt= v[i] + qomdt2*Eyl;
        wt= w[i] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;

        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

        // update position
        x[i] = xptilde + uptilde*dto2;
        y[i] = yptilde + vptilde*dto2;
        z[i] = zptilde + wptilde*dto2;


    } // end of iteration
    
    // update the final position and velocity
    u[i]= 2.0*uptilde - u[i];
    v[i]= 2.0*vptilde - v[i];
    w[i]= 2.0*wptilde - w[i];
    x[i] = xptilde + uptilde*dt_sub_cycling;
    y[i] = yptilde + vptilde*dt_sub_cycling;
    z[i] = zptilde + wptilde*dt_sub_cycling;


    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (x[i] > Lx){
        if (PERIODICX==true){ // PERIODIC
            x[i] = x[i] - Lx;
        } else { // REFLECTING BC
            u[i] = -u[i];
            x[i] = 2*Lx - x[i];
        }
    }

    if (x[i] < 0){
        if (PERIODICX==true){ // PERIODIC
            x[i] = x[i] + Lx;
        } else { // REFLECTING BC
            u[i] = -u[i];
            x[i] = -x[i];
        }
    }


    // Y-DIRECTION: BC particles
    if (y[i] > Ly){
        if (PERIODICY==true){ // PERIODIC
            y[i] = y[i] - Ly;
        } else { // REFLECTING BC
            v[i] = -v[i];
            y[i] = 2*Ly - y[i];
        }
    }

    if (y[i] < 0){
        if (PERIODICY==true){ // PERIODIC
            y[i] = y[i] + Ly;
        } else { // REFLECTING BC
            v[i] = -v[i];
            y[i] = -y[i];
        }
    }

    // Z-DIRECTION: BC particles
    if (z[i] > Lz){
        if (PERIODICZ==true){ // PERIODIC
            z[i] = z[i] - Lz;
        } else { // REFLECTING BC
            w[i] = -w[i];
            z[i] = 2*Lz - z[i];
        }
    }

    if (z[i] < 0){
        if (PERIODICZ==true){ // PERIODIC
            z[i] = z[i] + Lz;
        } else { // REFLECTING BC
            w[i] = -w[i];
            z[i] = -z[i];
        }
    }
}

/** particle mover for GPU*/
int mover_PC_GPU(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,int size)
{

    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

	int gridSize = grd->nxn * grd->nyn * grd->nzn;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;


    FPpart *x, *y, *z, *u, *v, *w;
    FPfield *XN_Flat, *YN_Flat, *ZN_Flat, *Ex_flat, *Ey_Flat, *Ez_Flat, *Bxn_flat, *Byn_Flat, *Bzn_Flat;

	//copy all necessary info to gpu
    cudaMalloc(&x, part->npmax * sizeof(FPpart));
    cudaMemcpy(x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMalloc(&y, part->npmax * sizeof(FPpart));
    cudaMemcpy(y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMalloc(&z, part->npmax * sizeof(FPpart));
    cudaMemcpy(z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMalloc(&u, part->npmax * sizeof(FPpart));
    cudaMemcpy(u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMalloc(&v, part->npmax * sizeof(FPpart));
    cudaMemcpy(v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 
    cudaMalloc(&w, part->npmax * sizeof(FPpart));
    cudaMemcpy(w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMalloc(&XN_Flat, gridSize * sizeof(FPfield));
    cudaMemcpy(XN_Flat, grd->XN_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&YN_Flat, gridSize * sizeof(FPfield));
    cudaMemcpy(YN_Flat, grd->YN_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&ZN_Flat, gridSize * sizeof(FPfield));
    cudaMemcpy(ZN_Flat, grd->ZN_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&Ex_flat, gridSize * sizeof(FPfield));
    cudaMemcpy(Ex_flat, field->Ex_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&Ey_Flat, gridSize * sizeof(FPfield));
    cudaMemcpy(Ey_Flat, field->Ey_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&Ez_Flat, gridSize * sizeof(FPfield));
    cudaMemcpy(Ez_Flat, field->Ez_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&Bxn_flat, gridSize * sizeof(FPfield));
    cudaMemcpy(Bxn_flat, field->Bxn_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&Byn_Flat, gridSize * sizeof(FPfield));
    cudaMemcpy(Byn_Flat, field->Byn_flat, gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMalloc(&Bzn_Flat,gridSize * sizeof(FPfield));
    cudaMemcpy(Bzn_Flat, field->Bzn_flat,gridSize * sizeof(FPfield), cudaMemcpyHostToDevice);

	int threadPerBlock = 64;
	int blocks = (part->npmax + threadPerBlock - 1) / threadPerBlock;
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){

        MOVER_KERNEL_BRUTEFORCE<<<blocks,threadPerBlock>>>(x, y, z,u, v, w, XN_Flat, YN_Flat, ZN_Flat, grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, grd->Lx, grd->Ly, grd->Lz, grd->invVOL, Ex_flat, Ey_Flat, Ez_Flat, Bxn_flat, Byn_Flat, Bzn_Flat, param->PERIODICX, param->PERIODICY, param->PERIODICZ, dt_sub_cycling, dto2, qomdt2, part->NiterMover, part->nop);

        cudaDeviceSynchronize();

    } 
	//copy back and free
    cudaMemcpy(part->x, x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
	cudaMemcpy(field->Ex_flat, Ex_flat, gridSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_Flat, gridSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_Flat, gridSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat, gridSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_Flat, gridSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_Flat, gridSize * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(u);
    cudaFree(v);
    cudaFree(w);
    cudaFree(XN_Flat);
    cudaFree(YN_Flat);
    cudaFree(ZN_Flat);
    cudaFree(Ex_flat);
    cudaFree(Ey_Flat);
    cudaFree(Ez_Flat);
    cudaFree(Bxn_flat);
    cudaFree(Byn_Flat);
    cudaFree(Bzn_Flat);

    return(0);
}













/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
