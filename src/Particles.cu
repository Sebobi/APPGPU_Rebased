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


/** particle kernel */

__global__ void single_particle_kernel(FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q, FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, int nxn, int nyn, int nzn, double xStart, double yStart, double zStart, FPfield invdx, FPfield invdy, FPfield invdz, double Lx, double Ly, double Lz, FPfield invVOL, FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat, FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat, bool PERIODICX, bool PERIODICY, bool PERIODICZ, FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2, int NiterMover, int npmax)
{
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    int flat_idx = 0;
    
    if(idx > npmax)
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

    xptilde = x[idx];
    yptilde = y[idx];
    zptilde = z[idx];

    // calculate the average velocity iteratively
    for(int innter=0; innter < NiterMover; innter++){
        
        // interpolation G-->P
        ix = 2 +  int((x[idx] - xStart)*invdx);
        iy = 2 +  int((y[idx] - yStart)*invdy);
        iz = 2 +  int((z[idx] - zStart)*invdz);

        // calculate weights

        flat_idx = get_idx(ix-1, iy, iz, nyn, nzn);
        xi[0]   = x[idx] - XN_flat[flat_idx];

        flat_idx = get_idx(ix, iy-1, iz, nyn, nzn);
        eta[0]  = y[idx] - YN_flat[flat_idx];

        flat_idx = get_idx(ix, iy, iz-1, nyn, nzn);
        zeta[0] = z[idx] - ZN_flat[flat_idx];

        flat_idx = get_idx(ix, iy, iz, nyn, nzn);
        xi[1]   = XN_flat[flat_idx] - x[idx];
        eta[1]  = YN_flat[flat_idx] - y[idx];
        zeta[1] = ZN_flat[flat_idx] - z[idx];

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
        ut= u[idx] + qomdt2*Exl;
        vt= v[idx] + qomdt2*Eyl;
        wt= w[idx] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;

        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

        // update position
        x[idx] = xptilde + uptilde*dto2;
        y[idx] = yptilde + vptilde*dto2;
        z[idx] = zptilde + wptilde*dto2;


    } // end of iteration
    
    // update the final position and velocity
    u[idx]= 2.0*uptilde - u[idx];
    v[idx]= 2.0*vptilde - v[idx];
    w[idx]= 2.0*wptilde - w[idx];
    x[idx] = xptilde + uptilde*dt_sub_cycling;
    y[idx] = yptilde + vptilde*dt_sub_cycling;
    z[idx] = zptilde + wptilde*dt_sub_cycling;


    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (x[idx] > Lx){
        if (PERIODICX==true){ // PERIODIC
            x[idx] = x[idx] - Lx;
        } else { // REFLECTING BC
            u[idx] = -u[idx];
            x[idx] = 2*Lx - x[idx];
        }
    }

    if (x[idx] < 0){
        if (PERIODICX==true){ // PERIODIC
            x[idx] = x[idx] + Lx;
        } else { // REFLECTING BC
            u[idx] = -u[idx];
            x[idx] = -x[idx];
        }
    }


    // Y-DIRECTION: BC particles
    if (y[idx] > Ly){
        if (PERIODICY==true){ // PERIODIC
            y[idx] = y[idx] - Ly;
        } else { // REFLECTING BC
            v[idx] = -v[idx];
            y[idx] = 2*Ly - y[idx];
        }
    }

    if (y[idx] < 0){
        if (PERIODICY==true){ // PERIODIC
            y[idx] = y[idx] + Ly;
        } else { // REFLECTING BC
            v[idx] = -v[idx];
            y[idx] = -y[idx];
        }
    }

    // Z-DIRECTION: BC particles
    if (z[idx] > Lz){
        if (PERIODICZ==true){ // PERIODIC
            z[idx] = z[idx] - Lz;
        } else { // REFLECTING BC
            w[idx] = -w[idx];
            z[idx] = 2*Lz - z[idx];
        }
    }

    if (z[idx] < 0){
        if (PERIODICZ==true){ // PERIODIC
            z[idx] = z[idx] + Lz;
        } else { // REFLECTING BC
            w[idx] = -w[idx];
            z[idx] = -z[idx];
        }
    }
}

/** particle mover for GPU*/
int mover_PC_GPU(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,int size)
{
    // print species and subcycling
    std::cout << "***GPU MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

    // allocate memory for variables on device

    FPpart *x_dev = NULL, *y_dev = NULL, *z_dev = NULL, *u_dev = NULL, *v_dev = NULL, *w_dev = NULL;
    FPinterp *q_dev = NULL;
    FPfield *XN_flat_dev = NULL, *YN_flat_dev = NULL, *ZN_flat_dev = NULL, *Ex_flat_dev = NULL, *Ey_flat_dev = NULL, *Ez_flat_dev = NULL, *Bxn_flat_dev = NULL, *Byn_flat_dev, *Bzn_flat_dev = NULL;

    cudaMalloc(&x_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(x_dev, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 

    cudaMalloc(&y_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(y_dev, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 

    cudaMalloc(&z_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(z_dev, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 

    cudaMalloc(&u_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(u_dev, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 

    cudaMalloc(&v_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(v_dev, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); 

    cudaMalloc(&w_dev, part->npmax * sizeof(FPpart));
    cudaMemcpy(w_dev, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&q_dev, part->npmax * sizeof(FPinterp));
    cudaMemcpy(q_dev, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);  

    cudaMalloc(&XN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(XN_flat_dev, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&YN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(YN_flat_dev, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&ZN_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(ZN_flat_dev, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    
    cudaMalloc(&Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ex_flat_dev, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ey_flat_dev, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Ez_flat_dev, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Bxn_flat_dev, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Byn_flat_dev, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(Bzn_flat_dev, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    std::cout<<"Before loop"<<std::endl;
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){

        // Call GPU kernel

        single_particle_kernel<<<(part->npmax + TPB - 1)/TPB, TPB>>>(x_dev, y_dev, z_dev,u_dev, v_dev, w_dev, q_dev, XN_flat_dev, YN_flat_dev, ZN_flat_dev, grd->nxn, grd->nyn, grd->nzn, grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, grd->Lx, grd->Ly, grd->Lz, grd->invVOL, Ex_flat_dev, Ey_flat_dev, Ez_flat_dev, Bxn_flat_dev, Byn_flat_dev, Bzn_flat_dev, param->PERIODICX, param->PERIODICY, param->PERIODICZ, dt_sub_cycling, dto2, qomdt2, part->NiterMover, part->nop);

        cudaDeviceSynchronize();

    } // end of one particle

    // copy memory back to CPU (only the parts that have been modified inside the kernel)

    cudaMemcpy(part->x, x_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, y_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, z_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, u_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, v_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, w_dev, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ex_flat, Ex_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, Bxn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_dev, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    
    // clean up
   
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);

    cudaFree(XN_flat_dev);
    cudaFree(YN_flat_dev);
    cudaFree(ZN_flat_dev);

    cudaFree(Ex_flat_dev);
    cudaFree(Ey_flat_dev);
    cudaFree(Ez_flat_dev);
    cudaFree(Bxn_flat_dev);
    cudaFree(Byn_flat_dev);
    cudaFree(Bzn_flat_dev);

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
