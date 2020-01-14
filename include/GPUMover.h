#ifndef GPUMOVER_H
#define GPUMOVER_H

#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"


void copyGrid(grid *cpu_grid,grid *gpu_grid, size_t size);

void copyParam(parameters *param, parameters *gpu_param);

void copyEMfield(EMfield *emf, EMfield *gpu_emf, size_t size);


#endif