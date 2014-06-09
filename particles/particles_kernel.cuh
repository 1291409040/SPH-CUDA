/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define FETCH(t, i) t[i]

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams {
    float3 gravity;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;

	float H;
	float H2;
	float c_poly6;
	float c_spiky_grad;
	float ARTI_PRESSUER;

	float restDens;
	float epsilon;

	float bedding_in;
	float bedding_out;
	float boundry_scale;

	float4 max_vel;
};

#endif
