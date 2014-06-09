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

/* 
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;


struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
		volatile float4 posTmpData = thrust::get<1>(t);
        volatile float4 velData = thrust::get<2>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 posTmp = make_float3(posTmpData.x, posTmpData.y, posTmpData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        vel += params.gravity * deltaTime;
        posTmp = pos + vel * deltaTime;
        // store new position and velocity
		thrust::get<1>(t) = make_float4(posTmp, posTmpData.w);
        thrust::get<2>(t) = make_float4(vel, velData.w);
    }
};

__device__ bool isValidValue(int x, int s, int e){
	if(x >= s && x < e) return true;
	return false;
}

__device__ int validValue(int x, int s, int e){
    if(x < s) x = s;
    if(x >= e) x = e - 1;
    return x;
}

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.H);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.H);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.H);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint*   gridParticleHash,  // output
               uint*   gridParticleIndex, // output
               float4* posTmp,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;
    
    volatile float4 p = posTmp[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));

	gridPos.x = validValue(gridPos.x, 0, params.gridSize.x);
	gridPos.y = validValue(gridPos.y, 0, params.gridSize.y);
	gridPos.z = validValue(gridPos.z, 0, params.gridSize.z);

    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
							      uint*   cellEnd,          // output: cell end index
								  float4* sortedPosTmp,     // output: sorted positions
                                  uint *  gridParticleHash, // input: sorted grid hashes
                                  uint *  gridParticleIndex,// input: sorted particle indices
								  float4* oldPosTmp,        // input: sorted position array
							      uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
    uint hash;
    // handle case when no. of particles not multiple of block size
    if (index < numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
	    sharedHash[threadIdx.x+1] = hash;

	    if (index > 0 && threadIdx.x == 0)
	    {
		    // first thread in block must load neighbor particle hash
		    sharedHash[0] = gridParticleHash[index-1];
	    }
	}

	__syncthreads();
	
	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

	    if (index == 0 || hash != sharedHash[threadIdx.x])
	    {
		    cellStart[hash] = index;
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
	    }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

	    // Now use the sorted index to reorder the pos and vel data
	    uint sortedIndex = gridParticleIndex[index];
		float4 posTmp = FETCH(oldPosTmp, sortedIndex);       // macro does either global read or texture fetch

		sortedPosTmp[index] = posTmp;
	}
}

__device__
float lenSquare(float3 vec)
{
	return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}
__device__
float len(float3 vec)
{
	return sqrt(lenSquare(vec));
}

__device__
float k_poly6(float3 vec)
{
	float tmp = params.H2 - lenSquare(vec);
	if(tmp <= 0.0) return  0.0;
    return params.c_poly6 * tmp * tmp * tmp;
}

__device__
float3 k_spiky_grad(float3 vec)
{
	float l = max(len(vec), 1e-8);
    float tmp = params.H - l;
    if(tmp <= 0.0) return make_float3(0.0, 0.0, 0.0);
    return (tmp * tmp * params.c_spiky_grad) * vec / l;
}

__global__
void calLambdaD(float4* posTmp,               // input: sorted positions
                float* lambda,               // output: lambda
                uint*   gridParticleIndex,    // input: sorted particle indices
                uint*   cellStart,
                uint*   cellEnd,
                uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    
    
    // read particle data from sorted arrays
	float3 pos = make_float3(FETCH(posTmp, index));
	//
	float sum_k_grad_Ci = 0.0;
    float rho = 0.0;
    float3 grad_pi_Ci = make_float3(0.0, 0.0, 0.0);
    // get address in grid
    int3 gridPos = calcGridPos(pos);

	gridPos.x = validValue(gridPos.x, 0, params.gridSize.x);
	gridPos.y = validValue(gridPos.y, 0, params.gridSize.y);
	gridPos.z = validValue(gridPos.z, 0, params.gridSize.z);

    // examine neighbouring cells
    for(int z=-1; z<=1; z++) {
        for(int y=-1; y<=1; y++) {
            for(int x=-1; x<=1; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);

				if(!isValidValue(neighbourPos.x, 0, params.gridSize.x)) continue;
				if(!isValidValue(neighbourPos.y, 0, params.gridSize.y)) continue;
				if(!isValidValue(neighbourPos.z, 0, params.gridSize.z)) continue;

				uint gridHash = calcGridHash(neighbourPos);

				// get start of bucket for this cell
				uint startIndex = FETCH(cellStart, gridHash);

				if (startIndex != 0xffffffff) {        // cell is not empty
					// iterate over particles in this cell
					uint endIndex = FETCH(cellEnd, gridHash);
					for(uint j=startIndex; j<endIndex; j++) {
						if (true) {              // check not colliding with self
							float3 pos2 = make_float3(FETCH(posTmp, j));

							// density
							rho += k_poly6(pos - pos2);
							// gradients of Ci
							float3 grad_pk_Ci = make_float3(0.0, 0.0, 0.0);
							grad_pk_Ci = k_spiky_grad(pos - pos2);
							grad_pk_Ci /= params.restDens;
							sum_k_grad_Ci += lenSquare(grad_pk_Ci);
							// k = i
							grad_pi_Ci += grad_pk_Ci;
						}
					}
				}
            }
        }
    }
	//
	sum_k_grad_Ci += lenSquare(grad_pi_Ci);
	float Ci = rho / params.restDens - 1;
    lambda[index] = -Ci / (sum_k_grad_Ci + params.epsilon);
}

__device__
float3 collision(float3 pos)
{
	float p;
	// x
	if(pos.x < params.worldOrigin.x + params.bedding_out){
        p = params.worldOrigin.x + params.bedding_out - pos.x;
		pos.x = params.worldOrigin.x + params.bedding_out - p / (p * p * params.boundry_scale + 1.0);
    }
    if(pos.x < params.worldOrigin.x + params.bedding_in){
        pos.x = params.worldOrigin.x + params.bedding_in;
    }
	if(pos.x > -params.worldOrigin.x - params.bedding_out){
        p = pos.x + params.worldOrigin.x + params.bedding_out;
        pos.x = p / (p * p * params.boundry_scale + 1.0) - params.worldOrigin.x - params.bedding_out;
    }
    if(pos.x > -params.worldOrigin.x - params.bedding_in){
        pos.x = -params.worldOrigin.x - params.bedding_in;
    }
	// y
	if(pos.y < params.worldOrigin.y + params.bedding_out){
        p = params.worldOrigin.y + params.bedding_out - pos.y;
		pos.y = params.worldOrigin.y + params.bedding_out - p / (p * p * params.boundry_scale + 1.0);
    }
    if(pos.y < params.worldOrigin.y + params.bedding_in){
        pos.y = params.worldOrigin.y + params.bedding_in;
    }
	if(pos.y > -params.worldOrigin.y - params.bedding_out){
        p = pos.y + params.worldOrigin.y + params.bedding_out;
        pos.y = p / (p * p * params.boundry_scale + 1.0) - params.worldOrigin.y - params.bedding_out;
    }
    if(pos.y > -params.worldOrigin.y - params.bedding_in){
        pos.y = -params.worldOrigin.y - params.bedding_in;
    }
	// z
	if(pos.z < params.worldOrigin.z + params.bedding_out){
        p = params.worldOrigin.z + params.bedding_out - pos.z;
		pos.z = params.worldOrigin.z + params.bedding_out - p / (p * p * params.boundry_scale + 1.0);
    }
    if(pos.z < params.worldOrigin.z + params.bedding_in){
        pos.z = params.worldOrigin.z + params.bedding_in;
    }
	if(pos.z > -params.worldOrigin.z - params.bedding_out){
        p = pos.z + params.worldOrigin.z + params.bedding_out;
        pos.z = p / (p * p * params.boundry_scale + 1.0) - params.worldOrigin.z - params.bedding_out;
    }
    if(pos.z > -params.worldOrigin.z - params.bedding_in){
        pos.z = -params.worldOrigin.z - params.bedding_in;
    }
	return pos;
}

__global__
void calDeltaP_Collision_UpdatePosD(float4* posTmp,  // input: sorted positions
				float4* posTmp2,  // input: sorted		
                float* lambda,               // output: lambda
                uint*   gridParticleIndex,    // input: sorted particle indices
                uint*   cellStart,
                uint*   cellEnd,
                uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    
    
    // read particle data from sorted arrays
	float3 pos = make_float3(FETCH(posTmp, index));
	float _lambda = FETCH(lambda, index);
	//
    float3 deltaP = make_float3(0.0, 0.0, 0.0);
    // get address in grid
    int3 gridPos = calcGridPos(pos);
	gridPos.x = validValue(gridPos.x, 0, params.gridSize.x);
	gridPos.y = validValue(gridPos.y, 0, params.gridSize.y);
	gridPos.z = validValue(gridPos.z, 0, params.gridSize.z);
    // examine neighbouring cells
    for(int z=-1; z<=1; z++) {
        for(int y=-1; y<=1; y++) {
            for(int x=-1; x<=1; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);

				if(!isValidValue(neighbourPos.x, 0, params.gridSize.x)) continue;
				if(!isValidValue(neighbourPos.y, 0, params.gridSize.y)) continue;
				if(!isValidValue(neighbourPos.z, 0, params.gridSize.z)) continue;

				uint gridHash = calcGridHash(neighbourPos);

				// get start of bucket for this cell
				uint startIndex = FETCH(cellStart, gridHash);

				if (startIndex != 0xffffffff) {        // cell is not empty
					// iterate over particles in this cell
					uint endIndex = FETCH(cellEnd, gridHash);
					for(uint j=startIndex; j<endIndex; j++) {
						if (j != index) {              // check not colliding with self
							float3 pos2 = make_float3(FETCH(posTmp, j));
							float _lambda2 = FETCH(lambda, j);
							//
							float corr = -0.1 * pow(k_poly6(pos - pos2) / params.ARTI_PRESSUER, 4);
							deltaP += (_lambda + _lambda2 + corr) * k_spiky_grad(pos - pos2);
						}
					}
				}
            }
        }
    }
	//
    deltaP /= params.restDens;
	pos += deltaP;
    pos = collision(pos);
	posTmp2[index] = make_float4(pos, 0.0f);
}

__global__
void updateVelPosD(float deltaTime,
			  float4* newPos,               // output: new velocity
              float4* newVel,               // input: sorted positions
              float4* sortedPosTmp,               // input: sorted velocities
              uint*   gridParticleIndex,    // input: sorted particle indices
              uint*   cellStart,
              uint*   cellEnd,
              uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;    
	uint originalIndex = gridParticleIndex[index];
    
    // read particle data from sorted arrays
	float3 pos = make_float3(FETCH(newPos, originalIndex));
	float3 posTmp = make_float3(FETCH(sortedPosTmp, index));

    // write new velocity back to original unsorted location
	float3 _vel = (posTmp - pos) / deltaTime;

	if(_vel.x > params.max_vel.x) _vel.x = params.max_vel.x;
	if(_vel.y > params.max_vel.y) _vel.y = params.max_vel.y;
	if(_vel.z > params.max_vel.z) _vel.z = params.max_vel.z;

    newVel[originalIndex] = make_float4(_vel, 0.0f);
	newPos[originalIndex] = make_float4(posTmp, 0.0f);
}

#endif
