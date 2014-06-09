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
 
 extern "C"
{
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


void setParameters(SimParams *hostParams);

void integrateSystem(float *pos,
					 float *posTmp,	
                     float *vel,
                     float deltaTime,
                     uint numParticles);

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* posTmp, 
              int    numParticles);

void reorderDataAndFindCellStart(uint*  cellStart,
							     uint*  cellEnd,
								 float* sortedPosTmp,
                                 uint*  gridParticleHash,
                                 uint*  gridParticleIndex,
								 float* oldPosTmp,
							     uint   numParticles,
							     uint   numCells);

void calLambda(float* sortedPosTmp,
			   float* lambda,
			   uint*  gridParticleIndex,
               uint*  cellStart,
               uint*  cellEnd,
               uint   numParticles,
               uint   numCells);

void calDeltaP_Collision_UpdatePos(float* sortedPosTmp,
			   float* sortedPosTmp2,
			   float* lambda,
			   uint*  gridParticleIndex,
               uint*  cellStart,
               uint*  cellEnd,
               uint   numParticles,
               uint   numCells);
void updateVelPos(float deltaTime,
			 float* newPos,
			 float* newVel,
             float* sortedPosTmp,
             uint*  gridParticleIndex,
             uint*  cellStart,
             uint*  cellEnd,
             uint   numParticles,
             uint   numCells);

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
