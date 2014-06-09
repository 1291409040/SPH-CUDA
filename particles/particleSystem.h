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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, uint iteration, bool bUseOpenGL);
    ~ParticleSystem();

    enum ParticleConfig
    {
	    CONFIG_RANDOM,
	    CONFIG_GRID,
	    _NUM_CONFIGS
    };

    enum ParticleArray
    {
        POSITION,
        VELOCITY,
    };

    void update(float deltaTime);
    void reset();

    void   setArray(const float* data, int start, int count);

    int    getNumParticles() const { return m_numParticles; }

    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    unsigned int getColorBuffer()       const { return m_colorVBO; }

    void * getCudaPosVBO()              const { return (void *)m_cudaPosVBO; }
    void * getCudaColorVBO()            const { return (void *)m_cudaColorVBO; }

    void setIterations(int i) { m_solverIterations = i; }

    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
	float getParticleRadius() {return 2.0/64.0; }

    void addSphere(int index, float *pos, float *vel, int r, float spacing);

protected: // methods
    ParticleSystem() {}
    uint createVBO(uint size);

    void _initialize(int numParticles);
    void _finalize();

    void initGrid(uint *size, float spacing, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint m_numParticles;

    // CPU data
    float* m_hPos;              // particle positions

    // GPU data
    float* m_dPos;
	float* m_dPosTmp;
    float* m_dVel;
	float* m_dLambda;

	float* m_dSortedPosTmp;
	float* m_dSortedPosTmp2;

    // grid data for sorting method
    uint*  m_dGridParticleHash; // grid hash value for each particle
    uint*  m_dGridParticleIndex;// particle index for each particle
    uint*  m_dCellStart;        // index of start of each cell in sorted list
    uint*  m_dCellEnd;          // index of end of cell

    uint   m_posVbo;            // vertex buffer object for particle positions
    uint   m_colorVBO;          // vertex buffer object for colors
    
    float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

    struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint m_numGridCells;

    uint m_timer;

    uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
