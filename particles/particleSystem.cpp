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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint iteration, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_dPos(0),
	m_dPosTmp(0),
    m_dVel(0),
	m_dLambda(0),
    m_timer(0),
    m_solverIterations(iteration)
{
	float surpportRadius = 1.5;
	float3 worldSize = make_float3(48.0, 48.0, 48.0);
	m_params.worldOrigin = make_float3(-worldSize.x/2.0, -worldSize.y/2.0, -worldSize.z/2.0);

	m_gridSize.x = worldSize.x / surpportRadius + 1;
	m_gridSize.y = worldSize.y / surpportRadius + 1;
	m_gridSize.z = worldSize.z / surpportRadius + 1;

    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

    // set simulation parameters
	m_params.gravity = make_float3(-9.8f*0.0, -9.8f, -9.8f*0.0);
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;

    m_params.H = surpportRadius;
	m_params.H2 = m_params.H * m_params.H;
	m_params.c_poly6 = 315.0 / (64.0 * CUDART_PI_F * pow(m_params.H, 9));
	m_params.c_spiky_grad = -45.0 / (CUDART_PI_F * pow(m_params.H, 6));
	m_params.ARTI_PRESSUER = m_params.c_poly6 * pow(m_params.H2 - 0.04*m_params.H2, 3);

	m_params.restDens = 0.8;
	m_params.epsilon = 1.0;

	m_params.bedding_in = 0.001 * m_params.H;
	m_params.bedding_out = 1.0 * m_params.H;
	m_params.boundry_scale = 0.05;

	m_params.max_vel = make_float4(30.0, 30.0, 30.0, 2.0);

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] = {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
	    { 1.0, 1.0, 0.0, },
	    { 0.0, 1.0, 0.0, },
	    { 0.0, 1.0, 1.0, },
	    { 0.0, 0.0, 1.0, },
	    { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL) {
        m_posVbo = createVBO(memSize);    
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    } else {
        cutilSafeCall( cudaMalloc( (void **)&m_cudaPosVBO, memSize )) ;
    }

	allocateArray((void**)&m_dPosTmp, memSize);
    allocateArray((void**)&m_dVel, memSize);
	allocateArray((void**)&m_dLambda, memSize / 4);

	allocateArray((void**)&m_dSortedPosTmp, memSize);
	allocateArray((void**)&m_dSortedPosTmp2, memSize);

    allocateArray((void**)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void**)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void**)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void**)&m_dCellEnd, m_numGridCells*sizeof(uint));

    if (m_bUseOpenGL) {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
		registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
        for(uint i=0; i<m_numParticles; i++) {
            float t = i / (float) m_numParticles;

            colorRamp(t, ptr);
            ptr+=3;
            *ptr++ = 1.0f;
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER);
    } else {
        cutilSafeCall( cudaMalloc( (void **)&m_cudaColorVBO, sizeof(float)*numParticles*4) );
    }

    cutilCheckError(cutCreateTimer(&m_timer));

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;

	freeArray(m_dPosTmp);
    freeArray(m_dVel);
	freeArray(m_dLambda);
	freeArray(m_dSortedPosTmp);
	freeArray(m_dSortedPosTmp2);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint*)&m_posVbo);
        glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
    } else {
        cutilSafeCall( cudaFree(m_cudaPosVBO) );
        cutilSafeCall( cudaFree(m_cudaColorVBO) );
    }
}

// step the simulation
void 
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL) {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    } else {
        dPos = (float *) m_cudaPosVBO;
    }

    // update constants
    setParameters(&m_params);

    // integrate
    integrateSystem(
        dPos,
		m_dPosTmp,
        m_dVel,
        deltaTime,
        m_numParticles);

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPosTmp,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
		m_dSortedPosTmp,
        m_dGridParticleHash,
        m_dGridParticleIndex,
		m_dPosTmp,
		m_numParticles,
		m_numGridCells);

	for(int i=0; i<m_solverIterations; i++){
		calLambda(
			m_dSortedPosTmp,
			m_dLambda,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_numParticles,
			m_numGridCells);

		calDeltaP_Collision_UpdatePos(
			m_dSortedPosTmp,
			m_dSortedPosTmp2,
			m_dLambda,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_numParticles,
			m_numGridCells);
	}

	updateVelPos(
		deltaTime,
		dPos,
		m_dVel,
        m_dSortedPosTmp,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);
    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL) {
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
}

void
ParticleSystem::setArray(const float* data, int start, int count)
{
    assert(m_bInitialized);
    if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
        glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    }    
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, uint numParticles)
{
    srand(1973);
    for(uint z=0; z<size[2]; z++) {
        for(uint y=0; y<size[1]; y++) {
            for(uint x=0; x<size[0]; x++) {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;
                if (i < numParticles) {
                    m_hPos[i*4] = 1.0 - (spacing * x) - 1.0f;
                    m_hPos[i*4+1] = 1.0 - (spacing * y) - 1.0f;
                    m_hPos[i*4+2] = 1.0 - (spacing * z) - 1.0f;
                }
            }
        }
    }
}

void
ParticleSystem::reset()
{
    uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
    uint gridSize[3];
    gridSize[0] = gridSize[1] = gridSize[2] = s;
    initGrid(gridSize, m_params.H*0.7f, m_numParticles);

    setArray(m_hPos, 0, m_numParticles);
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;
    for(int z=-r; z<=r; z++) {
        for(int y=-r; y<=r; y++) {
            for(int x=-r; x<=r; x++) {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                if ((l <= m_params.H * r) && (index < m_numParticles)) {
                    m_hPos[index*4]   = pos[0] + dx;
                    m_hPos[index*4+1] = pos[1] + dy;
                    m_hPos[index*4+2] = pos[2] + dz;
                    m_hPos[index*4+3] = pos[3];
                    index++;
                }
            }
        }
    }

    setArray(m_hPos, start, index);
}
