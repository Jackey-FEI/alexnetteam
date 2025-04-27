//
// File:        maxpooling_layer.c
// Description: Implementation of max pooling layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include "maxpooling_layer.h"
#include <cuda_runtime.h>
#include <float.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef struct mp_args
{
    max_pooling_op *op;
    short batch_id;
} mp_args;

#define WARP 32
#define THREADS_PER_BLOCK 512
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP)
#define TILE 8

__global__ void maxpool_forward_naive(const float *__restrict__ x,
                                      float *__restrict__ y,
                                      int N, int C, int H, int W,
                                      int OH, int OW, int P_S, int P_K)
{
    const int n = blockIdx.y;          // image id
    const int cg = blockIdx.x;         // 16-channel group
    const int warp = threadIdx.x >> 5; // 0..3
    const int lane = threadIdx.x & 31; // 0..31
    const int c = cg * WARPS_PER_BLOCK + warp;
    if (c >= C)
        return;

    // 4×8 warp tile coordinates
    const int WARP_H = 4, WARP_W = 8;
    const int local_h = lane / WARP_W; // 0..3
    const int local_w = lane % WARP_W; // 0..7
    const int tiles_h = (OH + WARP_H - 1) / WARP_H;
    const int tiles_w = (OW + WARP_W - 1) / WARP_W;

    for (int th = 0; th < tiles_h; ++th)
    {
        int oy = th * WARP_H + local_h;
        if (oy >= OH)
            continue;

        for (int tw = 0; tw < tiles_w; ++tw)
        {
            int ox = tw * WARP_W + local_w;
            if (ox >= OW)
                continue;

            int ih0 = oy * P_S;
            int iw0 = ox * P_S;

            float vmax = -FLT_MAX;
            for (int ky = 0; ky < P_K; ++ky)
                for (int kx = 0; kx < P_K; ++kx)
                {
                    int ih = ih0 + ky;
                    int iw = iw0 + kx;
                    if (ih < H && iw < W)
                    {
                        float v = x[((n * C + c) * H + ih) * W + iw];
                        vmax = fmaxf(vmax, v);
                    }
                }
            y[((n * C + c) * OH + oy) * OW + ox] = vmax;
        }
    }
}

void max_pooling_op_forward(max_pooling_op *op)
{
    mp_args args[op->batchsize + 1];
    int N = op->batchsize;
    int C = op->channels;
    int H = op->in_h;
    int W = op->in_w;
    int OH = op->out_h;
    int OW = op->out_w;

    size_t in_bytes = (size_t)N * C * H * W * sizeof(float);
    size_t out_bytes = (size_t)N * C * OH * OW * sizeof(float);

    /* ---------- allocate ---------- */
    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);

    /* ---------- H→D copy ---------- */
    cudaMemcpy(d_in, op->input, in_bytes, cudaMemcpyHostToDevice);

    constexpr int TPB = THREADS_PER_BLOCK;
    int num_c_blocks = (C + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 blockDim(TPB);
    dim3 gridDim(num_c_blocks, N);

    /* ---------- kernel launch ---------- */
    maxpool_forward_naive<<<gridDim, blockDim>>>(d_in, d_out, N, C, H, W, OH, OW, op->stride, op->kernel_size);
    cudaDeviceSynchronize();
    /* ---------- D→H copy ---------- */
    cudaMemcpy(op->output, d_out, out_bytes, cudaMemcpyDeviceToHost);
    /* ---------- free memory ---------- */
    cudaFree(d_in);
    cudaFree(d_out);
}


__global__ void maxpool_backward_naive(const float *__restrict__ x,
    const float *__restrict__ d_y,
    float *__restrict__ d_x,
    int N, int C, int H, int W,
    int OH, int OW, int P_S, int P_K)
{
const int n   = blockIdx.y;          // image id
const int cg  = blockIdx.x;          // 16-channel group
const int warp  = threadIdx.x >> 5;  // 0‥15
const int lane  = threadIdx.x & 31;  // 0‥31
const int c   = cg * WARPS_PER_BLOCK + warp;
if (c >= C) return;

/* 4 × 8 warp tile  */
constexpr int WARP_H = 4, WARP_W = 8;
const int local_h = lane / WARP_W;          // 0‥3
const int local_w = lane % WARP_W;          // 0‥7
const int tiles_h = (OH + WARP_H - 1) / WARP_H;
const int tiles_w = (OW + WARP_W - 1) / WARP_W;

for (int th = 0; th < tiles_h; ++th)
{
int oy = th * WARP_H + local_h;
if (oy >= OH) continue;

for (int tw = 0; tw < tiles_w; ++tw)
{
int ox = tw * WARP_W + local_w;
if (ox >= OW) continue;

/* ---------- locate max position in this pool window ---------- */
const int ih0 = oy * P_S;
const int iw0 = ox * P_S;

float vmax = -FLT_MAX;
int   imax = -1;              
for (int ky = 0; ky < P_K; ++ky)
for (int kx = 0; kx < P_K; ++kx)
{
int ih = ih0 + ky;
int iw = iw0 + kx;
if (ih < H && iw < W)
{
int idx = ((n * C + c) * H + ih) * W + iw;
float v = x[idx];
if (v > vmax)
{
vmax = v;
imax = idx;
}
}
}

/* ---------- accumulate gradient to that position ------------- */
const int oyx_idx = ((n * C + c) * OH + oy) * OW + ox;
float g = d_y[oyx_idx] / static_cast<float>(N); 
if (imax >= 0)
atomicAdd(&d_x[imax], g);
}
}
}


void max_pooling_op_backward(max_pooling_op *op)
{
int  N  = op->batchsize;
int  C  = op->channels;
int  H  = op->in_h;
int  W  = op->in_w;
int  OH = op->out_h;
int  OW = op->out_w;

size_t in_bytes  = (size_t)N * C * H  * W  * sizeof(float);
size_t out_bytes = (size_t)N * C * OH * OW * sizeof(float);

/* ---------- device buffers ---------- */
float *d_x, *d_dx, *d_dy;
cudaMalloc(&d_x,  in_bytes);  
cudaMalloc(&d_dx, in_bytes);       
cudaMalloc(&d_dy, out_bytes);     

cudaMemcpy(d_x,  op->input,   in_bytes,  cudaMemcpyHostToDevice);
cudaMemcpy(d_dy, op->d_output, out_bytes, cudaMemcpyHostToDevice);
cudaMemset(d_dx, 0, in_bytes);

/* ---------- launch grid identical to forward ---------- */
const int TPB = THREADS_PER_BLOCK;             
int num_c_blocks = (C + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
dim3 blockDim(TPB);
dim3 gridDim(num_c_blocks, N);

maxpool_backward_naive<<<gridDim, blockDim>>>(
d_x, d_dy, d_dx, N, C, H, W, OH, OW, op->stride, op->kernel_size);
cudaDeviceSynchronize();

/* ---------- copy result back ---------- */
cudaMemcpy(op->d_input, d_dx, in_bytes, cudaMemcpyDeviceToHost);

/* ---------- cleanup ---------- */
cudaFree(d_x);
cudaFree(d_dx);
cudaFree(d_dy);
}
