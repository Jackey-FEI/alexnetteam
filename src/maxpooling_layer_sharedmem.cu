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
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP) // 16
#define TILE 16

// P_S: stride
// P_K: kernel size
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

    const int SH_W = TILE + P_K - 1; // 10 when TILE = 8 and K = 3
    const int SH_H = SH_W;           // square
    const int SH_SIZE = SH_W * SH_H; // 100 floats
    extern __shared__ float sm[];    // shared memory for one Block
    float *warp_smem = sm + warp * SH_SIZE;

    for (int th = 0; th < tiles_h; ++th)
    {
        int oy0 = th * WARP_H; // top-left output y of tile
        int ih0 = oy0 * P_S;   // top-left input y

        for (int tw = 0; tw < tiles_w; ++tw)
        {
            int ox0 = tw * WARP_W;
            int iw0 = ox0 * P_S;

            // shared memory is not large enough to hold one image's single channel
            // so we need to load 10×10 tile (with halo) for each warp
            /* ---------- 1. load 10×10 tile (+halo) ---------- */
            // if condition: we only need certain threads in the warp to load
            // the shareed memory, this is deceided by the tile size, the kernel size and the stride
            if (((SH_W - P_K + 1) / P_S == 0) || lane % ((SH_W - P_K + 1) / P_S) == 0)
            {
                // load the first row
                for (int t = lane; t < SH_W; t += WARP)
                {
                    int dy = t / SH_W; // 0..9
                    int dx = t % SH_W; // 0..9
                    int ih = ih0 + dy;
                    int iw = iw0 + dx;

                    float v = (ih < H && iw < W)
                                  ? x[((n * C + c) * H + ih) * W + iw]
                                  : -FLT_MAX;
                    warp_smem[dy * SH_W + dx] = v;
                }
            }
            __syncwarp(); // tile ready

            /* ---------- 2. compute this warp’s 4×4 outputs ---------- */
            int oy = oy0 + local_h;
            int ox = ox0 + local_w;
            if (oy < OH && ox < OW)
            {
                float vmax = -FLT_MAX;
                for (int ky = 0; ky < P_K; ++ky)
                    for (int kx = 0; kx < P_K; ++kx)
                    {
                        float v = warp_smem[(local_h * P_S + ky) * SH_W + (local_w * P_S + kx)];
                        vmax = fmaxf(vmax, v);
                    }
                y[((n * C + c) * OH + oy) * OW + ox] = vmax;
            }
            __syncwarp(); // allow sm reuse
        }
    }
}

void max_pooling_op_forward(max_pooling_op *op)
{
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
    size_t smem_bytes = WARPS_PER_BLOCK * (TILE + op->kernel_size - 1) * (TILE + op->kernel_size - 1) * sizeof(float);
    maxpool_forward_naive<<<gridDim, blockDim, smem_bytes>>>(d_in, d_out, N, C, H, W, OH, OW, op->stride, op->kernel_size);
    cudaDeviceSynchronize();
    /* ---------- D→H copy ---------- */
    cudaMemcpy(op->output, d_out, out_bytes, cudaMemcpyDeviceToHost);
    /* ---------- free memory ---------- */
    cudaFree(d_in);
    cudaFree(d_out);
}

// Shared-memory backward kernel
__global__ void maxpool_backward_shared(const float *__restrict__ x,
    const float *__restrict__ dy,
    float *__restrict__ dx,
    int N, int C, int H, int W,
    int OH, int OW, int stride, int K)
{
    const int n = blockIdx.y;
    const int cg = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = cg * WARPS_PER_BLOCK + warp;
    if (c >= C) return;

    const int WARP_H = 4, WARP_W = 8;
    const int local_h = lane / WARP_W;
    const int local_w = lane % WARP_W;
    const int tiles_h = (OH + WARP_H - 1) / WARP_H;
    const int tiles_w = (OW + WARP_W - 1) / WARP_W;

    const int SH_W = TILE + K - 1;
    const int SH_H = SH_W;
    const int SH_SIZE = SH_W * SH_H;
    extern __shared__ float smem[];
    float *warp_smem = smem + warp * SH_SIZE;

    for (int th = 0; th < tiles_h; ++th) {
        int oy0 = th * WARP_H;
        int ih0 = oy0 * stride;
        for (int tw = 0; tw < tiles_w; ++tw) {
            int ox0 = tw * WARP_W;
            int iw0 = ox0 * stride;
            if (((SH_W - K + 1) / stride == 0) || lane % ((SH_W - K + 1) / stride) == 0) {
                for (int t = lane; t < SH_SIZE; t += WARP) {
                    int dy_s = t / SH_W;
                    int dx_s = t % SH_W;
                    int ih = ih0 + dy_s;
                    int iw = iw0 + dx_s;
                    float v = (ih < H && iw < W) ? x[((n * C + c) * H + ih) * W + iw] : -FLT_MAX;
                    warp_smem[dy_s * SH_W + dx_s] = v;
                }
            }
            __syncwarp();
            int oy = oy0 + local_h;
            int ox = ox0 + local_w;
            if (oy < OH && ox < OW) {
                float max_val = -FLT_MAX;
                int max_dy = 0, max_dx = 0;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        int idx_y = local_h * stride + ky;
                        int idx_x = local_w * stride + kx;
                        float v = warp_smem[idx_y * SH_W + idx_x];
                        if (v > max_val) {
                            max_val = v;
                            max_dy = idx_y;
                            max_dx = idx_x;
                        }
                    }
                }
                int out_idx = ((n * C + c) * OH + oy) * OW + ox;
                float grad = dy[out_idx] / N;
                int target_h = ih0 + max_dy;
                int target_w = iw0 + max_dx;
                atomicAdd(&dx[((n * C + c) * H + target_h) * W + target_w], grad);
            }
            __syncwarp();
        }
    }
}

void max_pooling_op_backward(max_pooling_op *op)
{
    int N = op->batchsize;
    int C = op->channels;
    int H = op->in_h;
    int W = op->in_w;
    int OH = op->out_h;
    int OW = op->out_w;
    int stride = op->stride;
    int K = op->kernel_size;

    size_t in_bytes = (size_t)N * C * H * W * sizeof(float);
    size_t out_bytes = (size_t)N * C * OH * OW * sizeof(float);

    float *d_x, *d_y, *d_dx;
    cudaMalloc(&d_x, in_bytes);
    cudaMalloc(&d_y, out_bytes);
    cudaMalloc(&d_dx, in_bytes);

    cudaMemcpy(d_x, op->input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, op->d_output, out_bytes, cudaMemcpyHostToDevice);

    cudaMemset(d_dx, 0, in_bytes);

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((C + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, N);
    size_t smem = WARPS_PER_BLOCK * (TILE + K - 1) * (TILE + K - 1) * sizeof(float);

    maxpool_backward_shared<<<gridDim, blockDim, smem>>>(d_x, d_y, d_dx, N, C, H, W, OH, OW, stride, K);
    cudaDeviceSynchronize();

    if (op->d_input) free(op->d_input);
    op->d_input = (float *)malloc(in_bytes);
    cudaMemcpy(op->d_input, d_dx, in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dx);
}