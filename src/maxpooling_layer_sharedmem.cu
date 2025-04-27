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

void max_pooling_op_backward(max_pooling_op *op)
{
    int channels = op->channels;
    int pool_size = op->kernel_size;
    int in_w = op->in_w;
    int in_h = op->in_h;
    int out_w = op->out_w;
    int out_h = op->out_h;
    register int iwih = in_w * in_h;
    register int owoh = out_w * out_h;

    int in_x, in_y;
    float max_value, cur_value;
    int x, y;
    register int in_shift, out_shift;
    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < op->out_w; i++)
        {
            for (int j = 0; j < op->out_h; j++)
            {
                for (int p = 0; p < op->batchsize; p++)
                {
                    //
                    // output[p][c][i][j]
                    //
                    x = i * pool_size;
                    y = j * pool_size;
                    max_value = -1111111;
                    while (x < MIN((i + 1) * pool_size, in_w))
                    {
                        while (y < MIN((j + 1) * pool_size, in_h))
                        {
                            cur_value = op->input[p * channels * iwih + c * iwih + y * in_w + x];
                            if (cur_value > max_value)
                            {
                                max_value = cur_value;
                                in_x = x;
                                in_y = y;
                            }
                            y++;
                        }
                        x++;
                    }

                    in_shift = c * iwih + in_y * in_w + in_x;
                    out_shift = c * owoh + j * out_w + i;
                    op->d_input[in_shift] += op->d_output[out_shift] / op->batchsize;
                }
            }
        }
    }
}