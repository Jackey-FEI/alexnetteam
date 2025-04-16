#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "convolution_layer.h"

__global__ void conv2d_forward_kernel(
    const float *input,
    const float *weights,
    const float *bias,
    float *output,
    int batchsize,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int ksize, int stride,
    int in_units, int out_units)
{
    int batch_id = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x / out_w;
    int ow = blockIdx.x % out_w;

    int output_idx = batch_id * out_units + oc * (out_h * out_w) + oh * out_w + ow;
    float sum = bias[oc];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int kw = 0; kw < ksize; ++kw) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                if (ih < in_h && iw < in_w) {
                    int input_idx = batch_id * in_units + ic * in_h * in_w + ih * in_w + iw;
                    int weight_idx = oc * (in_c * ksize * ksize) + ic * ksize * ksize + kh * ksize + kw;

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    output[output_idx] = sum;
}

static void img2col(const float *img, float *col, const conv_op *op)
{
    /**
     * Output
     *      col[ikk][owoh]
     * */

    //
    // Todo: simplify the code
    //
    register int input_offset;
    register int iwih = op->in_w*op->in_h;
    register int kk   = op->kernel_size* op->kernel_size;
    register int ikk  = op->in_channels * kk;
    register const float *input = img;
    register float *x_col = col;
    for (register unsigned short in_c = 0; in_c < op->in_channels; in_c++)
    {
        register int x_col_offset = in_c * kk;
        for (register int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride)
        {
            for (register int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, x_col_offset += ikk)
            {
                for (register unsigned short j = 0; j < op->kernel_size; j++)
                {
                    for (register unsigned short i = 0; i < op->kernel_size; i++, x_col_offset++)
                    {
                        if (!(st_x+i <op->in_w) | !(st_y+j <op->in_h))
                        {
                            x_col[x_col_offset] = 0;
                            continue;
                        }

                        input_offset = (st_x+i) + (st_y+j) * op->in_w + in_c * iwih;
                        x_col[x_col_offset] = input[input_offset];
                    }
                }
            }
        }
        ikk += kk;
    }
}

__host__ void conv_op_forward(conv_op *op) {
    op->input_col = (float *)calloc((op->batchsize)*(op->in_channels * op->kernel_size* op->kernel_size)*(op->out_w * op->out_h), sizeof(float));
    for(int p = 0; p < op->batchsize; p++) {
        float *x_col    = op->input_col + p * op->in_units;
        float *t_input  = op->input + p * op->in_units;
        img2col(t_input, x_col, op);
    }

    int in_units = op->in_channels * op->in_h * op->in_w;
    int out_units = op->out_channels * op->out_h * op->out_w;

    float *d_input, *d_weights, *d_bias, *d_output;

    size_t input_size = sizeof(float) * op->batchsize * in_units;
    size_t weight_size = sizeof(float) * op->out_channels * op->in_channels * op->kernel_size * op->kernel_size;
    size_t bias_size = sizeof(float) * op->out_channels;
    size_t output_size = sizeof(float) * op->batchsize * out_units;

    // Allocate device memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weights, weight_size);
    cudaMalloc(&d_bias, bias_size);
    cudaMalloc(&d_output, output_size);

    // Copy data to device
    cudaMemcpy(d_input, op->input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, op->weights, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, op->bias, bias_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Configure grid and block dimensions
    dim3 blockDim(1);
    dim3 gridDim(op->out_w * op->out_h, op->out_channels, op->batchsize);

    // Launch the kernel
    conv2d_forward_kernel<<<gridDim, blockDim>>>(
        d_input, d_weights, d_bias, d_output,
        op->batchsize,
        op->in_channels, op->in_h, op->in_w,
        op->out_channels, op->out_h, op->out_w,
        op->kernel_size, op->stride,
        in_units, out_units
    );

    // Copy the result back to host
    cudaMemcpy(op->output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}
