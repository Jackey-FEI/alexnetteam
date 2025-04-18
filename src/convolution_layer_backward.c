//
// File:        convolution_layer.c
// Description: Implementation of convolution layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "convolution_layer.h"
#include "matrix.h"
#define MIN(a,b) (((a) < (b)) ? (a) : (b))


typedef struct conv_args{
    conv_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} conv_args;


static void col2img(const float *col, float *img, const conv_op *op)
{
    //
    // Todo: simplify the code
    //
    register int input_offset;
    register int iwih = op->in_w * op->in_h;
    int kk  = op->kernel_size * op->kernel_size;
    int ikk = op->in_channels * kk;

    register int st_x=0;
    for (register unsigned short out_x = 0; out_x < op->out_w; out_x++)
    {
        register int st_y=0;
        for (register unsigned short out_y = 0; out_y < op->out_h; out_y++)
        {
            for (register unsigned short in_c = 0; in_c < op->in_channels; in_c++)
            {    
                register int x_col_offset = in_c * kk + out_x*out_y*ikk;
                for (register unsigned short j = 0; j < op->kernel_size; j++)
                {
                    for (register unsigned short i = 0; i < op->kernel_size; i++, x_col_offset++)
                    {
                        if (!(st_x+i < op->in_w) | !(st_y+j < op->in_h))
                            continue;
                        
                        input_offset = (st_x+i) + (st_y+j) * op->in_w + in_c * iwih;
                        img[input_offset] = col[x_col_offset];
                    }
                }
            }
            st_y += op->stride;
        }
        st_x += op->stride;
    }
}

static void* pthread_conv_op_backward(void *argv)
{
    /**
     * pthread conv_op_backward
     * */
    conv_args args;
    memcpy(&args, (conv_args *)argv, sizeof(conv_args));

    int oc   = args.op->out_channels,
        ikk  = args.op->in_channels * args.op->kernel_size * args.op->kernel_size, 
        owoh = args.op->out_w * args.op->out_h;

    // calculate delta_weights
    short internal     = args.ed_tunits - args.st_tunits;
    float *t_input_col = (float *)malloc( owoh * internal * sizeof(float));
    float *t_d_weights = (float *)malloc( oc * internal * sizeof(float));
    for (int p = 0; p < args.op->batchsize; p++)
    {
        for (int j = 0; j < owoh; j++)
        {
            memcpy((void *)(t_input_col+j*internal), 
                    (void *)(args.op->input_col+p*owoh*ikk + j*ikk + args.st_tunits),
                        sizeof(float)*internal);
        }
        memset(t_d_weights, 0,  oc * internal * sizeof(float));
        matrix_multiply(args.op->d_output, t_input_col, t_d_weights, oc, owoh, internal);

        for (int j = 0; j < oc; j++)
        {
            register int o_offset = j*internal;
            register int oo_offset = j*ikk + args.st_tunits;
            for (int i = 0; i < internal; i++)
                args.op->d_weights[oo_offset++] += t_d_weights[o_offset++] / args.op->batchsize; 
        }
    }
    free(t_d_weights);
    free(t_input_col);

    if (args.st_tunits == 0 )
    {
        // calculte delta_input and delta_bias
        for (int i = 0; i < args.op->out_channels; i++)
        {
            register int tmp=0;
            for (int p = i*owoh; p < (i+1)*owoh; p++)
                tmp += args.op->d_output[p];
            args.op->d_bias[i] = tmp;
        }

        float *d_x_col = (float *)calloc(ikk*owoh, sizeof(float));
        matrix_transpose(args.op->d_output, owoh, args.op->out_channels);
        matrix_multiply(args.op->d_output, args.op->weights, d_x_col, owoh, args.op->out_channels, ikk);
        col2img(d_x_col, args.op->d_input, args.op);
        free(d_x_col);
    }
}

void conv_op_backward(conv_op *op)
{
    /**
     * conv2d backward
     * 
     * Input:
     *      op->d_output
     * Output:
     *      op->d_weights
     *      op->d_bias
     *      op->d_input
     * */
    short tnum = 12; // number of threads
    if (op->in_channels * op->kernel_size * op->kernel_size < tnum)
    {
        conv_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->in_channels * op->kernel_size * op->kernel_size;
        pthread_conv_op_backward((void *)(&args));
    }else {
        conv_args args[tnum+1];
        pthread_t tid[tnum+1];
        short internal = ceil(1.0 * op->in_channels * op->kernel_size * op->kernel_size / tnum);

        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, op->in_channels * op->kernel_size * op->kernel_size);            
            pthread_create(&tid[p], NULL, pthread_conv_op_backward, (void *)(&args[p]));
        }

        for (int p = 0; p < tnum; p++)
            pthread_join(tid[p], NULL);
    }
    free(op->input_col);

}

void calloc_conv_weights(conv_op *op)
{
    op->weights = (float *)calloc(op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, sizeof(float));
    op->bias    = (float *)calloc(op->out_channels, sizeof(float));
}

void free_conv_weights(conv_op *op)
{
    free(op->weights);
    free(op->bias);
}

void calloc_conv_dweights(conv_op *op)
{
    op->d_weights = (float *)calloc(op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, sizeof(float));
    op->d_bias    = (float *)calloc(op->out_channels, sizeof(float));
}

void free_conv_dweights(conv_op *op)
{
    free(op->d_weights);
    free(op->d_bias);
}

void save_conv_weights(conv_op *op, FILE *fp)
{
    fwrite(op->weights, sizeof(float), op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, fp);
    fwrite(op->bias,    sizeof(float), op->out_channels, fp);
}


void load_conv_weights(conv_op *op, FILE *fp)
{
    fread(op->weights, sizeof(float), op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, fp);
    fread(op->bias,    sizeof(float), op->out_channels, fp);
}
