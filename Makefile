CC = gcc
NVCC = nvcc

TARGET = alexnet
OBJDIR = ./obj/
SRC = ./src/
CFLAGS = -w -lm -lpthread
NVCCFLAGS = -O2 -arch=sm_52

LAYER_OBJ = activation_layer.o batchnorm_layer.o convolution_layer_forward.o convolution_layer_backward.o dropout_layer.o maxpooling_layer.o fc_layer.o
LAYER_OBJS = $(addprefix $(OBJDIR), $(LAYER_OBJ))

TAR_OBJ = train.o inference.o data.o
TAR_OBJS = $(addprefix $(OBJDIR), $(TAR_OBJ))

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJDIR)alexnet.o $(OBJDIR)matrix.o $(TAR_OBJS) $(LAYER_OBJS)
	$(NVCC) -g -o $@ \
		$(OBJDIR)alexnet.o \
		$(OBJDIR)matrix.o \
		$(TAR_OBJS) \
		$(LAYER_OBJS) \
		-lm -lpthread

$(OBJDIR)alexnet.o: $(SRC)alexnet.c
	$(NVCC) -g -Xcompiler "$(CFLAGS)" -c $< -o $@

$(OBJDIR)matrix.o: $(SRC)matrix.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)train.o: $(SRC)train.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)inference.o: $(SRC)inference.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)data.o: $(SRC)data.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)activation_layer.o: $(SRC)activation_layer.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)batchnorm_layer.o: $(SRC)batchnorm_layer.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

# CUDA file compiled with nvcc
$(OBJDIR)convolution_layer_forward.o: $(SRC)convolution_layer_forward.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)convolution_layer_backward.o: $(SRC)convolution_layer_backward.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)dropout_layer.o: $(SRC)dropout_layer.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

$(OBJDIR)maxpooling_layer.o: $(SRC)maxpooling_layer_sharedmem.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)fc_layer.o: $(SRC)fc_layer.c
	$(CC) -g -o $@ -c $< $(CFLAGS)

all: $(OBJDIR) $(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET)
