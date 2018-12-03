.SUFFIXES: .cc .cu

CC = nvcc
CFLAGS = -std=c++11 -O3 -Xcompiler -ansi -Xcompiler -Ofast -Wno-deprecated-gpu-targets
INCLUDES = -I $(CUDA_HOME)/include/ -I ./common/
LDFLAGS = -lGL -lglut -lGLU -lGLEW

SOURCES = cpu.cc
SOURCES_CU = gpu.cu

OBJECTS=$(SOURCES:.cc=.o)
OBJECTSCU=$(SOURCES_CU:.cu=.o)

OUTDIR = build/
OUTFILE = nbody
MKDIR_P = mkdir -p

all: build

directory: $(OUTDIR)

build: directory $(SOURCES) $(OUTDIR)$(OUTFILE)

benchmark: nvcc -std=c++11 -I /usr/local/cuda-8.0/include/ -I ./common/ benchmark.cu -o build/nbody_b

$(OUTDIR):
	$(MKDIR_P) $(OUTDIR)

$(OUTDIR)$(OUTFILE) : $(OUTDIR) $(OBJECTS) $(OBJECTSCU)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJECTS) $(OBJECTSCU) $(LDFLAGS)

.cc.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -c $@

.cu.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -c $@

clean:
	@[ -f $(OUTDIR)$(OUTFILE) ] && rm $(OUTDIR)$(OUTFILE) || true
	rm *.o

rebuild: clean build