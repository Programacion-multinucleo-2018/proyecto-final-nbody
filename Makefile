.SUFFIXES: .cc .cu

CC = nvcc
CFLAGS = -std=c++11 -O3 -Xcompiler -ansi -Xcompiler -Ofast -Wno-deprecated-gpu-targets
INCLUDES = -I $(CUDA_HOME)/include/ -I ./common/
LDFLAGS = -lGL -lglut -lGLU -lGLEW
DEBUGF = $(CFLAGS) -ggdb

SOURCES = src/cpu/cpu.cc
SOURCES_CU = src/gpu/gpu.cu

OBJECTS=$(SOURCES:.cc=.o)
OBJECTSCU=$(SOURCES_CU:.cu=.o)

OUTDIR = build/
OUTFILE = nbody
OUTDEBUG = nbody_d
MKDIR_P = mkdir -p

all: build

directory: $(OUTDIR)

build: directory $(SOURCES) $(OUTDIR)$(OUTFILE)

$(OUTDIR):
	$(MKDIR_P) $(OUTDIR)

$(OUTDIR)$(OUTFILE) : $(OUTDIR) $(OBJECTS) $(OBJECTSCU)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJECTS) $(OBJECTSCU) $(LDFLAGS)

debug: directory $(SOURCES) $(OUTDIR)$(OUTDEBUG)

$(OUTDIR)$(OUTDEBUG) : $(OUTDIR) $(OBJECTS) $(OBJECTSCU)
	$(CC) $(DEBUGF) $(INCLUDES) -o $@ $(OBJECTS) $(OBJECTSCU) $(LDFLAGS) 

.cc.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -c $@

.cu.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -c $@

clean:
	@[ -f $(OUTDIR)$(OUTFILE) ] && rm $(OUTDIR)$(OUTFILE) || true
	@[ -f $(OUTDIR)$(OUTDEBUG)  ] && rm $(OUTDIR)$(OUTDEBUG)  || true
	rm *.o

rebuild: clean build

redebug: clean debug