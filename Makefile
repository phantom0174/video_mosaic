CXX = g++
NVCC = nvcc
HIPCC = hipcc

HOST_OPT = -O3 -fopenmp -Xlinker -rpath=./mosaic_libs/lib

CXXFLAGS = -pthread -fopenmp -fno-math-errno -funroll-loops -ftree-vectorize -DNDEBUG \
		   $(HOST_OPT)

NVFLAGS = -std=c++20 -O3 \
          --use_fast_math \
          -Xptxas "-O3 -v" \
          -Xcompiler "$(HOST_OPT)"

HIPCCFLAGS = -std=c++17 -O3 \
             -ffast-math \
             --offload-arch=gfx908 \
             -fopenmp -lrocblas \
             $(HOST_OPT)

LDFLAGS = -lm

EXES = make-cache opt-turbo opt-cuda

all: $(EXES)

clean:
	rm -f $(EXES)

make-cache: src/make_cache.cc src/common.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

# amd
opt-turbo: src/opt_turbo.cpp src/common.h src/timer.h
	$(HIPCC) $(HIPCCFLAGS) -I./mosaic_libs/include $(LDFLAGS) -L./mosaic_libs/lib -lturbojpeg -lrocblas -o $@ $<

# cuda
opt-cuda: src/opt_turbo.cu src/common.h src/timer.h
	$(NVCC) $(NVFLAGS) -I./mosaic_libs/include $(LDFLAGS) -L./mosaic_libs/lib -lturbojpeg -lcublas -o $@ $<