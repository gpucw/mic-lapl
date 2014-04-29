CC=icc

CFLAGS = -O3 -mmic -DMIC -std=gnu99 -openmp -opt-prefetch-distance=64,8 -opt-streaming-cache-evict=0 -opt-streaming-stores always

all: lapl_optim

lapl_optim: lapl_optim.o
	$(CC) $(CFLAGS)  $< -o lapl_optim

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o out* lapl_optim
