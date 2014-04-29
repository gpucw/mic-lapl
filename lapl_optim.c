#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>

#define WRITE_FREQ 1000000000000 
static int Lx,Ly;

typedef struct {
  float site4[4];
} supersite;

typedef struct {
	float site16[16];
} MICsupersite;

double stop_watch(double t0) {
  double time;
  struct timeval t;
  gettimeofday(&t, NULL);
  time = (double) t.tv_sec + (double) t.tv_usec * 1e-6;  
  return time-t0;
}

#ifndef MIC
void to_supersite(supersite *ssarr, float *arr) {
  for(int y=0; y<Ly; y++)
    for(int x=0; x<Lx/4; x++) {
      int vv = x + (Lx/4)*y;
      int v = x + (Lx)*y;
      ssarr[vv].site4[0] = arr[v+0*Lx/4];
      ssarr[vv].site4[1] = arr[v+1*Lx/4];
      ssarr[vv].site4[2] = arr[v+2*Lx/4];
      ssarr[vv].site4[3] = arr[v+3*Lx/4];
    }
  return;
}
#endif

#ifdef MIC
void to_supersite(MICsupersite *ssarr, float *arr) {
  for(int y=0; y<Ly; y++)
    for(int x=0; x<Lx/16; x++) {
      int vv = x + (Lx/16)*y;
      int v = x + (Lx)*y;
      ssarr[vv].site16[0] = arr[v+0*Lx/16];
      ssarr[vv].site16[1] = arr[v+1*Lx/16];
      ssarr[vv].site16[2] = arr[v+2*Lx/16];
      ssarr[vv].site16[3] = arr[v+3*Lx/16];
      ssarr[vv].site16[4] = arr[v+4*Lx/16];
      ssarr[vv].site16[5] = arr[v+5*Lx/16];
      ssarr[vv].site16[6] = arr[v+6*Lx/16];
      ssarr[vv].site16[7] = arr[v+7*Lx/16];
      ssarr[vv].site16[8] = arr[v+8*Lx/16];
      ssarr[vv].site16[9] = arr[v+9*Lx/16];
      ssarr[vv].site16[10] = arr[v+10*Lx/16];
      ssarr[vv].site16[11] = arr[v+11*Lx/16];
      ssarr[vv].site16[12] = arr[v+12*Lx/16];
      ssarr[vv].site16[13] = arr[v+13*Lx/16];
      ssarr[vv].site16[14] = arr[v+14*Lx/16];
      ssarr[vv].site16[15] = arr[v+15*Lx/16];
    }
  return;
}
#endif

#ifndef MIC
void from_supersite(float *arr, supersite *ssarr) {
  for(int y=0; y<Ly; y++)
    for(int x=0; x<Lx/4; x++) {
      int vv = x + (Lx/4)*y;
      int v = x + (Lx)*y;
      arr[v+0*Lx/4] = ssarr[vv].site4[0];
      arr[v+1*Lx/4] = ssarr[vv].site4[1];
      arr[v+2*Lx/4] = ssarr[vv].site4[2];
      arr[v+3*Lx/4] = ssarr[vv].site4[3];
    }
  return;
}
#endif

#ifdef MIC
void from_supersite(float *arr, MICsupersite *ssarr) {
  for(int y=0; y<Ly; y++)
    for(int x=0; x<Lx/16; x++) {
      int vv = x + (Lx/16)*y;
      int v = x + (Lx)*y;
      arr[v+0*Lx/16] = ssarr[vv].site16[0];
      arr[v+1*Lx/16] = ssarr[vv].site16[1];
      arr[v+2*Lx/16] = ssarr[vv].site16[2];
      arr[v+3*Lx/16] = ssarr[vv].site16[3];
      arr[v+4*Lx/16] = ssarr[vv].site16[4];
      arr[v+5*Lx/16] = ssarr[vv].site16[5];
      arr[v+6*Lx/16] = ssarr[vv].site16[6];
      arr[v+7*Lx/16] = ssarr[vv].site16[7];
      arr[v+8*Lx/16] = ssarr[vv].site16[8];
      arr[v+9*Lx/16] = ssarr[vv].site16[9];
      arr[v+10*Lx/16] = ssarr[vv].site16[10];
      arr[v+11*Lx/16] = ssarr[vv].site16[11];
      arr[v+12*Lx/16] = ssarr[vv].site16[12];
      arr[v+13*Lx/16] = ssarr[vv].site16[13];
      arr[v+14*Lx/16] = ssarr[vv].site16[14];
      arr[v+15*Lx/16] = ssarr[vv].site16[15];
    }
  return;
}
#endif


void read_from_file(float *arr, char fname[]) {
  FILE *fp = fopen(fname, "r");
  fread(arr, 4, Lx*Ly, fp);
  fclose(fp);
  return;
}

void write_to_file(char fname[], float *arr) {
  FILE *fp = fopen(fname, "w");
  fwrite(arr, 4, Lx*Ly, fp);
  fclose(fp);
  return;
}

void lapl_iter(float *out, float sigma, float *in) {
  float delta = sigma / (1+4*sigma);
  float norm = 1./(1+4*sigma);
  
  #pragma omp parallel
  {
		#pragma omp for nowait
		/* Do lapl iteration on volume */
		for(int y=1; y<Ly-1; y++)
			for(int x=1; x<Lx-1; x++) {
				int v00 = x+y*Lx;
				int v0p = v00 + 1;
				int v0m = v00 - 1;
				int vp0 = v00 + Lx;
				int vm0 = v00 - Lx;
				out[v00] = norm*in[v00] +	delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
			}
		
		#pragma omp for nowait
		for(int x=1; x<Lx-1; x++) {
			int y = 0;
			int v00 = x+y*Lx;
			int v0p = v00 + 1;
			int v0m = v00 - 1;
			int vp0 = v00 + Lx;
			int vm0 = x+(Ly-1)*Lx;
			out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}

		#pragma omp for nowait
		for(int x=1; x<Lx-1; x++) {
			int y = Ly-1;
			int v00 = x+y*Lx;
			int v0p = v00 + 1;
			int v0m = v00 - 1;
			int vp0 = x + 0*Lx;
			int vm0 = v00 - Lx;
			out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}
	
		#pragma omp for nowait
		for(int y=1; y<Ly-1; y++) {
			int x = 0;
			int v00 = x+y*Lx;
			int v0p = v00 + 1;
			int v0m = (Lx-1) + y*Lx;
			int vp0 = v00 + Lx;
			int vm0 = v00 - Lx;
			out[v00] = norm*in[v00] +
			delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}

		#pragma omp for nowait
		for(int y=1; y<Ly-1; y++) {
			int x = Lx-1;
			int v00 = x+y*Lx;
			int v0p = 0 + y*Lx;
			int v0m = v00 - 1;
			int vp0 = v00 + Lx;
			int vm0 = v00 - Lx;
			out[v00] = norm*in[v00] +  delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]); 
		}
	
		{
			int x = 0;
			int y = 0;
			int v00 = x+y*Lx;
			int v0p = v00 + 1;
			int v0m = Lx - 1;
			int vp0 = v00 + Lx;
			int vm0 = (Ly-1)*Lx;
			out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}
	
		{
			int x = 0;
			int y = Ly-1;
			int v00 = x+y*Lx;
			int v0p = v00 + 1;
			//int v0m = Lx - 1;
			int v0m = Lx*Ly - 1;
			int vp0 = x + 0*Lx;
			int vm0 = v00 - Lx;
			out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}
	
		{
			int x = Lx-1;
			int y = 0;
			int v00 = x+y*Lx;
			int v0p = 0 + y*Lx;
			int v0m = v00 - 1;
			int vp0 = v00 + Lx;
			//int vm0 = v00 - Lx;
			int vm0 = Lx*Ly - 1;    
			out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}
	
		{
			int x = Lx-1;
			int y = Ly-1;
			int v00 = x+y*Lx;
			int v0p = 0 + y*Lx;
			int v0m = v00 - 1;
			int vp0 = x + 0*Lx;
			int vm0 = v00 - Lx;
			out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
		}
	}
	return;
}

#ifndef MIC
void lapl_iter_supersite(supersite *out, float sigma, supersite *in){
  float delta = sigma / (1+4*sigma);
  float norm = 1./(1+4*sigma);
  
  __m128 register vnorm = _mm_load1_ps(&norm);
  __m128 register vdelta = _mm_load1_ps(&delta);
  
	#pragma omp parallel 
	{
		#pragma omp for nowait
		/* Do lapl iteration on volume */
		for(int y=0; y<Ly; y++)
			for(int x=1; x<Lx/4-1; x++) {
				int lx = Lx/4;
				int v00 = x+y*lx;
				int v0p = v00+1;
				int v0m = v00-1;
				int vp0 = x + ((y+1)%Ly)*lx;
				int vm0 = x + ((Ly+(y-1))%Ly)*lx;
      
				__m128 register in00 = _mm_load_ps(&in[v00].site4[0]);
				__m128 register in0p = _mm_load_ps(&in[v0p].site4[0]);
				__m128 register in0m = _mm_load_ps(&in[v0m].site4[0]);
				__m128 register inp0 = _mm_load_ps(&in[vp0].site4[0]);
				__m128 register inm0 = _mm_load_ps(&in[vm0].site4[0]);

				__m128 register hop = _mm_add_ps(inm0, inp0);
				hop = _mm_add_ps(hop, in0p);
				hop = _mm_add_ps(hop, in0m);
				hop = _mm_mul_ps(hop, vdelta);
				__m128 register dia = _mm_mul_ps(vnorm, in00);
				hop = _mm_add_ps(dia, hop);
				_mm_store_ps(&out[v00].site4[0], hop);
			}

		#pragma omp for nowait
		for(int y=0; y<Ly; y++) {
			int lx = Lx/4;
			int x = 0;
			int v00 = x+y*lx;
			int v0p = v00+1;
			int v0m = lx-1+y*lx;
			int vp0 = x + ((y+1)%Ly)*lx;
			int vm0 = x + ((Ly+(y-1))%Ly)*lx;    
      
			__m128 register in00 = _mm_load_ps(&in[v00].site4[0]);
			__m128 register in0p = _mm_load_ps(&in[v0p].site4[0]);
			__m128 register in0m = _mm_load_ps(&in[v0m].site4[0]);
			in0m = _mm_shuffle_ps(in0m, in0m, _MM_SHUFFLE(2,1,0,3));
			__m128 register inp0 = _mm_load_ps(&in[vp0].site4[0]);
			__m128 register inm0 = _mm_load_ps(&in[vm0].site4[0]);

			__m128 register hop = _mm_add_ps(inm0, inp0);
			hop = _mm_add_ps(hop, in0p);
			hop = _mm_add_ps(hop, in0m);
			hop = _mm_mul_ps(hop, vdelta);
			__m128 register dia = _mm_mul_ps(vnorm, in00);
			hop = _mm_add_ps(dia, hop);
			_mm_store_ps(&out[v00].site4[0], hop);
		}

		#pragma omp for nowait
		for(int y=0; y<Ly; y++) {
			int lx = Lx/4;
			int x = lx-1;
			int v00 = x+y*lx;
			int v0p = y*lx;
			int v0m = v00-1;
			int vp0 = x + ((y+1)%Ly)*lx;
			int vm0 = x + ((Ly+(y-1))%Ly)*lx;    
      
			__m128 register in00 = _mm_load_ps(&in[v00].site4[0]);
			__m128 register in0p = _mm_load_ps(&in[v0p].site4[0]);
			in0p = _mm_shuffle_ps(in0p, in0p, _MM_SHUFFLE(0,3,2,1));
			__m128 register in0m = _mm_load_ps(&in[v0m].site4[0]);
			__m128 register inp0 = _mm_load_ps(&in[vp0].site4[0]);
			__m128 register inm0 = _mm_load_ps(&in[vm0].site4[0]);

			__m128 register hop = _mm_add_ps(inm0, inp0);
			hop = _mm_add_ps(hop, in0p);
			hop = _mm_add_ps(hop, in0m);
			hop = _mm_mul_ps(hop, vdelta);
			__m128 register dia = _mm_mul_ps(vnorm, in00);
			hop = _mm_add_ps(dia, hop);
			_mm_store_ps(&out[v00].site4[0], hop);
		}
	}
	return;
}
#endif

#ifdef MIC
void lapl_iter_supersite(MICsupersite *out, float sigma, MICsupersite *in){
  float delta = sigma / (1+4*sigma);
  float norm = 1./(1+4*sigma);
  
	#pragma omp parallel 
	{
		__m512 register vnorm = _mm512_set1_ps(norm);
		__m512 register vdelta = _mm512_set1_ps(delta);
  
		/* Do lapl iteration on volume */
		#pragma omp for nowait
		for(int y=0; y<Ly; y++)
			for(int x=1; x<Lx/16-1; x++) {
				int lx = Lx/16;
				int v00 = x+y*lx;
				int v0p = v00+1;
				int v0m = v00-1;
				int vp0 = x + ((y+1)%Ly)*lx;
				int vm0 = x + ((Ly+(y-1))%Ly)*lx;
      
				__m512 register in00 = _mm512_load_ps(&in[v00].site16[0]);
				__m512 register in0p = _mm512_load_ps(&in[v0p].site16[0]);
				__m512 register in0m = _mm512_load_ps(&in[v0m].site16[0]);
				__m512 register inp0 = _mm512_load_ps(&in[vp0].site16[0]);
				__m512 register inm0 = _mm512_load_ps(&in[vm0].site16[0]);

				__m512 register hop = _mm512_add_ps(inm0, inp0);
				hop = _mm512_add_ps(hop, in0p);
				hop = _mm512_add_ps(hop, in0m);
				hop = _mm512_mul_ps(hop, vdelta);
				__m512 register dia = _mm512_mul_ps(vnorm, in00);
				hop = _mm512_add_ps(dia, hop);
				_mm512_store_ps(&out[v00].site16[0], hop);
			}

		#pragma omp for nowait
		for(int y=0; y<Ly; y++) {
			int lx = Lx/16;
			int x = 0;
			int v00 = x+y*lx;
			int v0p = v00+1;
			int v0m = lx-1+y*lx;
			int vp0 = x + ((y+1)%Ly)*lx;
			int vm0 = x + ((Ly+(y-1))%Ly)*lx;    
      
			__m512 register in00 = _mm512_load_ps(&in[v00].site16[0]);
			__m512 register in0p = _mm512_load_ps(&in[v0p].site16[0]);
			__m512 register in0m = _mm512_load_ps(&in[v0m].site16[0]);
		
			//shifting for MIC
			__m512i register shift = _mm512_set_epi32(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,15);
			in0m = (__m512) _mm512_permutevar_epi32(shift, (__m512i)in0m);
      
			//shifting for SSE
			//in0m = _mm_shuffle_ps(in0m, in0m, _MM_SHUFFLE(2,1,0,3));
      
			__m512 register inp0 = _mm512_load_ps(&in[vp0].site16[0]);
			__m512 register inm0 = _mm512_load_ps(&in[vm0].site16[0]);
		
			__m512 register hop = _mm512_add_ps(inm0, inp0);
			hop = _mm512_add_ps(hop, in0p);
			hop = _mm512_add_ps(hop, in0m);
			hop = _mm512_mul_ps(hop, vdelta);
			__m512 register dia = _mm512_mul_ps(vnorm, in00);
			hop = _mm512_add_ps(dia, hop);
			_mm512_store_ps(&out[v00].site16[0], hop);
		}
  
		#pragma omp for nowait
		for(int y=0; y<Ly; y++) {
			int lx = Lx/16;
			int x = lx-1;
			int v00 = x+y*lx;
			int v0p = y*lx;
			int v0m = v00-1;
			int vp0 = x + ((y+1)%Ly)*lx;
			int vm0 = x + ((Ly+(y-1))%Ly)*lx;    
      
			__m512 register in00 = _mm512_load_ps(&in[v00].site16[0]);
			__m512 register in0p = _mm512_load_ps(&in[v0p].site16[0]);
      
			//shifting for MIC
			__m512i register shift = _mm512_set_epi32(0,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
			in0p = (__m512) _mm512_permutevar_epi32(shift, (__m512i)in0p);
      
			//shifting for SSE
			//in0p = _mm_shuffle_ps(in0p, in0p, _MM_SHUFFLE(0,3,2,1));
	
			__m512 register in0m = _mm512_load_ps(&in[v0m].site16[0]);
			__m512 register inp0 = _mm512_load_ps(&in[vp0].site16[0]);
			__m512 register inm0 = _mm512_load_ps(&in[vm0].site16[0]);

			__m512 register hop = _mm512_add_ps(inm0, inp0);
			hop = _mm512_add_ps(hop, in0p);
			hop = _mm512_add_ps(hop, in0m);
			hop = _mm512_mul_ps(hop, vdelta);
			__m512 register dia = _mm512_mul_ps(vnorm, in00);
			hop = _mm512_add_ps(dia, hop);
			_mm512_store_ps(&out[v00].site16[0], hop);
		}
	}
	return;
}
#endif


void usage(char *argv[]) {
  fprintf(stderr, " Usage: %s LX LY NITER IN_FILE OUT_FILE\n", argv[0]);
  return;
}

int main(int argc, char *argv[]) {
	/* Check the number of command line arguments */
	if(argc != 6) {
		usage(argv);
		exit(1);
	}
  
	/* The number of processes in x and y are read from the command line */
	Lx = atoi(argv[1]);
	Ly = atoi(argv[2]);
	int niter = atoi(argv[3]);
	float sigma = 1.0;
	printf(" Threads = %d\n", omp_get_max_threads());
	printf(" Ly,Lx = %d,%d\n", Ly, Lx);
	printf(" niter = %d\n", niter);
	printf(" input file = %s\n", argv[4]);
	printf(" output file = %s\n", argv[5]);

	// SIMPLE //

	/* Allocate the buffer for the data */
	float *arr[2] = {malloc(sizeof(float)*Lx*Ly), malloc(sizeof(float)*Lx*Ly)};
	
	/* read file */
	read_from_file(arr[0], argv[4]);
	lapl_iter(arr[1], sigma, arr[0]);
	
	double t0 = stop_watch(0);
	
	for(int i=0; i<niter; i++) {
		lapl_iter(arr[(i+1)%2], sigma, arr[i%2]);
		//if(i % WRITE_FREQ == 0) {
		//	char fname[256];
		//	sprintf(fname, "%s.%08d", argv[5], i);
		//	write_to_file(fname, arr[(i+1)%2]);
		//}
	}
	t0 = stop_watch(t0);
	printf(" 0:	iters = %d,	(Lx,Ly) = (%d,%d)	t = %8.1f usec/iter,	BW = %6.3f GB/s,	P = %6.3f Gflop/s\n", niter, Lx, Ly, t0*1e6/(double)niter, Lx*Ly*sizeof(float)*2*niter/t0/1e9, (Lx*Ly*6.0*niter)/t0/1e9);
	
	/* write file */
	char fname[256];
	sprintf(fname, "%s.%08d", argv[5], niter);
	write_to_file(fname, arr[niter%2]);


	// Intrinsics //
	
	
	/* Allocate the buffer for the data */
	#ifndef MIC
		supersite *ssarr[2];
		posix_memalign((void**) &ssarr[0], 16, sizeof(supersite)*Lx*Ly/4);
		posix_memalign((void**) &ssarr[1], 16, sizeof(supersite)*Lx*Ly/4);
	#endif
	
	#ifdef MIC
		MICsupersite *ssarr[2];
		posix_memalign((void**) &ssarr[0], 64, sizeof(MICsupersite)*Lx*Ly/16);
		posix_memalign((void**) &ssarr[1], 64, sizeof(MICsupersite)*Lx*Ly/16);
	#endif
	
	
	/* read file */
	read_from_file(arr[0], argv[4]);
	
	/* convert to supersite-packed */
	to_supersite(ssarr[0], arr[0]);
	
	lapl_iter_supersite(ssarr[1], sigma, ssarr[0]);
	
	t0 = stop_watch(0);
	for(int i=0; i<niter; i++) {
		lapl_iter_supersite(ssarr[(i+1)%2], sigma, ssarr[i%2]);
		//if(i % WRITE_FREQ == 0) {
		//	char fname[256];
		//	sprintf(fname, "%s.ss%08d", argv[5], i);
		//	from_supersite(arr[0], ssarr[(i+1)%2]);
		//	write_to_file(fname, arr[0]);
		//}
	}
	t0 = stop_watch(t0);
	printf(" 1:	iters = %d,	(Lx,Ly) = (%d,%d)	t = %8.1f usec/iter,	BW = %6.3f GB/s,	P = %6.3f Gflop/s\n", niter, Lx, Ly, t0*1e6/(double)niter, Lx*Ly*sizeof(float)*2*niter/t0/1e9, (Lx*Ly*6.0*niter)/t0/1e9);
	
	
	/* write file */
	sprintf(fname, "%s.ss%08d", argv[5], niter);
	from_supersite(arr[0], ssarr[niter%2]);
	write_to_file(fname, arr[0]);

	for(int i=0; i<2; i++) {
		free(arr[i]);
		free(ssarr[i]);
	}

	return 0;
}
