#include <unistd.h>
#include <stddef.h>
static unsigned long count=0;static int armed=0;static void *host_handler=(void*)0x1234;
void g2ax_arm(int x){armed=x;} unsigned long g2ax_count(void){return count;}
void *g2ax_handler(void){return host_handler;}
static void hit(void){++count;if(!armed){write(2,"G2AX unexpected host numerical binding\n",37);_exit(97);}}
#define TRAP(n) void n(void){hit();}
TRAP(dgemv_) TRAP(dgesdd_) TRAP(xerbla_) TRAP(__xerbla)
TRAP(cblas_dgemv) TRAP(cblas_ddot) TRAP(cblas_dnrm2) TRAP(cblas_dgemm)
TRAP(gsl_linalg_SV_decomp) TRAP(gsl_rng_uniform) TRAP(gsl_rng_set)
void *gsl_set_error_handler(void *p){void *old=host_handler;hit();host_handler=p;return old;}
unsigned long gsl_rng_default_seed=0x13579;
