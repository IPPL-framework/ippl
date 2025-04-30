/* Definitions of real and integer variables: 
       if the code should be in double precision, here's 
       the only place where change should be made. 
   Also, some F90 intrinsics are defined here.
 
                      Zarija Lukic, February 2009
                           zarija@lanl.gov
*/

#ifndef TypesDefs_Header_Included
#define TypesDefs_Header_Included

#ifdef USENAMESPACE
namespace initializer {
#endif

#ifdef DOUBLE_REAL
typedef double real;
#define MY_MPI_REAL MPI_DOUBLE
#else
typedef float real;
#define MY_MPI_REAL MPI_FLOAT
#endif

#ifdef LONG_INTEGER
typedef long integer;
#define MY_MPI_INTEGER MPI_DOUBLE
#else
typedef int integer;
#define MY_MPI_INTEGER MPI_INT
#endif

typedef struct{
	double re;
	double im;
} my_fftw_complex;

inline int MOD(int x, int y) { return (x - y*(integer)(x/y));}

#ifdef USENAMESPACE
}
#endif

#endif
