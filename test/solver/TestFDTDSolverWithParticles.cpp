#include <cstddef>
using std::size_t;
#include <Kokkos_Core.hpp>
#include "Ippl.h"
#include "Types/Vector.h"
#include "Field/Field.h"
#include "MaxwellSolvers/FDTD.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
template<typename scalar1, typename... scalar>
    requires((std::is_floating_point_v<scalar1>))
KOKKOS_INLINE_FUNCTION float gauss(scalar1 mean, scalar1 stddev, scalar... x){
    uint32_t dim = sizeof...(scalar);
    ippl::Vector<scalar1, sizeof...(scalar)> vec{scalar1(x - mean)...};
    scalar1 vecsum(0);
    for(unsigned d = 0;d < dim;d++){
        vecsum += vec[d] * vec[d];
        
    }
    #ifndef __CUDA_ARCH__
    using std::exp;
    #endif
    return exp(-(vecsum) / (stddev * stddev)); 
}

int main(int argc, char* argv[]){
    ippl::initialize(argc, argv);
    {
        using scalar = float;
        //const unsigned dim = 3;
        //using vector_type = ippl::Vector<scalar, 3>;
        //using vector4_type = ippl::Vector<scalar, 4>;
        //using FourField = ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        //using ThreeField = ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        constexpr size_t n = 100;
        ippl::Vector<uint32_t, 3> nr{n / 2, n / 2, 2 * n};
        ippl::NDIndex<3> owned(nr[0], nr[1], nr[2]);
        ippl::Vector<scalar, 3> extents{scalar(1), scalar(1), scalar(1)};
        std::array<bool, 3> isParallel;
        isParallel.fill(false);
        isParallel[2] = true;

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        ippl::Vector<scalar, 3> hx;
        for(unsigned d = 0;d < 3;d++){
            hx[d] = extents[d] / (scalar)nr[d];
        }
        ippl::Vector<scalar, 3> origin = {0,0,0};
        ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);
        
        ippl::NSFDSolverWithParticles<scalar, ippl::absorbing> solver(layout, mesh, 1 << 17);
        for(int i = 0;i < 10;i++)
            solver.solve();
    }
    //exit:
    ippl::finalize();
}