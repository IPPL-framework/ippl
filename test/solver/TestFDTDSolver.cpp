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
        const unsigned dim = 3;
        using vector_type = ippl::Vector<scalar, 3>;
        using vector4_type = ippl::Vector<scalar, 4>;
        using FourField = ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using ThreeField = ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
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
        FourField source(mesh, layout);
        ThreeField E(mesh, layout);
        ThreeField B(mesh, layout);
        source = vector4_type(0);
        ippl::StandardFDTDSolver<ThreeField, FourField, ippl::absorbing> sfdsolver(source, E, B);
        sfdsolver.setPeriodicBoundaryConditions();
        auto aview = sfdsolver.A_n.getView();
        auto am1view = sfdsolver.A_nm1.getView();
        auto lDom = layout.getLocalNDIndex();
        size_t nghost = sfdsolver.A_n.getNghost();
        Kokkos::parallel_for(ippl::getRangePolicy(aview, 1), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            const size_t ig = i + lDom.first()[0] - nghost;
            const size_t jg = j + lDom.first()[1] - nghost;
            const size_t kg = k + lDom.first()[2] - nghost;
            scalar x = scalar(ig) / nr[0];
            scalar y = scalar(jg) / nr[1];
            scalar z = scalar(kg) / nr[2];
            //std::cout << x << ", " << y << ", " << z << "\n";
            (void)x;
            (void)y;
            (void)z;
            const scalar magnitude = gauss(scalar(0.5), scalar(0.05), z);
            aview  (i,j,k) = vector4_type{scalar(0), scalar(0), magnitude, scalar(0)};
            am1view(i,j,k) = vector4_type{scalar(0), scalar(0), magnitude, scalar(0)};
        });
        Kokkos::fence();
        sfdsolver.A_n.getFieldBC().apply(sfdsolver.A_n);
        //Kokkos::fence();
        //std::cout << sfdsolver.A_n.getView()(0,0,0) << "\n";
        //std::cout << sfdsolver.A_n.getView()(0,5,5) << "\n";
        //goto exit;
        sfdsolver.A_np1.getFieldBC().apply(sfdsolver.A_np1);
        sfdsolver.A_nm1.getFieldBC().apply(sfdsolver.A_nm1);
        sfdsolver.A_n.fillHalo();
        sfdsolver.A_nm1.fillHalo();
        int img_width  = 500;
        int img_height = 500;
        float* imagedata = new float[img_width * img_height * 3];
        std::fill(imagedata, imagedata + img_width * img_height * 3, 0.0f);
        uint8_t* imagedata_final = new uint8_t[img_width * img_height * 3];
        for(size_t s = 0;s < 4 * n;s++){
            auto ebh = sfdsolver.A_n.getHostMirror();
            Kokkos::deep_copy(ebh, sfdsolver.A_n.getView());
            for(int i = 1;i < img_width;i++){
                for(int j = 1;j < img_height;j++){
                    int i_remap = (double(i) / (img_width  - 1)) * (nr[2] - 4) + 2;
                    int j_remap = (double(j) / (img_height - 1)) * (nr[0] - 4) + 2;
                    if(i_remap >= lDom.first()[2] && i_remap <= lDom.last()[2]){
                        if(j_remap >= lDom.first()[0] && j_remap <= lDom.last()[0]){
                            ippl::Vector<scalar, 4> acc = ebh(j_remap + 1 - lDom.first()[0], nr[1] / 2, i_remap + 1 - lDom.first()[2]);
                            
                            
                            float normalized_colorscale_value = acc.norm();
                            //int index = (int)std::max(0.0f, std::min(normalized_colorscale_value * 255.0f, 255.0f));
                            imagedata[(j * img_width + i) * 3 + 0] = normalized_colorscale_value * 255.0f;
                            imagedata[(j * img_width + i) * 3 + 1] = normalized_colorscale_value * 255.0f;
                            imagedata[(j * img_width + i) * 3 + 2] = normalized_colorscale_value * 255.0f;
                        }
                    }
                }
            }
            std::transform(imagedata, imagedata + img_height * img_width * 3, imagedata_final, [](float x){return (unsigned char)std::min(255.0f, std::max(0.0f,x));});
            char output[1024] = {0};        
            snprintf(output, 1023, "%soutimage%.05lu.bmp", "renderdata/", s);
            if(s % 4 == 0)
                stbi_write_bmp(output, img_width, img_height, 3, imagedata_final);
            sfdsolver.solve();
        }
        double sum_error = 0;
        Kokkos::parallel_reduce(ippl::getRangePolicy(aview, 1), KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref){
            const size_t ig = i + lDom.first()[0] - nghost;
            const size_t jg = j + lDom.first()[1] - nghost;
            const size_t kg = k + lDom.first()[2] - nghost;
            scalar x = scalar(ig) / nr[0];
            scalar y = scalar(jg) / nr[1];
            scalar z = scalar(kg) / nr[2];
            (void)x;
            (void)y;
            (void)z;
            ref += Kokkos::abs(gauss(scalar(0.5), scalar(0.05), z) - aview(i,j,k)[2]);
        }, sum_error);
        std::cout << "Sum: " << sum_error / double(n * n * n) << "\n";

        
    }
    //exit:
    ippl::finalize();
}