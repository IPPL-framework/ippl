#include "Utility/Rendering.hpp"
#include "Utility/Colormaps.hpp"
using scalar = float;
KOKKOS_INLINE_FUNCTION scalar gaussian(scalar x, scalar y, scalar z, scalar sigma = 1.0,
                                       scalar mu = 0.5) {
    scalar pi        = std::acos(-1.0);
    scalar prefactor = (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    scalar r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
}
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {

        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<scalar, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt         = std::atoi(argv[1]);
        bool gauss_fct = std::atoi(argv[2]);
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);
        
        // Specifies SERIAL, PARALLEL dims
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // type definitions
        typedef ippl::Vector<scalar, dim> Vector_t;
        typedef ippl::Field<scalar, dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Vector<Vector_t, dim> Matrix_t;
        typedef ippl::Field<Matrix_t, dim, Mesh_t, Centering_t> MField_t;

        // domain [0,1]^3
        scalar dx       = 1.0 / scalar(pt);
        Vector_t hx     = {dx, dx, dx};
        Vector_t origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        Field_t field(mesh, layout, 1);
        MField_t result(mesh, layout, 1);
        MField_t exact(mesh, layout, 1);
        Kokkos::View<ippl::Vector<float, 3>*> position("ppositions", 8);

        Kokkos::parallel_for(position.extent(0), KOKKOS_LAMBDA(size_t i){
            position(i) = ippl::Vector<float, 3>{float(int(i / 4)), float((i / 2) % 2), float(i % 2) * 1.0f};
            position(i) *= 0.6f;
            position(i) += 0.2f;
        });
        using vec3 = rm::Vector<float, 3>;
        vec3 pos{-1.5,-1.0,-1.5};
        vec3 target{0.5,0.3,0.5};
        rm::camera cam(pos, target - pos);
        Font f(100);
        ippl::Image pimg = ippl::drawParticles(position, position.extent(0), 1000, 500, cam, 0.03f, KOKKOS_LAMBDA(ippl::Vector<float, 3> p){
            return p;
        }, position);
        ippl::Image primg = ippl::drawParticlesProjection(position, position.extent(0), 1000, 500, ippl::axis::x, ippl::getGlobalDomainBox(field), 5.0f, KOKKOS_LAMBDA(){
            return ippl::Vector<float, 3>{0,1,0};
        });
        pimg.transpose();
        ippl::drawTextOnto(pimg, "Z: 0.9 AV", 10, 10, f, ippl::Vector<float, 4>{1,1,1,1});
        primg.save_to("rojection.png");
        typename Field_t::view_type& view = field.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
        const int nghost               = field.getNghost();
        using mdrange_type             = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        for(int im = 0;im < 1;im++){
            Kokkos::parallel_for(
                "Assign field",
                mdrange_type({0, 0, 0}, {view.extent(0), view.extent(1), view.extent(2)}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    // local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;

                    scalar x = (ig + 0.5) * hx[0] + origin[0];
                    scalar y = (jg + 0.5) * hx[1] + origin[1];
                    scalar z = (kg + 0.5) * hx[2] + origin[2];

                    if (gauss_fct) {
                        view(i, j, k) = 0.1f * gaussian((x - 0.5 + im / 40.0), 0.5, 0.5, 0.01);
                    } else {
                        view(i, j, k) = x * y * z;
                    }
                }
            );
            
            (void)pos;
            (void)target;
            //ippl::Image img = ippl::drawFieldFog(field, 1000, 500, rm::camera(pos, target - pos), [](float x){
            //    return ippl::normalized_colormap(turbo_cm, Kokkos::sqrt(Kokkos::abs(x)) / 50.0f);
            //    //return ippl::alpha_extend(ippl::normalized_colormap(turbo_cm, Kokkos::abs(x) / 50.0f), clamp(Kokkos::abs(x) / 50.0f, 0.5f, 0.99f));
            //}, pimg);
            pimg.removeAlpha(ippl::Vector<float, 3>{0,0,0});
            pimg.save_to("particle.png");

            //ippl::Image img = ippl::drawFieldCrossSection(field, 600, 600, ippl::axis::y, 0.3f, [](float x){
            //    return ippl::normalized_colormap(turbo_cm, Kokkos::sqrt(Kokkos::abs(x)) / 50.0f);
            //});
            //img.removeAlpha(ippl::Vector<float, 3>{0,0,0});
            //img.save_to("field.png");
            //img.collectOnRank0();
            //if(ippl::Comm->rank() == 0){
            //    char buf[1024] = {0};
            //    snprintf(buf, 1024, "renderdataout%05d.bmp", im);
            //    img.save_to(buf);
            //}
        }
    }
    ippl::finalize();
}