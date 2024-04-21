#ifndef RENDERING_HPP
#define RENDERING_HPP
#include <cmath>

#include <Ippl.h>
#include <stb_image_write.hpp>
#include "Utility/Colormaps.hpp"
#define RM_INLINE KOKKOS_INLINE_FUNCTION
#include <rastmath.hpp>

#include <Field/Field.h>
template<typename T, unsigned N>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, N> rm_to_ippl(const rm::Vector<T, N>& rmvec){
    ippl::Vector<T, N> ret;
    for(unsigned i = 0;i < N;i++){
        ret[i] = rmvec[i];
    }
    return ret;
}
template<typename View, typename T>
KOKKOS_INLINE_FUNCTION Kokkos::pair<bool, typename View::value_type> field_at(const View& v, const ippl::Vector<T, 3>& pos, const ippl::Vector<T, 3>& origin, const ippl::Vector<T, 3>& hr, const ippl::NDIndex<3>& lDom){
    using vector_type = ippl::Vector<T, 3>;

    vector_type l;
    for(unsigned k = 0;k < 3;k++){
        l[k] = (pos[k] - origin[k]) / hr[k] + 1.0; //gather is implemented wrong
    }                     

    ippl::Vector<int, 3> index{int(l[0]), int(l[1]), int(l[2])};
    ippl::Vector<T, 3> whi = l - index;
    ippl::Vector<T, 3> wlo(1.0);
    wlo -= whi;
    //TODO: nghost
    ippl::Vector<size_t, 3> args = index - lDom.first() + 1;
    for(unsigned k = 0;k < 3;k++){
        if(args[k] >= v.extent(k) || args[k] == 0){
            return {false, typename View::value_type(0)};
        }
    }
    //std::cout << args << "\n";
    return {true, ippl::detail::gatherFromField(std::make_index_sequence<8>{}, v, wlo, whi, args)};

}
KOKKOS_INLINE_FUNCTION float clamp(float x, float lower = 0.0f, float upper = 1.0f){
    return x < lower ? lower : (x > upper ? upper : x);
}
namespace ippl{
    template<typename scalar, unsigned Dim>
    struct aabb{
        rm::Vector<scalar, Dim> start;
        rm::Vector<scalar, Dim> end;
        KOKKOS_INLINE_FUNCTION aabb(const rm::Vector<scalar, Dim>& s, const rm::Vector<scalar, Dim>& e) : start(s), end(e){}
        KOKKOS_INLINE_FUNCTION aabb(const ippl::Vector<scalar, Dim>& s, const ippl::Vector<scalar, Dim>& e){
            for(unsigned k = 0;k < Dim;k++){
                start[k] = s[k];
                end  [k] = e[k];
            }
        }
    };
    template<typename scalar, unsigned Dim>
    struct ray{
        rm::Vector<scalar, Dim> o, d, id;
        constexpr KOKKOS_INLINE_FUNCTION ray() = default;
        constexpr KOKKOS_INLINE_FUNCTION ray(const rm::Vector<scalar, Dim>& _o, const rm::Vector<scalar, Dim>& _d): o(_o), d(_d.normalized()){
            for(int i = 0;i < 3;i++){
                id[i] = scalar(1) / d[i];
            }
        }
    };
    template<typename scalar, unsigned Dim>
    KOKKOS_INLINE_FUNCTION Kokkos::pair<scalar, scalar> intersection_ts(const ray<scalar, Dim>& r, const aabb<scalar, Dim>& b) {
        scalar tmin = -INFINITY, tmax = INFINITY;
        
        for (unsigned i = 0; i < Dim; ++i) {
            scalar t1 = (b.start[i] - r.o[i]) * r.id[i];
            scalar t2 = (b.end  [i]   - r.o[i]) * r.id[i];
            tmin = Kokkos::max(tmin, Kokkos::min(t1, t2));
            tmax = Kokkos::min(tmax, Kokkos::max(t1, t2));
        }
        if(tmax < tmin){
            return {NAN, NAN};
        }
        return {tmin, tmax};
    }
    struct Image{
        using color_type = Vector<uint8_t, 4>;

        uint32_t width, height;
        using view_type = Kokkos::View<color_type*>;
        Kokkos::View<color_type*> color_buffer;

        Image() : width(0), height(0){}
        Image(uint32_t w, uint32_t h, color_type fill = Vector<uint8_t, 4>{0,120, 190, 255}) : width(w), height(h){
            initialize(fill);
        }
        void initialize(color_type fill){
            color_buffer = Kokkos::View<color_type*>("Image::color_buffer", width * height);
            auto cbuf = color_buffer;
            Kokkos::parallel_for(width * height, KOKKOS_LAMBDA(size_t i){
                cbuf(i) = fill;
            });
        }
        void save_to(const char* out_file_name){
            save_to(std::string(out_file_name));
        }
        void save_to(const std::string& out_file_name){
            Kokkos::View<color_type*>::host_mirror_type hmirror = Kokkos::create_mirror_view(color_buffer);
            Kokkos::deep_copy(hmirror, color_buffer);
            if(out_file_name.ends_with("png")){
                stbi_write_png(out_file_name.c_str(), width, height, 4, hmirror.data(), width * 4);
            }
            else if(out_file_name.ends_with("bmp")){
                stbi_write_bmp(out_file_name.c_str(), width, height, 4, hmirror.data());
            }
        }
        // Setter getter approach due to lambda shenanigans
        KOKKOS_INLINE_FUNCTION color_type get(uint32_t i, uint32_t j)const{
            return color_buffer(i * width + j);
        }

        KOKKOS_INLINE_FUNCTION void set(uint32_t i, uint32_t j, color_type c)const{
            color_buffer(i * width + j) = c;
        }
    };
    template<typename T>
    Image draw_scalar_field(const Field<T, 3, UniformCartesian<T, 3>, typename UniformCartesian<T, 3>::DefaultCentering>& f, const uint32_t width, const uint32_t height, rm::camera cam){
        Image ret(width, height, Vector<uint8_t, 4>{0,0,0,255});
        using exec_space       = typename Image::view_type::execution_space;
        using policy_type      = typename RangePolicy<2, exec_space>::policy_type;
        Kokkos::Array<uint32_t, 2> begin, end;
        begin[0] = 0;
        begin[1] = 0;
        end[0] = height;
        end[1] = width;
        auto fview = f.getView();
        ippl::Vector<T, 3> O = f.get_mesh().getOrigin();
        ippl::Vector<T, 3> hr = f.get_mesh().getMeshSpacing();
        NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
        ippl::Vector<T, 3> domain_begin = O;
        ippl::Vector<T, 3> domain_end = O;
        domain_begin += hr * lDom.first();
        domain_end   += hr * lDom.last();
        aabb<float, 3> domain_box(domain_begin, domain_end);
        //NDRegion<T, 1> lKlonk = f.getLayout().getDomain();
        //std::cout << lKlonk[0] << "\n";
        const float fovy = 1.0f;
        const float tanHalfFovy = std::tan(fovy / 2.0f);
        const float aspect = float(width) / float(height);
        const float xmul = tanHalfFovy * aspect;
        const float ymul = tanHalfFovy;
        rm::Vector<T, 3> pos = cam.pos.template cast<T>();
        auto look = cam.look_dir();
        //std::cout << "Luegin in direction" << look << "\n";
        auto left = cam.left();
        //std::cout << "Left" << left << "\n";
        rm::Vector<float, 3> up{0,1,0};
        //auto left = cam.;
        Kokkos::parallel_for(policy_type(begin, end), KOKKOS_LAMBDA(uint32_t y, uint32_t x){
            #ifndef __CUDA_ARCH__
            using std::isnan;
            using std::isinf;
            #endif //__CUDA_ARCH__
            int xi = int(x) - int(width / 2);
            int yi = int(y) - int(height / 2);
            rm::Vector<float, 3> tp = (look + (left * 2.0f * float(xi) * (1.0f / float(width)) * xmul) + up * (2.0f * ymul * float(yi) * (1.0f / float(height)))).template cast<T>().normalized();
            ray<float, 3> lray(pos, tp);
            auto [first_t, second_t] = intersection_ts(lray, domain_box);
            if(isnan(first_t)){
                ret.set(y, x, ippl::Vector<uint8_t, 4>{0,0,0,0});
                return;
            }
            constexpr float transparency = 0.3f;
            constexpr float luminance = 0.2f;
            constexpr float normalization_factor = 0.1f;
            int stepc = 300;
            float stepmul = 0.01f;
            ippl::Vector<float, 4> float_color{0,0,0,0};
            {
                for(int step = stepc - 1;step >= 0;step--){
                    T t = step * stepmul;
                    auto [inside, value] = field_at(fview, rm_to_ippl(pos + tp * t), O, hr, lDom);
                    if(!inside)continue;
                    float intensity = Kokkos::abs(value) * normalization_factor;
                    int intensity_index = clamp(intensity) * 254;
                    float red =   inferno_cm[intensity_index][0] * clamp(0.5f * (intensity - 1.0), 1.0f, 1000.0f);
                    float green = inferno_cm[intensity_index][1] * clamp(0.5f * (intensity - 1.0), 1.0f, 1000.0f);
                    float blue =  inferno_cm[intensity_index][2] * clamp(0.5f * (intensity - 1.0), 1.0f, 1000.0f);
                    //float opacity = intensity * 0.02f;
                    float_color[0] = float_color[0] * Kokkos::pow(transparency, stepmul) + red    * (1.0f - Kokkos::pow(luminance, stepmul));
                    float_color[1] = float_color[1] * Kokkos::pow(transparency, stepmul) + green  * (1.0f - Kokkos::pow(luminance, stepmul));
                    float_color[2] = float_color[2] * Kokkos::pow(transparency, stepmul) + blue   * (1.0f - Kokkos::pow(luminance, stepmul));
                    float_color[3] = 1.0f - (1.0f - float_color[3]) * Kokkos::pow(1.0f - 0.7f, stepmul);
                }
            }
            ret.set(y, x, ippl::Vector<uint8_t, 4>{
                uint8_t(int(clamp(float_color[0]) * 254) % 255), 
                uint8_t(int(clamp(float_color[1]) * 254) % 255), 
                uint8_t(int(clamp(float_color[2]) * 254) % 255), 
                uint8_t(int(clamp(float_color[3]) * 254) % 255)});
        });
        Kokkos::fence();
        return ret;
    }
}
#endif