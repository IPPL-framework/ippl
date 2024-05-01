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
/**
 * @brief Helper function to interpolate a field to a point
 * 
 * @tparam View view_type of the field
 * @tparam T Positional scalar type 
 * @param v View of the field
 * @param pos Sampling position
 * @param origin Field origin
 * @param hr Field mesh spacing
 * @param lDom Local domain
 * @return Value interpolated to point @pos
 */
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
/**
 * @brief Device overload of std::clamp
 * 
 * @param x Value
 * @param lower lower limit
 * @param upper upper limit
 * @return Clamped value 
 */
KOKKOS_INLINE_FUNCTION float clamp(float x, float lower = 0.0f, float upper = 1.0f){
    return x < lower ? lower : (x > upper ? upper : x);
}
namespace ippl{
    /**
     * @brief Struct describing an axis-aligned hypercube
     * 
     * @details Holds two N-Dimensional Vectors start and end to denote the intervals 
     * 
     *
     * @tparam scalar Scalar type
     * @tparam Dim Dimension
     */
    template<typename scalar, unsigned Dim>
    struct aabb{
        rm::Vector<scalar, Dim> start;
        rm::Vector<scalar, Dim> end;
        KOKKOS_INLINE_FUNCTION constexpr aabb(const rm::Vector<scalar, Dim>& s, const rm::Vector<scalar, Dim>& e) : start(s), end(e){}
        KOKKOS_INLINE_FUNCTION constexpr aabb(const ippl::Vector<scalar, Dim>& s, const ippl::Vector<scalar, Dim>& e){
            for(unsigned k = 0;k < Dim;k++){
                start[k] = s[k];
                end  [k] = e[k];
            }
        }
        KOKKOS_INLINE_FUNCTION constexpr scalar extent(unsigned d)const noexcept{
            assert(d < Dim);
            return end[d] - start[d];
        }

        KOKKOS_INLINE_FUNCTION constexpr scalar max_extent()const noexcept{
            assert(d < Dim);
            using Kokkos::max;
            scalar ext = extent(0);
            for(unsigned d = 1;d < Dim;d++){
                ext = max(ext, extent(d));
            }
            return ext;
        }
        KOKKOS_INLINE_FUNCTION bool contains(const rm::Vector<scalar, Dim>& v)const noexcept{
            for(unsigned d = 0;d < Dim;d++){
                if(v[d] < start[d] || v[d] > end[d])return false;
            }
            return true;
        }
        KOKKOS_INLINE_FUNCTION bool contains(const ippl::Vector<scalar, Dim>& v)const noexcept{
            for(unsigned d = 0;d < Dim;d++){
                if(v[d] < start[d] || v[d] > end[d])return false;
            }
            return true;
        }

    };
    /**
     * @brief Struct representing a line through N Dimensional space
     * 
     * @details Consists of origin and direction vector
     * 
     * @tparam scalar Scalar type
     * @tparam Dim Dimension
     */
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
    /**
     * @brief Computes the two intersection points of a line with a box
     * 
     * @tparam scalar The floating-point type
     * @tparam Dim dimension of the box and ray
     * @param r The line parametrization
     * @param b The box
     * @return The two intersection t parameters, returns NaN for no intersection 
     */
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
    /**
     * @brief Struct representing an image
     * 
     */
    struct Image{
        //Color type, representing an RGBA color with values in [0,1]
        using color_type = Vector<float, 4>;
        //Depth type, representing framebuffer depth
        using depth_type = float;

        //Resolution of the image
        uint32_t width, height;


        using view_type = Kokkos::View<color_type*>;
        using depth_view_type = Kokkos::View<color_type*>;

        //Color buffer, width * height elements
        Kokkos::View<color_type*> color_buffer;
        //Depth buffer, width * height elements
        Kokkos::View<depth_type*> depth_buffer;

        //Inexpensive default ctor
        Image() : width(0), height(0){}

        //Initialize empty image with unicolor
        Image(uint32_t w, uint32_t h, color_type fill = color_type{0.1f, 0.4f, 0.5f, 1.0f}) : width(w), height(h){
            initialize(fill);
        }
        /**
         * @brief Initialize the image with one color
         * 
         * @param fill Fill color
         */
        void initialize(color_type fill){
            color_buffer = Kokkos::View<color_type*>("Image::color_buffer", width * height);
            auto cbuf = color_buffer;
            Kokkos::parallel_for(width * height, KOKKOS_LAMBDA(size_t i){
                cbuf(i) = fill;
            });

            depth_buffer = Kokkos::View<depth_type*>("Image::depth_buffer", width * height);
            auto dbuf = depth_buffer;
            depth_type infin = std::numeric_limits<depth_type>::infinity();
            Kokkos::parallel_for(width * height, KOKKOS_LAMBDA(size_t i){
                dbuf(i) = infin;
            });
        }
        /**
         * @brief Save the image to a file
         * 
         * @param out_file_name filename **including ending** !
         */
        void save_to(const char* out_file_name){
            save_to(std::string(out_file_name));
        }
        /**
         * @brief Save the image to a file
         * 
         * @param out_file_name filename **including ending** !
         */
        void save_to(const std::string& out_file_name){
            using output_color = ippl::Vector<uint8_t, 4>;
            Kokkos::View<color_type*> cbview = color_buffer;
            Kokkos::View<output_color*> oview("output_view", color_buffer.extent(0));
            Kokkos::parallel_for(width * height, KOKKOS_LAMBDA(size_t i){
                oview(i)[0] = uint8_t(Kokkos::min(255u, uint32_t(clamp(cbview(i)[0], 0.0f, 1.0f) * 256.0f)));
                oview(i)[1] = uint8_t(Kokkos::min(255u, uint32_t(clamp(cbview(i)[1], 0.0f, 1.0f) * 256.0f)));
                oview(i)[2] = uint8_t(Kokkos::min(255u, uint32_t(clamp(cbview(i)[2], 0.0f, 1.0f) * 256.0f)));
                oview(i)[3] = uint8_t(Kokkos::min(255u, uint32_t(clamp(cbview(i)[3], 0.0f, 1.0f) * 256.0f)));
            });
            Kokkos::fence();
            Kokkos::View<output_color*>::host_mirror_type hmirror = Kokkos::create_mirror_view(oview);
            Kokkos::deep_copy(hmirror, oview);
            Kokkos::fence();
            if(out_file_name.ends_with("png")){
                stbi_write_png(out_file_name.c_str(), width, height, 4, hmirror.data(), width * 4);
            }
            else if(out_file_name.ends_with("bmp")){
                stbi_write_bmp(out_file_name.c_str(), width, height, 4, hmirror.data());
            }
        }
        // Setter getter approach due to lambda shenanigans
        /**
         * @brief Gets a pixel color
         * 
         * @param i Y-Coordinate of the pixel
         * @param j X-Coordinate of the pixel
         * @return Pixel color 
         */
        KOKKOS_INLINE_FUNCTION color_type get(uint32_t i, uint32_t j)const{
            return color_buffer(i * width + j);
        }
        /**
         * @brief Sets a pixel without depth check
         * 
         * @param i Y-Coordinate of the pixel
         * @param j X-Coordinate of the pixel
         * @param c Color
         */
        KOKKOS_INLINE_FUNCTION void set(uint32_t i, uint32_t j, color_type c)const{
            color_buffer(i * width + j) = c;
        }
        /**
         * @brief Sets a pixel with depth check
         * 
         * @param i Y-Coordinate of the pixel
         * @param j X-Coordinate of the pixel
         * @param c Color
         * @param d Depth value
         */
        KOKKOS_INLINE_FUNCTION void set(uint32_t i, uint32_t j, color_type c, depth_type d)const{
            if(depth_buffer(i * width + j) > d){
                depth_buffer(i * width + j) = d;
                color_buffer(i * width + j) = c;
            }
        }
    };
    /**
     * @brief Maps a float from [0,1] to an int in [0,255] and accesses the colormap at that position. 
     * Linearly interpolates the two nearest colors.
     * 
     * @param cmap Colormap values
     * @param value Normalized access value in [0,1]
     * @return A 3-D vector representing RGB in [0,1]
     */
    KOKKOS_INLINE_FUNCTION ippl::Vector<float, 3> normalized_colormap(const float (&cmap)[256][3], float value){
        int intensity_index = int(clamp(value, 0.0f, 1.0f) * 255.0f);
        float fractional = clamp(value, 0.0f, 1.0f) * 255.0f - float(intensity_index);
        ippl::Vector<float, 3> lower{
            float(cmap[intensity_index][0]),
            float(cmap[intensity_index][1]),
            float(cmap[intensity_index][2]),
        };
        if(intensity_index == 255)return lower;
        lower *= (1.0f - fractional);
        lower += ippl::Vector<float, 3>{
            float(cmap[intensity_index + 1][0] * fractional),
            float(cmap[intensity_index + 1][1] * fractional),
            float(cmap[intensity_index + 1][2] * fractional),
        };
        return lower;
    }
    /**
     * @brief Extends a 3D color to a 4D witha given alpha value
     * 
     * @param color Color input
     * @param alpha alpha input
     * @return [Red, Green, Blue, Alpha] 
     */
    KOKKOS_INLINE_FUNCTION ippl::Vector<float, 4> alpha_extend(ippl::Vector<float, 3> color, float alpha){
        ippl::Vector<float, 4> ret;
        ret[0] = color[0];
        ret[1] = color[1];
        ret[2] = color[2];
        ret[3] = alpha;
        return ret;
    }
    enum struct axis : unsigned{
        x, y, z
    };
    /**
     * @brief Draws an axis-aligned cross section of a field
     * 
     * @tparam T Scalar type of the mesh, should be an actual scalar and not a vector
     * @tparam field_valuetype Value type of the field
     * @tparam color_map A function mapping field_valuetype to either Vector<float, 3> or Vector<float, 4>
     * @param width Image width
     * @param height Image height
     * @param axis_index 0 for yz plane, 1 for xz plane, 2 for xy plane
     * @param offset Offset along cross section plane normal
     */
    template<typename T, typename field_valuetype, typename color_map>
        requires (std::is_invocable_r_v<ippl::Vector<float, 3>, color_map, field_valuetype> || 
                  std::is_invocable_r_v<ippl::Vector<float, 4>, color_map, field_valuetype>)
    Image drawFieldCrossSection(const Field<field_valuetype, 3, UniformCartesian<T, 3>, typename UniformCartesian<T, 3>::DefaultCentering>& f, const uint32_t width, const uint32_t height, axis perpendicular_to, T offset, color_map cmap){
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
        NDIndex<3> gDom = f.getLayout().getDomain();
        ippl::Vector<T, 3> domain_begin = O;
        ippl::Vector<T, 3> global_domain_begin = O;
        ippl::Vector<T, 3> domain_end = O;
        ippl::Vector<T, 3> global_domain_end = O;
        domain_begin += hr * lDom.first();
        domain_end   += hr * lDom.last();
        global_domain_begin += hr * gDom.first();
        global_domain_end   += hr * gDom.last();
        aabb<float, 3> domain_box(domain_begin, domain_end);
        aabb<float, 3> global_domain_box(global_domain_begin, global_domain_end);

        ippl::Vector<T, 2> transverse_sizes;
        if(perpendicular_to == axis::x){
            transverse_sizes = Vector<T, 2>{global_domain_box.extent(1), global_domain_box.extent(2)};
        }
        if(perpendicular_to == axis::y){
            transverse_sizes = Vector<T, 2>{global_domain_box.extent(0), global_domain_box.extent(2)};
        }
        if(perpendicular_to == axis::z){
            transverse_sizes = Vector<T, 2>{global_domain_box.extent(0), global_domain_box.extent(1)};
        }
        
        //i -> y
        //j -> x
        Kokkos::parallel_for(policy_type(begin, end), KOKKOS_LAMBDA(uint32_t i, uint32_t j){
            Vector<T, 2> plane_remap{
                float(j) * transverse_sizes[0] / width,
                float(i) * transverse_sizes[1] / height,
            };
            Vector<T, 3> sample_pos;
            if(perpendicular_to == axis::x){
                sample_pos = Vector<T, 3>{offset, plane_remap[0], plane_remap[1]};
            }
            if(perpendicular_to == axis::y){
                sample_pos = Vector<T, 3>{plane_remap[0], offset, plane_remap[1]};
            }
            if(perpendicular_to == axis::z){
                sample_pos = Vector<T, 3>{plane_remap[0], plane_remap[1], offset};
            }
            auto [inside, value] = field_at(fview, sample_pos, O, hr, lDom);
            if(!inside)return;
            if constexpr(std::is_same_v<ippl::Vector<float, 3>, std::remove_all_extents_t<std::invoke_result_t<color_map, field_valuetype>>>){
                auto col = alpha_extend(cmap(value), 1.0f);
                ret.set(i, j, col);
            }
            else if constexpr(std::is_same_v<ippl::Vector<float, 4>, std::remove_all_extents_t<std::invoke_result_t<color_map, field_valuetype>>>){
                auto col = cmap(value);
                ret.set(i, j, col);
            }
        });
        return ret;
    }
    /**
     * @brief Draw a fog representation of a 3D Field with raymarching
     * 
     * @tparam T Scalar type of the mesh, should be an actual scalar and not a vector
     * @tparam field_valuetype Value type of the field
     * @tparam color_map A function mapping field_valuetype to either Vector<float, 3> or Vector<float, 4>
     * @return An image containing the rendered field. 
     * @details The contained depth info is as if the field was an opaque cube.
     */
    template<typename T, typename field_valuetype, typename color_map>
        requires (std::is_invocable_r_v<ippl::Vector<float, 3>, color_map, field_valuetype> || 
                  std::is_invocable_r_v<ippl::Vector<float, 4>, color_map, field_valuetype>)
    Image drawFieldFog(const Field<field_valuetype, 3, UniformCartesian<T, 3>, typename UniformCartesian<T, 3>::DefaultCentering>& f, const uint32_t width, const uint32_t height, rm::camera cam, color_map cmap){
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
        NDIndex<3> gDom = f.getLayout().getDomain();
        ippl::Vector<T, 3> domain_begin = O;
        ippl::Vector<T, 3> global_domain_begin = O;
        ippl::Vector<T, 3> domain_end = O;
        ippl::Vector<T, 3> global_domain_end = O;
        domain_begin += hr * lDom.first();
        domain_end   += hr * lDom.last();

        global_domain_begin += hr * gDom.first();
        global_domain_end   += hr * gDom.last();
        aabb<float, 3> domain_box(domain_begin, domain_end);
        aabb<float, 3> global_domain_box(global_domain_begin, global_domain_end);
        const float distance_normalization = global_domain_box.max_extent();
        //f.getLayout().getDomain().first()[0];
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
            int xi = int(x) - int(width / 2);
            int yi = int(y) - int(height / 2);
            rm::Vector<float, 3> tp = (look + (left * 2.0f * float(xi) * (1.0f / float(width)) * xmul) + up * (2.0f * ymul * float(yi) * (1.0f / float(height)))).template cast<T>().normalized();
            ray<float, 3> lray(pos, tp);
            auto [first_t, second_t] = intersection_ts(lray, domain_box);
            if(Kokkos::isnan(first_t)){
                ret.set(y, x, ippl::Vector<uint8_t, 4>{0,0,0,0});
                return;
            }
            
            //const float transparency = 1.0f - alfa;
            const float luminance = 0.1f;
            //constexpr float normalization_factor = 0.1f;
            int stepc = 300;
            float stepmul = 0.01f * distance_normalization;
            ippl::Vector<float, 4> float_color{0,0,0,0};
            {
                for(int step = stepc - 1;step >= 0;step--){
                    T t = step * stepmul;
                    auto [inside, value] = field_at(fview, rm_to_ippl(pos + tp * t), O, hr, lDom);
                    if(!inside)continue;
                    auto col = cmap(value);

                    float alfa = 0;
                    if constexpr(std::is_same_v<ippl::Vector<float, 3>, std::remove_all_extents_t<std::invoke_result_t<color_map, field_valuetype>>>){
                        alfa = 0.8f;
                    }
                    else if constexpr(std::is_same_v<ippl::Vector<float, 4>, std::remove_all_extents_t<std::invoke_result_t<color_map, field_valuetype>>>){
                        alfa = col[3];
                    }
                    float transparency = 1.0f - alfa;
                    const float transparency_remaining  = Kokkos::pow(transparency, stepmul / distance_normalization);
                    const float luminance_gained = 1.0f - Kokkos::pow(luminance,    stepmul / distance_normalization);

                    float_color[0] = float_color[0] * transparency_remaining + col[0] * luminance_gained;
                    float_color[1] = float_color[1] * transparency_remaining + col[1] * luminance_gained;
                    float_color[2] = float_color[2] * transparency_remaining + col[2] * luminance_gained;
                    float_color[3] = 1.0f - (1.0f - float_color[3]) * transparency_remaining;
                }
            }
            ret.set(y, x, float_color, first_t);
        });
        Kokkos::fence();
        return ret;
    }
}
#endif