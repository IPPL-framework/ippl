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
    KOKKOS_INLINE_FUNCTION ippl::Vector<float, 3> alpha_remove(ippl::Vector<float, 4> color){
        ippl::Vector<float, 3> ret;
        ret[0] = color[0];
        ret[1] = color[1];
        ret[2] = color[2];
        return ret;
    }
    KOKKOS_INLINE_FUNCTION ippl::Vector<float, 4> porterDuff(const ippl::Vector<float, 4>& A4, const ippl::Vector<float, 4>& B4){
        float alpha_A = A4[3];
        float alpha_B = B4[3];
        float alpha_C;

        ippl::Vector<float, 3> A = alpha_remove(A4);
        ippl::Vector<float, 3> B = alpha_remove(B4);
        ippl::Vector<float, 3> C;
        alpha_C = alpha_A + (1.0f - alpha_A) * alpha_B;
        if(alpha_C != 0.0f){
            C[0] = (1.0f / alpha_C) * (A[0] * alpha_A + B[0] * (1.0f - alpha_A) * alpha_B);
            C[1] = (1.0f / alpha_C) * (A[1] * alpha_A + B[1] * (1.0f - alpha_A) * alpha_B);
            C[2] = (1.0f / alpha_C) * (A[2] * alpha_A + B[2] * (1.0f - alpha_A) * alpha_B);
        }
        return alpha_extend(C, alpha_C);
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


        using color_buffer_type = Kokkos::View<color_type*>;
        using depth_buffer_type = Kokkos::View<depth_type*>;

        //Color buffer (Kokkos::View<color_type*>), width * height elements
        color_buffer_type color_buffer;
        //Depth buffer (Kokkos::View<depth_type*>), width * height elements
        depth_buffer_type depth_buffer;

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
            //depth_type infin = std::numeric_limits<depth_type>::infinity();
            Kokkos::parallel_for(width * height, KOKKOS_LAMBDA(size_t i){
                dbuf(i) = 1e20;
            });
        }
        /**
         * @brief Save the image to a file
         * 
         * @param out_file_name filename **including ending** !
         * @return 0 on failure, not 0 otherwise
         */
        bool save_to(const char* out_file_name)const noexcept{
            return save_to(std::string(out_file_name));
        }
        /**
         * @brief Get the RangePolicy for the image
         * 
         * @return RangePolicy<2, typename Image::color_buffer_type::execution_space>::policy_type 
         */
        RangePolicy<2, typename Image::color_buffer_type::execution_space>::policy_type getRangePolicy()const noexcept{
            Kokkos::Array<uint32_t, 2> begin, end;
            begin[0] = 0;
            begin[1] = 0;
            end[0] = height;
            end[1] = width;
            return RangePolicy<2, typename Image::color_buffer_type::execution_space>::policy_type(begin, end);
        }
        /**
         * @brief Save the image to a file
         * 
         * @param out_file_name filename **including ending** !
         * @return 0 on failure, not 0 otherwise
         */
        bool save_to(const std::string& out_file_name)const noexcept{
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
                return stbi_write_png(out_file_name.c_str(), width, height, 4, hmirror.data(), width * 4);
            }
            else if(out_file_name.ends_with("bmp")){
                return stbi_write_bmp(out_file_name.c_str(), width, height, 4, hmirror.data());
            }
            std::cerr << "Unsupported format: " + out_file_name + "\n";
            return 1;
        }
        /**
         * @brief Writes the image to a FILE, useful for pipes to ffmpeg
         * 
         * @param file File descriptor
         * @param format Output format, supported values are "bmp" and "png"
         * @return 0 on failure, not 0 otherwise
         */
        bool save_to(FILE* file, const char (&format)[4] = "bmp")const noexcept{
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
            if(format[0] == 'b'){ //bmp
                return stbi_write_bmp_to_func(
                [](void* context, void* data, int size) {
                    FILE* fdr = reinterpret_cast<FILE*>(context);
                    std::fwrite(data, 1, size, fdr);
                },
                file, width, height, 4, hmirror.data());
            }
            if(format[1] == 'p'){ //png
                return stbi_write_png_to_func(
                [](void* context, void* data, int size) {
                    FILE* fdr = reinterpret_cast<FILE*>(context);
                    std::fwrite(data, 1, size, fdr);
                },
                file, width, height, 4, hmirror.data(), width * 4);
            }
            std::cerr << "Unsupported format: " + std::string(format) + "\n";
            return 1;
        }
        /**
         * @brief Performs a depth blend with another image
         * Changes this image! 
         * @details The method used to blend two colors is 
         *
         * @param o Other image
         */
        void depthBlend(const Image& o){
            if(o.width != width || o.height != height){
                std::cerr << "Cannot depth blend images of mismatching size\n";
                std::cerr << width << " x " << height << " vs " << o.width << " x " << o.height << "\n";
                std::abort();
            }
            const uint32_t width  = this->width;
            //uint32_t height = this->height;
            auto this_cb = this->color_buffer;
            auto this_db = this->depth_buffer;
            auto othr_cb =     o.color_buffer;
            auto othr_db =     o.depth_buffer;
            if(true){
                Kokkos::parallel_for(this->getRangePolicy(), KOKKOS_LAMBDA(uint32_t i, uint32_t j){
                    float alpha_A;
                    float alpha_B;
                    float alpha_C;
                    ippl::Vector<float, 3> A, B, C{0.0f,0.0f,0.0f};
                    if(othr_db(i * width + j) < this_db(i * width + j)){
                        alpha_A = othr_cb(i * width + j)[3];
                        alpha_B = this_cb(i * width + j)[3];
                        A = alpha_remove(othr_cb(i * width + j));
                        B = alpha_remove(this_cb(i * width + j));
                    }
                    else{
                        alpha_B = othr_cb(i * width + j)[3];
                        alpha_A = this_cb(i * width + j)[3];
                        B = alpha_remove(othr_cb(i * width + j));
                        A = alpha_remove(this_cb(i * width + j));
                    }
                    alpha_C = alpha_A + (1.0f - alpha_A) * alpha_B;
                    if(alpha_C != 0.0f){

                        //C[0] = A[0] + B[0] * (1.0f - alpha_A);
                        //C[1] = A[1] + B[1] * (1.0f - alpha_A);
                        //C[2] = A[2] + B[2] * (1.0f - alpha_A);
                        C[0] = (1.0f / alpha_C) * (A[0] * alpha_A + B[0] * (1.0f - alpha_A) * alpha_B);
                        C[1] = (1.0f / alpha_C) * (A[1] * alpha_A + B[1] * (1.0f - alpha_A) * alpha_B);
                        C[2] = (1.0f / alpha_C) * (A[2] * alpha_A + B[2] * (1.0f - alpha_A) * alpha_B);
                    }
                    this_cb(i * width + j) = alpha_extend(C, alpha_C);
                });
            }
            Kokkos::fence();
        }
        Image& operator+=(const Image& o){
            if(o.width != width || o.height != height){
                std::cerr << "Cannot depth blend images of mismatching size\n";
                std::cerr << width << " x " << height << " vs " << o.width << " x " << o.height << "\n";
                std::abort();
            }
            const uint32_t width  = this->width;
            auto this_cb = this->color_buffer;
            auto othr_cb =     o.color_buffer;

            Kokkos::parallel_for(this->getRangePolicy(), KOKKOS_LAMBDA(uint32_t i, uint32_t j){
                
                this_cb(i * width + j) += othr_cb(i * width + j);
            });
            return *this;
        }
        void collectOnRank0(){
            int size = ippl::Comm->size();
            int rank = ippl::Comm->rank();
            //Every color element has 4 floats
            int csendsize = width * height * 4;

            //Every depth element has 1 floats
            int dsendsize = width * height;
            for (int mask = 1; mask < size; mask <<= 1) {
                int partner = rank ^ mask;
                if ((rank & mask) == 0 && partner < size) {
                    Image tmp(width, height);
                    // Receive data from partner
                    MPI_Recv(tmp.color_buffer.data(), csendsize, MPI_FLOAT, partner, 0, ippl::Comm->getCommunicator(), MPI_STATUS_IGNORE);
                    MPI_Recv(tmp.depth_buffer.data(), dsendsize, MPI_FLOAT, partner, 1, ippl::Comm->getCommunicator(), MPI_STATUS_IGNORE);
                    
                    depthBlend(tmp);
                    
                } else if (rank & mask) {
                    // Send data to partner
                    MPI_Send(color_buffer.data(), csendsize, MPI_FLOAT, partner, 0, ippl::Comm->getCommunicator());
                    MPI_Send(depth_buffer.data(), dsendsize, MPI_FLOAT, partner, 1, ippl::Comm->getCommunicator());
                }
            }
        }

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
            depth_buffer(i * width + j) = 0;
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
    template<typename field_type>
    KOKKOS_INLINE_FUNCTION aabb<float, 3> getLocalDomainBox(field_type f){
        auto fview = f.getView();
        ippl::Vector<field_type, 3> O = f.get_mesh().getOrigin();
        ippl::Vector<field_type, 3> hr = f.get_mesh().getMeshSpacing();
        NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
        NDIndex<3> gDom = f.getLayout().getDomain();
        ippl::Vector<field_type, 3> domain_begin = O;
        ippl::Vector<field_type, 3> global_domain_begin = O;
        ippl::Vector<field_type, 3> domain_end = O;
        ippl::Vector<field_type, 3> global_domain_end = O;
        domain_begin += hr * lDom.first();
        domain_end   += hr * lDom.last();
        domain_end   += hr;

        global_domain_begin += hr * gDom.first();
        global_domain_end   += hr * gDom.last();
        aabb<float, 3> domain_box(domain_begin, domain_end);
        aabb<float, 3> global_domain_box(global_domain_begin, global_domain_end);
        return domain_box;
    }
    template<typename field_type>
    KOKKOS_INLINE_FUNCTION aabb<float, 3> getGlobalDomainBox(field_type f){
        auto fview = f.getView();
        auto c_c = f.get_mesh().getOrigin();
        using scalar_type = typename decltype(c_c)::value_type;
        ippl::Vector<scalar_type, 3> O = f.get_mesh().getOrigin();
        ippl::Vector<scalar_type, 3> hr = f.get_mesh().getMeshSpacing();
        NDIndex<3> lDom = f.getLayout().getLocalNDIndex();
        NDIndex<3> gDom = f.getLayout().getDomain();
        ippl::Vector<scalar_type, 3> domain_begin = O;
        ippl::Vector<scalar_type, 3> global_domain_begin = O;
        ippl::Vector<scalar_type, 3> domain_end = O;
        ippl::Vector<scalar_type, 3> global_domain_end = O;
        domain_begin += hr * lDom.first();
        domain_end   += hr * lDom.last();
        domain_end   += hr;

        global_domain_begin += hr * gDom.first();
        global_domain_end   += hr * gDom.last();
        aabb<float, 3> domain_box(domain_begin, domain_end);
        aabb<float, 3> global_domain_box(global_domain_begin, global_domain_end);
        return global_domain_box;
    }
    
    
    enum struct axis : unsigned{
        x, y, z
    };
    /**
     * @brief Project particles onto a plane and draw them. Requires a domain for scaling
     * 
     * @param position_attrib View containing the particles
     * @param count Amount of particles to be drawn, must be < position_attrib.extent(0)
     * @param width Image width
     * @param height Image height
     * @param axis Projection plane axis, either ippl::axis::x, ippl::axis::y or ippl::axis::z
     * @param domain the domain to fit onto the screen
     * @param particle_radius On-Screen radii of drawn particles
     * @param particle_color On-Screen fill color of particles
     * 
     * 
     * @tparam vector_type Position vector type
     */
    template<typename vector_type>
        requires(vector_type::dim == 3)
    Image drawParticlesProjection(Kokkos::View<vector_type*> position_attrib, size_t count, int width, int height, const axis orthogonal_to, const aabb<typename vector_type::value_type, 3> dom, float particle_radius, Vector<float, 4> particle_color){
        Image ret(width, height, ippl::Vector<float, 4>{0,0,0,0});
        auto cbuffer = ret.color_buffer;
        ippl::Vector<float, 2> img_extents{(float)width, (float)height};
        
        Kokkos::parallel_for(count, KOKKOS_LAMBDA(size_t idx){
            vector_type pos = position_attrib(idx);
            unsigned k = 0;
            ippl::Vector<float, 2> pos_remap;
            for(unsigned d = 0;d < 3;d++){
                if(d == (unsigned)orthogonal_to)continue;
                pos_remap[k] = (pos[d] - dom.start[d]) / (dom.end[d] - dom.start[d]) * img_extents[k];
                ++k;
            }
            int j = int(pos_remap[0]);
            int i = int(pos_remap[1]);
            std::cout << i << ", " << j << "\n";
            float corrected_radius = particle_radius;
            using Kokkos::ceil;
            using Kokkos::min;
            using Kokkos::exp;
            int ill = i - 2 * Kokkos::ceil(corrected_radius);
            int iul = i + 2 * Kokkos::ceil(corrected_radius);
            int jll = j - 2 * Kokkos::ceil(corrected_radius);
            int jul = j + 2 * Kokkos::ceil(corrected_radius);
            
            //Inner loop for filling the circle 
            for(int _i = ill;_i <= iul;_i++){
                for(int _j = jll;_j <= jul;_j++){
                    if(_i >= 0 && _i < height && _j >= 0 && _j < width){
                        float pdist = float(_i - i) * float(_i - i) + float(_j - j) * float(_j - j);
                        if(pdist < corrected_radius * corrected_radius){
                            float inten = Kokkos::exp(-pdist * pdist / (corrected_radius * corrected_radius));
                            cbuffer(_i * width + _j)[0] = particle_color[0];
                            cbuffer(_i * width + _j)[1] = particle_color[1];
                            cbuffer(_i * width + _j)[2] = particle_color[2];
                            Kokkos::atomic_add(&(cbuffer(_i * width + _j)[3]), inten);
                            //std::cout << i << ", " << j << "\n";
                        }
                    }
                }
            }
        });
        return ret;
    }
    /**
     * @brief Project particles onto a plane and draw them. Requires a domain for scaling
     * 
     * @param position_attrib ParticleAttribute describing the particles' positions
     * @param count Amount of particles to be drawn, must be < position_attrib.extent(0)
     * @param width Image width
     * @param height Image height
     * @param axis Projection plane axis, either ippl::axis::x, ippl::axis::y or ippl::axis::z
     * @param domain the domain to fit onto the screen
     * @param particle_radius On-Screen radii of drawn particles
     * @param particle_color On-Screen fill color of particles
     * 
     * 
     * @tparam vector_type Position vector type
     */
    template<typename vector_type, class... Properties>
        requires(vector_type::dim == 3)
    Image drawBunchProjection(ippl::ParticleAttrib<vector_type, Properties...> position_attrib, int width, int height, const axis orthogonal_to, aabb<typename vector_type::value_type, 3> dom, float particle_radius, Vector<float, 4> particle_color){
        return drawParticlesProjection(position_attrib.getView(), position_attrib.getParticleCount(), width, height, orthogonal_to, dom, particle_radius, particle_color);
    }
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
     * @param swap_axis_order Whether the axis order should be swapped from x->y->z to its reverse. 
     * Will cause cross section drawings to be mirrored along the diagonal.
     */
    template<typename T, typename field_valuetype, typename color_map>
        requires (std::is_invocable_r_v<ippl::Vector<float, 3>, color_map, field_valuetype> || 
                  std::is_invocable_r_v<ippl::Vector<float, 4>, color_map, field_valuetype>)
    Image drawFieldCrossSection(const Field<field_valuetype, 3, UniformCartesian<T, 3>, typename UniformCartesian<T, 3>::DefaultCentering>& f, const uint32_t width, const uint32_t height, axis perpendicular_to, T offset, color_map cmap, bool swap_axis_order = false){
        Image ret(width, height, Vector<uint8_t, 4>{0,0,0,255});
        
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
        ippl::Vector<T, 2> transverse_origins;
        if(perpendicular_to == axis::x){
            transverse_sizes = Vector<T, 2>  {global_domain_box.extent(1), global_domain_box.extent(2)};
            transverse_origins = Vector<T, 2>{global_domain_box.start [1],  global_domain_box.start[2]};
        }
        if(perpendicular_to == axis::y){
            transverse_sizes = Vector<T, 2>{global_domain_box.extent (0), global_domain_box.extent(2)};
            transverse_origins = Vector<T, 2>{global_domain_box.start[0], global_domain_box.start [2]};
        }
        if(perpendicular_to == axis::z){
            transverse_sizes = Vector<T, 2>{global_domain_box.extent (0), global_domain_box.extent(1)};
            transverse_origins = Vector<T, 2>{global_domain_box.start[0], global_domain_box.start [1]};
        }
        if(swap_axis_order){
            std::swap(transverse_origins[0], transverse_origins[1]);
            std::swap(transverse_sizes[0], transverse_sizes[1]);
        }
        
        //i -> y
        //j -> x
        Kokkos::parallel_for(ret.getRangePolicy(), KOKKOS_LAMBDA(uint32_t i, uint32_t j){
            Vector<T, 2> plane_remap{
                (float(j) / width ) * transverse_sizes[0] + transverse_origins[0],
                (float(i) / height) * transverse_sizes[1] + transverse_origins[1],
            };
            Vector<T, 3> sample_pos;
            if(swap_axis_order){
                T tmp = plane_remap[0];
                plane_remap[0] = plane_remap[1];
                plane_remap[1] = tmp;
            }
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
            if(!inside){
                //ret.set(i, j, Vector<float, 4>{0.2,0.4,0.5,1.0});
                return;
            }
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
    Image drawFieldFog(const Field<field_valuetype, 3, UniformCartesian<T, 3>, typename UniformCartesian<T, 3>::DefaultCentering>& f, const uint32_t width, const uint32_t height, rm::camera cam, color_map cmap, Image already_drawn = Image()){
        if((already_drawn.width != 0 && already_drawn.height != 0) && (already_drawn.width != width || already_drawn.height != height)){
            std::cerr << "Framebuffer passed as already_drawn mismathes in size\n";
            std::abort();
        }
        Image ret(width, height, Vector<uint8_t, 4>{0,0,0,0});
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
        domain_end   += hr;

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
        std::cout << "Luegin in direction" << look << "\n";
        auto left = cam.left().normalized();
        //std::cout << "Left" << left << "\n";

        rm::Vector<float, 3> up{0,1,0};
        up = left.cross(look).normalized();
        //auto left = cam.;
        bool check_for_already_existing_depth_buffer = already_drawn.width > 0;
        auto preexisting_depthbuffer = already_drawn.depth_buffer;
        Kokkos::parallel_for(ret.getRangePolicy(), KOKKOS_LAMBDA(uint32_t y, uint32_t x){
            int xi = int(x) - int(width / 2);
            int yi = int(y) - int(height / 2);
            rm::Vector<float, 3> tp = (look + (left * 2.0f * float(xi) * (1.0f / float(width)) * xmul) + up * (2.0f * ymul * float(yi) * (1.0f / float(height)))).template cast<T>().normalized();
            ray<float, 3> lray(pos, tp.normalized());
            auto [first_t, second_t] = intersection_ts(lray, domain_box);
            if(Kokkos::isnan(first_t)){// || (check_for_already_existing_depth_buffer && (first_t > preexisting_depthbuffer(y * width + x)))){
                if(check_for_already_existing_depth_buffer){
                    ret.color_buffer(y * width + x) = already_drawn.color_buffer(y * width + x);
                    ret.depth_buffer(y * width + x) = already_drawn.depth_buffer(y * width + x);

                }
                
                return;
            }
            
            //const float transparency = 1.0f - alfa;
            const float luminance = 0.1f;
            //constexpr float normalization_factor = 0.1f;
            int stepc = 1000;
            float stepmul = 0.01f * distance_normalization;
            ippl::Vector<float, 4> float_color{0,0,0,0};
            {
                int step = stepc - 1;
                if(check_for_already_existing_depth_buffer){
                    if(already_drawn.depth_buffer(y * width + x) < 1e17 && already_drawn.depth_buffer(y * width + x) >= ((step - 1) * stepmul)){
                        step = Kokkos::min(step, int(already_drawn.depth_buffer(y * width + x) / stepmul));
                        if(step < stepc){
                            float_color = already_drawn.color_buffer(y * width + x);
                        }
                    }
                }
                for(;step >= 0;step--){
                    T t = step * stepmul;
                    auto [inside, value] = field_at(fview, rm_to_ippl(pos + tp * t), O, hr, lDom);
                    if(inside){
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
                    if(check_for_already_existing_depth_buffer){
                        if(already_drawn.depth_buffer(y * width + x) > (step * stepmul) && already_drawn.depth_buffer(y * width + x) < ((step+1) * stepmul)){
                            step = Kokkos::min(step, int(already_drawn.depth_buffer(y * width + x) / stepmul));
                            if(step < stepc){
                                float_color = porterDuff(already_drawn.color_buffer(y * width + x), float_color);
                            }
                        }
                    }
                }
            }
            ret.set(y, x, float_color, first_t);
        });
        Kokkos::fence();
        return ret;
    }
    /**
     * @brief Draw particles as size-corrected circles to the screen as seen from a certain point
     * 
     * @tparam vector_type 
     * @param position_view Kokkos View containing position vectors
     * @param count Amount of particles to be visualized, must be <= position_view.extent(0)
     * @param width Output image width
     * @param height Output image height
     * @param cam Point of view camera
     * @param radius Visible particle radius
     * @param particle_color Visible particle color
     */
    template<typename vector_type>
        requires(vector_type::dim == 3)
    Image drawParticles(Kokkos::View<vector_type*> position_view, size_t count, int width, int height, rm::camera cam, float particle_radius, Vector<float, 4> particle_color){
        //particle_radius = Kokkos::max(1.0f, particle_radius);
        (void)particle_color;
        (void)particle_radius;
        Image ret(width, height, Vector<float, 4>{0.0f,0.0f,0.0f,0.0f});
        auto ret_cb = ret.color_buffer;
        auto ret_db = ret.depth_buffer;
        using mat4 = rm::Matrix<float, 4, 4>;
        
        //mat4 ortho_matrix = rm::ortho<float>(0, width, 0, height, -1, 1);
        
        mat4 camera_matrix = cam.matrix(width, height);
        rm::Vector<float, 3> camera_forward_vector = cam.look_dir().normalized();
        Kokkos::parallel_for(count, KOKKOS_LAMBDA(size_t particle_idx){

            //Obtain particle position
            vector_type acc = position_view(particle_idx);
            rm::Vector<float, 3> ppos3{
                (float)acc[0],
                (float)acc[1],
                (float)acc[2]
            };
            const float depth_value = (ppos3 - cam.pos).dot(camera_forward_vector);
            
            //Map to clip space ([-1,1] x [-1,1]), mind the division by w() to homogenize
            rm::Vector<float, 4> in_clipspace = camera_matrix * ppos3.template one_extend<float>();
            in_clipspace = in_clipspace * (1.0f / in_clipspace.w());

            //Map clip space to pixel space ([0,width] x [0, height]), where i represents height and j width!
            int j = int(((float)in_clipspace[0] + 1.0f) * (float(width) / 2));
            int i = int(((float)in_clipspace[1] + 1.0f) * (float(height) / 2));
            float corrected_radius = particle_radius * height / depth_value;
            if(Kokkos::isnan(corrected_radius) || Kokkos::isinf(corrected_radius)){
                corrected_radius = 0.0f;
            }
            using Kokkos::ceil;
            using Kokkos::min;
            using Kokkos::exp;
            int ill = i - 2 * Kokkos::ceil(corrected_radius);
            int iul = i + 2 * Kokkos::ceil(corrected_radius);
            int jll = j - 2 * Kokkos::ceil(corrected_radius);
            int jul = j + 2 * Kokkos::ceil(corrected_radius);
            
            //Inner loop for filling the circle 
            for(int _i = ill;_i <= iul;_i++){
                for(int _j = jll;_j <= jul;_j++){
                    if(_i >= 0 && _i < height && _j >= 0 && _j < width){
                        float pdist = Kokkos::hypotf(float(_i - i), float(_j - j));
                        if(pdist < corrected_radius){
                            ret_cb(_i * width + _j) = Vector<float, 4>(
                                particle_color * Kokkos::exp(-pdist * pdist / (corrected_radius * corrected_radius))
                            );
                            Kokkos::atomic_min(&ret_db(_i * width + _j), depth_value);
                        }
                    }
                }
            }
        });
        return ret;
    }
    template<typename vector_type, class... Properties>
        requires(vector_type::dim == 3)
    Image drawParticles(ippl::ParticleAttrib<vector_type, Properties...> position_attrib, int width, int height, rm::camera cam, float particle_radius, Vector<float, 4> particle_color){
        return drawParticles(position_attrib.getView(), position_attrib.getParticleCount(), width, height, cam, particle_radius, particle_color);
    }
}
#endif