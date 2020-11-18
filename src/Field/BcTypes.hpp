


namespace ippl {
    namespace detail {

        template<typename T, unsigned Dim, class Mesh, class Cell>
        BCondBase<T, Dim, Mesh, Cell>::BCondBase(unsigned int face)
        : face_m(face)
        , changePhysical_m(false)
        { }


        template<typename T, unsigned Dim, class Mesh, class Cell>
        inline std::ostream&
        operator<<(std::ostream& os, const BCondBase<T, Dim, Mesh, Cell>& bc)
        {
            bc.write(os);
            return os;
        }

    }

    template<typename T, unsigned Dim, class Mesh, class Cell>
    void ExtrapolateFace<T, Dim, Mesh, Cell>::apply(Field<T, Dim, Mesh, Cell>& field) 
    {
        //We only support constant extrapolation for the moment, other higher order 
        //extrapolation stuffs need to be added.
        unsigned d = this->face_m / 2;
        typename Field<T, Dim, Mesh, Cell>::view_type& view = field.getView();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        int src, dest;
        if(this->face_m & 1) {
            src  = view.extent(d) - 2;
            dest = src + 1;
        }
        else {
            src  = 1;
            dest = src - 1;
        }
        switch(d) {
            case 0:
                Kokkos::parallel_for("Assign extrapolate BC X", 
                                      mdrange_type({0, 0},
                                                   {view.extent(1),
                                                    view.extent(2)}),
                                      KOKKOS_CLASS_LAMBDA(const int j, const int k){
                                      
                                      view(dest, j, k) =  slope_m * view(src, j, k) + offset_m; 
                                      
                                      });
                break;

            case 1:
                Kokkos::parallel_for("Assign extrapolate BC Y", 
                                      mdrange_type({0, 0},
                                                   {view.extent(0),
                                                    view.extent(2)}),
                                      KOKKOS_CLASS_LAMBDA(const int i, const int k){

                                      view(i, dest, k) =  slope_m * view(i, src, k) + offset_m; 
                                       
                                      });
                break;
            case 2:
                Kokkos::parallel_for("Assign extrapolate BC Z", 
                                      mdrange_type({0, 0},
                                                   {view.extent(0),
                                                    view.extent(1)}),
                                      KOKKOS_CLASS_LAMBDA(const int i, const int j){

                                      view(i, j, dest) =  slope_m * view(i, j, src) + offset_m; 
                                       
                                      });
                break;
            default:
                break;
        }
    }
    
    template<typename T, unsigned Dim, class Mesh, class Cell>
    void ExtrapolateFace<T, Dim, Mesh, Cell>::write(std::ostream& out) const
    {
        out << "Constant Extrapolation Face"
            << ", Face = " << this->face_m;
    }

    template<typename T, unsigned Dim, class Mesh, class Cell>
    void NoBcFace<T, Dim, Mesh, Cell>::write(std::ostream& out) const
    {
        out << "NoBcFace"
            << ", Face = " << this->face_m;
    }


    template<typename T, unsigned Dim, class Mesh, class Cell>
    void ConstantFace<T, Dim, Mesh, Cell>::write(std::ostream& out) const
    {
        out << "ConstantFace"
            << ", Face = " << this->face_m
            << ", Constant = " << this->offset_m;
    }


    template<typename T, unsigned Dim, class Mesh, class Cell>
    void ZeroFace<T, Dim, Mesh, Cell>::write(std::ostream& out) const
    {
        out << "ZeroFace"
            << ", Face = " << this->face_m;
    }


    template<typename T, unsigned Dim, class Mesh, class Cell>
    void PeriodicFace<T, Dim, Mesh, Cell>::write(std::ostream& out) const
    {
        out << "PeriodicFace"
            << ", Face = " << this->face_m;
    }
    
    template<typename T, unsigned Dim, class Mesh, class Cell>
    void PeriodicFace<T, Dim, Mesh, Cell>::apply(Field<T, Dim, Mesh, Cell>& field)
    {
       unsigned d = this->face_m / 2;
       typename Field<T, Dim, Mesh, Cell>::view_type& view = field.getView();
       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
       int N = view.extent(d);
       
       switch (d) {
           case 0:
               Kokkos::parallel_for("Assign periodic field BC X", 
                                     mdrange_type({0, 0},
                                                  {view.extent(1),
                                                   view.extent(2)}),
                                     KOKKOS_CLASS_LAMBDA(const int j, const int k){
                                    
                                     view(0, j, k)   = view(N-2, j, k); 
                                     view(N-1, j, k) = view(1, j, k); 
                                   
                                     });
               break;
           case 1:
               Kokkos::parallel_for("Assign periodic field BC Y", 
                                     mdrange_type({0, 0},
                                                  {view.extent(0),
                                                   view.extent(2)}),
                                     KOKKOS_CLASS_LAMBDA(const int i, const int k){

                                     view(i, 0, k)   = view(i, N-2, k); 
                                     view(i, N-1, k) = view(i, 1, k); 
                              
                                    });
               break;
           case 2:
               Kokkos::parallel_for("Assign periodic field BC Z", 
                                     mdrange_type({0, 0},
                                                  {view.extent(0),
                                                   view.extent(1)}),
                                     KOKKOS_CLASS_LAMBDA(const int i, const int j){

                                     view(i, j, 0)    = view(i, j, N-2); 
                                     view(i, j, N-1)  = view(i, j, 1); 
                              
                                    });
               break;
           default:
               break;
       }
    }
}
