


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


        template<typename T, unsigned Dim, class Mesh, class Cell>
        void ExtrapolateFace<T, Dim, Mesh, Cell>::apply(Field<T, Dim, Mesh, Cell>& /*field*/) {
            //TODO
        }
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

       if(d == 0) {
           int Nx = view.extent(0);
           Kokkos::parallel_for("Assign periodic field BC X", 
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                                                     {view.extent(1),
                                                                      view.extent(2)}),
                                KOKKOS_LAMBDA(const int j, const int k){

                                view(0,  j, k) = view(Nx-2, j, k); 
                                view(Nx-1, j, k) = view(1, j, k); 
                              
                              });
       }
       else if(d == 1) {
           int Ny = view.extent(1);
           Kokkos::parallel_for("Assign periodic field BC Y", 
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(2)}),
                                KOKKOS_LAMBDA(const int i, const int k){

                                view(i,  0, k) = view(i, Ny-2, k); 
                                view(i, Ny-1, k) = view(i, 1, k); 
                              
                              });


       }
       else {
           int Nz = view.extent(2);
           Kokkos::parallel_for("Assign periodic field BC Z", 
                                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                                                     {view.extent(0),
                                                                      view.extent(1)}),
                                KOKKOS_LAMBDA(const int i, const int j){

                                view(i,  j, 0)  = view(i, j, Nz-2); 
                                view(i,  j, Nz-1) = view(i, j, 1); 
                              
                              });

       }
    }
}
