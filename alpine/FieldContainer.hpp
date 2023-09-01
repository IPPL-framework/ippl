#ifndef IPPL_PIC_MANAGER
#define IPPL_PIC_MANAGER

#include <memory>
#include "Manager/BaseManager.h"

namespace ippl {
    // Define the FieldsContainer class
    template <class FieldLayout_t, typename T, unsigned Dim = 3>
    class FieldContainer{
    
    public:
        FieldContainer(Vector_t<double, Dim> hr, Vector_t<double, Dim> rmin,
                        Vector_t<double, Dim> rmax, ippl::e_dim_tag decomp[Dim], double Q)
            : hr_m(hr), rmin_m(rmin), rmax_m(rmax), Q_m(Q) {
            for (unsigned int i = 0; i < Dim; i++) {
                decomp_m[i] = decomp[i];
            }
     }
    
    VField_t<T, Dim> E_m;
    Field_t<Dim> rho_m;
    Field<T, Dim> phi_m;

    Vector_t<T, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    };
}  // namespace ippl

#endif
