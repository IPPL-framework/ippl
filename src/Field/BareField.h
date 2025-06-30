//
// Class BareField
//   A BareField represents a field.
//
#ifndef IPPL_BARE_FIELD_H
#define IPPL_BARE_FIELD_H

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

#include "Types/IpplTypes.h"

#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Utility/ViewUtils.h"

#include "Expression/IpplExpressions.h"

#include "Field/HaloCells.h"
#include "FieldLayout/FieldLayout.h"

namespace ippl {
    class Index;

    /*!
     * @file BareField.h
     * A BareField represents a real field.
     */

    /*!
     * @class BareField
     * @tparam T data type
     * @tparam Dim field dimension
     * @warning The implementation currently only supports 3-dimensional fields. The reason are
     * runtime issues with "if constrexpr" in the assignment operator when running on GPU.
     */
    template <typename T, unsigned Dim, class... ViewArgs>
    class BareField : public detail::Expression<
                          BareField<T, Dim, ViewArgs...>,
                          sizeof(typename detail::ViewType<T, Dim, ViewArgs...>::view_type)> {
    public:
        using Layout_t = FieldLayout<Dim>;

        //! Domain type specifying the index region
        using Domain_t = NDIndex<Dim>;

        //! View type storing the data
        using view_type = typename detail::ViewType<T, Dim, ViewArgs...>::view_type;
        typedef typename view_type::memory_space memory_space;
        typedef typename view_type::execution_space execution_space;
        using HostMirror = typename view_type::host_mirror_type;
        template <class... PolicyArgs>
        using policy_type = typename RangePolicy<Dim, PolicyArgs...>::policy_type;

        using halo_type = detail::HaloCells<T, Dim, ViewArgs...>;

        using value_type              = T;
        constexpr static unsigned dim = Dim;

        /*! A default constructor, which should be used only if the user calls the
         * 'initialize' function before doing anything else.  There are no special
         * checks in the rest of the BareField methods to check that the field has
         * been properly initialized.
         */
        BareField();

        BareField(const BareField&) = default;

        /*! Constructor for a BareField. The default constructor is deleted.
         * @param l of field
         * @param nghost number of ghost layers
         */
        BareField(Layout_t& l, int nghost = 1);

        /*!
         * Creates a new BareField with the same properties and contents
         * @return A deep copy of the field
         */
        BareField deepCopy() const;

        // Destroy the BareField.
        ~BareField() = default;

        /*!
         * Dimension independent view resize function which calls Kokkos.
         * @tparam Args... variadic template specifying the individiual
         * dimension arguments
         */
        template <typename... Args>
        void resize(Args... args);

        /*!
         * Initialize the field, if it was constructed from the default constructor.
         * This should NOT be called if the field was constructed by providing
         * a FieldLayout.
         * @param l of field
         * @param nghost number of ghost layers
         */
        void initialize(Layout_t& l, int nghost = 1);

        // ML
        void updateLayout(Layout_t&, int nghost = 1);

        /*!
         * Local field size.
         * @param d the dimension
         * @returns the number of grid points in the given dimension.
         */
        detail::size_type size(unsigned d) const { return owned_m[d].length(); }

        /*!
         * Index domain of the local field.
         * @returns the index domain.
         */
        const Domain_t& getOwned() const { return owned_m; }

        /*!
         * Index domain of the allocated field.
         * @returns the allocated index domain (including ghost cells)
         */
        const Domain_t getAllocated() const { return owned_m.grow(nghost_m); }

        int getNghost() const { return nghost_m; }

        void fillHalo();
        void accumulateHalo();
        void accumulateHalo_noghost(int nghost = 1);

        auto& getCommunicator() const { return getLayout().comm; }

        // Access to the layout.
        Layout_t& getLayout() const {
            PAssert(layout_m != 0);
            return *layout_m;
        }

        const Index& getIndex(unsigned d) const { return getLayout().getDomain()[d]; }
        const NDIndex<Dim>& getDomain() const { return getLayout().getDomain(); }

        halo_type& getHalo() { return halo_m; }

        // Assignment from a constant.
        BareField& operator=(T x);

        /*!
         * Assign an arbitrary BareField expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>
        BareField& operator=(const detail::Expression<E, N>& expr);

        /*!
         * Assign another field.
         * @tparam Args... variadic template to specify an access index for
         * a view element.
         * @param args view indices
         * @returns a view element
         */
        template <typename... Args>
        KOKKOS_INLINE_FUNCTION T operator()(Args... args) const {
            return dview_m(args...);
        }

        view_type& getView() { return dview_m; }

        const view_type& getView() const { return dview_m; }

        HostMirror getHostMirror() const { return Kokkos::create_mirror(dview_m); }

        /*!
         * Generate the range policy for iterating over the field,
         * excluding ghost layers
         * @tparam PolicyArgs... additional template parameters for the range policy
         * @param nghost Number of ghost layers to include in the range policy (default 0)
         * @return Range policy for iterating over the field and nghost of the ghost layers
         */
        template <class... PolicyArgs>
        policy_type<execution_space, PolicyArgs...> getFieldRangePolicy(
            const int nghost = 0) const {
            PAssert_LE(nghost, nghost_m);
            const size_t shift = nghost_m - nghost;
            return getRangePolicy(dview_m, shift);
        }

        /*!
         * Print the BareField.
         * @param out stream
         */
        void write(std::ostream& out = std::cout) const;

        /*!
         * Print the BareField
         * @param inf Inform object
         */
        void write(Inform& inf) const;

        T sum(int nghost = 0) const;
        T max(int nghost = 0) const;
        T min(int nghost = 0) const;
        T prod(int nghost = 0) const;

    private:
        //! Number of ghost layers on each field boundary
        int nghost_m;

        //! Actual field data
        view_type dview_m;

        //! Domain of the data
        Domain_t owned_m;

        halo_type halo_m;

        /*!
         * Allocate field.
         */
        void setup();

        //! How the arrays are laid out.
        Layout_t* layout_m = nullptr;
    };

}  // namespace ippl

#include "Field/BareField.hpp"
#include "Field/BareFieldOperations.hpp"

#endif
