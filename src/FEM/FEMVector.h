// Class FEMVector
//    This class represents a one dimensional vector which can be used in the 
//    context of FEM to represent a field defined on the DOFs of a mesh.


#ifndef IPPL_FEMVECTOR_H
#define IPPL_FEMVECTOR_H

#include "Types/ViewTypes.h"
#include "Field/HaloCells.h"

namespace ippl {

    /**
     * @brief 1D vector used in the context of FEM.
     * 
     * @tparam T The datatype which the vector is storing.
     */
    template <typename T>
    class FEMVector {
    public:

        /**
         * @brief Constructor taking size, neighbors, and halo exchange indices.
         * 
         * Constrcutor of a FEMVector taking in the size and information about
         * the neighboring MPI ranks. 
         * 
         * @param n The size of the vector.
         * @param neighbors The ranks of the neighboring MPI tasks.
         * @param sendIdxs The indices for which the data should be send to the
         * MPI neighbors.
         * @param recvIdxs The halo cell indices.
         */
        FEMVector(size_t n, std::vector<size_t> neighbors,
            std::vector< Kokkos::View<size_t*> > sendIdxs,
            std::vector< Kokkos::View<size_t*> > recvIdxs);
        

        /**
         * @brief Copy values from neighboring ranks into local halo.
         * 
         * This function takes local values and copies to them to the
         * corresponding halo cells of the neighbors.
         */
        void fillHalo();

        /**
         * @brief Accumulate halo values in neighbor.
         * 
         * This function takes the local halo values and sums them to the
         * corresponding values of the neighbor ranks.
         */
        void accumulateHalo();

        /**
         * @brief Set the halo cells to \p clearValue.
         * 
         * @param clearValue The value to which the halo cells should be set.
         */
        void clearHalo(T clearValue);


        /**
         * @brief Set all the values of the vector to \p value.
         * 
         * Sets all the values in the vector to \p value this also includes the
         * halo cells.
         * 
         * @param value The value to which the entries should be set.
         */
        void operator= (T value);


        /**
         * @brief Get underlying data view.
         * 
         * Returns a constant reference to the underlying data the FEMVector is
         * storing, this corresponds to a \c Kokkos::View living on the default
         * device.
         */
        const Kokkos::View<T*>& getView() const;
        

        /**
         * @brief Pack data into \p FEMVector::commBuffer_m for
         * MPI communication.
         * 
         * This function takes data from the vector accoding to \p idxStore and
         * stores it inside of \p FEMVector::commBuffer_m.
         * 
         * @param idxStore A 2D Kokkos view which stores the the indices for
         * \p FEMVector::data_m which we want to send.
         */
        void pack(const Kokkos::View<size_t*>& idxStore);
        
        
        /**
         * @brief Unpack data from \p FEMVector::commBuffer_m into
         * \p FEMVector::data_m after communication.
         * 
         * This function takes data from \p FEMVector::commBuffer_m and stores
         * it accoding to \p idxStore in
         * \p FEMVector::data_m.
         * 
         * @param idxStore A 1D Kokkos view which stores the the indices for
         * \p FEMVector::data_m to which we want to store.
         * 
         * @tparam Op The operator to use in order to update the values in
         * \p FEMVector::data_m.
         */
        template <typename Op>
        void unpack(const Kokkos::View<size_t*>& idxStore);


        /**
         * @brief Struct for assigment operator to be used with
         * \p FEMVector::unpack.
         */
        struct Assign{
            KOKKOS_INLINE_FUNCTION void operator()(T& lhs, const T& rhs) const {
                lhs = rhs;
            }   
        };


        /**
         * @brief Struct for addition+assignment operator to be used with
         * \p FEMVector::unpack.
         */
        struct AssignAdd{
            KOKKOS_INLINE_FUNCTION void operator()(T& lhs, const T& rhs) const {
                lhs += rhs;
            }   
        };


    private:
        /**
         * @brief Data this object is storing.
         * 
         * The data which the \c FEMVector is storing, it is represented by a
         * one dimensional \c Kokkos::View and lives on the default device.
         */
        Kokkos::View<T*> data_m;

        /**
         * @brief Stores the ranks of the neighboring MPI tasks.
         * 
         * A vector storing the ranks of the neighboring MPI tasks. This is used
         * during halo operators in combination with \p FEMVector::sendIdxs_m
         * and \p FEMVector::recvIdxs_m in order to send the correct data to the
         * correct ranks.
         */
        std::vector<size_t> neighbors_m;

        /**
         * @brief Stores indices of \p FEMVector::data_m which need to be send
         * to the MPI neighbors.
         * 
         * This is a 2D list which stores the indices of the
         * \p FEMVector::data_m variable which need to be send to the MPI
         * neighbors. The first dimension goes over all the neighbors and should
         * be used in combination with \p FEMVector::neighbors_m while the
         * second dimension goes over the actual indices.
         * 
         * This corresponds to the indices which belong to this rank but are
         * shared by the halos of the other ranks.
         */
        std::vector< Kokkos::View<size_t*> > sendIdxs_m;

        /**
         * @brief Stores indices of \p FEMVector::data_m which are part of the
         * halo.
         * 
         * This is a 2D list which stores the indices of the
         * \p FEMVector::data_m variable which are part of the halo. The first
         * dimension goes over all the
         * neighbors and should be used in combination with
         * \p FEMVector::neighbors_m while the second dimension goes over the
         * actual indices.
         * 
         * It is called recvIdxs to be in line with \c ippl::detail::HaloCells
         * and because it stores the indices for which we recive data from the
         * neighbors.
         */
        std::vector< Kokkos::View<size_t*> > recvIdxs_m;


        /**
         * @brief Buffer for MPI communication.
         * 
         * This buffer is used during MPI communication in order to store the
         * values which should be send to the neighbors. We are using a
         * \c ippl::detail::FieldBufferData even though we do not have a
         * \c ippl::Field , but this buffer struct is general enough to allow
         * for such things.
         */
        detail::FieldBufferData<T> commBuffer_m;

    };
}   // namespace ippl


#include "FEMVector.hpp"

#endif  // IPPL_FEMVECTOR_H