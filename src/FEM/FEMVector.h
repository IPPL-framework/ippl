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
    class FEMVector : public detail::Expression<
                        FEMVector<T>,
                        sizeof(typename detail::ViewType<T, 1>::view_type)>{
    public:
        /**
         * @brief Dummy parameter in order for the \p detail::Expression defined
         * operators to work.
         * 
         * In the file IpplOperations.h a bunch of operations are defined, we
         * want to be able to use them with the FEMVector, the problem is that
         * they require the class to have a \p dim parameter, therefore we have
         * one here.
         */
        static constexpr unsigned dim = 1;
        
        /**
         * @brief Dummy type definition in order for the \p detail::Expression
         * defined operators to work.
         * 
         * In the file IpplOperations.h a bunch of operations are defined, we
         * want to be able to use them with the FEMVector, the problem is that
         * they require the class to define a \p value_type type. So therefore
         * we define it here.
         */
        using value_type = T;

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
         * @brief Set all the values of this vector to the values of the other
         * vector
         * 
         * Set the values of this vector to the values of \p expr
         * 
         * @param otherVector The expression from which to copy the values
         * 
         * @note Here we have to check how efficient this is, because in theory
         * we are copying a FEMVector onto the device when we are calling the
         * operator[] function inside of the Kokkos::parallel_for.
         */
        template <typename E, size_t N>
        void operator= (const detail::Expression<E, N>& expr);



        /**
         * @brief Subscript operator to get value at position \p i.
         * 
         * This function returns the value of the vector at index \p i, it is
         * equivalent to \p FEMVector::operator().
         * 
         * @param i The index off the value to retrieve.
         */
        KOKKOS_INLINE_FUNCTION T operator[] (size_t i) const;
        

        /**
         * @brief Subscript operator to get value at position \p i.
         * 
         * This function returns the value of the vector at index \p i, it is
         * equivalent to \p FEMVector::operator().
         * 
         * @param i The index off the value to retrieve.
         */
        KOKKOS_INLINE_FUNCTION T operator() (size_t i) const;


        /**
         * @brief Get underlying data view.
         * 
         * Returns a constant reference to the underlying data the FEMVector is
         * storing, this corresponds to a \c Kokkos::View living on the default
         * device.
         */
        const Kokkos::View<T*>& getView() const;


        /**
         * @brief Dummy function used such that the CG with periodic boundary
         * compiles.
         * 
         * @todo implement properly.
         * 
         * Currently this is a dummy function such that the CG code compiles.
         */
        T getVolumeAverage() const;
        

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
         * \p FEMVector::unpack().
         */
        struct Assign{
            KOKKOS_INLINE_FUNCTION void operator()(T& lhs, const T& rhs) const {
                lhs = rhs;
            }   
        };


        /**
         * @brief Struct for addition+assignment operator to be used with
         * \p FEMVector::unpack().
         */
        struct AssignAdd{
            KOKKOS_INLINE_FUNCTION void operator()(T& lhs, const T& rhs) const {
                lhs += rhs;
            }   
        };


    private:
        /**
         * @brief Structure holding MPI neighbor and boundary information.
         * 
         * This struct holds all the information regarding MPI (neighbor list, 
         * indecies which need to be exchanged...) and additionaly boundary
         * information, like what type of boundary we have. The \c FEMVector
         * class then has a pointer to an object of this, the reason behind is
         * that this allows us to copy a \c FEMVector onto the device cheaply,
         * without having to worry about copying much additional information.
         */
        struct BoundaryInfo {

            /**
             * @brief constructor for a \c BoundaryInfo object.
             * 
             * Constructor to be used to create an object of type
             * \c BoundaryInfo.
             * 
             * @param neighbors The ranks of the neighboring MPI tasks.
             * @param sendIdxs The indices for which the data should be send to
             * the MPI neighbors.
             * @param recvIdxs The halo cell indices. 
             */
            BoundaryInfo (std::vector<size_t> neighbors,
                std::vector< Kokkos::View<size_t*> > sendIdxs,
                std::vector< Kokkos::View<size_t*> > recvIdxs_m);

            
            /**
             * @brief Stores the ranks of the neighboring MPI tasks.
             * 
             * A vector storing the ranks of the neighboring MPI tasks. This is
             * used during halo operators in combination with
             * \p BoundaryInfo::sendIdxs_m and
             * \p BoundaryInfo::recvIdxs_m in order to send the
             * correct data to the correct ranks.
             */
            std::vector<size_t> neighbors_m;

            /**
             * @brief Stores indices of \p FEMVector::data_m which need to be
             * send to the MPI neighbors.
             * 
             * This is a 2D list which stores the indices of the
             * \p FEMVector::data_m variable which need to be send to the MPI
             * neighbors. The first dimension goes over all the neighbors and
             * should be used in combination with \p BounderyInfo::neighbors_m
             * while the second dimension goes over the actual indices.
             * 
             * This corresponds to the indices which belong to this rank but are
             * shared by the halos of the other ranks.
             */
            std::vector< Kokkos::View<size_t*> > sendIdxs_m;

            /**
             * @brief Stores indices of \p FEMVector::data_m which are part of
             * the halo.
             * 
             * This is a 2D list which stores the indices of the
             * \p FEMVector::data_m variable which are part of the halo. The
             * first dimension goes over all the
             * neighbors and should be used in combination with
             * \p BoundaryInfo::neighbors_m while the second dimension goes
             * over the actual indices.
             * 
             * It is called recvIdxs to be in line with
             * \c ippl::detail::HaloCells and because it stores the indices for
             * which we recive data from the neighbors.
             */
            std::vector< Kokkos::View<size_t*> > recvIdxs_m;


            /**
             * @brief Buffer for MPI communication.
             * 
             * This buffer is used during MPI communication in order to store
             * the values which should be send to the neighbors. We are using a
             * \c ippl::detail::FieldBufferData even though we do not have a
             * \c ippl::Field , but this buffer struct is general enough to
             * allow for such things.
             */
            detail::FieldBufferData<T> commBuffer_m;
        };
        
        /**
         * @brief Data this object is storing.
         * 
         * The data which the \c FEMVector is storing, it is represented by a
         * one dimensional \c Kokkos::View and lives on the default device.
         */
        Kokkos::View<T*> data_m;


        /**
         * @brief Struct holding all the MPI and boundary infromation.
         * 
         * Pointer to a struct holding all the infromation required for MPI
         * communication and general boundary information. The reason for it
         * beeing a pointer, is such that when this \c FEMVector object is
         * copied to device only a pointer and not all the data needs to be 
         * copied to device.
         */
        BoundaryInfo* boundaryInfo_m;

    };


    /**
     * @brief Calculate the inner product between two \c ippl::FEMVector(s).
     * 
     * Calculates the inner product \f$ a^T b\f$ between the
     * \c ippl::FEMVector(s) \p a and \p b. Note that during the
     * inner product computations the halo cells are included, if this should
     * not be the case the hallo cells should be set to 0 using the
     * \p ippl::FEMVector::clearHalo() function.
     * 
     * @param a First field.
     * @param b Second field.
     * 
     * @return The value \f$a^Tb\f$.
     */
    template <typename T>
    T innerProduct(const FEMVector<T>& a, const FEMVector<T>& b) {
        T localSum = 0;
        auto aView = a.getView();
        auto bView = b.getView();
        size_t n = aView.extent(0);
        Kokkos::parallel_reduce("FEMVector innerProduct", n,
            KOKKOS_LAMBDA(const size_t i, T& val){
                val += aView(i)*bView(i);
            },
            Kokkos::Sum<T>(localSum)
        );

        T globalSum = 0;
        ippl::Comm->allreduce(localSum, globalSum, 1, std::plus<T>());
        return globalSum;
    }
}   // namespace ippl


#include "FEMVector.hpp"

#endif  // IPPL_FEMVECTOR_H