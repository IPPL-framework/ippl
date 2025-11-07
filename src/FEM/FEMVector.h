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
     * This class represents a 1D vector which stores elements of type \p T and
     * provides functionalities to handle halo cells and their exchanges.
     * It can conceptually be though of being a mathemtical vector, and to this
     * extend is used during fem to represent the vectors \f$x\f$, \f$b\f$ when
     * solving the linear system \f$Ax = b\f$.
     * 
     * We use this instead of an \c ippl::Field, because for basis
     * functions which have DoFs at non-vertex positions a representation as a
     * field is not easily possible.
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
         * want to be able to use them with a \c FEMVector, the problem is that
         * they require the class to have a \p dim parameter, therefore we have
         * one here.
         */
        static constexpr unsigned dim = 1;
        
        /**
         * @brief Dummy type definition in order for the \p detail::Expression
         * defined operators to work.
         * 
         * In the file IpplOperations.h a bunch of operations are defined, we
         * want to be able to use them with a \c FEMVector, the problem is that
         * they require the class to define a \p value_type type. So therefore
         * we define it here.
         */
        using value_type = T;

        /**
         * @brief Constructor taking size, neighbors, and halo exchange indices.
         * 
         * Constructor of a FEMVector taking in the size and information about
         * the neighboring MPI ranks. 
         * 
         * @param n The size of the vector.
         * @param neighbors The ranks of the neighboring MPI tasks.
         * @param sendIdxs The indices for which the data should be sent to the
         * MPI neighbors.
         * @param recvIdxs The halo cell indices.
         */
        FEMVector(size_t n, std::vector<size_t> neighbors,
            std::vector< Kokkos::View<size_t*> > sendIdxs,
            std::vector< Kokkos::View<size_t*> > recvIdxs);
        
        
        /**
         * @brief Copy constructor (shallow).
         * 
         * Creates a shallow copy of the other vector, by copying the underlying
         * \c Kokkos::View and boundary infromation.
         * 
         * @param other The other vector we are copying from.
         */
        KOKKOS_FUNCTION FEMVector(const FEMVector<T>& other);

        
        /**
         * @brief Constructor only taking size, does not create any MPI/boundary
         * information.
         * 
         * This constructor only takes the size of the vector and allocates the
         * appropriate number of elements, it does not sotre any MPI
         * communication or boundary infromation. This constructor is useful
         * if only a simply vector for arithmetic is needed without the need
         * for any communication.
         */
        FEMVector(size_t n);
        

        /**
         * @brief Copy values from neighboring ranks into local halo.
         * 
         * This function takes the local values which are part of other ranks
         * halos and copies them to the corresponding halo cells of those
         * neighbors.
         */
        void fillHalo();

        /**
         * @brief Accumulate halo values in neighbor.
         * 
         * This function takes the local halo values (which are part of another
         * ranks values) and sums them to these corresponding values of the
         * neighbor ranks.
         */
        void accumulateHalo();

        /**
         * @brief Set the halo cells to \p setValue.
         * 
         * @param setValue The value to which the halo cells should be set.
         */
        void setHalo(T setValue);


        /**
         * @brief Set all the values of the vector to \p value.
         * 
         * Sets all the values in the vector to \p value this also includes the
         * halo cells.
         * 
         * @param value The value to which the entries should be set.
         */
        FEMVector<T>& operator= (T value);


        /**
         * @brief Set all the values of this vector to the values of the
         * expression.
         * 
         * Set the values of this vector to the values of \p expr
         * 
         * @param expr The expression from which to copy the values
         * 
         * @note Here we have to check how efficient this is, because in theory
         * we are copying a FEMVector onto the device when we are calling the
         * operator[] function inside of the Kokkos::parallel_for.
         */
        template <typename E, size_t N>
        FEMVector<T>& operator= (const detail::Expression<E, N>& expr);

        
        /**
         * @brief Copy the values from another \c FEMVector to this one.
         * 
         * Sets the element values of this vector to the ones of \p v, only the
         * values are set everything else (MPI config, boundaries) are ignored.
         * 
         * @param v The other vector to copy values from.
        */
        FEMVector<T>& operator= (const FEMVector<T>& v);



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
         * @brief Get the size (number of elements) of the vector.
         */
        size_t size() const;


        /**
         * @brief Create a deep copy, where all the information of this vector
         * is copied to a new one.
         * 
         * Returns a \c FEMVector which is an exact copy of this one. All the
         * information (elements, MPI information, etc.) are explicitly copied
         * (i.e. deep copy).
         * 
         * @returns An exact copy of this vector
        */
        FEMVector<T> deepCopy() const;
        

        /**
         * @brief Create a new \c FEMVector with different data type, but same
         * size and boundary infromation.
         * 
         * This function is used to create a new \c FEMVector with same size and
         * boundary infromation, but of different data type. The boundary 
         * information is copied over via a deep copy fashion.
         * 
         * @tparam K The data type of the new vector.
         * 
         * @returns A vector of same structure but new data type.
         */
        template <typename K>
        FEMVector<K> skeletonCopy() const;
        

        /**
         * @brief Pack data into \p BoundaryInfo::commBuffer_m for
         * MPI communication.
         * 
         * This function takes data from the vector accoding to \p idxStore and
         * stores it inside of \p BoundaryInfo::commBuffer_m.
         * 
         * @param idxStore A 1D Kokkos view which stores the the indices for
         * \p FEMVector::data_m which we want to send.
         */
        void pack(const Kokkos::View<size_t*>& idxStore);
        
        
        /**
         * @brief Unpack data from \p BoundaryInfo::commBuffer_m into
         * \p FEMVector::data_m after communication.
         * 
         * This function takes data from \p BoundaryInfo::commBuffer_m and stores
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
             * @param sendIdxs The indices for which the data should be sent to
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
             * \p FEMVector::data_m variable which need to be sent to the MPI
             * neighbors. The first dimension goes over all the neighbors and
             * should be used in combination with \p BoundaryInfo::neighbors_m
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
         * @brief Struct holding all the MPI and boundary information.
         * 
         * Pointer to a struct holding all the information required for MPI
         * communication and general boundary information. The reason for it
         * beeing a pointer, is such that when this \c FEMVector object is
         * copied to device only a pointer and not all the data needs to be 
         * copied to device.
         */
        std::shared_ptr<BoundaryInfo> boundaryInfo_m;

    };


    /**
     * @brief Calculate the inner product between two \c ippl::FEMVector(s).
     * 
     * Calculates the inner product \f$ a^T b\f$ between the
     * \c ippl::FEMVector(s) \p a and \p b. Note that during the
     * inner product computations the halo cells are included, if this should
     * not be the case the hallo cells should be set to 0 using the
     * \p ippl::FEMVector::setHalo() function.
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

    template <typename T>
    T norm(const FEMVector<T>& v, int p = 2) {

        T local = 0;
        auto view = v.getView();
        size_t n = view.extent(0);
        switch (p) {
            case 0: {
                Kokkos::parallel_reduce("FEMVector l0 norm", n,
                    KOKKOS_LAMBDA(const size_t i, T& val) {
                        val = Kokkos::max(val, Kokkos::abs(view(i)));
                    },
                    Kokkos::Max<T>(local)
                );
                T globalMax = 0;
                ippl::Comm->allreduce(local, globalMax, 1, std::greater<T>());
                return globalMax;
            }
            default: {
                Kokkos::parallel_reduce("FEMVector lp norm", n,
                    KOKKOS_LAMBDA(const size_t i, T& val) {
                        val += std::pow(Kokkos::abs(view(i)), p);
                    },
                    Kokkos::Sum<T>(local)
                );
                T globalSum = 0;
                ippl::Comm->allreduce(local, globalSum, 1, std::plus<T>());
                return std::pow(globalSum, 1.0 / p);
            }
        }
    }
}   // namespace ippl


#include "FEMVector.hpp"

#endif  // IPPL_FEMVECTOR_H
