#ifndef IPPL_HDF5_STREAM_H
#define IPPL_HDF5_STREAM_H

#include "Stream/BaseStream.h"
#include "hdf5.h"

namespace ippl {

    namespace hdf5 {

        template <typename>
        struct is_hdf5_datatype : std::false_type {};

        template <typename T>
        hid_t get_hdf5_datatype(const T& /*x*/) {
            static_assert(is_hdf5_datatype<T>::value, "type isn't a HDF5 type");
            return get_hdf5_datatype(T());
        }

#define IPPL_HDF5_DATATYPE(CppType, H5Type)                   \
    template <>                                               \
    inline hid_t get_hdf5_datatype<CppType>(const CppType&) { \
        return H5Type;                                        \
    }                                                         \
                                                              \
    template <>                                               \
    struct is_hdf5_datatype<CppType> : std::true_type {};
    }  // namespace hdf5

    IPPL_HDF5_DATATYPE(char, H5T_NATIVE_CHAR);

    IPPL_HDF5_DATATYPE(int, H5T_NATIVE_INT);

    IPPL_HDF5_DATATYPE(double, H5T_NATIVE_DOUBLE);

    IPPL_HDF5_DATATYPE(float, H5T_NATIVE_FLOAT);

    IPPL_HDF5_DATATYPE(long double, H5T_NATIVE_LDOUBLE);

    class Hdf5Stream : public BaseStream<ParticleBase>, BaseStream<FieldContainer> {
    public:
        void open(std::string filename) override;

        void close() override;

        void operator<<(const ParticleBase& obj) override;

        void operator>>(ParticleBase& obj) override;

        void operator<<(const FieldContainer& obj) override;

        void operator>>(FieldContainer& obj) override;

    private:
        hid_t file_id; /* file identifier */
        herr_t status;
    };
}  // namespace ippl

#endif
