#ifndef IPPL_HDF5_STREAM_H
#define IPPL_HDF5_STREAM_H

#include "Stream/BaseStream.h"
#include "hdf5.h"

#include <type_traits>

namespace ippl {

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

    IPPL_HDF5_DATATYPE(char, H5T_NATIVE_CHAR);

    IPPL_HDF5_DATATYPE(int, H5T_NATIVE_INT);

    IPPL_HDF5_DATATYPE(double, H5T_NATIVE_DOUBLE);

    IPPL_HDF5_DATATYPE(float, H5T_NATIVE_FLOAT);

    IPPL_HDF5_DATATYPE(long double, H5T_NATIVE_LDOUBLE);

    template <class Object>
    class Hdf5Stream : public BaseStream<Object> {
    public:

        void create(const fs::path& path, bool overwrite = false) final;

        void open(const fs::path& path, char access) final;

        void close() final;

    protected:
        hid_t file; /* file identifier */
        herr_t status;
    };



    /* Create a new file using default properties. */
    template <class Object>
    void Hdf5Stream<Object>::create(const fs::path& path, bool overwrite) {

        BaseStream<Object>::create(path, overwrite);

        std::string filename = path.filename().string();

        if (overwrite) {
            file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        } else {
            file = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
        }

        if (file == H5I_INVALID_HID) {
            throw IpplException("Hdf5Stream::create", "Unable to create " + filename);
        }
    }

    template <class Object>
    void Hdf5Stream<Object>::open(const fs::path& path, char access) {

        BaseStream<Object>::open(path, access);

        std::string filename = path.filename().string();

        unsigned flags = H5F_ACC_RDWR;

        switch (access) {
            case 'r':
                flags = H5F_ACC_RDONLY;
                break;
            default:
                flags = H5F_ACC_RDWR;
        }

        file = H5Fopen(filename.c_str(), flags, H5P_DEFAULT);

        if (file == H5I_INVALID_HID) {
            throw IpplException("Hdf5Stream::open", "Unable to open " + filename);
        }
    }

    template <class Object>
    void Hdf5Stream<Object>::close() {
        /* Terminate access to the file. */
        status = H5Fclose(file);

        if (status < 0) {
            throw IpplException("Hdf5Stream::close", "Unable to close " + this->path_m.filename().string());
        }
    }

}  // namespace ippl

#endif
