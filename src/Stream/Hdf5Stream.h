#ifndef IPPL_HDF5_STREAM_H
#define IPPL_HDF5_STREAM_H

#include <type_traits>

#include "Stream/BaseStream.h"
#include "hdf5.h"

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
        Hdf5Stream();

        virtual ~Hdf5Stream() = default;

        void create(const fs::path& path, const ParameterList& param) final;

        void open(const fs::path& path) final;

        void close() final;

    protected:
        hid_t file; /* file identifier */
        herr_t status;
    };

    template <class Object>
    Hdf5Stream<Object>::Hdf5Stream()
        : BaseStream<Object>() {}

    /* Create a new file using default properties. */
    template <class Object>
    void Hdf5Stream<Object>::create(const fs::path& path, const ParameterList& param) {
        BaseStream<Object>::create(path, param);

        std::string filename = path.filename().string();

        // not clear why the keyword "template" is needed here"
        bool overwrite = this->param_m.template get<bool>("overwrite");

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
    void Hdf5Stream<Object>::open(const fs::path& path) {
        BaseStream<Object>::open(path);

        std::string filename = path.filename().string();

        file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

        if (file == H5I_INVALID_HID) {
            throw IpplException("Hdf5Stream::open", "Unable to open " + filename);
        }
    }

    template <class Object>
    void Hdf5Stream<Object>::close() {
        /* Terminate access to the file. */
        status = H5Fclose(file);

        if (status < 0) {
            throw IpplException("Hdf5Stream::close",
                                "Unable to close " + this->path_m.filename().string());
        }
    }

}  // namespace ippl

#endif
