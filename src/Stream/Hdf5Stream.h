#ifndef IPPL_HDF5_STREAM_H
#define IPPL_HDF5_STREAM_H

#include <typeinfo>

#include "H5Cpp.h"
#include "Stream/BaseStream.h"

namespace ippl {

    namespace hdf5 {
        H5::PredType get_hdf5_type(const std::type_info& tinfo) {
            if (typeid(int) == tinfo) {
                return H5::PredType::NATIVE_INT;
            } else if (typeid(double) == tinfo) {
                return H5::PredType::NATIVE_DOUBLE;
            } else if (typeid(float) == tinfo) {
                return H5::PredType::NATIVE_FLOAT;
            } else if (typeid(long double) == tinfo) {
                return H5::PredType::NATIVE_LDOUBLE;
            }
            return H5::PredType::NATIVE_CHAR;
        }
    }  // namespace hdf5

    template <class Object>
    class Hdf5Stream : public BaseStream<Object> {
    public:
        Hdf5Stream();

        virtual ~Hdf5Stream() = default;

        void create(const fs::path& path, const ParameterList& param) final;

        void open(const fs::path& path) final;

        void close() final;

    protected:
        H5::H5File h5file_m;
    };

    template <class Object>
    Hdf5Stream<Object>::Hdf5Stream()
        : BaseStream<Object>() {
        // Turn off auto-printing
        H5::Exception::dontPrint();
    }

    /* Create a new file using default properties. */
    template <class Object>
    void Hdf5Stream<Object>::create(const fs::path& path, const ParameterList& param) {
        BaseStream<Object>::create(path, param);

        std::string filename = path.filename().string();

        try {
            // not clear why the keyword "template" is needed here"
            bool overwrite = this->param_m.template get<bool>("overwrite");

            unsigned int flags = (overwrite) ? H5F_ACC_TRUNC : H5F_ACC_EXCL;

            h5file_m = H5::H5File(filename, flags, H5::FileCreatPropList::DEFAULT,
                                  H5::FileAccPropList::DEFAULT);
        } catch (...) {
            throw IpplException("Hdf5Stream::create", "Unable to create " + filename);
        }
    }

    template <class Object>
    void Hdf5Stream<Object>::open(const fs::path& path) {
        BaseStream<Object>::open(path);

        std::string filename = path.filename().string();

        try {
            h5file_m.openFile(filename.c_str(), H5F_ACC_RDWR, H5::FileAccPropList::DEFAULT);

        } catch (...) {
            throw IpplException("Hdf5Stream::open", "Unable to open " + filename);
        }
    }

    template <class Object>
    void Hdf5Stream<Object>::close() {
        try {
            h5file_m.close();
        } catch (...) {
            throw IpplException("Hdf5Stream::close",
                                "Unable to close " + this->path_m.filename().string());
        }
    }

}  // namespace ippl

#endif
