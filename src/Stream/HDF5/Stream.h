#ifndef IPPL_HDF5_STREAM_H
#define IPPL_HDF5_STREAM_H

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "H5Cpp.h"
#include "Stream/BasicFileStream.h"
#include "Stream/BasicStreams.h"
#include "Stream/Format/Format.h"

namespace ippl {

    namespace hdf5 {

        namespace core {
            const std::unordered_map<std::type_index, H5::DataType> predefined_types = {
                {std::type_index(typeid(char)), H5::PredType::NATIVE_CHAR},
                {std::type_index(typeid(int)), H5::PredType::NATIVE_INT},
                {std::type_index(typeid(double)), H5::PredType::NATIVE_DOUBLE},
                {std::type_index(typeid(float)), H5::PredType::NATIVE_FLOAT},
                {std::type_index(typeid(long double)), H5::PredType::NATIVE_LDOUBLE},
                {std::type_index(typeid(std::string)), H5::StrType(0, H5T_VARIABLE)}};

            H5::DataType get_hdf5_type(const std::type_index& tindex) {
                H5::DataType type = H5::PredType::NATIVE_CHAR;
                try {
                    type = predefined_types.at(tindex);
                } catch (...) {
                    throw IpplException("ippl::hdf5::get_hdf5_type", "No such H5 type available.");
                }
                return type;
            }

            template <typename T>
            H5::DataType get_hdf5_type(const T& /*value*/) {
                return get_hdf5_type(std::type_index(typeid(T)));
            }
        }  // namespace core

        template <class Object>
        class Stream : public BasicFileStream, public basic_iostream<Object> {
        public:
            Stream() = delete;

            Stream(std::unique_ptr<Format> format);

            virtual ~Stream() = default;

            void create(const fs::path& path, const ParameterList& param) final;

            void open(const fs::path& path) final;

            void close() final;

            template <typename T>
            void operator<<(const std::pair<std::string, T>& pair);

        protected:
            H5::H5File h5file_m;
            std::unique_ptr<Format> format_m;
        };

        template <class Object>
        Stream<Object>::Stream(std::unique_ptr<Format> format)
            : BasicFileStream()
            , basic_iostream<Object>()
            , format_m(std::move(format)) {
            // Turn off auto-printing
            H5::Exception::dontPrint();
        }

        /* Create a new file using default properties. */
        template <class Object>
        void Stream<Object>::create(const fs::path& path, const ParameterList& param) {
            BasicFileStream::create(path, param);

            std::string filename = path.filename().string();

            try {
                // not clear why the keyword "template" is needed here"
                bool overwrite = this->param_m.template get<bool>("overwrite");

                unsigned int flags = (overwrite) ? H5F_ACC_TRUNC : H5F_ACC_EXCL;

                h5file_m = H5::H5File(filename, flags, H5::FileCreatPropList::DEFAULT,
                                      H5::FileAccPropList::DEFAULT);
            } catch (...) {
                throw IpplException("hdf5::Stream::create", "Unable to create " + filename);
            }

            this->open(path);

            ParameterList pp;
            format_m->header(&pp);
            *this << pp;

            this->close();
        }

        template <class Object>
        void Stream<Object>::open(const fs::path& path) {
            BasicFileStream::open(path);

            std::string filename = path.filename().string();

            try {
                h5file_m.openFile(filename.c_str(), H5F_ACC_RDWR, H5::FileAccPropList::DEFAULT);

            } catch (...) {
                throw IpplException("hdf5::Stream::open", "Unable to open " + filename);
            }
        }

        template <class Object>
        void Stream<Object>::close() {
            try {
                h5file_m.close();
            } catch (...) {
                throw IpplException("hdf5::Stream::close",
                                    "Unable to close " + this->path_m.filename().string());
            }
        }

        template <class Object>
        template <typename T>
        void Stream<Object>::operator<<(const std::pair<std::string, T>& pair) {
            try {
                const std::string& key = pair.first;
                const T& value         = pair.second;

                H5::DataSpace dspace(H5S_SCALAR);  // FIXME We might also write arrays etc.
                std::cout << "key " << key << std::endl;
                H5::DataType type = core::get_hdf5_type(value);

                H5::Attribute attr = this->h5file_m.createAttribute(key, type, dspace);

                attr.write(type, &value);

                attr.close();

            } catch (...) {
                throw IpplException("hdf5::Stream::operator<<", "Unable to write attribute");
            }
        }
    }  // namespace hdf5
}  // namespace ippl

#endif
