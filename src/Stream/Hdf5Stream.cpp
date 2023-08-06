#ifndef IPPL_HDF5_STREAM_H
#define IPPL_HDF5_STREAM_H

#include "Stream/BaseStream.h"

namespace ippl {

    void open(std::string filename) {
        /* Create a new file using default properties. */
        file_id = H5Fcreate(filename´, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }

    void close() {
        /* Terminate access to the file. */
        status = H5Fclose(file_id);
    }

    void Hdf5Stream::operator<<(const ParticleBase& obj){};

    void Hdf5Stream::operator>>(ParticleBase& obj){};

    void Hdf5Stream::operator<<(const FieldContainer& obj){};

    void Hdf5Stream::operator>>(FieldContainer& obj){};
}  // namespace ippl

#endif
