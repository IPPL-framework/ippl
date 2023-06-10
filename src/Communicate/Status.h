//
// Class Status
//   A communication status handle for non-blocking communication.
//
//
// Copyright (c) 2023, Matthias Frey, University of St Andrews, UK
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_MPI_STATUS_H
#define IPPL_MPI_STATUS_H

#include <mpi.h>
#include <optional>

namespace ippl {
    namespace mpi {

        class Status {
        public:
            Status()
                : status_m()
                , count_m(-1){};

            Status(const Status&) = default;

            Status& operator=(Status& other) = default;

            int source() const noexcept { return status_m.MPI_SOURCE; }

            int tag() const noexcept { return status_m.MPI_TAG; }

            int error() const noexcept { return status_m.MPI_ERROR; }

            template <typename T>
            std::optional<int> count();

            // https://en.cppreference.com/w/cpp/language/cast_operator
            operator MPI_Status*() noexcept { return &status_m; }

            operator const MPI_Status*() const noexcept { return &status_m; }

        private:
            MPI_Status status_m;
            int count_m;
        };

        template <typename T>
        std::optional<int> Status::count() {
            if (count_m != -1) {
                return count_m;
            }

            int count             = MPI_UNDEFINED;
            MPI_Datatype datatype = get_mpi_datatype<T>(T());
            MPI_Get_count(&status_m, datatype, &count);

            if (count == MPI_UNDEFINED) {
                return std::optional<int>();
            }
            count_m = count;
            return count_m;
        }
    }  // namespace mpi
}  // namespace ippl

#endif
