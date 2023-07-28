#ifndef CONNECTOR_H
#define CONNECTOR_H

#include "PICManager/PICManager.hpp"

/**
 * @namespace Connector
 * @brief Namespace for the Connector classes.
 */
namespace Connector {

    /**
     * @enum CONNECTION_TYPE
     * @brief Enum for the types of connections.
     */
    enum CONNECTION_TYPE {
        VTK,
        HDF5,
        CSV,
        INSITU
    };

    /**
     * @class Connector
     * @brief Base class for the Connector classes.
     * @tparam T The type of the data.
     * @tparam Dim The dimension of the data.
     */
    template <typename T, unsigned Dim = 3>
    class Connector {
        const char* testName_m;
        const CONNECTION_TYPE connType_m;
        std::map<int, std::unique_ptr<Inform> > streams_m;
        int streamIdCounter_m;

    public:
        Inform* msg;

        /**
         * @brief Constructor for the Connector class.
         * @param TestName The name of the test.
         * @param connType The type of the connection.
         */
        Connector(const char* TestName, const CONNECTION_TYPE connType)
            : testName_m(TestName)
            , connType_m(connType)
            , streamIdCounter_m(0)
            , msg(nullptr) {
            //
            msg = new Inform("Connector ");
        }

        /**
         * @brief Destructor for the Connector class.
         */
        ~Connector() {
            std::cout << "delete " << streamIdCounter_m << " streams" << std::endl;
            if (ippl::Comm->rank() == 0) {
                for (int i = 0; i < streamIdCounter_m; i++)
                    deleteStream(i);
            }
            if (msg)
                delete msg;
        }

        /**
         * @brief Open a connection.
         * @param fn The name of the file to connect to.
         * @return The status of the operation.
         */
        virtual int open(std::string fn) = 0;

        /**
         * @brief Close the connection.
         * @return The status of the operation.
         */
        virtual int close() = 0;

        /**
         * @brief Get the status of the connection.
         * @return The status of the connection.
         */
        virtual int status() = 0;

        /**
         * @brief Write data to the connection.
         * @return The status of the operation.
         */
        virtual int write() = 0;

        /**
         * @brief Read data from the connection.
         * @return The status of the operation.
         */
        virtual int read() = 0;

        /**
         * @brief Get the name of the test.
         * @return The name of the test.
         */
        const char* getName() { return testName_m; }

        /**
         * @brief Get the name of the connection type.
         * @return The name of the connection type.
         */
        const char* getConnTypeName() {
            switch (connType_m) {
                case CONNECTION_TYPE::VTK:
                    return "VTK";
                case CONNECTION_TYPE::HDF5:
                    return "HDF5 not yet avaidable";
                case CONNECTION_TYPE::CSV:
                    return "CSV";
                case CONNECTION_TYPE::INSITU:
                    return "INSITY not yet avaidable";
            };

            /* soon we can do this :)
               #include <iostream>
               #include <reflexpr>   (not yet in C++20
               return std::meta::get_base_name_v<
                  std::meta::get_element_m<
                  std::meta::get_enumerators_m<reflexpr(connType_m)>,
                  0>
               >;
            */
        }

        /**
         * @brief Request a stream from the pool.
         * @return Reference to the requested stream.
         */
        int requestStreamId(std::string fileName) {
            std::unique_ptr<Inform> outputStream(
                new Inform(NULL, fileName.c_str(), Inform::OVERWRITE, ippl::Comm->rank()));
            if (outputStream)
                std::cout << "Open " << outputStream << " with id " << streamIdCounter_m
                          << std::endl;
            int streamId = streamIdCounter_m;
            streamIdCounter_m++;
            streams_m[streamId] = std::move(outputStream);
            return streamId;
        }

        /**
         * @brief Write a message to the stream with the given ID.
         * @param streamId The ID of the stream to write to.
         * @param message The message to write.
         */
        void write(int streamId, const std::string& message) {
            if (ippl::Comm->rank() == 0) {
                auto it = streams_m.find(streamId);
                if (it != streams_m.end()) {
                    (*it->second) << message;
                }
            }
        }

        /**
         * @brief Delete a stream from the pool.
         * @param streamId The ID of the stream to delete.
         */
        void deleteStream(int streamId) {
            if (ippl::Comm->rank() == 0) {
                auto it = streams_m.find(streamId);
                if (it != streams_m.end()) {
                    streams_m.erase(it);
                }
            }
        }
    };

    // ... other classes ...

}  // namespace Connector
#endif
