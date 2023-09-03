#include "Ippl.h"

#include <iostream>

#include "Utility/Configuration.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        try {
            ippl::Configuration config;

            // default:
            config.add<int>("nx", 0);
            config.add<int>("ny", 0);
            config.add<int>("nz", 0);
            config.add<bool>("verbose", false);

            config.parse(argv[1]);

            std::cout << "Configuration:" << std::endl;
            std::cout << config << std::endl;

            std::cout << "in code:" << std::endl;
            int nx = config.get<int>("nx");
            std::cout << nx << std::endl;

        } catch (const IpplException& e) {
            std::cout << e.what() << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
