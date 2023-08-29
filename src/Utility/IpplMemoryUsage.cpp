//
// Class IpplMemoryUsage
//   A simple singleton class which lets the user watch the memory consumption of a program.
//   ATTENTION: We use following memory convention
//
//              8 bit = 1 byte = 1e-3 kB = 1e-6 MB
//                             = 1e-3 / 1.024 KiB (KibiByte)
//                             = 1e-3 / 1.024 / 1.024 MiB (MebiByte)
//
//              instead of the usually taken but wrong relation 1024 kB = 1 MB.
//
//   General usage
//   1) create the instance using IpplMemoryUsage::getInstance(unit, reset).
//      The reset boolean indicates wether the memory at creation should be
//      subtracted from the measurements later on.
//      The class is based on t getrusage() that returns the memory
//      consumption in kB (KiloByte). You can specify the return value of
//      IpplMemoryUsage by the first argument.
//      Although one can use those input parameters they are only applied
//      at the first call. Additional calls with different input do NOT
//      modify the instance.
//   2) At any point in the program you can call IpplMemoryUsage::sample()
//      to collect the data.
//
//
#include "Utility/IpplMemoryUsage.h"

//////////////////////////////////////////////////////////////////////
IpplMemoryUsage::IpplMemoryUsage() {}

IpplMemoryUsage::IpplMemoryUsage(Unit unit, bool reset)
    : who_m(RUSAGE_SELF) {
    globalMemPerCore_m = std::unique_ptr<double[]>(new double[Comm->getNodes()]);

    switch (unit) {
        case Unit::BIT:
            conversion_factor_m = 8.0e3;
            unit_m              = "bit";
            break;
        case Unit::B:
            conversion_factor_m = 1.0e3;
            unit_m              = "B";
            break;
        case Unit::KB:
            conversion_factor_m = 1.0;
            unit_m              = "kB";
            break;
        case Unit::KiB:
            conversion_factor_m = 1.0 / 1.024;
            unit_m              = "KiB";
            break;
        case Unit::MB:
            conversion_factor_m = 1.0e-3;
            unit_m              = "MB";
            break;
        case Unit::MiB:
            conversion_factor_m = 1.0 / (1.024 * 1024.0);
            unit_m              = "MiB";
            break;
        case Unit::GiB:
            conversion_factor_m = 1.0 / (1.024 * 1024.0 * 1024.0);
            unit_m              = "GiB";
            break;
        case Unit::GB:
        default:
            conversion_factor_m = 1.0e-6;
            unit_m              = "GB";
            break;
    }

    initial_memory_m = 0.0;
    if (reset) {
        this->sample_m();
        initial_memory_m = max_rss_m;
    }
}

IpplMemoryUsage::IpplMemory_p IpplMemoryUsage::getInstance(Unit unit, bool reset) {
    static IpplMemory_t instance_mp = IpplMemory_t(new IpplMemoryUsage(unit, reset));
    return instance_mp.get();
}

double IpplMemoryUsage::getMemoryUsage(int core) const {
    return globalMemPerCore_m[core];
}

void IpplMemoryUsage::sample() {
    // update max_rss_m
    this->sample_m();

    for (int i = 0; i < Comm->getNodes(); i++)
        globalMemPerCore_m[i] = 0;

    double localMemPerCore = max_rss_m;

    gather(&localMemPerCore, &globalMemPerCore_m[0], 1);
}

const std::string& IpplMemoryUsage::getUnit() const {
    return unit_m;
}

void IpplMemoryUsage::sample_m() {
    rusage usage;
    if (getrusage(who_m, &usage) == -1)
        throw std::runtime_error("IpplMemoryUsage::sample_m(): Error in collecting memory!");

    max_rss_m = usage.ru_maxrss * conversion_factor_m - initial_memory_m;
}
