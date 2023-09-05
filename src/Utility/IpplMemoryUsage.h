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
#ifndef IPPL_MEMPRYUSAGE_H
#define IPPL_MEMPRYUSAGE_H

#include "Ippl.h"

#include <memory>
#include <sys/resource.h>
#include <sys/time.h>  // not required but increases portability

class IpplMemoryUsage {
public:
    typedef IpplMemoryUsage* IpplMemory_p;
    typedef std::unique_ptr<IpplMemoryUsage> IpplMemory_t;

    enum Unit {
        BIT,  ///< Bit
        B,    ///< Byte
        KB,   ///< KiloByte
        KiB,  ///< KibiByte
        MB,   ///< MegaByte
        MiB,  ///< MebiByte
        GB,   ///< GigaByte
        GiB   ///< GebiByte
    };

public:
    /*!
     * Create / Get pointer to instance
     * @param unit of memory
     * @param reset the memory to zero. (see constructor documentation)
     */
    static IpplMemory_p getInstance(Unit unit = Unit::GB, bool reset = true);

    /*!
     * Get the memory of a specific core (only valid call for root core 0)
     * @param core we want memory of.
     * @returns the max. resident set size
     */
    double getMemoryUsage(int core) const;

    /*!
     * Collect the memory data of all cores.
     */
    void sample();

    /*!
     * @returns the unit string.
     */
    const std::string& getUnit() const;

private:
    /*!
     * Does nothing.
     */
    IpplMemoryUsage();

    /*!
     * Create an instance.
     * @param unit we want to have
     * @param reset the memory to zero. The value at construction time
     * is subtracted at every sampling.
     */
    IpplMemoryUsage(Unit unit = Unit::GB, bool reset = true);

    /*!
     * Obtain the memory consumption. It throws an exception if
     * an error ocurred.
     */
    void sample_m();

private:
    static IpplMemory_t instance_mp;               ///< *this
    std::unique_ptr<double[]> globalMemPerCore_m;  ///< memory of all cores
    std::string unit_m;                            ///< what's the unit of the memory
    double initial_memory_m;     ///< memory usage at construction time [GB] or [GiB]
    double max_rss_m;            ///< max. resident set size [GB] or [GiB]
    double conversion_factor_m;  ///< to various units. getrusage() returns in kB
    double who_m;                ///< RUSAGE_SELF, RUSAGE_CHILDREN (, RUSAGE_THREAD)
};

#endif
