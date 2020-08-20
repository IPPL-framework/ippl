#ifndef STATIC_IPPL_INFO_H
#define STATIC_IPPL_INFO_H

#include <mpi.h>

class Communicate;
class IpplStats;
class Inform;

class StaticIpplInfo {
public:
    StaticIpplInfo();
    ~StaticIpplInfo();

    // Inform *Info;
    // Inform *Warn;
    // Inform *Error;
    // Inform *Debug;

    // the parallel communication object
    Communicate *Comm;

    // the statistics collection object
    IpplStats *Stats;

    Inform *Info;
    Inform *Warn;
    Inform *Error;
    Inform *Debug;

    // flag telling whether to use optimization for reducing
    // communication by deferring guard cell fills.
    bool deferGuardCellFills;

    // flag telling whether to turn off compression in the Field classes.
    bool noFieldCompression;

    // flag telling whether to try to (pseudo-)randomly offset the
    // LField blocks to try to avoid cache conflicts.
    bool offsetStorage;

    // flag telling whether to try to do a TryCompress after each
    // individual LField has been processed in an expression.
    bool extraCompressChecks;

    MPI_Comm communicator_m;

    // counter indicating how many IpplInit objects have been created.
    // When this gets back to zero, it's time to delete the Comm and quit.
    int NumCreated;

    // flag indicating whether this class has been created with
    // argc,argv specified ever.  This should only be done once.
    bool CommInitialized;

    // flag indicating whether we should print out stats info at the
    // end of the program.
    bool PrintStats;

    // flag indicating if we need to delete the comm object at the end.
    bool NeedDeleteComm;

    // flag indicating whether to use checksums on messages
    bool UseChecksums;

    // flag indicating whether to retransmit messages when errors occur
    bool Retransmit;

    // data with argc and argv
    int MyArgc;
    char **MyArgv;

    // data with my node number and total number of nodes.  These are
    // only changed when a new Communicate object is created.
    int MyNode;
    int TotalNodes;

    // data with SMP information.  These are changed after a new
    // Communicate object is created.
    int NumSMPs;
    int *SMPIDList;
    int *SMPNodeList;

    // data about a limit to the number of nodes that should be used
    // in FFT operations.  If this is <= 0 or > number of nodes, it is ignored.
    int MaxFFTNodes;

    // Maximum read chunk size
    int ChunkSize;

    // A boolean setting for whether we should attempt to use parallel
    // I/O within a single SMP, for example by having multipple processors
    // try to read from a single file (vs just having one node do it).
    bool PerSMPParallelIO;

};

#endif
