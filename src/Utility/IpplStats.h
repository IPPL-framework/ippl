// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_STATS_H
#define IPPL_STATS_H

/***************************************************************************
 * IpplStats keeps statistics about a given IPPL job, and can report on
 * a summary of these statistics when asked.
 *
 * To add a new type of statistic, do these steps:
 *   1. Add a new private variable to accumulate the stats
 *   2. Add a new public method to call to add values to the stat
 *   3. In the constructor, initialize the stats object with a name in the
 *      colon-initializer list.
 *   4. Add commands to the rest of ippl to change the stat as needed.
 *      Example:
 *                 #ifndef IPPL_NO_STATS
 *                 Ippl::Stats.incMessageSent();
 *                 #endif
 *
 * This interface is extensible ... you can add new types of statistics
 * by calling 'addStat(description, initval)' with a string description
 * of the stat, and the initial value it should have.  This will return a
 * unique integer identifier for that stat.  Then, you can modify the stat
 * by calling incStat(statindex, amount) and decStat(statindex, amount).
 * At the end, then, this statistic will be printed out just like all the
 * others.
 ***************************************************************************/

// include files
#include "Utility/IpplInfo.h"
#include "Utility/Timer.h"
#include "Utility/Inform.h"

#include <vector>

class IpplStats {

public:
  // constructor: initialize statistics, and start run timer
  IpplStats();

  // destructor
  ~IpplStats();

  // print out the statistics to the given Inform object
  void print(Inform &);

  // add a statistics object to our list of stats ... return an integer
  // which is the index of the stat in our list, which can be used with
  // the 'incStat' and 'decStat' methods to change that statistic
  int addStat(const char *description, long initval = 0) {
    StatList.push_back(new StatData(description, initval, true));
    return (StatList.size() - 1);
  }

  // increment or decrement the statistic by the given value.  Return the
  // current value
  long incStat(int statindx, long val = 1) {
    StatList[statindx]->Value += val;
    return StatList[statindx]->Value;
  }
  long decStat(int statindx, long val = 1) {
    StatList[statindx]->Value -= val;
    return StatList[statindx]->Value;
  }

  //
  // general IPPL operation information
  //

  // return a ref to the timer, so that it can be turned on and off
  Timer &getTime() { return Time; }

  //
  // communication statistics operations
  //

  void incMessageSent() { ++MessagesSent.Value; }
  void incMessageSentToOthers() { ++MessagesSentToOthers.Value; }
  void incMessageSentToSelf() { ++MessagesSentToSelf.Value; }
  void incMessageReceived() { ++MessagesReceived.Value; }
  void incMessageReceivedFromNetwork() { ++MessagesReceivedFromNetwork.Value;}
  void incMessageReceivedFromQueue() { ++MessagesReceivedFromQueue.Value; }
  void incMessageReceiveChecks() { ++MessageReceiveChecks.Value; }
  void incMessageReceiveChecksFailed() { ++MessageReceiveChecksFailed.Value; }
  void incMessageBytesSent(long bytes) { BytesSent.Value += bytes; }
  void incMessageBytesReceived(long bytes) { BytesReceived.Value += bytes; }
  void incBarriers() { ++Barriers.Value; }
  void incReductions() { ++Reductions.Value; }
  void incScatters() { ++Scatters.Value; }

  //
  // BareField statistics operations
  //

  void incBareFields() { ++BareFields.Value; }
  void incLFields() { ++LFields.Value; }
  void incLFieldBytes(long bytes) { LFieldBytes.Value += bytes; }
  void incFieldLayouts() { ++FieldLayouts.Value; }
  void incRepartitions() { ++Repartitions.Value; }
  void incExpressions()            { ++Expressions.Value; }
  void incBFEqualsExpression()     { ++BFEqualsExpression.Value; }
  void incIBFEqualsExpression()    { ++IBFEqualsExpression.Value; }
  void incParensEqualsExpression() { ++ParensEqualsExpression.Value; }
  void incBFEqualsBF()             { ++BFEqualsBF.Value; }
  void incIBFEqualsIBF()           { ++IBFEqualsIBF.Value; }
  void incSubEqualsExpression()    { ++SubEqualsExpression.Value; }
  void incFFTs() { ++FFTs.Value; }
  void incGuardCellFills() { ++GuardCellFills.Value; }
  void incBoundaryConditions() { ++BoundaryConditions.Value; }
  void incCompresses() { ++Compresses.Value; }
  void incDecompresses() { ++Decompresses.Value; }
  void incCompressionCompares(long c) { CompressionCompares.Value += c; }
  void incCompressionCompareMax(long c) { CompressionCompareMax.Value += c; }
  void incBareFieldIterators() { ++BareFieldIterators.Value; }
  void incDefaultBareFieldIterators() { ++DefaultBareFieldIterators.Value; }
  void incBeginScalarCodes() { ++BeginScalarCodes.Value; }
  void incEndScalarCodes() { ++EndScalarCodes.Value; }
  //
  // Particle statistics operations
  //

  void incParticleAttribs() { ++ParticleAttribs.Value; }
  void incIpplParticleBases() { ++IpplParticleBases.Value; }
  void incParticleUpdates() { ++ParticleUpdates.Value; }
  void incParticleExpressions() { ++ParticleExpressions.Value; }
  void incParticleGathers() { ++ParticleGathers.Value; }
  void incParticleScatters() { ++ParticleScatters.Value; }
  void incParticlesCreated(long num) { ParticlesCreated.Value += num; }
  void incParticlesDestroyed(long num) { ParticlesDestroyed.Value += num; }
  void incParticlesSwapped(long num) { ParticlesSwapped.Value += num; }

private:
  // a simple object used to accumulate a stat, with a name.
  struct StatData {
    // constructor
    StatData(std::vector<StatData *> &datalist, const char *nm, long initval = 0,
	     bool needDelete = false)
      : Value(initval), Name(nm), NeedDelete(needDelete) {
      // add ourselves to the list of statistics objects
      datalist.push_back(this);
    }

    // another constructor, without the vector
    StatData(const char *nm, long initval = 0, bool needDelete = false)
      : Value(initval), Name(nm), NeedDelete(needDelete) { }

    // default constructor
    StatData() : Value(0), Name("") { }

    // destructor
    ~StatData() { }

    // value and name
    long   Value;
    std::string Name;

    // if this is true, we need to be deleted by the Stats class
    bool NeedDelete;
  };

  // a vector of statistics data objects, which will be used to print
  // out the results at the end.  All stats should be declared as StatData
  // variables below; in their constructor, they will put themselves in the
  // list of statistics objects.
  std::vector<StatData *> StatList;

  // a timer to time the whole program (although other timers may certainly be
  // created throughout the program)
  Timer Time;

  // build-in ippl statistics objects
  StatData MessagesSent;
  StatData MessagesSentToOthers;
  StatData MessagesSentToSelf;
  StatData MessagesReceived;
  StatData MessagesReceivedFromNetwork;
  StatData MessagesReceivedFromQueue;
  StatData MessageReceiveChecks;
  StatData MessageReceiveChecksFailed;
  StatData BytesSent;
  StatData BytesReceived;
  StatData Barriers;
  StatData Reductions;
  StatData Scatters;

  StatData BareFields;
  StatData LFields;
  StatData LFieldBytes;
  StatData FieldLayouts;
  StatData Repartitions;
  StatData Expressions;
  StatData BFEqualsExpression;
  StatData IBFEqualsExpression;
  StatData ParensEqualsExpression;
  StatData BFEqualsBF;
  StatData IBFEqualsIBF;
  StatData SubEqualsExpression;
  StatData FFTs;
  StatData GuardCellFills;
  StatData BoundaryConditions;
  StatData Compresses;
  StatData Decompresses;
  StatData CompressionCompares;
  StatData CompressionCompareMax;
  StatData BareFieldIterators;
  StatData DefaultBareFieldIterators;
  StatData BeginScalarCodes;
  StatData EndScalarCodes;

  StatData ParticleAttribs;
  StatData IpplParticleBases;
  StatData ParticleUpdates;
  StatData ParticleExpressions;
  StatData ParticleGathers;
  StatData ParticleScatters;
  StatData ParticlesCreated;
  StatData ParticlesDestroyed;
  StatData ParticlesSwapped;

};

// simple macros used to increment a stat, which is turned on or off
// by the IPPL_NO_STATS flag
#ifndef IPPL_NO_STATS
# define INCIPPLSTAT(stat)        Ippl::Stats->stat()
# define ADDIPPLSTAT(stat,amount) Ippl::Stats->stat(amount)
#else
# define INCIPPLSTAT(stat)
# define ADDIPPLSTAT(stat,amount)
#endif

#endif
  
/***************************************************************************
 * $RCSfile: IpplStats.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: IpplStats.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/

