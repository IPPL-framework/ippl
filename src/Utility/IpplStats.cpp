// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 * This program was prepared by PSI. 
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "Utility/IpplStats.h"
#include "Utility/IpplInfo.h"
#include "Utility/Inform.h"
#include "Utility/Timer.h"


//////////////////////////////////////////////////////////////////////////
// constructor: Initialize all the ippl-specific statistics objects,
// and add them to our list of stats
IpplStats::IpplStats() : StatList(), Time(),
  MessagesSent(StatList, "Messages sent"),
  MessagesSentToOthers(StatList, "Messages sent to other nodes"),
  MessagesSentToSelf(StatList, "Messages sent to our own node"),
  MessagesReceived(StatList, "Messages received"),
  MessagesReceivedFromNetwork(StatList, "Messages received from network"),
  MessagesReceivedFromQueue(StatList, "Messages received from queue"),
  MessageReceiveChecks(StatList, "Message receive polls"),
  MessageReceiveChecksFailed(StatList, "Message receive polls which failed"),
  BytesSent(StatList, "Message total bytes sent"),
  BytesReceived(StatList, "Message total bytes received"),
  Barriers(StatList, "Barriers performed"),
  Reductions(StatList, "General reductions performed"),
  Scatters(StatList, "General scatters performed"),

  BareFields(StatList, "BareField objects created"),
  LFields(StatList, "LField objects created"),
  LFieldBytes(StatList, "LField bytes of storage allocated"),
  FieldLayouts(StatList, "FieldLayout objects created"),
  Repartitions(StatList, "BareField objects repartitioned"),
  Expressions(StatList, "BareField expressions evaluated"),
  BFEqualsExpression(StatList, "BF=Expression expressions evaluated"),
  IBFEqualsExpression(StatList, "IBF=Expression expressions evaluated"),
  ParensEqualsExpression(StatList, "Parens=Expression expressions evaluated"),
  BFEqualsBF(StatList, "General BF=BF expressions evaluated"),
  IBFEqualsIBF(StatList, "General IBF=IBF expressions evaluated"),
  SubEqualsExpression(StatList, "SubField=Expression expressions evaluated"),
  FFTs(StatList, "FFTs performed"),
  GuardCellFills(StatList, "Number of times guard cells were filled"),
  BoundaryConditions(StatList, "Number of times boundary conditions applied"),
  Compresses(StatList, "Number of LFields compressed"),
  Decompresses(StatList, "Number of LFields decompressed"),
  CompressionCompares(StatList, "Number of compression comparisons"),
  CompressionCompareMax(StatList, "Maximum possible compression comparisons"),
  BareFieldIterators(StatList, "BareField Iterators created"),
  DefaultBareFieldIterators(StatList, "Default BareField Iterators created"),
  BeginScalarCodes(StatList, "Number of scalar code section initializations"),
  EndScalarCodes(StatList, "Number of scalar code section finalizes"),

  ParticleAttribs(StatList, "ParticleAttrib objects created"),
  IpplParticleBases(StatList, "IpplParticleBase objects created"),
  ParticleUpdates(StatList, "Particle object updates"),
  ParticleExpressions(StatList, "Particle expressions evaluted"),
  ParticleGathers(StatList, "Particle/Field gather operations"),
  ParticleScatters(StatList, "Particle/Field scatter operations"),
  ParticlesCreated(StatList, "Particles created"),
  ParticlesDestroyed(StatList, "Particles destroyed"),
  ParticlesSwapped(StatList, "Particles swapped to another node")

{
  // start the timer
  Time.stop();
  Time.clear();
  Time.start();
}


//////////////////////////////////////////////////////////////////////////
// destructor: delete all the necessary StatData's
IpplStats::~IpplStats() {
  for (unsigned int i=0; i < StatList.size(); ++i) {
    if (StatList[i]->NeedDelete)
      delete (StatList[i]);
  }
}


//////////////////////////////////////////////////////////////////////////
// print out the statistics to the given Inform object
void IpplStats::print(Inform &o) {

  // if we have no stats, just return
  if (StatList.size() == 0)
    return;

  // for each statistic, print out the description, a set of ...'s, and
  // the stat, right-justified to 10 places
  o << "Runtime statistics summary:" << endl;
  for (unsigned int i=0; i < StatList.size(); ++i) {
    o << StatList[i]->Name << " ";
    int numperiods = 48 - strlen(StatList[i]->Name.c_str());
    if (numperiods < 2)
      numperiods = 2;
    for (int j=0; j < numperiods; ++j)
      o << ".";
    o << " " << std::setw(12) << StatList[i]->Value << endl;
  }
}


/***************************************************************************
 * $RCSfile: IpplStats.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: IpplStats.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/

