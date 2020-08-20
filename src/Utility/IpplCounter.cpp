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
#include "Utility/IpplCounter.h"
#include "Utility/IpplInfo.h"
#include "Message/GlobalComm.h"
#include "PETE/IpplExpressions.h"

// forward references for counter routines
#ifdef SGI_HW_COUNTERS
extern "C" int start_counters( int e0, int e1);
extern "C" int read_counters( int e0, long long *c0, int e1, long long *c21);
#endif


//////////////////////////////////////////////////////////////////////
// constructor
IpplCounter::IpplCounter(const char *category)
    : totalcyc_m(0), totalinst_m(0), c0_m(0), c21_m(0),
#ifdef SGI_HW_COUNTERS
      e0_m(0), e1_m(21),
#endif
      gen_start_m(0), gen_read_m(0),
      category_m(category), msg_m("Counter")
{ }


//////////////////////////////////////////////////////////////////////
// destructor
IpplCounter::~IpplCounter()
{ }


//////////////////////////////////////////////////////////////////////
// start a mflops counter going
void IpplCounter::startCounter()
{
#ifdef SGI_HW_COUNTERS
  gen_start_m = start_counters(e0_m, e1_m);
#endif
}


//////////////////////////////////////////////////////////////////////
// stop a mflops counter, and accumulate results
void IpplCounter::stopCounter()
{
#ifdef SGI_HW_COUNTERS
  gen_read_m = read_counters(e0_m, &c0_m, e1_m, &c21_m);
#endif

  if(gen_read_m != gen_start_m) {
    msg_m << "Lost counters! MFLOPS Counters not working." << endl;
  } else {
    totalcyc_m += c0_m;
    totalinst_m += c21_m;
  }
}


//////////////////////////////////////////////////////////////////////
// print out mflops
void IpplCounter::printIt()
{
  double cpu_mhz = 195.0, totalmflops = 0.0, totalsofar = 0.0;
  double maxmflops = 0.0, minmflops = 0.0;
  double mflops = 0.0, runMflops = 0.0;

  if (c0_m != 0)
    mflops = double(c21_m) / double(c0_m) * cpu_mhz;

  if (totalcyc_m != 0)
    runMflops = double(totalinst_m) / double(totalcyc_m) * cpu_mhz;

  reduce(mflops, totalmflops, OpAddAssign());
  reduce(mflops, maxmflops, OpMaxAssign());
  reduce(mflops, minmflops, OpMinAssign());
  reduce(runMflops, totalsofar, OpAddAssign());

  msg_m << category_m << " MFLOPS -> total = " << totalmflops;
  msg_m << ", pernode = " << totalmflops/(double)(Ippl::getNodes());
  msg_m << ", min = " << minmflops << ", max = " << maxmflops;
  msg_m << ", duration = " << totalsofar << endl;
}


/***************************************************************************
 * $RCSfile: IpplCounter.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/
