// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_COUNTER_H
#define IPPL_COUNTER_H

/***************************************************************************
 * IpplCounter - a simple megaflops counter that accesses hardware counters
 * for measureing megaflop performance.
 *
 * To use these counters:
 *   1. Create a counter.
 *       example:   Counter FFTcounter("FFT");
 *
 *   2. Locate the function which you would like to measure and start the
 *       counter by placing the startCounter() method before it.  Then, stop
 *       the counter after it by using the stopCounter() method.
 *       example:   FFTcounter.startCounter();
 *                  fft->transform(....);
 *                  FFTcounter.stopCounter();
 *
 *   3. Use the printIt() method to print to the screen.
 *       example:   FFTcounter.printIt();
 *
 ***************************************************************************/

// include files
#include "Utility/Inform.h"

class IpplCounter
{
public:
  // constructor
  IpplCounter(const char *category);

  // destructor
  ~IpplCounter();

  // counter operations
  void startCounter();
  void stopCounter();
  void printIt();

private:
  typedef long long CounterLong;

  CounterLong totalcyc_m;
  CounterLong totalinst_m;
  CounterLong c0_m;
  CounterLong c21_m;

#ifdef SGI_HW_COUNTERS
  int e0_m;
  int e1_m;
#endif
  int gen_start_m;
  int gen_read_m;

  std::string category_m;
  Inform msg_m;
};

#endif

/***************************************************************************
 * $RCSfile: IpplCounter.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 ***************************************************************************/

/***************************************************************************
 * $RCSfile: addheaderfooter,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:17 $
 * IPPL_VERSION_ID: $Id: addheaderfooter,v 1.1.1.1 2003/01/23 07:40:17 adelmann Exp $
 ***************************************************************************/
