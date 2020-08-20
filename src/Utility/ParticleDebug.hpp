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
//-----------------------------------------------------------------------------
// The IPPL Framework - Visit http://people.web.psi.ch/adelmann/ for more details
//
// This program was prepared by the Regents of the University of California at
// ParticleDebug.cpp , Tim Williams 8/6/1998
// Helper functions to print out (formatted ASCII) ParticleAttrib elements.
// Intended mainly for use from within a debugger, called interactively, but
// also callable as template functions from source code. To call from many
// debuggers, the user has to provide nontemplate wrapper functions, as
// described in ParticleDebugFunctions.cpp.

// include files
#include "Utility/ParticleDebug.h"
#include "Utility/Inform.h"

#include "Particle/ParticleAttrib.h"

#include <iostream>
#include <iomanip> // need format fcns setf() and setprecision() from here


//----------------------------------------------------------------------
// Print a ParticleAttrib
//----------------------------------------------------------------------
template<class T>
void pap(ParticleAttrib<T>& pattr, bool docomm) {

  // Set Inform ptr if not set:
  if (!PtclDbgInformIsSet) {
    if (!FldDbgInformIsSet) {
      PtclDbgInform = new Inform(NULL, INFORM_ALL_NODES);
    }
    else {
      setPtclDbgInform(*FldDbgInform);
    }
  }

  if (docomm) {
    int mype = IpplInfo::myNode();
    int npes = IpplInfo::getNodes();
    int myNumPtcles = pattr.size();
    int numPtcles = pattr.size();
    int tag = Ippl::Comm->next_tag(IPPL_APP_TAG0, IPPL_APP_CYCLE);
    int tag2 = Ippl::Comm->next_tag(IPPL_APP_TAG0, IPPL_APP_CYCLE);
    Message *msg, *msg2;
    if (mype == 0) {
      int otherNumPtcles = 0;
      for (int pe = 1; pe < npes; pe++) {
        msg = IpplInfo::Comm->receive_block(pe, tag);
        msg->get(otherNumPtcles);
        delete msg;
        numPtcles += otherNumPtcles;
      }
      msg2 = new Message;
      msg2->put(numPtcles);
      IpplInfo::Comm->broadcast_others(msg2, tag2);
    }
    else {
      msg = new Message;
      msg->put(myNumPtcles);
      IpplInfo::Comm->send(msg, 0, tag);
      int pe0 = 0;
      msg2 = IpplInfo::Comm->receive_block(pe0, tag2);
      msg2->get(numPtcles);
      delete msg2;
    }
    IpplInfo::Comm->barrier();
    spap(pattr, 0, numPtcles - 1, 1, docomm);

  }
  else {

    spap(pattr, 0, pattr.size() - 1, 1, docomm);

  }
}

//----------------------------------------------------------------------
// Print a single element of a ParticleAttrib
//----------------------------------------------------------------------
template<class T>
void epap(ParticleAttrib<T>& pattr, int i, bool docomm) {


  // Set Inform ptr if not set:
  if (!PtclDbgInformIsSet) {
    if (!FldDbgInformIsSet) {
      PtclDbgInform = new Inform(NULL, INFORM_ALL_NODES);
    }
    else {
      setPtclDbgInform(*FldDbgInform);
    }
  }
  spap(pattr, i, i, 1, docomm);
}

//----------------------------------------------------------------------
// Print a strided subrange of a ParticleAttrib
//----------------------------------------------------------------------
template<class T>
void spap(ParticleAttrib<T>& pattr,
          int ibase, int ibound, int istride, bool docomm) {



  // Set Inform ptr if not set:
  if (!PtclDbgInformIsSet) {
    if (!FldDbgInformIsSet) {
      PtclDbgInform = new Inform(NULL, INFORM_ALL_NODES);
    }
    else {
      setPtclDbgInform(*FldDbgInform);
    }
  }

  // Check input parameters for errors and unimplemented values:
  bool okParameters = true;
  if (ibase < -1) {
    (*PtclDbgInform) << "spap() error: ibase (= " << ibase
                    << ") < lowest index value (= " << 0 << ")" << endl;
    okParameters = false;
  }
  //tjw??? Can't check if i greater than total num ptcles, because this number
  //isn't available in ParticleAttrib
  if (istride < 0) {
    (*PtclDbgInform) << "spap() error: istride < 0 not implemented yet."
                     << endl;
    okParameters = false;
  }
  else {
    if ((ibound < ibase) && !((ibase == 0) && (ibound == -1))) {
      (*PtclDbgInform) << "spap() error: ibase (= " << ibase
                       << ") > ibound (=  " << ibound
                       << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (istride == 0) {
    if (((ibound - ibase) != 0) && !((ibase == 0) && (ibound == -1))) {
      (*PtclDbgInform) << "spap() error: istride = 0 but (ibound - ibase) = "
                       << (ibound - ibase) << endl;
      okParameters = false;
    }
    else {
      istride = 1; // Allow specifying stride 0 for 1-element range; set=1
    }
  }

  if (!okParameters) return; // Exit if problem with input parameters

  if (docomm) {

    // With communication; assume a GLOBAL particle index range. Find which PEs
    // own parts of it and have those PEs print out their values:
    int myNumPtcles = pattr.size();
    int npes = IpplInfo::getNodes();
    int* numsPtcles = new int[npes];
    int mype = IpplInfo::myNode();
    for (int pe=0; pe<npes; pe++) {
      numsPtcles[pe] = 0;
      if (pe == mype) numsPtcles[pe] = myNumPtcles;
    }
    int tag = Ippl::Comm->next_tag(IPPL_APP_TAG0, IPPL_APP_CYCLE);
    int tag2 = Ippl::Comm->next_tag(IPPL_APP_TAG0, IPPL_APP_CYCLE);
    Message *msg, *msg2;
    if (mype == 0) {
      int otherNumPtcles = 0;
      for (int pe=1; pe<npes; pe++) {
        msg = IpplInfo::Comm->receive_block(pe, tag);
        msg->get(otherNumPtcles);
        delete msg;
        numsPtcles[pe] = numsPtcles[pe - 1] + otherNumPtcles;
      }
      msg2 = new Message;
      msg2->putmsg((void *)numsPtcles, sizeof(int), npes);
      IpplInfo::Comm->broadcast_others(msg2, tag2);
    }
    else {
      msg = new Message;
      msg->put(myNumPtcles);
      IpplInfo::Comm->send(msg, 0, tag);
      int pe0 = 0;
      msg2 = IpplInfo::Comm->receive_block(pe0, tag2);
      msg2->getmsg(numsPtcles);
      delete msg2;
    }
    // Find out if I (pe) own part of the stated global particle index range:
    int myPtcleIndexBegin, myPtcleIndexEnd;
    if (mype == 0) {
      myPtcleIndexBegin = 0;
      myPtcleIndexEnd = myNumPtcles - 1;
    }
    else {
      myPtcleIndexBegin = numsPtcles[mype - 1];
      myPtcleIndexEnd = myPtcleIndexBegin + myNumPtcles - 1;
    }
    // Construct Index objects for convenience of using Index::touches, etc:
    Index myRange(myPtcleIndexBegin, myPtcleIndexEnd, 1);
    Index requestedRange(ibase, ibound, istride);
    for (int pe=0; pe < npes; pe++) {
      if (mype == pe) {
        if (myNumPtcles > 0) {
          if (myRange.touches(requestedRange)) {
            Index myRequestedRange = requestedRange.intersect(myRange);
            int mybase = myRequestedRange.first();
            int mybound = myRequestedRange.last();
            *PtclDbgInform << "....PE = " << mype
                          << " GLOBAL ptcle index subrange (" << mybase
                          << " : " << mybound << " : " << istride
                          << ")...." << endl;
            for (int p = mybase; p <= mybound; p += istride*elementsPerLine) {
              for (int item = 0; ((item < elementsPerLine) &&
                                  ((p+item*istride) <= mybound)); item++) {
//                                (item < mylength)); item++) {
                *PtclDbgInform << std::setprecision(digitsPastDecimal)
                               << std::setw(widthOfElements)
                               << pattr[p + item*istride] << " ";
              }

              *PtclDbgInform << endl;
            }
          }
        }
        else {
          //don't         *PtclDbgInform << "....PE = " << mype
          //don't			<< " has no particles ...." << endl;
        }
      }
      IpplInfo::Comm->barrier();
    }
    if (mype == 0) *PtclDbgInform << endl;
    delete [] numsPtcles;
  }
  else {

    // No communication; assume calling pe(s) print data for their particle
    // data values having LOCAL index range (ibase,ibound,istride):
    int mype = IpplInfo::myNode();
    int myNumPtcles = pattr.size();
    if (PtclDbgInform->getPrintNode() != INFORM_ALL_NODES) {
      WARNMSG(endl << "spap(): Currently, if docomm=false you must specify "
              << "an Inform object having INFORM_ALL_NODES as its "
              << "printing-node specifier if you want to see output from "
              << "any processor calling [e,s]pap(); the Inform object "
              << "you're trying to use has "
              << PtclDbgInform->getPrintNode() << " specified. "
              << "N.B.: If you called setInform() and didn't also call "
              << "setPtclDbgInform() you are getting the FldDbgInform object "
              << "you set with setInform, which you may not have constructed "
              << "with INFORM_ALL_NODES." << endl << endl);
    }

    if (myNumPtcles > 0) {
      *PtclDbgInform << "....PE = " << mype
                    << " LOCAL ptcle index range (" << ibase
                    << " : " << ibound << " : " << istride << ")...." << endl;
      int length = (ibound - ibase)/istride + 1;
      for (int p = ibase; p <= ibound; p += istride*elementsPerLine) {
        for (int item = 0; ((item < elementsPerLine) &&
                            (item < length)); item++) {
          *PtclDbgInform << std::setprecision(digitsPastDecimal)
                         << pattr[p + item*istride] << " ";
        }
        *PtclDbgInform << endl;
      }
      *PtclDbgInform << endl;
    } else {
      *PtclDbgInform << "....PE = " << mype
                    << " has no particles ...." << endl;
    }

  }
}