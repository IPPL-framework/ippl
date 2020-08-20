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
#include "FieldLayout/BinaryBalancer.h"
#include "FieldLayout/FieldLayout.h"
#include "Field/BareField.h"
#include "Utility/PAssert.h"


//////////////////////////////////////////////////////////////////////

/*

  Implementation of BinaryBalancer.

  The general strategy is that you do log(P) splits on the domain.  It
  starts with the whole domain, does a reduction to find where to
  split it, then does reductions on each of the resulting domains to
  find where to split those, then reductions on those to split them,
  and so on until it is done.

  Suppose you're on the n'th split, so there are 2**n domains figured
  out so far, and after the split there will be 2**(n+1) splits.  In
  each of those 2**n domains you need to find

  a) The axis to split on. This is done by just finding the longest
     axis in that domain.

  b) The location within that domain to make the split.  This is done
     by reducing the weights on all dimensions except the axis to be
     split and finding the location within that array that puts half
     the weight on each side.

  The reduction for b) is done is a scalable way.  It is a parallel
  reduction, and if there are 2**n domains being split, the reductions
  are accumulated onto processors 0..2**n-1.  Those processors
  calculate the split locations and broadcast them to all the
  processors.

  At every stage of the process all the processors know all the
  domains.  This is necessary because the weight array could be
  distributed arbitrarily, so the reductions could involve any
  processors.

  Nevertheless, the reductions are performed efficiently.  Using
  DomainMaps, only the processors that need to participate in a
  reduction do participate.

*/

//////////////////////////////////////////////////////////////////////

//
// Given an NDIndex<Dim> find the axis with the longest length, that is
// not SERIAL.
//
template <unsigned Dim>
static int
FindCutAxis(const NDIndex<Dim> &domain, const FieldLayout<Dim> &layout)
{



  // CutAxis will be the dimension to cut.
  int cutAxis=-1;
  // MaxLength will have the maximum length of any dimension.
  unsigned int maxLength=0;
  // Loop over dimension.
  for (unsigned int d=0; d<Dim; ++d) {
    if (layout.getDistribution(d) != SERIAL ||
	layout.getRequestedDistribution(d) != SERIAL) {
      // Check if this axis is longer than the current max.
      unsigned int length = domain[d].length();
      if ( maxLength < length ) {
	// If so, remember it.
	cutAxis = d;
	maxLength = length;
      }
    }
  }

  // Make sure we found one.
  //PAssert_GE(cutAxis, 0);

  if(cutAxis<0)
    throw BinaryRepartitionFailed();

  // Return the longest axis.
  return cutAxis;
}


//
// Find the median point in a container.
// The first two arguments are begin and end iterators.
// The third is a dummy argument of type T, where the
// container is of objects of type T.
//

template<class RandomIterator, class T>
static RandomIterator
FindMedian(int nprocs,RandomIterator begin, RandomIterator end, T)
{
  // First find the total weight.
  T w = 0;
  // Use w to find T's name


  // If we have only one processor, cut at the left.
  if ( nprocs == 1 )
    return begin;

  int lprocs = nprocs/2;
  RandomIterator rp, median;
  for (rp=begin; rp!=end; ++rp)
    w += *rp;

  // If the total weight is zero, we need to do things specially.
  if ( w==0 )
    {
      // The total weight is zero.
      // Put about as much zero weight stuff on the left and the right.
      median = begin + ((end-begin)*lprocs)/nprocs;
    }
  else
    {
      // The total weight is nonzero.
      // Put equal amounts on the left and right processors.
      T w2 = (w*lprocs)/nprocs;
      // Find the point with half the weight to the left.
      bool found = false;
      w = T(0);
      for (rp=begin; (rp!=end)&&(!found); ++rp) {
	// Add current element to running total
	w += *rp;
	if (w>=w2) {
	  found = true;
	  if ( (w-w2) > (*rp/T(2)) )
	    median = rp;
	  else
	    median = (rp+1 != end) ? rp+1 : rp;
	}
      }
    }
  // Found it.  Exit.
  return median;
}

//////////////////////////////////////////////////////////////////////

//
// Routines for doing reductions over all dimensions but one of a
// local BrickIterator.
//
// These will only work for 1, 2 and 3 dimensions right now.
// I'm sure there is a way to make this work efficiently
// for arbitrary dimension, but this works for now.
//

// Reduce over a 3 dimensional BrickIterator
// given the axis not to reduce on
// and where in that dimension to reduce.

static inline double
PerpReduce(BrickIterator<double,3>& data, int i, int cutAxis)
{
  double r=0;
  if (cutAxis==0)
    {
      int l1 = data.size(1);
      int l2 = data.size(2);
      if ( (l1>0) && (l2>0) )
        for (int j2=0; j2<l2; ++j2)
          for (int j1=0; j1<l1; ++j1)
            r += data.offset(i,j1,j2);
    }
  else if (cutAxis==1)
    {
      int l0 = data.size(0);
      int l2 = data.size(2);
      if ( (l0>0) && (l2>0) )
        for (int j2=0; j2<l2; ++j2)
          for (int j0=0; j0<l0; ++j0)
            r += data.offset(j0,i,j2);
    }
  else if (cutAxis==2)
    {
      int l0 = data.size(0);
      int l1 = data.size(1);
      if ( (l0>0) && (l1>0) )
        for (int j1=0; j1<l1; ++j1)
          for (int j0=0; j0<l0; ++j0)
            r += data.offset(j0,j1,i);
    }
  return r;
}

// Reduce over a 2 dimensional BrickIterator
// given the axis not to reduce on
// and where in that dimension to reduce.

static inline double
PerpReduce(BrickIterator<double,2>& data, int i, int cutAxis)
{
  double r=0;
  if (cutAxis==0)
    {
      int length = data.size(1);
      for (int j=0; j<length; ++j)
        r += data.offset(i,j);
    }
  else
    {
      int length = data.size(0);
      for (int j=0; j<length; ++j)
        r += data.offset(j,i);
    }
  return r;
}

//
// Reduce over a 1 dimensional BrickIterator
// given the axis not to reduce on
// and where in that dimension to reduce.
//

static inline double
PerpReduce(BrickIterator<double,1>& data, int i, int )
{
  return data.offset(i);
}

//
// Reduce over all the dimensions but one of a brick iterator.
// Put the results in an array of doubles.
//

template<unsigned Dim>
static void
LocalReduce(double *reduced, int cutAxis, BrickIterator<double,Dim> data)
{



  int length = data.size(cutAxis);
  for (int i=0; i<length; ++i)
    reduced[i] = PerpReduce(data,i,cutAxis);
}

//////////////////////////////////////////////////////////////////////

//
// For each domain, do the local reduction of
// data from weights, and send that to the node
// that is accumulating stuff for that domain.
//
// The local reductions take place all across the machine.
// The reductions for each domain are finished on a single processor.
// Each of those final reductions are on different processors.
//

template<class IndexIterator, unsigned Dim>
static void
SendReduce(IndexIterator domainsBegin, IndexIterator domainsEnd,
           BareField<double,Dim>& weights, int tag)
{

  // Buffers to store up domains and blocks of reduced data.
  std::vector<double*> reducedBuffer;
  std::vector<Index> domainBuffer;
  // Loop over all of the domains.  Keep a counter of which one you're on.
  int di;
  IndexIterator dp;
  /*out << "SendReduce, ndomains=" << domainsEnd-domainsBegin << endl;*/
  for (dp=domainsBegin, di=0; dp!=domainsEnd; ++dp, ++di)
    {
      /*out << "SendReduce, domain=" << *dp << endl;*/
      // Find the dimension we'll be cutting on.
      // We'll reduce in the dimensions perpendicular to this.
      int cutAxis = FindCutAxis(*dp, weights.getLayout());
      // Find the LFields on this processor that touch this domain.
      typename BareField<double,Dim>::iterator_if lf_p;
      for (lf_p=weights.begin_if(); lf_p != weights.end_if(); ++lf_p)
        if ( (*dp).touches( (*lf_p).second->getOwned() ) )
          {
            // Find the intersection with this LField.
            NDIndex<Dim> intersection =
	      (*dp).intersect( (*lf_p).second->getOwned() );
            // Allocate the accumulation buffer.
            int length = intersection[cutAxis].length();
            double *reduced = new double[length];
            // Reduce into the local buffer.
	    /*out << "LocalReduce " << intersection << endl;*/
            LocalReduce(reduced,cutAxis,(*lf_p).second->begin(intersection));
            // Save the domain and the data.
            reducedBuffer.push_back(reduced);
            domainBuffer.push_back(intersection[cutAxis]);
          }

      // If we found any hits, send them out.
      int nrdomains = reducedBuffer.size();
      /*out << "nrdomains=" << nrdomains << endl;*/
      if ( nrdomains>0 )
        {
          // Build a message to hold everything for this domain.
          Message *mess = new Message;
          // The number of reduced domains is the first thing in the message.
          mess->put(nrdomains);
          // Loop over the reduced domains, storing in the message each time.
          std::vector<Index>::iterator dbp = domainBuffer.begin();
          std::vector<double*>::iterator rbp = reducedBuffer.begin();
          for (int i=0; i<nrdomains; ++i, ++dbp, ++rbp)
            {
              // First store the domain.
	      /*out << "putMessage " << *dbp << endl;*/
              putMessage(*mess,*dbp);
              // Then the reduced data using begin/end iterators.
              // Tell the message to delete the memory when it is done.
              double *p = *rbp;
              mess->setCopy(false).setDelete(true).put(p,p+(*dbp).length());
            }
          // Send the message to proc di.
	  Ippl::Comm->send(mess, di, tag);
        }
      // Clear out the buffers.
      domainBuffer.erase( domainBuffer.begin(), domainBuffer.end() );
      reducedBuffer.erase( reducedBuffer.begin(), reducedBuffer.end() );
      /*out << "Bottom of SendReduce loop" << endl;*/
    }
}

//////////////////////////////////////////////////////////////////////

//
// Receive the messages with reduced data sent out in SendReduce.
// Finish the reduction.
// Return begin and end iterators for the reduced data.
//

template<unsigned Dim>
static void
ReceiveReduce(NDIndex<Dim>& domain, BareField<double,Dim>& weights,
              int reduce_tag, int nprocs,
	      int& cutLoc, int& cutAxis)
{


  // Build a place to accumulate the reduced data.
  cutAxis = FindCutAxis(domain, weights.getLayout());
  /*out << "ReceiveReduce, cutAxis=" << cutAxis << endl;*/
  int i, length = domain[cutAxis].length();
  int offset = domain[cutAxis].first();
  std::vector<double> reduced(length);
  std::vector<double> subReduced(length);
  for (i=0; i<length; ++i)
    reduced[i] = 0;

  // Build a count of the number of messages to expect.
  // We get *one message* from each node that has a touch.
  int expected = 0;
  int nodes = Ippl::getNodes();
  int mynode = Ippl::myNode();
  bool* found_touch = new bool[nodes];
  for (i=0; i<nodes; ++i) found_touch[i] = false;
  // First look in the local vnodes of weights.
  typename BareField<double,Dim>::iterator_if lf_p, lf_end = weights.end_if();
  for (lf_p = weights.begin_if();
       lf_p != lf_end && !(found_touch[mynode]); ++lf_p) {
    // Expect a message if it touches.
    if ( (*lf_p).second->getOwned().touches(domain) )
      found_touch[mynode] = true;
  }
  // Now look in the remote parts of weights.
  typename FieldLayout<Dim>::touch_iterator_dv rf_p;
  // Get the range of remote vnodes that touch domain.
  typename FieldLayout<Dim>::touch_range_dv range =
    weights.getLayout().touch_range_rdv( domain );
  // Record the processors who have touches
  for (rf_p = range.first; rf_p != range.second ; ++rf_p) {
    int owner = (*((*rf_p).second)).getNode();
    found_touch[owner] = true;
  }
  // now just count up the number of messages to receive
  for (i=0; i<nodes; ++i)
    if (found_touch[i]) expected++;
  delete [] found_touch;

  // Receive messages until we're done.
  while ( --expected >= 0 )
    {
      // Receive a message.
      int any_node = COMM_ANY_NODE;
      Message *mess = Ippl::Comm->receive_block(any_node,reduce_tag);
      PAssert(mess);
      // Loop over all the domains in this message.
      int received_domains = 0;
      mess->get(received_domains);
      while ( --received_domains>=0 )
        {
          // Get the domain for the next part.
          Index rdomain;
          getMessage( *mess, rdomain );
	  /*out << "ReceiveReduce, rdomain=" << rdomain << endl;*/
          // Get the incoming reduced data.
          int rfirst = rdomain.first() - offset;
          mess->get(subReduced[rfirst]);
          // Accumulate it with the rest.
          int rlast = rdomain.last() - offset;
          for (int i=rfirst; i<=rlast; ++i)
            reduced[i] += subReduced[i];
        }
      // Delete the message, we're done with it
      delete mess;
    }

  // Get the median.
  cutLoc =
    FindMedian(nprocs,reduced.begin(),reduced.begin()+length,double())
    -reduced.begin() + domain[cutAxis].first();
  /*out << "ReceiveReduce, cutLoc=" << cutLoc << endl;*/
}

//////////////////////////////////////////////////////////////////////

//
// Given the location and axis of the cut,
// Broadcast to everybody.
//

inline void
BcastCuts(int cutLoc, int cutAxis, int bcast_tag)
{


  // Make a message.
  Message *mess = new Message();
  // Add the data to it.
  mess->put(cutLoc);
  mess->put(cutAxis);
  // Send it out.
  Ippl::Comm->broadcast_all(mess,bcast_tag);
}

//////////////////////////////////////////////////////////////////////

//
// Receive the broadcast cuts.
// Cut up each of the domains using the cuts.
//

template<unsigned Dim>
static void
ReceiveCuts(std::vector< NDIndex<Dim> > &domains,
	    std::vector< int >& nprocs,
	    int bcast_tag)
{



  // Make a container to hold the split domains.
  int nDomains = domains.size();
  std::vector< NDIndex<Dim> > cutDomains(nDomains*2);
  std::vector<int> cutProcs(std::vector<int>::size_type(nDomains*2));

  // Everybody receives the broadcasts.
  // There will be one for each domain in the list.
  for (int expected = 0; expected < nDomains; ++expected)
    {
      // Receive each broadcast.
      // The processor number will correspond to the location
      // in the domains vector.
      int whichDomain = COMM_ANY_NODE;
      int cutLocation = 0, cutAxis = 0;
      Message *mess = Ippl::Comm->receive_block(whichDomain,bcast_tag);
      PAssert(mess);
      mess->get(cutLocation);
      mess->get(cutAxis);
      delete mess;

      // Split this domain.
      const NDIndex<Dim>& domain = domains[whichDomain];
      NDIndex<Dim>& left = cutDomains[ whichDomain*2 ];
      NDIndex<Dim>& right = cutDomains[ whichDomain*2+1 ];
      // Build the left and right domains.
      left = domain ;
      right = domain ;
      /*out << "Build indexes from : "
	  << domain[cutAxis].first() << " "
	  << cutLocation<< " "
	  << domain[cutAxis].last()<< " "
	  << endl;*/
      left[ cutAxis ] = Index( domain[cutAxis].first(), cutLocation-1 );
      right[ cutAxis ] = Index( cutLocation, domain[cutAxis].last() );

      int procs = nprocs[whichDomain];
      cutProcs[ whichDomain*2 ] = procs/2;
      cutProcs[ whichDomain*2+1 ] = procs - procs/2;
    }

  // Put the domains you've just built into the input containers.
  // Strip out the domains with no processors assigned.
  domains.clear();
  nprocs.clear();
  PAssert_EQ(cutProcs.size(), cutDomains.size());
  for (unsigned int i=0; i<cutProcs.size(); ++i)
    {
      if ( cutProcs[i] != 0 )
	{
	  domains.push_back(cutDomains[i]);
	  nprocs.push_back(cutProcs[i]);
	}
      else
	{
	  PAssert_EQ(cutDomains[i].size(), 0);
	}
    }
}

//////////////////////////////////////////////////////////////////////

//
// Sweep through a list of domains, splitting each one
// according to the weights in a BareField.
//

template<unsigned Dim>
static void
CutEach(std::vector< NDIndex<Dim> >& domains,
	std::vector< int >& nprocs,
	BareField<double,Dim>& weights)
{

  // Get tags for the reduction and the broadcast.
  int reduce_tag = Ippl::Comm->next_tag( F_REDUCE_PERP_TAG , F_TAG_CYCLE );
  int bcast_tag  = Ippl::Comm->next_tag( F_REDUCE_PERP_TAG , F_TAG_CYCLE );
  /*out << "reduce_tag=" << reduce_tag << endl;*/
  /*out << "bcast_tag=" << bcast_tag << endl;*/

  // Do the sends for the reduces.
  SendReduce(domains.begin(),domains.end(),weights,reduce_tag);

  // On the appropriate processors, receive the data for the reduce,
  // and broadcast the cuts.
  unsigned int mynode = Ippl::Comm->myNode();
  if ( mynode < domains.size() )
    {
      // Receive partially reduced data, finish the reduction, find the median.
      int cutAxis, cutLoc;
      ReceiveReduce(domains[mynode],weights,reduce_tag,
		    nprocs[mynode],cutLoc,cutAxis);
      // Broadcast those cuts out to everybody.
      BcastCuts(cutLoc,cutAxis,bcast_tag);
    }

  // Receive the broadcast cuts and slice up the domains.
  ReceiveCuts(domains,nprocs,bcast_tag);
}

//////////////////////////////////////////////////////////////////////

template<unsigned Dim>
NDIndex<Dim>
CalcBinaryRepartition(FieldLayout<Dim>& layout, BareField<double,Dim>& weights)
{
// Build a list of domains as we go.
  std::vector< NDIndex<Dim> > domains; // used by TAU_TYPE_STRING
  std::vector<int> procs;

  /*out << "Starting CalcBinaryRepartition, outstanding msgs="
      << Ippl::Comm->getReceived()
      << endl;*/

  // Get the processors we'll be dealing with.
  int nprocs = Ippl::Comm->getNodes();
  int myproc = Ippl::Comm->myNode();
  domains.reserve(nprocs);
  procs.reserve(nprocs);
  // Start the list with just the top level domain.
  domains.push_back( layout.getDomain() );
  procs.push_back( nprocs );

  // mprocs is the max number of procs assigned to a domain.
  int mprocs=nprocs;

  // Loop as long as some domain has more than one proc assigned to it.
  while ( mprocs>1 )
    {
      // Cut all the domains in half.
      CutEach(domains,procs,weights);

      // Find the max number of procs assigned to a domain.
      mprocs = 0;
      for (unsigned int i=0; i<procs.size(); ++i)
	if (mprocs<procs[i]) mprocs = procs[i];
    }
  // Return the domain on this processor.


  //seriously dirty fix
  typename std::vector< NDIndex<Dim> >::iterator i;

  bool degenerated = false;

  for(i = domains.begin();i!=domains.end();++i)
  {
	  for(unsigned int d = 0;d<Dim;++d)
		if((*i)[d].first() == (*i)[d].last())
		{
			degenerated = true;
			break;
		}
	if(degenerated)
		break;
  }

  if(!degenerated)
	return domains.begin()[myproc];
  else
	{
		throw BinaryRepartitionFailed();
	}

}

//////////////////////////////////////////////////////////////////////


/***************************************************************************
 * $RCSfile: BinaryBalancer.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: BinaryBalancer.cpp,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $
 ***************************************************************************/
