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
#include "Field/AssignDefs.h"
#include "Index/NDIndex.h"
#include "Utility/PAssert.h"


//////////////////////////////////////////////////////////////////////
// BrickCounter methods

template<unsigned Dim>
BrickCounter<Dim>::BrickCounter(const NDIndex<Dim>& counted)
{
  for (unsigned d=0; d<Dim; ++d)
    {
      BrickCounter<Dim>::Counters[d] = 0;
      BrickCounter<Dim>::Counts[d] = counted[d].length();
    }
}

//////////////////////////////////////////////////////////////////////

template<unsigned Dim>
void
BrickCounter<Dim>::op_pp()
{
  // Remove this profiling because this is too lightweight.
  // 
  // 
  for (unsigned d=0; d<Dim; ++d) {
    step(d);
    if ( BrickCounter<Dim>::Counters[d] != BrickCounter<Dim>::Counts[d] )
      return;
    rewind(d);
  }
  // If we get to here we have looped over the whole thing.
}


//////////////////////////////////////////////////////////////////////
// BrickIterator methods

template<class T, unsigned Dim>
BrickIterator<T,Dim>::BrickIterator(T* p,
				    const NDIndex<Dim>& counted,
				    const NDIndex<Dim>& domalloc)
: BrickCounter<Dim>(counted), Current(p), Whole(true)
{
  // Remove this profiling because this is too lightweight.
  // 
  // 

  // Calculate the strides,carriage returns and offset for the pointer.
  int n=1;
  for (unsigned d=0; d<Dim; ++d) {
    Strides[d] = n*counted[d].stride();
    Current += n*(counted[d].first() - domalloc[d].first());
    n *= domalloc[d].length();
    Whole = (Whole && counted[d] == domalloc[d]);
  }
}

//////////////////////////////////////////////////////////////////////

template< class T, unsigned Dim >
BrickIterator<T,Dim>::BrickIterator(T* p, const vec<int,Dim>& sizes)
{
  // Remove this profiling because this is too lightweight.
  //
  //

  int n = 1;
  for (unsigned d=0; d<Dim; ++d) {
    BrickCounter<Dim>::Counts[d] = sizes[d];
    BrickCounter<Dim>::Counters[d] = 0;
    Strides[d] = n;
    n *= sizes[d];
  }
  Current = p;
  Whole = true;
}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned Dim>
void
BrickIterator<T,Dim>::op_pp()
{
  // Remove this profiling because this is too lightweight.
  // 
  // 

  for (unsigned d=0; d<Dim; ++d) {
    step(d);
    if ( BrickCounter<Dim>::Counters[d] != BrickCounter<Dim>::Counts[d] )
      return;
    rewind(d);
  }
  // If we get to here we have looped over the whole thing.
  Current = 0;
}


//////////////////////////////////////////////////////////////////////
// put data into a message to send to another node
// ... for putMessage, the second argument
// is used for an optimization for when the entire brick is being
// sent.  If that is so, do not copy data into a new buffer, just
// put the pointer into the message.  USE WITH CARE.  The default is
// tohave putMessage make a copy of the data in the brick before adding
// it to the message.  In many situations, this is required since the
// data referred to by the iterator is not contiguous.  getMessage
// has no such option, it always does the most efficient thing it can.
template<class T, unsigned Dim>
Message& BrickIterator<T,Dim>::putMessage(Message& m, bool makecopy) {
  
  

  int n, s[Dim];
  unsigned int i;

  // put in the size of the data, and calculate total number of elements
  for (n=1, i=0; i < Dim; ++i) {
    s[i] = BrickCounter<Dim>::Counts[i];
    n *= BrickCounter<Dim>::Counts[i];
  }
  m.put(s, s + Dim);

  // Only add in the actual data if there is something to add.
  if (n > 0) {
    // If we are not required to make a copy, check if we can just add in
    // the pointer.  We can do this ONLY if we are adding in the entire
    // domain, and not a subset, of the original domain.
    if (!makecopy && whole()) {
      // We are iterating over the whole brick (not just a subset), and
      // we are not required to make a copy, so, well, don't.  Just put
      // in the pointer to the brick and ask the message NOT to delete
      // the data when the message is sent.
      m.setCopy(false);
      m.setDelete(false);
      m.putmsg((void *)Current, sizeof(T), n);
    } else {
      // FIX THIS!! Better to give a BrickIterator begin/end pair to
      // putMessage than to do this.
      T* cpydata = (T*) malloc(sizeof(T)*n);
      T* cpy     = cpydata;
      T* cpyend  = cpydata + n;
      BrickIterator<T, Dim> bi = *this;
      for (; cpy != cpyend; ++cpy, ++bi)
	new (cpy) T(*bi);

      // put data into this message
      m.setCopy(false);
      m.setDelete(true);
      m.putmsg((void *)cpydata, sizeof(T), n);
    }
  }

  return m;
}


//////////////////////////////////////////////////////////////////////
// get data out from a message
template<class T, unsigned Dim>
Message& BrickIterator<T,Dim>::getMessage(Message& m) {
  
  
  int n, s[Dim];
  unsigned int i;
  // this will only work if this iterator does not yet point
  // at any data
  PInsist(Current == 0,
          "Iterator already has data in BrickIterator::getMessage!!");

  // retrieve size of data in message, and set counters, strides, etc.
  m.get((int*) s);
  for (n=1, i=0; i < Dim; ++i) {
    BrickCounter<Dim>::Counts[i] = s[i];
    BrickCounter<Dim>::Counters[i] = 0;
    Strides[i] = n;
    n *= s[i];
  }

  // retrieve actual data into an allocated buffer
  // NOTE: This just stores a pointer to the data that is actually stored
  // in the Message item.  It does not make its own copy.  So, if you use
  // a BrickIterator to get data from a message, you must use the data in the
  // BrickIterator BEFORE you delete the Message.  Otherwise, you'll be
  // pointing at deallocated memory.
  /*
  Current = new T[n];
  ::getMessage_iter(m, Current);
  */
  Current = static_cast<T *>(m.item(0).data());
  m.get();
  return m;
}



/***************************************************************************
 * $RCSfile: BrickIterator.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: BrickIterator.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
