// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef REF_COUNTED_H
#define REF_COUNTED_H

//////////////////////////////////////////////////////////////////////

class RefCounted
{
public: 

  // The reference count.
  int RefCount;

  // The constructor just initializes the reference count.
  RefCounted()
    : RefCount(0)
      {
      }

};

//////////////////////////////////////////////////////////////////////

template < class T >
class RefCountedP
{
public: 

  // Null ctor constructs with a null ptr.
  RefCountedP()
    : p(0)
      {
      }

  // Initialize from a pointer to the pointed class.
  RefCountedP(T* pp)
    : p(pp) 
      {
	if (pp)
	  ++(pp->RefCount);
      }

  // Copy ctor.
  RefCountedP(const RefCountedP<T>& pp)
    : p(pp.p)
      {
	if(p)
	  ++(p->RefCount);
      }

  // When you destroy one of these, decrement the RefCount.
  ~RefCountedP()
    {
      if (p) {
	if ( (--(p->RefCount)) == 0 )
	  delete p;
	p = 0;
      }
    }

  // Assignment operator increments the refcount.
  RefCountedP& operator=(const RefCountedP& rhs)
    {
      // First unlink from the one we're pointing to now.
      if ( p && (--(p->RefCount)==0) )
	delete p;
      // Assign the new one.
      if ((p = rhs.p))
	++(p->RefCount);
      return *this;
    }

  // Assignment operator increments the refcount.
  RefCountedP& operator=(T* pp)
    {
      // First unlink from the one we're pointing to now.
      if ( p && (--(p->RefCount)==0) )
	delete p;
      // Assign the new one.
      if ((p = pp))
	++(p->RefCount);
      return *this;
    }

  // Two ways to use the pointer.
  T* operator->() const { return p;  }
  T& operator*()  const { return *p; }

  // Just extract the pointer if you need it.
  operator T*() { return p; }
  operator const T*() const { return p; }

  void invalidate()
    {
      if ( p && (--(p->RefCount)==0) )
	delete p;
      p = 0;
    }
  bool valid()
    {
      return p!=0;
    }

  // Utility function for implementing copy-on-write semantics.
  RefCountedP<T>& CopyForWrite()
    {
      // If more than one thing is referring to this one
      if ( p && (p->RefCount > 1) )
	{
          // Unlink from the existing object.
          --(p->RefCount);
	  // Copy it.
	  p = new T( *p );
	  // Inform it that we are watching.
	  p->RefCount = 1;
	}
      // Return yourself for further processing.
      return *this;
    }

private:
  // The pointer itself.
  T *p;
};

//////////////////////////////////////////////////////////////////////

#endif // REF_COUNTED_H

/***************************************************************************
 * $RCSfile: RefCounted.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: RefCounted.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
