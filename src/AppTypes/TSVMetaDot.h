// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef TSV_META_DOT_H
#define TSV_META_DOT_H

//////////////////////////////////////////////////////////////////////
//
// Definition of the struct TSV_MetaDot.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2> struct TSV_MetaDot {};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektor dot Vektor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Vektor<T1,D> , Vektor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Vektor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    T0 dot = lhs[0]*rhs[0];
    for (unsigned d=1; d<D; ++d)
      dot += lhs[d]*rhs[d];
    return dot;
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,1> , Vektor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Vektor<T1,1>& lhs, const Vektor<T2,1>& rhs) {
    return lhs[0]*rhs[0];
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,2> , Vektor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Vektor<T1,2>& lhs, const Vektor<T2,2>& rhs) {
    return lhs[0]*rhs[0] + lhs[1]*rhs[1];
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,3> , Vektor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static T0
  apply(const Vektor<T1,3>& lhs, const Vektor<T2,3>& rhs) {
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzor dot Tenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Tenzor<T1,D> , Tenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,D>
  apply(const Tenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    Tenzor<T0,D> dot = typename Tenzor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=0; j<D; ++j) {
	T0 sum = lhs(i,0) * rhs(0,j);
	for (unsigned int k=1; k<D; ++k)
	  sum += lhs(i,k) * rhs(k,j);
	dot(i,j) = sum;
      }
    return dot;
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,1> , Tenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,1>
  apply(const Tenzor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    return Tenzor<T0,1>(lhs[0]*rhs[0]);
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,2> , Tenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,2>
  apply(const Tenzor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    return Tenzor<T0,2>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0),
			lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1),
			lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0),
			lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1));
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,3> , Tenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,3>
  apply(const Tenzor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    return Tenzor<T0,3>( lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0) + lhs(0,2)*rhs(2,0) ,
			 lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1) + lhs(0,2)*rhs(2,1) ,
			 lhs(0,0)*rhs(0,2) + lhs(0,1)*rhs(1,2) + lhs(0,2)*rhs(2,2) ,
			 lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0) + lhs(1,2)*rhs(2,0) ,
			 lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1) + lhs(1,2)*rhs(2,1) ,
			 lhs(1,0)*rhs(0,2) + lhs(1,1)*rhs(1,2) + lhs(1,2)*rhs(2,2) ,
			 lhs(2,0)*rhs(0,0) + lhs(2,1)*rhs(1,0) + lhs(2,2)*rhs(2,0) ,
			 lhs(2,0)*rhs(0,1) + lhs(2,1)*rhs(1,1) + lhs(2,2)*rhs(2,1) ,
			 lhs(2,0)*rhs(0,2) + lhs(2,1)*rhs(1,2) + lhs(2,2)*rhs(2,2) );
			
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzor dot SymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< SymTenzor<T1,D> , SymTenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,D>
  apply(const SymTenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    Tenzor<T0,D> dot = typename Tenzor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=i; j<D; ++j) {
	T0 sum = lhs.HL(i,0) * rhs.HL(j,0);
	unsigned int k=1;
	for ( ; k<i; ++k )
	  sum += lhs.HL(i,k) * rhs.HL(j,k);
	for ( ; k<j; ++k )
	  sum += lhs.HL(k,i) * rhs.HL(j,k);
	for ( ; k<D; ++k )
	  sum += lhs.HL(k,i) * rhs.HL(k,j);
	dot(i,j) = sum;
      }
    return dot;
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,1> , SymTenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,1>
  apply(const SymTenzor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    return Tenzor<T0,1>(lhs[0]*rhs[0]);
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,2> , SymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,2>
  apply(const SymTenzor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    return Tenzor<T0,2>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0),
			lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1),
			lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0),
			lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1));
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,3> , SymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,3>
  apply(const SymTenzor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    return 
      Tenzor<T0,3>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0) + lhs(0,2)*rhs(2,0) ,
                   lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1) + lhs(0,2)*rhs(2,1) ,
                   lhs(0,0)*rhs(0,2) + lhs(0,1)*rhs(1,2) + lhs(0,2)*rhs(2,2) ,
                   lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0) + lhs(1,2)*rhs(2,0) ,
                   lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1) + lhs(1,2)*rhs(2,1) ,
                   lhs(1,0)*rhs(0,2) + lhs(1,1)*rhs(1,2) + lhs(1,2)*rhs(2,2) ,
                   lhs(2,0)*rhs(0,0) + lhs(2,1)*rhs(1,0) + lhs(2,2)*rhs(2,0) ,
                   lhs(2,0)*rhs(0,1) + lhs(2,1)*rhs(1,1) + lhs(2,2)*rhs(2,1) ,
                   lhs(2,0)*rhs(0,2) + lhs(2,1)*rhs(1,2) + lhs(2,2)*rhs(2,2));
			
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzor dot Vektor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Tenzor<T1,D> , Vektor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,D>
  apply(const Tenzor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    Vektor<T0,D> ret = typename Vektor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i) {
      T0 sum = lhs(i,0)*rhs[0];
      for (unsigned int j=1; j<D; ++j)
	sum += lhs(i,j)*rhs[j];
      ret[i] = sum;
    }
    return ret;
  }
};


template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,1> , Vektor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,1>
  apply(const Tenzor<T1,1>& lhs, const Vektor<T2,1>& rhs) {
    return Vektor<T0,1>( lhs[0]*rhs[0] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,2> , Vektor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,2>
  apply(const Tenzor<T1,2>& lhs, const Vektor<T2,2>& rhs) {
    return Vektor<T0,2>( lhs(0,0)*rhs[0] + lhs(0,1)*rhs[1] ,
			 lhs(1,0)*rhs[0] + lhs(1,1)*rhs[1] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,3> , Vektor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const Tenzor<T1,3>& lhs, const Vektor<T2,3>& rhs) {
    return Vektor<T0,3>( lhs(0,0)*rhs[0] + lhs(0,1)*rhs[1] + lhs(0,2)*rhs[2],
			 lhs(1,0)*rhs[0] + lhs(1,1)*rhs[1] + lhs(1,2)*rhs[2],
			 lhs(2,0)*rhs[0] + lhs(2,1)*rhs[1] + lhs(2,2)*rhs[2] );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektor dot Tenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Vektor<T1,D> , Tenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,D>
  apply(const Vektor<T2,D>& lhs, const Tenzor<T1,D>& rhs) {
    Vektor<T0,D> ret = typename Vektor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i) {
      T0 sum = lhs[0]*rhs(0,i);
      for (unsigned int j=1; j<D; ++j)
	sum += lhs[j]*rhs(j,i);
      ret[i] = sum;
    }
    return ret;
  }
};


template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,1> , Tenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,1>
  apply(const Vektor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    return Vektor<T0,1>( lhs[0]*rhs[0] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,2> , Tenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,2>
  apply(const Vektor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    return Vektor<T0,2>( lhs[0]*rhs(0,0) + lhs[1]*rhs(1,0) ,
			 lhs[0]*rhs(0,1) + lhs[1]*rhs(1,1) );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,3> , Tenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    return Vektor<T0,3>( lhs[0]*rhs(0,0) + lhs[1]*rhs(1,0) + lhs[2]*rhs(2,0),
			 lhs[0]*rhs(0,1) + lhs[1]*rhs(1,1) + lhs[2]*rhs(2,1),
			 lhs[0]*rhs(0,2) + lhs[1]*rhs(1,2) + lhs[2]*rhs(2,2) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzor dot Vektor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< SymTenzor<T1,D> , Vektor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,D>
  apply(const SymTenzor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    Vektor<T0,D> ret = typename Vektor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i) {
      T0 sum = lhs.HL(i,0)*rhs[0];
      unsigned int j=1;
      for ( ; j<i; ++j)
	sum += lhs.HL(i,j)*rhs[j];
      for ( ; j<D; ++j)
	sum += lhs.HL(j,i)*rhs[j];
      ret[i] = sum;
    }
    return ret;
  }
};


template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,1> , Vektor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,1>
  apply(const SymTenzor<T1,1>& lhs, const Vektor<T2,1>& rhs) {
    return Vektor<T0,1>( lhs[0]*rhs[0] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,2> , Vektor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,2>
  apply(const SymTenzor<T1,2>& lhs, const Vektor<T2,2>& rhs) {
    return Vektor<T0,2>( lhs(0,0)*rhs[0] + lhs(0,1)*rhs[1] ,
			 lhs(1,0)*rhs[0] + lhs(1,1)*rhs[1] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,3> , Vektor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const SymTenzor<T1,3>& lhs, const Vektor<T2,3>& rhs) {
    return Vektor<T0,3>( lhs(0,0)*rhs[0] + lhs(0,1)*rhs[1] + lhs(0,2)*rhs[2],
			 lhs(1,0)*rhs[0] + lhs(1,1)*rhs[1] + lhs(1,2)*rhs[2],
			 lhs(2,0)*rhs[0] + lhs(2,1)*rhs[1] + lhs(2,2)*rhs[2] );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektor dot SymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Vektor<T1,D> , SymTenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,D>
  apply(const Vektor<T2,D>& lhs, const SymTenzor<T1,D>& rhs) {
    Vektor<T0,D> ret = typename Vektor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i) {
      T0 sum = lhs[0]*rhs[i*(i+1)/2];
      unsigned int j=1;
      for ( ; j<i; ++j)
	sum += lhs[j]*rhs[i*(i+1)/2+j];
      for ( ; j<D; ++j)
	sum += lhs[j]*rhs[j*(j+1)/2+i];
      ret[i] = sum;
    }
    return ret;
  }
};


template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,1> , SymTenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,1>
  apply(const Vektor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    return Vektor<T0,1>( lhs[0]*rhs[0] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,2> , SymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,2>
  apply(const Vektor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    return Vektor<T0,2>( lhs[0]*rhs(0,0) + lhs[1]*rhs(1,0) ,
			 lhs[0]*rhs(0,1) + lhs[1]*rhs(1,1) );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,3> , SymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    return Vektor<T0,3>( lhs[0]*rhs(0,0) + lhs[1]*rhs(1,0) + lhs[2]*rhs(2,0),
			 lhs[0]*rhs(0,1) + lhs[1]*rhs(1,1) + lhs[2]*rhs(2,1),
			 lhs[0]*rhs(0,2) + lhs[1]*rhs(1,2) + lhs[2]*rhs(2,2) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for SymTenzor dot Tenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< SymTenzor<T1,D> , Tenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,D>
  apply(const SymTenzor<T1,D>& lhs, const Tenzor<T2,D>& rhs) {
    Tenzor<T0,D> dot = typename Tenzor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=0; j<D; ++j) {
	T0 sum = lhs.HL(i,0) * rhs(0,j);
	unsigned int k = 1;
	for (; k<i; ++k)
	  sum += lhs.HL(i,k) * rhs(k,j);
	for (; k<D; ++k)
	  sum += lhs.HL(k,i) * rhs(k,j);
	dot(i,j) = sum;
      }
    return dot;
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,1> , Tenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,1>
  apply(const SymTenzor<T1,1>& lhs, const Tenzor<T2,1>& rhs) {
    return Tenzor<T0,1>(lhs[0]*rhs[0]);
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,2> , Tenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,2>
  apply(const SymTenzor<T1,2>& lhs, const Tenzor<T2,2>& rhs) {
    return Tenzor<T0,2>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0),
			lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1),
			lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0),
			lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1));
  }
};

template<class T1, class T2>
struct TSV_MetaDot< SymTenzor<T1,3> , Tenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,3>
  apply(const SymTenzor<T1,3>& lhs, const Tenzor<T2,3>& rhs) {
    return 
      Tenzor<T0,3>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0) + lhs(0,2)*rhs(2,0) ,
                   lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1) + lhs(0,2)*rhs(2,1) ,
                   lhs(0,0)*rhs(0,2) + lhs(0,1)*rhs(1,2) + lhs(0,2)*rhs(2,2) ,
                   lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0) + lhs(1,2)*rhs(2,0) ,
                   lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1) + lhs(1,2)*rhs(2,1) ,
                   lhs(1,0)*rhs(0,2) + lhs(1,1)*rhs(1,2) + lhs(1,2)*rhs(2,2) ,
                   lhs(2,0)*rhs(0,0) + lhs(2,1)*rhs(1,0) + lhs(2,2)*rhs(2,0) ,
                   lhs(2,0)*rhs(0,1) + lhs(2,1)*rhs(1,1) + lhs(2,2)*rhs(2,1) ,
                   lhs(2,0)*rhs(0,2) + lhs(2,1)*rhs(1,2) + lhs(2,2)*rhs(2,2));
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tenzor dot SymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Tenzor<T1,D> , SymTenzor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,D>
  apply(const Tenzor<T1,D>& lhs, const SymTenzor<T2,D>& rhs) {
    Tenzor<T0,D> dot = typename Tenzor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i)
      for (unsigned int j=0; j<D; ++j) {
	T0 sum = lhs(i,0) * rhs.HL(j,0);
	unsigned int k=1;
	for (; k<j; ++k)
	  sum += lhs(i,k) * rhs.HL(j,k);
	for (; k<D; ++k)
	  sum += lhs(i,k) * rhs.HL(k,j);
	dot[i*D+j] = sum;
      }
    return dot;
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,1> , SymTenzor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,1>
  apply(const Tenzor<T1,1>& lhs, const SymTenzor<T2,1>& rhs) {
    return Tenzor<T0,1>(lhs[0]*rhs[0]);
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,2> , SymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,2>
  apply(const Tenzor<T1,2>& lhs, const SymTenzor<T2,2>& rhs) {
    return Tenzor<T0,2>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0),
			lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1),
			lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0),
			lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1));
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Tenzor<T1,3> , SymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Tenzor<T0,3>
  apply(const Tenzor<T1,3>& lhs, const SymTenzor<T2,3>& rhs) {
    return 
      Tenzor<T0,3>(lhs(0,0)*rhs(0,0) + lhs(0,1)*rhs(1,0) + lhs(0,2)*rhs(2,0) ,
                   lhs(0,0)*rhs(0,1) + lhs(0,1)*rhs(1,1) + lhs(0,2)*rhs(2,1) ,
                   lhs(0,0)*rhs(0,2) + lhs(0,1)*rhs(1,2) + lhs(0,2)*rhs(2,2) ,
                   lhs(1,0)*rhs(0,0) + lhs(1,1)*rhs(1,0) + lhs(1,2)*rhs(2,0) ,
                   lhs(1,0)*rhs(0,1) + lhs(1,1)*rhs(1,1) + lhs(1,2)*rhs(2,1) ,
                   lhs(1,0)*rhs(0,2) + lhs(1,1)*rhs(1,2) + lhs(1,2)*rhs(2,2) ,
                   lhs(2,0)*rhs(0,0) + lhs(2,1)*rhs(1,0) + lhs(2,2)*rhs(2,0) ,
                   lhs(2,0)*rhs(0,1) + lhs(2,1)*rhs(1,1) + lhs(2,2)*rhs(2,1) ,
                   lhs(2,0)*rhs(0,2) + lhs(2,1)*rhs(1,2) + lhs(2,2)*rhs(2,2));
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for Vektor dot AntiSymTenzor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< Vektor<T1,D> , AntiSymTenzor<T2,D> >
{
    typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
    inline static Vektor<T0,D>
    apply(const Vektor<T2,D>& lhs, const AntiSymTenzor<T1,D>& rhs) {
        Vektor<T0,D> ret = typename Vektor<T0,D>::DontInitialize();
        for (unsigned int j=0; j<D; ++j) {
            double sum = 0;
            for (unsigned int i=0; i<j; i++)
                sum -= lhs[i]*rhs[((j-1)*j/2)+i];
            for (unsigned int i=j+1; i<D; ++i)
                sum += lhs[i]*rhs[((i-1)*i/2)+j];
            ret[j] = sum;
        }
        return ret;
    }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,2> , AntiSymTenzor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,2>
  apply(const Vektor<T1,2>& lhs, const AntiSymTenzor<T2,2>& rhs) {
    return Vektor<T0,2>( lhs[0]*rhs(0,0) + lhs[1]*rhs(1,0) ,
			 lhs[0]*rhs(0,1) + lhs[1]*rhs(1,1) );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< Vektor<T1,3> , AntiSymTenzor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const Vektor<T1,3>& lhs, const AntiSymTenzor<T2,3>& rhs) {
    return Vektor<T0,3>( lhs[0]*rhs(0,0) + lhs[1]*rhs(1,0) + lhs[2]*rhs(2,0),
			 lhs[0]*rhs(0,1) + lhs[1]*rhs(1,1) + lhs[2]*rhs(2,1),
			 lhs[0]*rhs(0,2) + lhs[1]*rhs(1,2) + lhs[2]*rhs(2,2) );
  }
};

//////////////////////////////////////////////////////////////////////
//
// Specializations for AntiSymTenzor dot Vektor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct TSV_MetaDot< AntiSymTenzor<T1,D> , Vektor<T2,D> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,D>
  apply(const AntiSymTenzor<T1,D>& lhs, const Vektor<T2,D>& rhs) {
    Vektor<T0,D> ret = typename Vektor<T0,D>::DontInitialize();
    for (unsigned int i=0; i<D; ++i) {
      T0 sum = 0;
      for (unsigned int j=0; j<i; ++j)
	sum += lhs[((i-1)*i/2)+j]*rhs[j];
      for (unsigned int j=i+1; j<D; ++j)
	sum -= lhs[((j-1)*j/2)+i]*rhs[j];
      ret[i] = sum;
    }
    return ret;
  }
};


template<class T1, class T2>
struct TSV_MetaDot< AntiSymTenzor<T1,1> , Vektor<T2,1> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,1>
  apply(const AntiSymTenzor<T1,1>& lhs, const Vektor<T2,1>& rhs) {
    return Vektor<T0,1>( lhs[0]*rhs[0] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< AntiSymTenzor<T1,2> , Vektor<T2,2> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,2>
  apply(const AntiSymTenzor<T1,2>& lhs, const Vektor<T2,2>& rhs) {
    return Vektor<T0,2>( lhs(0,0)*rhs[0] + lhs(0,1)*rhs[1] ,
			 lhs(1,0)*rhs[0] + lhs(1,1)*rhs[1] );
  }
};

template<class T1, class T2>
struct TSV_MetaDot< AntiSymTenzor<T1,3> , Vektor<T2,3> >
{
  typedef typename PETEBinaryReturn<T1,T2,OpMultipply>::type T0;
  inline static Vektor<T0,3>
  apply(const AntiSymTenzor<T1,3>& lhs, const Vektor<T2,3>& rhs) {
    return Vektor<T0,3>( lhs(0,0)*rhs[0] + lhs(0,1)*rhs[1] + lhs(0,2)*rhs[2],
			 lhs(1,0)*rhs[0] + lhs(1,1)*rhs[1] + lhs(1,2)*rhs[2],
			 lhs(2,0)*rhs[0] + lhs(2,1)*rhs[1] + lhs(2,2)*rhs[2] );
  }
};

//////////////////////////////////////////////////////////////////////

#endif // TSV_META_DOT_H

/***************************************************************************
 * $RCSfile: TSVMetaDot.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: TSVMetaDot.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/

