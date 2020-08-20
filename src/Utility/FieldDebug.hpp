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

// FieldDebug.cpp , Tim Williams 10/23/1996

// include files
#include "Utility/FieldDebug.h"
#include "Utility/FieldDebugPrint.h"
#include "Utility/Inform.h"

#include "Field/BareField.h"

#include <iostream>
#include <iomanip> // need format fcns setf() and setprecision() from here


//----------------------------------------------------------------------
// Print a 1D Field
//----------------------------------------------------------------------
template<class T>
void fp1(BareField<T,1U>& field, bool docomm) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  int base0   = field.getLayout().getDomain()[0].first();
  int bound0  = field.getLayout().getDomain()[0].last();
  int stride0 = field.getLayout().getDomain()[0].stride();
  sfp1(field, base0, bound0, stride0, docomm);
}
//----------------------------------------------------------------------
// Print a 2D Field
//----------------------------------------------------------------------
template<class T>
void fp2(BareField<T,2U>& field, bool docomm) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  int base0   = field.getLayout().getDomain()[0].first();
  int bound0  = field.getLayout().getDomain()[0].last();
  int stride0 = field.getLayout().getDomain()[0].stride();
  int base1   = field.getLayout().getDomain()[1].first();
  int bound1  = field.getLayout().getDomain()[1].last();
  int stride1 = field.getLayout().getDomain()[1].stride();
  sfp2(field, base0, bound0, stride0, base1, bound1, stride1, docomm);
}
//----------------------------------------------------------------------
// Print a 3D Field
//----------------------------------------------------------------------
template<class T>
void fp3(BareField<T,3U>& field, bool docomm) {
  
  
  int base0   = field.getLayout().getDomain()[0].first();
  int bound0  = field.getLayout().getDomain()[0].last();
  int stride0 = field.getLayout().getDomain()[0].stride();
  int base1   = field.getLayout().getDomain()[1].first();
  int bound1  = field.getLayout().getDomain()[1].last();
  int stride1 = field.getLayout().getDomain()[1].stride();
  int base2   = field.getLayout().getDomain()[2].first();
  int bound2  = field.getLayout().getDomain()[2].last();
  int stride2 = field.getLayout().getDomain()[2].stride();
  sfp3(field, base0, bound0, stride0, base1, bound1, stride1, 
       base2, bound2, stride2, docomm);
}
//----------------------------------------------------------------------
// Print a 1D Field, including global guard layers
//----------------------------------------------------------------------
template<class T>
void ggfp1(BareField<T,1U>& field, bool docomm) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  int stride0 = field.getLayout().getDomain()[0].stride();
  int base0   = field.getLayout().getDomain()[0].first() - 
    field.leftGuard(0)*stride0;
  int bound0  = field.getLayout().getDomain()[0].last() + 
    field.leftGuard(0)*stride0;
  sfp1(field, base0, bound0, stride0, docomm);
}
//----------------------------------------------------------------------
// Print a 2D Field, including global guard layers
//----------------------------------------------------------------------
template<class T>
void ggfp2(BareField<T,2U>& field, bool docomm) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  int stride0 = field.getLayout().getDomain()[0].stride();
  int base0   = field.getLayout().getDomain()[0].first() - 
    field.leftGuard(0)*stride0;
  int bound0  = field.getLayout().getDomain()[0].last() + 
    field.leftGuard(0)*stride0;
  int stride1 = field.getLayout().getDomain()[1].stride();
  int base1   = field.getLayout().getDomain()[1].first() - 
    field.leftGuard(1)*stride1;
  int bound1  = field.getLayout().getDomain()[1].last() + 
    field.leftGuard(1)*stride1;
  sfp2(field, base0, bound0, stride0, base1, bound1, stride1, docomm);
}
//----------------------------------------------------------------------
// Print a 3D Field, including global guard layers
//----------------------------------------------------------------------
template<class T>
void ggfp3(BareField<T,3U>& field, bool docomm) {
  
  
  int stride0 = field.getLayout().getDomain()[0].stride();
  int base0   = field.getLayout().getDomain()[0].first() - 
    field.leftGuard(0)*stride0;
  int bound0  = field.getLayout().getDomain()[0].last() + 
    field.leftGuard(0)*stride0;
  int stride1 = field.getLayout().getDomain()[1].stride();
  int base1   = field.getLayout().getDomain()[1].first() - 
    field.leftGuard(1)*stride1;
  int bound1  = field.getLayout().getDomain()[1].last() + 
    field.leftGuard(1)*stride1;
  int stride2 = field.getLayout().getDomain()[2].stride();
  int base2   = field.getLayout().getDomain()[2].first() - 
    field.leftGuard(2)*stride2;
  int bound2  = field.getLayout().getDomain()[2].last() + 
    field.leftGuard(2)*stride2;
  sfp3(field, base0, bound0, stride0, base1, bound1, stride1, 
       base2, bound2, stride2, docomm);
}
//----------------------------------------------------------------------
// Print a 1D Field, including global and internal guard layers
//----------------------------------------------------------------------
template<class T>
void agfp1(BareField<T,1U>& field) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  FieldDebugWriteb(field);
}
//----------------------------------------------------------------------
// Print a 2D Field, including global and internal guard layers
//----------------------------------------------------------------------
template<class T>
void agfp2(BareField<T,2U>& field) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  FieldDebugWriteb(field);
}
//----------------------------------------------------------------------
// Print a 3D Field, including global and internal guard layers
//----------------------------------------------------------------------
template<class T>
void agfp3(BareField<T,3U>& field) {
  
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.
  FieldDebugWriteb(field);
}
//----------------------------------------------------------------------
// Print a single element of a 1D Field
//----------------------------------------------------------------------
template<class T>
void efp1(BareField<T,1U>& field, int i, bool docomm) {
  
  
  sfp1(field, i, i, 1, docomm);
}
//----------------------------------------------------------------------
// Print a single element of a 2D Field
//----------------------------------------------------------------------
template<class T>
void efp2(BareField<T,2U>& field, int i, int j, bool docomm) {
  
  
  sfp2(field, i, i, 1, j, j, 1, docomm);
}
//----------------------------------------------------------------------
// Print a single element of a 3D Field
//----------------------------------------------------------------------
template<class T>
void efp3(BareField<T,3U>& field, int i, int j, int k, bool docomm) {
  
  
  sfp3(field, i, i, 1, j, j, 1, k, k, 1, docomm);
}
//----------------------------------------------------------------------
// Print a strided slice (range) of a 1D Field
//----------------------------------------------------------------------
template<class T>
void sfp1(BareField<T,1U>& field,
	  int ibase, int ibound, int istride, bool docomm) {
  
  

  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.

  // Check input parameters for errors and unimplemented values:
  bool okParameters = true;
  int first0 = field.getLayout().getDomain()[0].first() - 
    field.leftGuard(0) * field.getDomain()[0].stride();
  int last0 = field.getLayout().getDomain()[0].last() + field.leftGuard(0);
  if (ibase < first0) {
    (*FldDbgInform) << "sfp() error: ibase (= " << ibase
		    << ") < lowest index value (= " << first0 << ")" << endl;
    okParameters = false;
  }
  if (ibound > last0) {
    (*FldDbgInform) << "sfp() error: ibound (= " << ibound 
		    << ") > highest index value (= " << last0 << ")" << endl;
    okParameters = false;
  }
  if (istride < 0) {
    (*FldDbgInform) << "sfp() error: istride < 0 not implemented yet." << endl;
    okParameters = false;
  } else {
    if (ibound < ibase) {
      (*FldDbgInform) << "sfp() error: ibase (= " << ibase
		      << ") > ibound (=  " << ibound << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (istride == 0) {
    if ( (ibound - ibase) != 0 ) {
      (*FldDbgInform) << "sfp() error: istride = 0 but (ibound - ibase) = " 
		      << (ibound - ibase) << endl;
      okParameters = false;
    } else {
      istride = 1; // Allow specifying stride 0 for 1-element range; set=1 
    }
  }

  if (okParameters) {
    NDIndex<1U> ndi(Index(ibase,ibound,istride));
    FieldDebugPrint<T,1U> dfp(true,widthOfElements,digitsPastDecimal,
			      elementsPerLine);
    dfp.print(field, ndi, *FldDbgInform, docomm);
  }
}
//----------------------------------------------------------------------
// Print a strided slice (range) of a 2D Field
//----------------------------------------------------------------------
template<class T>
void sfp2(BareField<T,2U>& field,
	  int ibase, int ibound, int istride,
	  int jbase, int jbound, int jstride, bool docomm) {
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.

  // Check input parameters for errors and unimplemented values:
  bool okParameters = true;
  int first0 = field.getLayout().getDomain()[0].first() - 
    field.leftGuard(0) * field.getDomain()[0].stride();
  int last0 = field.getLayout().getDomain()[0].last() + field.leftGuard(0);
  if (ibase < first0) {
    (*FldDbgInform) << "sfp() error: ibase (= " << ibase
		    << ") < lowest index value (= " << first0 << ")" << endl;
    okParameters = false;
  }
  if (ibound > last0) {
    (*FldDbgInform) << "sfp() error: ibound (= " << ibound 
		    << ") > highest index value (= " << last0 << ")" << endl;
    okParameters = false;
  }
  if (istride < 0) {
    (*FldDbgInform) << "sfp() error: istride < 0 not implemented yet." << endl;
    okParameters = false;
  } else {
    if (ibound < ibase) {
      (*FldDbgInform) << "sfp() error: ibase (= " << ibase
		      << ") > ibound (=  " << ibound << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (istride == 0) {
    if ( (ibound - ibase) != 0 ) {
      (*FldDbgInform) << "sfp() error: istride = 0 but (ibound - ibase) = " 
		      << (ibound - ibase) << endl;
      okParameters = false;
    } else {
      istride = 1; // Allow specifying stride 0 for 1-element range; set=1 
    }
  }
  int first1 = field.getLayout().getDomain()[1].first() - 
    field.leftGuard(1) * field.getDomain()[1].stride();
  int last1 = field.getLayout().getDomain()[1].last() + field.leftGuard(1);
  if (jbase < first1) {
    (*FldDbgInform) << "sfp() error: jbase (= " << jbase
		    << ") < lowest index value (= " << first1 << ")" << endl;
    okParameters = false;
  }
  if (jbound > last1) {
    (*FldDbgInform) << "sfp() error: jbound (= " << jbound 
		    << ") > highest index value (= " << last1 << ")" << endl;
    okParameters = false;
  }
  if (jstride < 0) {
    (*FldDbgInform) << "sfp() error: jstride < 0 not implemented yet." << endl;
    okParameters = false;
  } else {
    if (jbound < jbase) {
      (*FldDbgInform) << "sfp() error: jbase (= " << jbase
		      << ") > jbound (=  " << jbound << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (jstride == 0) {
    if ( (jbound - jbase) != 0 ) {
      (*FldDbgInform) << "sfp() error: jstride = 0 but (jbound - jbase) = " 
		      << (jbound - jbase) << endl;
      okParameters = false;
    } else {
      jstride = 1; // Allow specifying stride 0 for 1-element range; set=1 
    }
  }

  if (okParameters) {
    NDIndex<2U> ndi(Index(ibase,ibound,istride),Index(jbase,jbound,jstride));
    FieldDebugPrint<T,2U> dfp(true,widthOfElements,digitsPastDecimal,
			      elementsPerLine);
    dfp.print(field, ndi, *FldDbgInform, docomm);
  }
}
//----------------------------------------------------------------------
// Print a strided slice (range) of a 3D Field
//----------------------------------------------------------------------
template<class T>
void sfp3(BareField<T,3U>& field,
	  int ibase, int ibound, int istride,
	  int jbase, int jbound, int jstride,
	  int kbase, int kbound, int kstride, bool docomm) {
  
  if (!FldDbgInformIsSet) setInform(*IpplInfo::Info); // Set ptr if not set.

  // Check input parameters for errors and unimplemented values:
  bool okParameters = true;
  int first0 = field.getLayout().getDomain()[0].first() - 
    field.leftGuard(0) * field.getDomain()[0].stride();
  int last0 = field.getLayout().getDomain()[0].last() + field.leftGuard(0);
  if (ibase < first0) {
    (*FldDbgInform) << "sfp() error: ibase (= " << ibase
		    << ") < lowest index value (= " << first0 << ")" << endl;
    okParameters = false;
  }
  if (ibound > last0) {
    (*FldDbgInform) << "sfp() error: ibound (= " << ibound 
		    << ") > highest index value (= " << last0 << ")" << endl;
    okParameters = false;
  }
  if (istride < 0) {
    (*FldDbgInform) << "sfp() error: istride < 0 not implemented yet." << endl;
    okParameters = false;
  } else {
    if (ibound < ibase) {
      (*FldDbgInform) << "sfp() error: ibase (= " << ibase
		      << ") > ibound (=  " << ibound << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (istride == 0) {
    if ( (ibound - ibase) != 0 ) {
      (*FldDbgInform) << "sfp() error: istride = 0 but (ibound - ibase) = " 
		      << (ibound - ibase) << endl;
      okParameters = false;
    } else {
      istride = 1; // Allow specifying stride 0 for 1-element range; set=1 
    }
  }
  int first1 = field.getLayout().getDomain()[1].first() - 
    field.leftGuard(1) * field.getDomain()[1].stride();
  int last1 = field.getLayout().getDomain()[1].last() + field.leftGuard(1);
  if (jbase < first1) {
    (*FldDbgInform) << "sfp() error: jbase (= " << jbase
		    << ") < lowest index value (= " << first1 << ")" << endl;
    okParameters = false;
  }
  if (jbound > last1) {
    (*FldDbgInform) << "sfp() error: jbound (= " << jbound 
		    << ") > highest index value (= " << last1 << ")" << endl;
    okParameters = false;
  }
  if (jstride < 0) {
    (*FldDbgInform) << "sfp() error: jstride < 0 not implemented yet." << endl;
    okParameters = false;
  } else {
    if (jbound < jbase) {
      (*FldDbgInform) << "sfp() error: jbase (= " << jbase
		      << ") > jbound (=  " << jbound << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (jstride == 0) {
    if ( (jbound - jbase) != 0 ) {
      (*FldDbgInform) << "sfp() error: jstride = 0 but (jbound - jbase) = " 
		      << (jbound - jbase) << endl;
      okParameters = false;
    } else {
      jstride = 1; // Allow specifying stride 0 for 1-element range; set=1 
    }
  }
  int first2 = field.getLayout().getDomain()[2].first() - 
    field.leftGuard(2) * field.getDomain()[2].stride();
  int last2 = field.getLayout().getDomain()[2].last() + field.leftGuard(2);
  if (kbase < first2) {
    (*FldDbgInform) << "sfp() error: kbase (= " << kbase
		    << ") < lowest index value (= " << first2 << ")" << endl;
    okParameters = false;
  }
  if (kbound > last2) {
    (*FldDbgInform) << "sfp() error: kbound (= " << kbound 
		    << ") > highest index value (= " << last2 << ")" << endl;
    okParameters = false;
  }
  if (kstride < 0) {
    (*FldDbgInform) << "sfp() error: kstride < 0 not implemented yet." << endl;
    okParameters = false;
  } else {
    if (kbound < kbase) {
      (*FldDbgInform) << "sfp() error: kbase (= " << kbase
		      << ") > kbound (=  " << jbound << ") not implemented yet." << endl;
      okParameters = false;
    }
  }
  if (kstride == 0) {
    if ( (kbound - kbase) != 0 ) {
      (*FldDbgInform) << "sfp() error: kstride = 0 but (kbound - kbase) = " 
		      << (kbound - kbase) << endl;
      okParameters = false;
    } else {
      kstride = 1; // Allow specifying stride 0 for 1-element range; set=1 
    }
  }

  if (okParameters) {
    NDIndex<3U> ndi(Index(ibase,ibound,istride),Index(jbase,jbound,jstride),
		    Index(kbase,kbound,kstride));
    FieldDebugPrint<T,3U> dfp(true,widthOfElements,digitsPastDecimal,
			      elementsPerLine);
    dfp.print(field, ndi, *FldDbgInform, docomm);
  }
}

//----------------------------------------------------------------------
// An output function which writes out a BareField a vnode at a time and
// includes the border information, using fp[1,2,3]-like formatting as much as
// possible. Patterned after BareField::writeb()
//----------------------------------------------------------------------

template< class T, unsigned Dim >
inline void FieldDebugWriteb(BareField<T,Dim>& F)
{
  
  

  int ibase,ibound,istride,jbase,jbound,jstride,kbase,kbound,kstride;
  istride = F.getLayout().getDomain()[0].stride();
  ibase   = F.getLayout().getDomain()[0].first() - F.leftGuard(0)*istride;
  ibound  = F.getLayout().getDomain()[0].last() + F.leftGuard(0)*istride;
  if (Dim >= 2) {
    jstride = F.getLayout().getDomain()[1].stride();
    jbase   = F.getLayout().getDomain()[1].first() - F.leftGuard(1)*jstride;
    jbound  = F.getLayout().getDomain()[1].last() + F.leftGuard(1)*jstride;
  }
  if (Dim >= 3) {
    kstride = F.getLayout().getDomain()[2].stride();
    kbase   = F.getLayout().getDomain()[2].first() - F.leftGuard(2)*kstride;
    kbound  = F.getLayout().getDomain()[2].last() + F.leftGuard(2)*kstride;
  }
  (*FldDbgInform) << "~~~~~~~~ field slice ("
		  << ibase << ":" << ibound << ":" << istride <<  ", "
		  << jbase << ":" << jbound << ":" << jstride <<  ", "
		  << kbase << ":" << kbound << ":" << kstride <<  ") "
		  << "~~~~~~~~" << endl;
  int icount = 0;
  int iBlock;
  typename BareField<T,Dim>::const_iterator_if l_i;
  for (l_i = F.begin_if(); l_i != F.end_if(); ++l_i) {
    // find the offset within the global space
    LField<T,Dim> *ldf = (*l_i).second.get();
    const NDIndex<Dim> &Owned = ldf->getOwned();

    (*FldDbgInform) << "****************************************"
		    << "***************************************" << endl;
    (*FldDbgInform) << "********* vnode = " << icount++ << " *********" 
		    << endl;
    (*FldDbgInform) << "****************************************"
		    << "***************************************" << endl;
    (*FldDbgInform) << "Owned = " << endl << Owned << endl;
    typename LField<T,Dim>::iterator lf_bi = ldf->begin();
    if ( Dim==1 ) {
      int n0 = ldf->size(0);
      int l0 = -F.leftGuard(0);   // tjw: assumes stride 1, right???
      int r0 = n0 + F.rightGuard(0);
      int ifirst = Owned[0].first() - F.leftGuard(0)*istride;
      int ilast = Owned[0].last() + F.rightGuard(0);
      (*FldDbgInform) << "- - - - - - - - - - - - - - - - - - - - - - - - - "
		      << "I = " << ifirst << ":" << ilast << ":"
		      << istride << endl;
      for ( iBlock = l0 ; iBlock < r0 ; iBlock += elementsPerLine) {
	for (int i0=iBlock; 
	     ((i0 < iBlock + elementsPerLine*istride) && (i0 < r0)); 
	     i0 = i0 + istride) {
	  (*FldDbgInform) << std::setprecision(digitsPastDecimal) 
			  << std::setw(widthOfElements)
			  << lf_bi.offset(i0) << " ";
	}
	(*FldDbgInform) << endl;
      }
    } else if ( Dim==2 ) {
      int n0 = ldf->size(0);
      int n1 = ldf->size(1);
      int l0 = -F.leftGuard(0);
      int l1 = -F.leftGuard(1);
      int r0 = n0 + F.rightGuard(0);
      int r1 = n1 + F.rightGuard(1);
      int j = 0;
      int ifirst = Owned[0].first() - F.leftGuard(0)*istride;
      int ilast = Owned[0].last() + F.rightGuard(0);
      int jfirst = Owned[1].first();
      // (re)define jbase, jbound, jlast w/o guards
      int jlast = Owned[1].last();
      jbase = F.getLayout().getDomain()[1].first();
      jbound  = F.getLayout().getDomain()[1].last();
      for (int i1=l1; i1<r1; ++i1) {
	j = jfirst + i1;
	if ((j < jfirst) || (j > jlast)) {
	  if ((j < jbase) || (j > jbound)) {
	    (*FldDbgInform) 
	      << "--------------------------------global guard------";
	  } else {
	    (*FldDbgInform) 
	      << "---------------------------------------guard------";
	  }
	} else {
	  (*FldDbgInform) 
	    << "--------------------------------------------------";
	}
	(*FldDbgInform) << "- - - - - - - - - - - - - - - - - - - - - - - - - "
			<< "I = " << ifirst << ":" << ilast << ":" 
			<< istride << endl;
	for ( iBlock = l0 ; iBlock < r0 ; iBlock += elementsPerLine) {
	  for (int i0=iBlock; 
	       ((i0 < iBlock + elementsPerLine*istride) && (i0 < r0)); 
	       i0 = i0 + istride) {
	    (*FldDbgInform) << std::setprecision(digitsPastDecimal) 
			    << std::setw(widthOfElements)
			    << lf_bi.offset(i0,i1) << " ";
	  }
	  (*FldDbgInform) << endl;
	}
	(*FldDbgInform) << endl;
      }
    } else if ( Dim==3 ) {
      int n0 = ldf->size(0);
      int n1 = ldf->size(1);
      int n2 = ldf->size(2);
      int l0 = -F.leftGuard(0);
      int l1 = -F.leftGuard(1);
      int l2 = -F.leftGuard(2);
      int r0 = n0 + F.rightGuard(0);
      int r1 = n1 + F.rightGuard(1);
      int r2 = n2 + F.rightGuard(2);
      int j = 0;
      int k = 0;
      int ifirst = Owned[0].first() - F.leftGuard(0)*istride;
      int ilast = Owned[0].last() + F.rightGuard(0);
      int jfirst = Owned[1].first();
      // (re)define jbase, jbound, jlast w/o guards
      int jlast = Owned[1].last();
      jbase = F.getLayout().getDomain()[1].first();
      jbound  = F.getLayout().getDomain()[1].last();
      int kfirst = Owned[2].first();
      // (re)define kbase, kbound, klast w/o guards
      int klast = Owned[2].last();
      kbase = F.getLayout().getDomain()[2].first();
      kbound  = F.getLayout().getDomain()[2].last();
      for (int i2=l2; i2<r2; ++i2) {
	k = kfirst + i2;
	if ((k < kfirst) || (k > klast)) {
	  if ((k < kbase) || (k > kbound)) {
	    (*FldDbgInform) 
	      << "================================global guard======";
	  } else {
	    (*FldDbgInform) 
	      << "=======================================guard======";
	  }
	} else {
	  (*FldDbgInform) 
	    << "==================================================";
	}
	(*FldDbgInform) << "K = " << k << endl;
	for (int i1=l1; i1<r1; ++i1) {
	  j = jfirst + i1;
	  if ((j < jfirst) || (j > jlast)) {
	    if ((j < jbase) || (j > jbound)) {
	      (*FldDbgInform) 
		<< "--------------------------------global guard------";
	    } else {
	      (*FldDbgInform) 
		<< "---------------------------------------guard------";
	    }
	  } else {
	    (*FldDbgInform) 
	      << "--------------------------------------------------";
	  }
	  (*FldDbgInform) << "J = " << j << endl;
	  (*FldDbgInform) 
	    << "- - - - - - - - - - - - - - - - - - - - - - - - - "
	    << "I = " << ifirst << ":" << ilast << ":" << istride << endl;
	  for ( iBlock = l0 ; iBlock < r0 ; iBlock += elementsPerLine) {
	    for (int i0=iBlock; 
		 ((i0 < iBlock + elementsPerLine*istride) && (i0 < r0)); 
		 i0 = i0 + istride) {
	      (*FldDbgInform) << std::setprecision(digitsPastDecimal) 
			      << std::setw(widthOfElements)
			      << lf_bi.offset(i0,i1,i2) << " ";
	    }
	    (*FldDbgInform) << endl;
	  }
	  (*FldDbgInform) << endl;
	}
	(*FldDbgInform) << endl;
      }
    } else {
      ERRORMSG(" can not write for larger than three dimensions " << endl);
    }
  }
}


/***************************************************************************
 * $RCSfile: FieldDebug.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: FieldDebug.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
