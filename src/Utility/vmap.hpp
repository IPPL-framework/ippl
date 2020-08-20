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
#include "Utility/vmap.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////
//
// Copy ctor
//
//////////////////////////////////////////////////////////////////////
template<class Key, class T, class Compare>
vmap<Key,T,Compare>::vmap( const vmap<Key,T,Compare>& x_ac )
  : Lt(x_ac.Lt)			// Copy the comparison object.
{
  
  
  V_c = x_ac.V_c;		// Copy the vector of data.
}

//////////////////////////////////////////////////////////////////////
//
// Assignment
//
//////////////////////////////////////////////////////////////////////
template<class Key, class T, class Compare>
vmap<Key,T,Compare>&
vmap<Key,T,Compare>::operator=( const vmap<Key,T,Compare>& x_ac )
{
  
  
  Lt = x_ac.Lt;			// Copy the comparison object.
  V_c = x_ac.V_c;		// Copy the vector of data.
  return *this;
}

//////////////////////////////////////////////////////////////////////
//
// Insert operations.
//
//////////////////////////////////////////////////////////////////////

//
// Insert a single value.
// Return ptr to the inserted and a bool for success.
//

template<class Key, class T, class Compare>
std::pair<typename vmap<Key,T,Compare>::iterator,bool>
vmap<Key,T,Compare>::insert(const value_type& value)
{
  iterator f_i;
  
  

  bool success;
  f_i = lower_bound( value.first );            // Look for it.
  if ( f_i == end() )			       // Are we at the end?
    {					       // Yes, we can't deref f_i.
      V_c.push_back(value);	               // Append. Could realloc
      f_i = begin() + size() - 1;
      success = true;
    } 
  // BFH: Ge -> Le
  else if ( Le((*f_i).first,value.first))      // Did you find it?
    {
      success = false;
    }
  else 
    {
      size_type n = f_i - begin();	       // Not there, remember location.
      V_c.insert( f_i , value );	       // Insert. Could cause realloc.
      f_i = begin() + n;
      success = true;
    }
  return std::pair<iterator,bool>(f_i,success);     // Return ptr to inserted.
}

//
// Insert a single value given a hint for where it should go.
// It is robust: even if the hint is wrong it will insert in the right place.
// If you start with values sorted in increasing order and give it 
// end() as the hint then it will insert in constant time.
// Return a ptr to the inserted position.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::iterator
vmap<Key,T,Compare>::insert(iterator hint_i, const value_type& value)
{
    
    
    iterator low_i = begin();	// The bounds for the search range
    iterator high_i = end();	// to find where to really put it.
    if ( hint_i == high_i ) {	// Is the hint to append?
        if ( empty() || Ge( value.first , hint_i[-1].first ) ) { // Is that right? Yes!
            V_c.push_back(value);	// Quick append.  Could cause realloc!
            return end()-1;		// return where we put it.
        }
    }
    else {				                // Hint is not for append.
        if ( Lt(value.first,(*hint_i).first) )      // Is it to the left?
            high_i = hint_i;                          //   range = (begin,hint)
        else			                // It is to the right.
            low_i = hint_i;		                //   range = (hint,end)
    }
    // Given the range figured out above, find where to really put it.
    hint_i = std::upper_bound( low_i, high_i, value, value_compare(Lt) );
    size_type n = hint_i - begin();            // Remember the location.
    V_c.insert(hint_i,value);                  // Insert here, could realloc.
    return V_c.begin() + n;	             // return iterator.
}

//
// Insert a range of values, with append as the hint.
// If the values being inserted are sorted and all greater than or equal
// to the elements in the vmap this works out to a fast append.
//

template<class Key, class T, class Compare>
void
vmap<Key,T,Compare>::insert(const value_type *first_i,const value_type *last_i)
{
  
  
  for (; first_i != last_i; ++first_i)
    insert( end() , *first_i );
}

//////////////////////////////////////////////////////////////////////
//
// erase operations.
//
//////////////////////////////////////////////////////////////////////

//
// Erase a single element.
//

template<class Key, class T, class Compare>
void
vmap<Key,T,Compare>::erase(iterator p_i)
{
  
  
  V_c.erase(p_i);
}

//
// Erase a range of elements.
//

template<class Key, class T, class Compare>
void
vmap<Key,T,Compare>::erase(iterator first_i, iterator last_i)
{
  
  
  V_c.erase(first_i, last_i);
}

//
// Erase all the values with key equal to the given.
// Return the number erased.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::size_type
vmap<Key,T,Compare>::erase(const key_type& key)
{
  
  
  std::pair<iterator,iterator> range( equal_range(key) );
  erase(range.first, range.second);
  return range.second - range.first;
}

//////////////////////////////////////////////////////////////////////
//
// Map operations.
//
//////////////////////////////////////////////////////////////////////

//
// Find a value if it is there.
// Return end() if it is not.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::iterator
vmap<Key,T,Compare>::find(const key_type& key)
{
  
  
  iterator f_i = lower_bound(key); // Look for it.
  if ( f_i == end() )		   // Are we at the end?
    return end();		   //   Yes, don't try to deref.
  // BFH: Ge -> Le
  if ( Le((*f_i).first , key) )	   // Did you find it?
    return f_i;			   //   yes, return it.
  else				   // or...
    return end();		   //   no, return end().
}

//
// Return ptr to the first element that is >= the given one.
// Can also think of it as the first place it could be put
// without violating the sequence.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::iterator
vmap<Key,T,Compare>::lower_bound(const key_type& key)
{
  
  
  return std::lower_bound(begin(),end(),value_type(key,T()),value_compare(Lt));
}

//
// Return ptr to the first element > the given one.
// Can also think of it as the last place it could be put
// without violating the sequence.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::iterator
vmap<Key,T,Compare>::upper_bound(const key_type& key)
{
  
  
  return std::upper_bound(begin(),end(),value_type(key,T()),value_compare(Lt));
}

//
// Return a lower_bound, upper_bound pair.
//

template<class Key, class T, class Compare>
std::pair<typename vmap<Key,T,Compare>::iterator,
          typename vmap<Key,T,Compare>::iterator>
vmap<Key,T,Compare>::equal_range(const key_type& key)
{
  
  return std::equal_range(begin(),end(),value_type(key,T()),value_compare(Lt));
}

//
// Look up an element given a key.
// If there is not one there already, make space for it.
//

template<class Key, class T, class Compare>
T&
vmap<Key,T,Compare>::operator[](const key_type& key)
{
  
  
  size_type n;			// We will need to calculate the offset.
  iterator f_i;			// Pointer to the data we find.
  if ( V_c.empty() )		// Is there anything here?
    {				//    No.  Can't dereference.
      n = 0;			//    The new one will be at the beginning.
      f_i = begin();		//    Point to the beginning.
    }
  else				       // Yes, there is something here.
    {				       // 
      f_i = lower_bound(key);	       // Search the whole range.
      // BFH: Ge -> Le
      if ((f_i!=end()) && Le((*f_i).first,key)) // Did we find this one?
	return (*f_i).second;	       //    Yes, return ref to data.
      n = f_i - begin();	       // Not there, remember location.
    }
  V_c.insert( f_i , value_type(key,T()) ); // Insert. Could cause realloc.
  return V_c[n].second;		           // Return ref to data.
}

//////////////////////////////////////////////////////////////////////
//
// Const map operations.
//
//////////////////////////////////////////////////////////////////////

//
// Find how many have the given key.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::size_type
vmap<Key,T,Compare>::count(const key_type& key) const
{
  
  
  std::pair<const_iterator,const_iterator>
    range( equal_range(key) );	     // Find the bounds.
  return range.second - range.first; // Difference to get the count.
}

//
// Find a value if it is there.
// Return end() if it is not.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::const_iterator
vmap<Key,T,Compare>::find(const key_type& key) const
{
  
  
  const_iterator f_i = lower_bound(key); // Look for it.
  // BFH: Ge -> Le
  if ( Le((*f_i).first , key ))	         // Did you find it?
    return f_i;			         //   yes, return it.
  else				         // or...
    return end();		         //   no, return end().
}

//
// Return ptr to the first element that is >= the given one.
// Can also think of it as the first place it could be put
// without violating the sequence.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::const_iterator
vmap<Key,T,Compare>::lower_bound(const key_type& key) const
{
  
  
  return std::lower_bound(begin(),end(),value_type(key,T()),value_compare(Lt));
}

//
// Return ptr to the first element > the given one.
// Can also think of it as the last place it could be put
// without violating the sequence.
//

template<class Key, class T, class Compare>
typename vmap<Key,T,Compare>::const_iterator
vmap<Key,T,Compare>::upper_bound(const key_type& key) const
{
  
  
  return std::upper_bound(begin(),end(),value_type(key,T()),value_compare(Lt));
}

//
// Return a lower_bound, upper_bound pair.
//

template<class Key, class T, class Compare>
std::pair<typename vmap<Key,T,Compare>::const_iterator,
          typename vmap<Key,T,Compare>::const_iterator>
vmap<Key,T,Compare>::equal_range(const key_type& key) const
{
  
  return std::equal_range(begin(),end(),value_type(key,T()),value_compare(Lt));
}


//
// Look up an element given a key.
// Scream and die if it is not there.
//

template<class Key, class T, class Compare>
const T&
vmap<Key,T,Compare>::operator[](const key_type& key) const
{
  
  
  const_iterator f_i = lower_bound(key);   // Search the whole range.
  return (*f_i).second;		     // Return what you found.
}

/***************************************************************************
 * $RCSfile: vmap.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:34 $
 * IPPL_VERSION_ID: $Id: vmap.cpp,v 1.1.1.1 2003/01/23 07:40:34 adelmann Exp $ 
 ***************************************************************************/
