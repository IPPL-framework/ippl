// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef VMAP_H
#define VMAP_H

//////////////////////////////////////////////////////////////////////
/*

  class vmap

  A class with the same interface as a map but implemented 
  with a sorted vector.

  This then has less overhead than a map for access but higher overhead 
  for insert and delete operations.  It is appropriate for cases where
  the contents of the vmap are read much more than they are written.

  Searching for a tag takes log time because a binary search is used.

  Insert and delete take linear time when inserting in the middle, but 
  constant time for inserting at the end.  The efficient way to fill 
  the map is to:
  1. Call reserve(size_type) with the number of elements you will be adding.
  2. Add elements in increasing order with end() as the hint.
  This will fill the vmap in linear time.

*/
//////////////////////////////////////////////////////////////////////

// include files
#include <vector>
#include <utility>
#include <functional>

//////////////////////////////////////////////////////////////////////

template<class Key>
class dummy_less
{
public:
  bool operator()(const Key l, const Key r) const
  {
    return l < r;
  }
};

//////////////////////////////////////////////////////////////////////

template<class Key, class T, class Compare = dummy_less<Key> >
class vmap
{
public:
  // typedefs:

  typedef Key key_type;
  typedef std::pair<Key, T> value_type;
  typedef Compare key_compare;
    
  class value_compare : public 

std::binary_function<value_type, value_type, bool>
  {
  private:
    Compare comp;
  public:
    value_compare(const Compare &c) : comp(c) {}
    bool operator()(const value_type& x, const value_type& y) const
    {
      return comp(x.first, y.first);
    }
  };

private:
  typedef std::vector< value_type > rep_type;

  // Here is the actual storage.
  rep_type V_c;

  // The comparator object and some convenient permutations.
  Compare Lt;
  bool Gt(const key_type& a, const key_type& b) const { return Lt(b,a); }
  bool Ge(const key_type& a, const key_type& b) const { return !Lt(a,b); }
  bool Le(const key_type& a, const key_type& b) const { return !Lt(b,a); }

public: 

  // More typedefs.

  //  typedef typename rep_type::pointer pointer;
  typedef typename rep_type::reference reference;
  typedef typename rep_type::const_reference const_reference;
  typedef typename rep_type::iterator iterator;
  typedef typename rep_type::const_iterator const_iterator;
  typedef typename rep_type::reverse_iterator reverse_iterator;
  typedef typename rep_type::const_reverse_iterator const_reverse_iterator;
  typedef typename rep_type::size_type size_type;
  typedef typename rep_type::difference_type difference_type;

  // accessors:

  iterator         begin()  { return V_c.begin(); }
  iterator         end()    { return V_c.end(); }
  reverse_iterator rbegin() { return V_c.rbegin(); }
  reverse_iterator rend()   { return V_c.rend(); }

  const_iterator         begin()  const { return V_c.begin(); }
  const_iterator         end()    const { return V_c.end(); }
  const_reverse_iterator rbegin() const { return V_c.rbegin(); }
  const_reverse_iterator rend()   const { return V_c.rend(); }

  key_compare   key_comp()   const { return Lt; }
  value_compare value_comp() const { return value_compare(Lt); }
  bool          empty()      const { return V_c.empty(); }
  size_type     size()       const { return V_c.size(); }
  size_type     max_size()   const { return V_c.max_size(); }
  size_type     capacity()   const { return V_c.capacity(); }

  void swap(vmap<Key, T, Compare>& x) { V_c.swap(x.V_c); }
  void reserve( size_type n ) { V_c.reserve(n); }

  // allocation/deallocation/assignment

  vmap(const Compare& comp = Compare()) : Lt(comp) {}
  vmap(const vmap<Key, T, Compare>& x);
  vmap<Key, T, Compare>& operator=(const vmap<Key, T, Compare>& x);

  // insert functions.

  std::pair<iterator,bool> insert(const value_type& x);
  iterator            insert(iterator hint_i, const value_type& x); 
  void                insert(const value_type* first, const value_type* last);

  // erase functions.

  void      erase(iterator position_i);
  void      erase(iterator first_i, iterator last_i);
  size_type erase(const key_type& x);

  // map operations:

  T&        operator[](const key_type& k);
  iterator  find(const key_type& x);      
  iterator  lower_bound(const key_type& x);
  iterator  upper_bound(const key_type& x);
  std::pair<iterator,iterator> equal_range(const key_type& x);

  // const map operations.

  size_type      count(const key_type& x) const;
  const T&       operator[](const key_type& k) const;
  const_iterator find(const key_type& x) const;
  const_iterator lower_bound(const key_type& x) const;
  const_iterator upper_bound(const key_type& x) const;
  std::pair<const_iterator,const_iterator> equal_range(const key_type& x) const;

};

template <class Key, class T, class Compare>
inline bool operator==(const vmap<Key, T, Compare>& x, 
                       const vmap<Key, T, Compare>& y) {
    return x.size() == y.size() && equal(x.begin(), x.end(), y.begin());
}

template <class Key, class T, class Compare>
inline bool operator<(const vmap<Key, T, Compare>& x, 
                      const vmap<Key, T, Compare>& y) {
    return lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}


//////////////////////////////////////////////////////////////////////

#include "Utility/vmap.hpp"

#endif // VMAP_H

/***************************************************************************
 * $RCSfile: vmap.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:34 $
 * IPPL_VERSION_ID: $Id: vmap.h,v 1.1.1.1 2003/01/23 07:40:34 adelmann Exp $ 
 ***************************************************************************/
