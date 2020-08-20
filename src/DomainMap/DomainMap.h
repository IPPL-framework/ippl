// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef DOMAIN_MAP_H
#define DOMAIN_MAP_H

/***********************************************************************

Steve Karmesin
Dakota Scientific Software
February 1996

A DomainMap holds a bunch of Domains of type Key in such a way that you
can quickly look up which Keys's touch a given one and an object 
associated with that Key.

Key must have the following operations defined:

bool touches(const Key&, const Key&);
Return true if the Keys touch, false if they don't.

bool contains(const Key& l, const Key& r);
Return true if l contains r, false otherwise.

bool split(pair<const Key,const Key>&, const Key& k);
Split one Key into two, and return a boolean for success.

split and containes need to have the following properties for 
consistent behavior:

1. If Key A is split into A1 and A2, then 
   contains(A,A1)==true and contains(A,A2)==true.

2. If Key A is split into A1 and A2, then there is no Key B 
   for which contains(A1,B) and contains(A2,B) are both true.

3. If contains(A,B)==true and contains(B,C)==true then we must have
   contains(A,C)==true.

Each Key is associated with a T. We enter those Ts into the
DomainMap and later pull out the ones that touch a given Key.

touches(const Key&) must do two things:
1) It must be fast.
2) It must be right.

"Fast" means that it has to be at worst linear in the number of hits,
times a log of the total number of elements in the Map.

"Right" means that it has to find all of the elements that touch the
given Key exactly once.

This currently has very small subset of the capabilites of an STL map..
It should gradually get expanded to include as many as make sense. 

***********************************************************************/

// include files
#include "Utility/PAssert.h"

#include <list>
#include <iterator>
#include <utility>
#include <cstddef>

template < class Key, class T , class Touches, class Contains, class Split > 
class DomainMap 
{
public:
    typedef DomainMap<Key,T,Touches,Contains,Split> DomainMap_t;
    typedef unsigned size_type;
    
    typedef Key key_type;
    typedef T   mapped_type;
    
    struct value_type {
        Key first;
        T   second;
        value_type() {}
        value_type(const Key& k, const T& t)
                : first(k), 
                  second(t){
        }
    };
    

private:
    // A class for the nodes of the tree.
    // Not visible from outside of DomainMap
    class Node
    {
    public: 
        typedef std::list<value_type> cont_type; // The type for the container
        Key MyDomain;			  // This Node's Domain.
        Node *Left,*Right;		  // Make this a binary tree.
        Node *Parent;			  // Have this around for traversals.
        cont_type cont;			  // The values in this Node.
        
        // Construct with the domain and the parent.
        Node(const Key& d, Node* p=0) : MyDomain(d),Left(0),Right(0),Parent(p) {}
        // When you die take you children with you.
        ~Node() {
            if (Left ) delete Left;
            if (Right) delete Right;
        }

        // Insert a pair<Key,T> into the tree.
        // tjw: added noSplit flag 3/24/1998
        //mwerks      void insert(const value_type& d, bool noSplit=false);
        //////////////////////////////////////////////////////////////////////
        // Insert a T into the tree.
        // This isn't nearly as bad as it looks. It is just careful
        // to not build any nodes it doesn't have to, and to minimize
        // the number of times it calls split.  There is some duplicated
        // code here to keep the threading through the if statements smooth.
        void  insert(const value_type& d, bool noSplit=false) {
                
            Key left_domain;		// When splitting a node, we'll need a spot
            Key right_domain;		// to store the left and right domains.
                
            // ----------------------------------------------------------------------
            // tjw:
            // First, make sure that the domain isn't of total extent 1, which will
            // cause an assertion failure in split() functions like Index::split() in
            // the code below. For this special case, just insert straightaway. Use
            // the newly-added flag noSplit to request this; noSplit defaults to
            // false, so no existing code should break:
                
            if (noSplit) {
                cont.push_back(d);
                return;
            }
                
            // tjw.
            //-----------------------------------------------------------------------
                
            if ( Left && Right ) {	
                // If they both exist already the algorithm is simple.
                if ( Contains::test(Left->MyDomain,d.first) )
                    Left->insert(d);
                else if ( Contains::test(Right->MyDomain,d.first) )
                    Right->insert(d);
                else
                    cont.push_back(d);
                    
            }
            else if ( Left==0 && Right==0 ) {
                // If neither exist, split and check.
                if ( Split::test(left_domain, right_domain, MyDomain) ) {
                    if ( Contains::test(left_domain, d.first) ) {
                        // It is on the left. Build it and go.
                        Left = new Node( left_domain, this );
                        Left->insert(d);
                    }
                    else if ( Contains::test(right_domain, d.first) ) {
                        // It is on the right. Build it and go.
                        Right = new Node( right_domain, this );
                        Right->insert(d);
                    }
                    else {
                        // It is not in either. Put it here.
                        cont.push_back(d);
                    }
                }
                else {
                    // Couldn't split. Put it here.
                    cont.push_back(d);
                }
                    
            }
            else if ( Right ) {
                // Only the one on the right exists already. Check it first.
                if ( Contains::test(Right->MyDomain, d.first) ) {
                    // It is there, go down.
                    Right->insert(d);
                }
                else {
                    // Not on the right. Check the left.
                    Split::test(left_domain, right_domain, MyDomain);
                    if ( Contains::test(left_domain,d.first) ) {
                        // It is there. Build the left and drop into it.
                        Left = new Node( left_domain, this );
                        Left->insert(d);
                    }
                    else {
                        // Not there either. Keep it here.
                        cont.push_back(d);
                    }
                }
                    
            }
            else if ( Left ) {
                // Only the one on the left exists already. Check it first.
                if ( Contains::test(Left->MyDomain, d.first) ) {
                    // It is there. Drop into it.
                    Left->insert(d);
                }
                else {
                    // It isn't on the left. Split to see if it is on the right.
                    Split::test(left_domain,right_domain,MyDomain);
                    if ( Contains::test(right_domain,d.first) ) {
                        // It is there. Build it and drop into it.
                        Right = new Node( right_domain, this );
                        Right->insert(d);
                    }
                    else {
                        // Not on the right either.  Keep it here.
                        cont.push_back(d);
                    }
                }
            }
        }
        
    };
    
    friend class Node;
    
public: 
    
    // forward declaration of const_iterator
    class const_iterator;

    // Forward iterator.
    // It could be bidirectional but I'm to lazy to add operator--().
    class iterator {
        friend class DomainMap<Key,T,Touches,Contains,Split>;
        friend class const_iterator;

    public:

        // Null ctor initializes p to zero so we can tell it is null.
        iterator() : p(0) { }

        // Create one with the two pointers it needs.
        iterator(Node *pp, typename Node::cont_type::iterator vv)
                : p(pp), v(vv)
            {
            }

        // equality just tests each element.
        bool operator==(const iterator& rhs) const
            {
                return (p==rhs.p)&&((p==0)||(v==rhs.v));
            }
        bool operator!=(const iterator& rhs) const
            {
                return !(*this == rhs);
            }

        // Get the current one.
        value_type& operator*()
            {
                PAssert(p != 0);
                return *v;
            }

        // Increment the iterator.
        iterator& operator++() { op_pp(); return *this; }

        //mwerks private:
        //mwerks    void op_pp();
        //////////////////////////////////////////////////////////////////////
        // Increment an iterator.
        void op_pp()
            {
                 
                

                PAssert(p != 0);
                // First try to increment inside this node.
                if ( (++v) == p->cont.end() ) {
                    // If that one is at its end, go to the next node.
                    do {
                        // First look right.
                        if ( p->Right ) {
                            // If it is there, go there and then all the way left.
                            p = p->Right;
                            while ( p->Left )
                                p = p->Left;
                        }
                        else {
                            // If there is no right, go up until you can go right.
                            Node *y = p->Parent; // DomainMap<Key,T,Touches,Contains,Split>::Node *y;
                            while (y && (p==y->Right)) {
                                p = y;
                                y = y->Parent;
                            }
                            p = y;
                        }
                        // If the node found has nothing in it, keep looking.
                    } while (p && ((v=p->cont.begin())==p->cont.end()));
                }
            }
        Node *p;			              // Which Node are we in.
        typename Node::cont_type::iterator v;     // Within that node where are we.
    };

    // const_iterator: like iterator, but does return by value
    class const_iterator {
        friend class DomainMap<Key,T,Touches,Contains,Split>;

    public:
        // Null ctor initializes p to zero so we can tell it is null.
        const_iterator() : p(0) {}

        // Create one with the two pointers it needs.
        const_iterator(Node *pp, typename Node::cont_type::iterator vv)
                : p(pp), v(vv) {}

        // Create a const_iterator from an iterator
        const_iterator(const iterator& iter) : p(iter.p), v(iter.v) {}

        // equality just tests each element.
        bool operator==(const const_iterator& rhs) const
            {
                return (p==rhs.p)&&((p==0)||(v==rhs.v));
            }
        bool operator!=(const const_iterator& rhs) const
            {
                return !(*this == rhs);
            }

        // Get the current one.
        value_type operator*() const
            {
                PAssert(p != 0);
                return *v;
            }

        // Increment the iterator.
        const_iterator& operator++() { op_pp(); return *this; }

    private:
        //mwerks    void op_pp();
        //////////////////////////////////////////////////////////////////////
        // Increment a const_iterator.
        void op_pp()
            {
                 
                

                PAssert(p != 0);
                // First try to increment inside this node.
                if ( (++v) == p->cont.end() ) {
                    // If that one is at its end, go to the next node.
                    do {
                        // First look right.
                        if ( p->Right ) {
                            // If it is there, go there and then all the way left.
                            p = p->Right;
                            while ( p->Left )
                                p = p->Left;
                        }
                        else {
                            // If there is no right, go up until you can go right.
                            // DomainMap<Key,T,Touches,Contains,Split>::Node *y = p->Parent;
                            Node *y = p->Parent;
                            while (y && (p==y->Right)) {
                                p = y;
                                y = y->Parent;
                            }
                            p = y;
                        }
                        // If the node found has nothing in it, keep looking.
                    } while (p && ((v=p->cont.begin())==p->cont.end()));
                }
            }
        Node *p;			             // Which Node are we in.
        typename Node::cont_type::iterator v;    // Within that node where are we.
    };

    // Touch iterator.
    // Given a Key find all those that touch it.
    class touch_iterator {
        friend class DomainMap<Key,T,Touches,Contains,Split>;
    public: 

        typedef std::forward_iterator_tag    iterator_category;
        typedef typename DomainMap_t::value_type value_type;
        typedef typename DomainMap_t::value_type *pointer;
        typedef typename DomainMap_t::value_type &reference;
        typedef ptrdiff_t               difference_type;

        // Null ctor initializes p to zero so we can tell it is null.
        touch_iterator() : p(0) { }

        // equality just tests each element.
        bool operator==(const touch_iterator& rhs) const
            {
                return (p==rhs.p)&&((p==0)||(v==rhs.v));
            }
        bool operator!=(const touch_iterator& rhs) const
            {
                return !(*this == rhs);
            }

        // Get the current one.
        value_type& operator*()
            {
                PAssert(p != 0);
                return *v;
            }

        value_type* operator->()
            {
                PAssert(p != 0);
                return &(*v);
            }

        // Convert to a regular iterator.
        operator iterator()
            {
                return iterator(p,v);
            }

        // Increment the iterator.
        touch_iterator& operator++() { op_pp(); return *this; }

    private:
        //mwerks    void op_pp();
        //////////////////////////////////////////////////////////////////////
        // Increment a touch iterator.
        void op_pp()
            {
                 
                

                PAssert(p != 0);

                // Try to find one inside this node.
                while ( (++v) != p->cont.end() )
                    if ( Touches::test(TouchThis,(*v).first) )
                        return;			// Found one!  Return it.
  
                do {
                    // Not here. Look through the tree.
                    // First look right.
                     // DomainMap<Key,T,Touches,Contains,Split>::Node *y = p->Right;
                    Node *y = p->Right;
                    if ( y && Touches::test(TouchThis,y->MyDomain) ) {
                        // If it is there and we touch it, go there and then left.
                        p = y;
                        for (y=y->Left; y && Touches::test(TouchThis,y->MyDomain); y=y->Left )
                            p = y;
                    }
                    else {
                        // If there is no right, go up until you can go right.
                        // No need to test for touching on the way up because we
                        // would not be here if the parent didn't touch.
                        for (y=p->Parent; y && (p==y->Right); y=y->Parent) 
                            p = y;
                        p = y;
                    }
                    // Point to the beginning of the T's in this node.
                    if (p) {
                        for ( v = p->cont.begin() ; v != p->cont.end() ; ++v )
                            if ( Touches::test(TouchThis,(*v).first) )
                                return;		// Found one!  Return it.
                    }

                } while (p);

            }
        Node *p;			              // Which Node are we in.
        typename Node::cont_type::iterator v;     // Within that node where are we.
        Key TouchThis;		              // The Key that is being touched.
    };

    // Create an DomainMap with a Key that has the bounding box
    // of the whole domain. It should be true that all of the Keys
    // inserted into the DomainMap are contained in the 
    // original Key.  This bounding box is enough to determine 
    // how the recursive bisection proceeds.
    DomainMap(const Key& d)
            : Root(new Node(d)), Size(0)
        {
        }

    DomainMap()
            : Root(0), Size(0)
        {
        }

    // Copy ctor and assign do deep copies.
    DomainMap( const DomainMap<Key,T,Touches,Contains,Split>& );
    void operator=(const DomainMap<Key,T,Touches,Contains,Split>& );

    // When you die take the tree with you.
    ~DomainMap() { delete Root; }

    // Iterators to count over all the elements.
    iterator begin()       { return Leftmost; }
    iterator end()         { return iterator(); }
    const_iterator begin() const { return const_iterator(Leftmost); }
    const_iterator end()   const { return const_iterator(); }

    // Add a pair<Key,T> to it.
    // tjw: added noSplit flag 3/24/1998
    void insert(const value_type& d, bool noSplit=false);

    // Find the range that touches a given T.
    std::pair<touch_iterator,touch_iterator> touch_range(const Key& t) const;

    // return how many we have.
    size_type size() const { return Size; }

private: 

    friend class iterator;
    friend class const_iterator;
    friend class touch_iterator;

    Touches touches;
    Contains contains;
    Split split;

    Node* Root;
    iterator Leftmost;
    unsigned Size;

    // Add a pair<Key,T> to it without updating leftmost.
    void insert_noupdate(const value_type& d)
        {
            Root->insert(d);
            Size += 1;
        }
    // Reset the pointer to the first one.
    void update_leftmost();
};

//////////////////////////////////////////////////////////////////////

#include "DomainMap/DomainMap.hpp"

#endif // DOMAIN_MAP_H

