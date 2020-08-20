// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef PARTICLE_ATTRIB_H
#define PARTICLE_ATTRIB_H

/*
 * ParticleAttrib - Templated class for all particle attribute classes.
 *
 * This templated class is used to represent a single particle attribute.
 * An attribute is one data element within a particle object, and is
 * stored as an array.  This class stores the type information for the
 * attribute, and provides methods to create and destroy new items, and
 * to perform operations involving this attribute with others.  It also
 * provides iterators to allow the user to operate on single particles
 * instead of the entire array.
 *
 * ParticleAttrib is the primary element involved in expressions for
 * particles (just as Field is the primary element there).  This file
 * defines the necessary templated classes and functions to make
 * ParticleAttrib a capable expression-template participant.
 *
 * For some types such as Vektor, Tenzor, etc. which have multiple items,
 * we want to involve just the Nth item from each data element in an
 * expression.  The () operator here returns an object of type
 * ParticleAttribElem, which will use the () operator on each individual
 * element to access an item over which one can iterate and get just the
 * Nth item from each data element.  For example, if we have an attribute
 * like this:
 *             ParticleAttrib< Vektor<float, 4> > Data
 * we can involve just the 2nd item of each Vektor in an expression by
 * referring to this as
 *             Data(1)
 * which returns an object of type ParticleAttribElem that knows to return
 * just the 2nd item from each Vektor.  ParticleAttribElem is also expression-
 * template-aware; in fact, it is intended primarily for use in expressions
 * and not in many other situations.  The ParticleAttribElem will use the
 * () operator to get the Nth item out of each data element, so this requires
 * the user to define operator () for the particle attribute type being
 * used (for Vektor, Tenzor, etc., this has already been done).  This same
 * thing has been done for operator () involving either one or two indices,
 * which is needed to get the i,j element of a Tenzor, for example.
 *
 * To perform gather/scatter type operations involving sparse indices, in
 * which the sparse indices represent a list of points in a dense field
 * onto which we want to gather/scatter values, you can use the [] operator
 * to get a SubParticleAttrib object that knows about the particle
 * elements and associated sparse index points.  This allows us to have
 * the syntax
 *    P[S] = expr(A[S])
 * where P is a ParticleAttrib, A is some other object such as a Field that
 * can be indexed by an SIndex, and S is an SIndex object.  In this case,
 * the length of the ParticleAttrib would be changed to match the number
 * of local points in the SIndex, and the expression would be evaluated at
 * all the points in the SIndex and stored into P.  It also allows the
 * syntax
 *    A[S] = expr(B[S], P[S])
 * where A, B are things like Field, S in an SIndex, and P is a ParticleAttrib.
 * Here, the LHS is assigned, at all the points in the SIndex, to the values
 * of the expression, which can include a ParticleAttrib only if it is
 * indexed by an SIndex.  This is because SubParticleAttrib contains the
 * ability to provide an iterator with the right interface for the expression
 * evaluation.
 */

// include files
#include "Particle/ParticleAttribBase.h"
#include "Particle/ParticleAttribElem.h"
#include "SubParticle/SubParticleAttrib.h"
#include "DataSource/DataSource.h"
#include "DataSource/MakeDataSource.h"
#include "PETE/IpplExpressions.h"
#include "Index/NDIndex.h"
#include "Utility/DiscType.h"
#include "Utility/Inform.h"
#include "Utility/IpplStats.h"

#include <vector>
#include <utility>

// forward declarations
template<class T, unsigned Dim> class Vektor;
template<class T, unsigned Dim, class M, class C> class Field;
template <class T> class ParticleAttribIterator;
template <class T> class ParticleAttribConstIterator;


// ParticleAttrib class definition
template <class T>
class ParticleAttrib : public ParticleAttribBase, public DataSource,
		       public PETE_Expr< ParticleAttrib<T> >
{

    friend class ParticleAttribIterator<T>;
    friend class ParticleAttribConstIterator<T>;

public:
    // useful typedefs for type of data contained here
    typedef T Return_t;
    typedef std::vector<T> ParticleList_t;
    typedef ParticleAttribIterator<T> iterator;
    typedef ParticleAttribConstIterator<T> const_iterator;
    typedef ParticleAttribBase::SortListIndex_t  SortListIndex_t;
    typedef ParticleAttribBase::SortList_t       SortList_t;

public:
    // default constructor
    ParticleAttrib() : ParticleAttribBase(sizeof(T), DiscType<T>::str()), LocalSize(0), attributeIsDirty_(false) {
        INCIPPLSTAT(incParticleAttribs);
    }

    // copy constructor
    ParticleAttrib(const ParticleAttrib<T>& pa)
        : ParticleAttribBase(pa), ParticleList(pa.ParticleList), LocalSize(pa.LocalSize), attributeIsDirty_(false)  {
        INCIPPLSTAT(incParticleAttribs);
    }

    // destructor: delete the storage of the attribute array
    ~ParticleAttrib() { }

    //
    // bracket operators to access the Nth particle data
    //

    /// notify user that attribute has changed. The user is responsible to
    /// define the meaning of the dirt and clean state
    bool isDirty() {
        reduce(attributeIsDirty_, attributeIsDirty_, OpOr());
        return attributeIsDirty_;
    }

    void resetDirtyFlag() {
        attributeIsDirty_ = false;
    }

    typename ParticleList_t::reference
    operator[](size_t n) {
        attributeIsDirty_ = true;
        return ParticleList[n];
    }

    typename ParticleList_t::const_reference
    operator[](size_t n) const {
        return ParticleList[n];
    }

    //
    // bracket operator to refer to an attrib and an SIndex object
    //

    template<unsigned Dim>
    SubParticleAttrib<ParticleAttrib<T>, T, Dim>
    operator[](const SIndex<Dim> &s) const {
        ParticleAttrib<T> &a = const_cast<ParticleAttrib<T> &>(*this);
        return SubParticleAttrib<ParticleAttrib<T>, T, Dim>(a, s);
    }

    //
    // PETE interface.
    //
    enum { IsExpr = 0 };
    typedef const_iterator PETE_Expr_t;
    PETE_Expr_t MakeExpression() const { return cbegin(); }

    // Get begin and end point iterators
    iterator begin() { return iterator(this); }
    iterator end()   { return iterator(this, LocalSize); }

    const_iterator cbegin() const { const_iterator A(this); return A;}
    const_iterator cend()   const { const_iterator A(this, LocalSize); return A; }

    size_t size(void) const { return LocalSize; }

    //
    // methods to allow the user to access components of multiple-item attribs
    //

    // Create a ParticleAttribElem to allow the user to access just the Nth
    // element of the attribute stored here.
    ParticleAttribElem<T,1U> operator()(unsigned);

    // Same as above, but specifying two indices
    ParticleAttribElem<T,2U> operator()(unsigned, unsigned);

    // Same as above, but specifying three indices
    ParticleAttribElem<T,3U> operator()(unsigned, unsigned, unsigned);

    //
    // Particle <-> Field interaction methods
    //

    // scatter the data from this attribute onto the given Field, using
    // the given Position attribute
    template <unsigned Dim, class M, class C, class PT, class IntOp>
    void
    scatter(Field<T,Dim,M,C>& f,
            const ParticleAttrib< Vektor<PT,Dim> >& pp,
            const IntOp& /*intop*/) const {


        // make sure field is uncompressed and guard cells are zeroed
        f.Uncompress();
        T zero = 0;
        f.setGuardCells(zero);

        const M& mesh = f.get_mesh();
        // iterate through ParticleAttrib data and call scatter operation
        typename ParticleList_t::const_iterator curr, last = ParticleList.begin()+LocalSize;
        typename ParticleAttrib< Vektor<PT,Dim> >::const_iterator ppiter=pp.cbegin();
        for (curr = ParticleList.begin(); curr != last; ++curr, ++ppiter)
            IntOp::scatter(*curr,f,*ppiter,mesh);

        // accumulate values in guard cells (and compress result)
        f.accumGuardCells();

        INCIPPLSTAT(incParticleScatters);
        return;
    }

    // scatter the data from this attribute onto the given Field, using
    // the given Position attribute, and store the mesh information
    template <unsigned Dim, class M, class C, class PT,
              class IntOp, class CacheData>
    void
    scatter(Field<T,Dim,M,C>& f,
            const ParticleAttrib< Vektor<PT,Dim> >& pp,
            const IntOp& /*intop*/,
            ParticleAttrib<CacheData>& cache) const {



        // make sure field is uncompressed and guard cells are zeroed
        f.Uncompress();
        T zero = 0;
        f.setGuardCells(zero);

        const M& mesh = f.get_mesh();
        // iterate through ParticleAttrib data and call scatter operation
        typename ParticleList_t::const_iterator curr, last = ParticleList.begin()+LocalSize;
        typename ParticleAttrib< Vektor<PT,Dim> >::const_iterator ppiter=pp.cbegin();
        typename ParticleAttrib<CacheData>::iterator citer=cache.begin();
        for (curr = ParticleList.begin(); curr != last; ++curr, ++ppiter, ++citer)
            IntOp::scatter(*curr,f,*ppiter,mesh,*citer);

        // accumulate values in guard cells (and compress result)
        f.accumGuardCells();

        INCIPPLSTAT(incParticleScatters);
        return;
    }

    // scatter the data from this attribute onto the given Field, using
    // the precomputed mesh information
    template <unsigned Dim, class M, class C, class IntOp, class CacheData>
    void
    scatter(Field<T,Dim,M,C>& f, const IntOp& /*intop*/,
            const ParticleAttrib<CacheData>& cache) const {



        // make sure field is uncompressed and guard cells are zeroed
        f.Uncompress();
        T zero = 0;
        f.setGuardCells(zero);

        // iterate through ParticleAttrib data and call scatter operation
        typename ParticleList_t::const_iterator curr, last = ParticleList.begin()+LocalSize;
        typename ParticleAttrib<CacheData>::const_iterator citer=cache.cbegin();
        for (curr = ParticleList.begin(); curr != last; ++curr, ++citer)
            IntOp::scatter(*curr,f,*citer);

        // accumulate values in guard cells (and compress result)
        f.accumGuardCells();

        INCIPPLSTAT(incParticleScatters);
        return;
    }

    // gather the data from the given Field into this attribute, using
    // the given Position attribute
    template <unsigned Dim, class M, class C, class PT, class IntOp>
    void
    gather(const Field<T,Dim,M,C>& f,
           const ParticleAttrib< Vektor<PT,Dim> >& pp,
           const IntOp& /*intop*/) {


        // make sure field is uncompressed
        f.Uncompress();
        // fill guard cells if they are dirty
        if (f.isDirty())
            f.fillGuardCells(true);

        const M& mesh = f.get_mesh();
        // iterate through ParticleAttrib data and call gather operation
        typename ParticleList_t::iterator curr, last = ParticleList.begin()+LocalSize;
        typename ParticleAttrib< Vektor<PT,Dim> >::const_iterator ppiter=pp.cbegin();
        for (curr = ParticleList.begin(); curr != last; ++curr, ++ppiter)
            IntOp::gather(*curr,f,*ppiter,mesh);

        // try to compress the Field again
        f.Compress();

        INCIPPLSTAT(incParticleGathers);
        return;
    }

    // gather the data from the given Field into this attribute, using
    // the given Position attribute, and store the mesh information
    template <unsigned Dim, class M, class C, class PT,
              class IntOp, class CacheData>
    void
    gather(const Field<T,Dim,M,C>& f,
           const ParticleAttrib< Vektor<PT,Dim> >& pp,
           const IntOp& /*intop*/,
           ParticleAttrib<CacheData>& cache) {


        // make sure field is uncompressed
        f.Uncompress();
        // fill guard cells if they are dirty
        if (f.isDirty())
            f.fillGuardCells(true);

        const M& mesh = f.get_mesh();
        // iterate through ParticleAttrib data and call gather operation
        typename ParticleList_t::iterator curr, last = ParticleList.begin()+LocalSize;
        typename ParticleAttrib< Vektor<PT,Dim> >::const_iterator ppiter=pp.cbegin();
        typename ParticleAttrib<CacheData>::iterator citer=cache.begin();
        for (curr=ParticleList.begin(); curr != last; ++curr,++ppiter,++citer)
            IntOp::gather(*curr,f,*ppiter,mesh,*citer);

        // try to compress the Field again
        f.Compress();

        INCIPPLSTAT(incParticleGathers);
        return;
    }

    // gather the data from the given Field into this attribute, using
    // the precomputed mesh information
    template <unsigned Dim, class M, class C, class IntOp, class CacheData>
    void
    gather(const Field<T,Dim,M,C>& f, const IntOp& /*intop*/,
           const ParticleAttrib<CacheData>& cache) {


        // make sure field is uncompressed
        f.Uncompress();
        // fill guard cells if they are dirty
        if (f.isDirty())
            f.fillGuardCells(true);

        // iterate through ParticleAttrib data and call gather operation
        typename ParticleList_t::iterator curr, last = ParticleList.begin()+LocalSize;
        typename ParticleAttrib<CacheData>::const_iterator citer=cache.cbegin();
        for (curr = ParticleList.begin(); curr != last; ++curr, ++citer)
            IntOp::gather(*curr,f,*citer);

        // try to compress the Field again
        f.Compress();

        INCIPPLSTAT(incParticleGathers);
        return;
    }

    //
    // Assignment operators
    //

    // assign a general expression
    template<class T1>
    const ParticleAttrib<T>& operator=(const PETE_Expr<T1>& rhs) {
        assign(*this,rhs);
        return *this;
    }

    // assignment of a ParticleAttrib
    const ParticleAttrib<T>& operator=(const ParticleAttrib<T>& rhs) {
        if (size() != rhs.size()) {
            ERRORMSG("Attempting to copy particle attributes with unequal sizes.");
            ERRORMSG("\n" << size() << " != " << rhs.size() << endl);
        }
        assign(*this,rhs);
        return *this;
    }

    // assignment of a scalar
    const ParticleAttrib<T>& operator=(T rhs) {
        assign(*this,rhs);
        return *this;
    }

    //
    // methods used to manipulate the normal attrib data
    //

    // Create storage for M particle attributes.  The storage is uninitialized.
    // New items are appended to the end of the array.
    virtual void create(size_t);

    // Delete the attribute storage for M particle attributes, starting at
    // the position I.
    // This really erases the data, which will change local indices
    // of the data.  It actually just copies the data from the end of the
    // storage into the selected block
    // Boolean flag indicates whether to use optimized destroy method
    virtual void destroy(size_t M, size_t I, bool optDestroy=true);

    // This version takes a list of particle destroy events
    // Boolean flag indicates whether to use optimized destroy method
    virtual void destroy(const std::vector< std::pair<size_t,size_t> >& dlist,
                         bool optDestroy=true);

    // puts M particle's data starting from index I into a Message.
    // Return the number of particles put into the message.
    virtual size_t putMessage(Message&, size_t, size_t);

    // Another version of putMessage, which takes list of indices
    // Return the number of particles put into the message.
    virtual size_t putMessage(Message&, const std::vector<size_t>&);

    // Get data out of a Message containing N particle's attribute data,
    // and store it here.  Data is appended to the end of the list.  Return
    // the number of particles retrieved.
    virtual size_t getMessage(Message&, size_t);

    //
    // methods used to manipulate the ghost particle data
    //

    // Delete the ghost attrib storage for M particles, starting at pos I.
    // Items from the end of the list are moved up to fill in the space.
    // Return the number of items actually destroyed.
    virtual size_t ghostDestroy(size_t, size_t);/* {
                                                   return 0;
                                                   }*/
    virtual void ghostCreate(size_t);/*
                                       {

                                       }*/
    // puts M particle's data starting from index I into a Message.
    // Return the number of particles put into the message.  This is for
    // when particles are being swapped to build ghost particle interaction
    // lists.
    virtual size_t ghostPutMessage(Message&, size_t, size_t);/* {
                                                                return 0;
                                                                }*/
    // puts data for a list of particles into a Message, for interaction lists.
    // Return the number of particles put into the message.
    virtual size_t ghostPutMessage(Message&, const std::vector<size_t>&);/* {
                                                                            return 0;
                                                                            }*/

    // Get ghost particle data from a message.
    virtual size_t ghostGetMessage(Message&, size_t);/* {
                                                        return 0;
                                                        }*/

    //
    // virtual methods used to sort data
    //

    // Calculate a "sort list", which is an array of data of the same
    // length as this attribute, with each element indicating the
    // (local) index wherethe ith particle shoulkd go.  For example,
    // if there are four particles, and the sort-list is {3,1,0,2}, that
    // means the particle currently with index=0 should be moved to the third
    // position, the one with index=1 should stay where it is, etc.
    // The optional second argument indicates if the sort should be ascending
    // (true, the default) or descending (false).
    virtual void calcSortList(SortList_t &slist, bool ascending = true);

    // Process a sort-list, as described for "calcSortList", to reorder
    // the elements in this attribute.  All indices in the sort list are
    // considered "local", so they should be in the range 0 ... localnum-1.
    // The sort-list does not have to have been calculated by calcSortList,
    // it could be calculated by some other means, but it does have to
    // be in the same format.  Note that the routine may need to modify
    // the sort-list temporarily, but it will return it in the same state.
    virtual void sort(SortList_t &slist);

    //
    // other functions
    //

    // Print out information for debugging purposes.
    virtual void printDebug(Inform&);

protected:
    // a virtual function which is called by this base class to get a
    // specific instance of DataSourceObject based on the type of data
    // and the connection method (the argument to the call).
    virtual DataSourceObject *createDataSourceObject(const char *nm,
                                                     DataConnect *dc, int tm) {
        return make_DataSourceObject(nm, dc, tm, *this);
    }

    // storage for particle data
    ParticleList_t ParticleList;
    size_t LocalSize;

private:
    bool attributeIsDirty_;
};


//
// iterator for data in a ParticleAttrib
//
//~
//~ template <class T>
//~ class ParticleAttribIterator : public PETE_Expr< ParticleAttribIterator<T> >
//~ {
//~ public:
//~ typedef typename ParticleAttrib<T>::ParticleList_t ParticleList_t;
//~
//~ ParticleAttribIterator() : myList(0), curr(0) { }
//~
//~ ParticleAttribIterator(ParticleList_t& pa)
//~ : myList(&pa), curr(&(*pa.begin())) { }
//~
//~ ParticleAttribIterator(ParticleList_t& pa, size_t offset)
//~ : myList(&pa), curr(&(*(pa.begin()+offset))) { }
//~
//~ ParticleAttribIterator(const ParticleAttribIterator<T>& i)
//~ : myList(i.myList), curr(i.curr) { }
//~
//~ // PETE interface.
//~ typedef ParticleAttribIterator<T> PETE_Expr_t;
//~ typedef T PETE_Return_t;
//~ PETE_Expr_t MakeExpression() const { return *this; }
//~ PETE_Return_t& operator*(void) const { return *curr; }
//~
//~ ParticleAttribIterator<T>& operator++(void) {
//~ ++curr;
//~ return *this;
//~ }
//~ ParticleAttribIterator<T>& at_end(void) {
//~ curr = &*myList->end();
//~ return *this;
//~ }
//~ bool operator==(const ParticleAttribIterator<T>& a) const {
//~ return  (curr == a.curr);
//~ }
//~ bool operator!=(const ParticleAttribIterator<T>& a) const {
//~ return !(curr == a.curr);
//~ }
//~ const ParticleList_t& getParticleList() const { return *myList; }
//~ T* getP() const { return curr; }
//~
//~ private:
//~ ParticleList_t* myList;        // ParticleList I iterate over
//~ T* curr;                       // iterator current position
//~ };


template <class T>
class ParticleAttribIterator : public PETE_Expr< ParticleAttribIterator<T> >
{
public:
    typedef typename ParticleAttrib<T>::ParticleList_t ParticleList_t;
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;
    typedef std::random_access_iterator_tag iterator_category;

    ParticleAttribIterator() : attrib(0) { }

    ParticleAttribIterator(ParticleAttrib<T> *pa)
        : attrib(pa), curr(pa->ParticleList.begin()) { }

    ParticleAttribIterator(ParticleAttrib<T> *pa, size_t offset)
        : attrib(pa), curr(pa->ParticleList.begin()+offset) { }

    ParticleAttribIterator(const ParticleAttribIterator<T>& i)
        : attrib(i.attrib), curr(i.curr) { }

    // PETE interface.
    typedef ParticleAttribIterator<T> PETE_Expr_t;
    typedef T PETE_Return_t;
    PETE_Expr_t MakeExpression() const { return *this; }
    PETE_Return_t& operator*(void) const { return *curr; }
    T* operator->() const { return getP(); }

    ParticleAttribIterator<T>& operator++(void) {
        ++curr;
        return *this;
    }

    ParticleAttribIterator<T> operator++(int) {
        ParticleAttribIterator<T> tmp(*this);
        ++curr;
        return tmp;
    }

    ParticleAttribIterator<T>& operator--(void) {
        --curr;
        return *this;
    }

    ParticleAttribIterator<T> operator--(int) {
        ParticleAttribIterator<T> tmp(*this);
        --curr;
        return tmp;
    }

    ParticleAttribIterator<T>& operator+=(size_t n) {
        curr += n;
        return *this;
    }

    ParticleAttribIterator<T> operator+(size_t n) const {
        ParticleAttribIterator<T> tmp(*this);
        tmp += n;
        return tmp;
    }

    ParticleAttribIterator<T>& operator-=(size_t n) {
        curr -= n;
        return *this;
    }

    ParticleAttribIterator<T> operator-(size_t n) const {
        ParticleAttribIterator<T> tmp(*this);
        tmp -= n;
        return tmp;
    }

    size_t operator-(const ParticleAttribIterator<T>& a) const {
        return (curr - a.curr);
    }

    ParticleAttribIterator<T> operator[](size_t n) const {
        return (*this + n);
    }

    ParticleAttribIterator<T>& at_end(void) {
        curr = attrib->ParticleList.begin()+attrib->LocalSize;
        return *this;
    }

    bool operator==(const ParticleAttribIterator<T>& a) const {
        return  (curr == a.curr);
    }

    bool operator!=(const ParticleAttribIterator<T>& a) const {
        return !(curr == a.curr);
    }

    bool operator<(const ParticleAttribIterator<T>& a) const {
        return (curr < a.curr);
    }

    bool operator<=(const ParticleAttribIterator<T>& a) const {
        return (curr <= a.curr);
    }

    bool operator>(const ParticleAttribIterator<T>& a) const {
        return (curr > a.curr);
    }

    bool operator>=(const ParticleAttribIterator<T>& a) const {
        return (curr >= a.curr);
    }
    //const ParticleList_t& getParticleList() const { return attrib->ParticleList; }
    size_t size() const { return attrib->LocalSize; }
    T* getP() const { return &(*curr); }

private:
    ParticleAttrib<T> *attrib;        // ParticleList I iterate over
    typename ParticleList_t::iterator curr;    // iterator current position
};

template <class T>
ParticleAttribIterator<T> operator+(size_t n, const ParticleAttribIterator<T> & a) {
    return (a + n);
}

template <class T>
class ParticleAttribConstIterator : public PETE_Expr< ParticleAttribConstIterator<T> >
{
public:
    typedef typename ParticleAttrib<T>::ParticleList_t ParticleList_t;
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;
    typedef std::random_access_iterator_tag iterator_category;

    ParticleAttribConstIterator() : attrib(0) { }

    ParticleAttribConstIterator(const ParticleAttrib<T> * pa)
        : attrib(pa), curr(pa->ParticleList.begin()) { }

    ParticleAttribConstIterator(const ParticleAttrib<T> * pa, size_t offset)
        : attrib(pa), curr(pa->ParticleList.begin()+offset) { }

    ParticleAttribConstIterator(const ParticleAttribConstIterator<T>& i)
        : attrib(i.attrib), curr(i.curr) { }

    // PETE interface.
    typedef ParticleAttribConstIterator<T> PETE_Expr_t;
    typedef T PETE_Return_t;
    PETE_Expr_t MakeExpression() const { return *this; }
    const PETE_Return_t& operator*(void) const { return *curr; }

    T const * operator->() const { return getP(); }

    ParticleAttribConstIterator<T>& operator=(const ParticleAttribConstIterator<T>&) = default;


    ParticleAttribConstIterator<T>& operator++(void) {
        ++curr;
        return *this;
    }

    ParticleAttribConstIterator<T> operator++(int) {
        ParticleAttribConstIterator<T> tmp(*this);
        ++curr;
        return tmp;
    }

    ParticleAttribConstIterator<T>& operator--(void) {
        --curr;
        return *this;
    }

    ParticleAttribConstIterator<T> operator--(int) {
        ParticleAttribConstIterator<T> tmp(*this);
        --curr;
        return tmp;
    }

    ParticleAttribConstIterator<T>& operator+=(size_t n) {
        curr += n;
        return *this;
    }

    ParticleAttribConstIterator<T> operator+(size_t n) const {
        ParticleAttribConstIterator<T> tmp(*this);
        tmp += n;
        return tmp;
    }

    ParticleAttribConstIterator<T>& operator-=(size_t n) {
        curr -= n;
        return *this;
    }

    ParticleAttribConstIterator<T> operator-(size_t n) const {
        ParticleAttribConstIterator<T> tmp(*this);
        tmp -= n;
        return tmp;
    }

    size_t operator-(const ParticleAttribConstIterator<T>& a) const {
        return (curr - a.curr);
    }

    ParticleAttribConstIterator<T> operator[](size_t n) const {
        return (*this + n);
    }

    ParticleAttribConstIterator<T>& at_end(void) {
        curr = attrib->ParticleList.begin()+attrib->LocalSize;
        return *this;
    }

    bool operator==(const ParticleAttribConstIterator<T>& a) const {
        return  (curr == a.curr);
    }

    bool operator!=(const ParticleAttribConstIterator<T>& a) const {
        return !(curr == a.curr);
    }

    bool operator<(const ParticleAttribConstIterator<T>& a) const {
        return (curr < a.curr);
    }

    bool operator<=(const ParticleAttribConstIterator<T>& a) const {
        return (curr <= a.curr);
    }

    bool operator>(const ParticleAttribConstIterator<T>& a) const {
        return (curr > a.curr);
    }

    bool operator>=(const ParticleAttribConstIterator<T>& a) const {
        return (curr >= a.curr);
    }

    //const ParticleList_t& getParticleList() const { return attrib->ParticleList; }
    size_t size() const { return attrib->LocalSize; }
    T const * getP() const { return &(*curr); }

private:
    ParticleAttrib<T> const * attrib;        // ParticleList I iterate over
    typename ParticleList_t::const_iterator curr;    // iterator current position
};

template <class T>
ParticleAttribConstIterator<T> operator+(size_t n, const ParticleAttribConstIterator<T> & a) {
    return (a + n);
}

// Global template functions for gather/scatter operations

// scatter the data from the given attribute onto the given Field, using
// the given Position attribute
template <class FT, unsigned Dim, class M, class C, class PT, class IntOp>
inline
void scatter(const ParticleAttrib<FT>& attrib, Field<FT,Dim,M,C>& f,
  	     const ParticleAttrib< Vektor<PT,Dim> >& pp, const IntOp& intop) {
    attrib.scatter(f, pp, intop);
}

// scatter the data from the given attribute onto the given Field, using
// the given Position attribute, and save the mesh information for reuse
template <class FT, unsigned Dim, class M, class C, class PT,
          class IntOp, class CacheData>
inline
void scatter(const ParticleAttrib<FT>& attrib, Field<FT,Dim,M,C>& f,
  	     const ParticleAttrib< Vektor<PT,Dim> >& pp, const IntOp& intop,
             ParticleAttrib<CacheData>& cache) {
    attrib.scatter(f, pp, intop, cache);
}

// scatter the data from the given attribute onto the given Field, using
// the precomputed mesh information
template <class FT, unsigned Dim, class M, class C,
          class IntOp, class CacheData>
inline
void scatter(const ParticleAttrib<FT>& attrib, Field<FT,Dim,M,C>& f,
  	     const IntOp& intop, const ParticleAttrib<CacheData>& cache) {
    attrib.scatter(f, intop, cache);
}

// gather the data from the given Field into the given attribute, using
// the given Position attribute
template <class FT, unsigned Dim, class M, class C, class PT, class IntOp>
inline
void gather(ParticleAttrib<FT>& attrib, const Field<FT,Dim,M,C>& f,
            const ParticleAttrib< Vektor<PT,Dim> >& pp, const IntOp& intop) {
    attrib.gather(f, pp, intop);
}

// gather the data from the given Field into the given attribute, using
// the given Position attribute, and save the mesh information for reuse
template <class FT, unsigned Dim, class M, class C, class PT,
          class IntOp, class CacheData>
inline
void gather(ParticleAttrib<FT>& attrib, const Field<FT,Dim,M,C>& f,
            const ParticleAttrib< Vektor<PT,Dim> >& pp, const IntOp& intop,
            ParticleAttrib<CacheData>& cache) {
    attrib.gather(f, pp, intop, cache);
}

// gather the data from the given Field into the given attribute, using
// the precomputed mesh information
template <class FT, unsigned Dim, class M, class C,
          class CacheData, class IntOp>
inline
void gather(ParticleAttrib<FT>& attrib, const Field<FT,Dim,M,C>& f,
            const IntOp& intop, const ParticleAttrib<CacheData>& cache) {
    attrib.gather(f, intop, cache);
}

// This scatter function computes the particle number density by
// scattering the scalar value val for each particle into the Field.
template <class FT, unsigned Dim, class M, class C, class PT, class IntOp>
void scatter(Field<FT,Dim,M,C>& f, const ParticleAttrib< Vektor<PT,Dim> >& pp,
	     const IntOp& intop, FT val);

// version which also caches mesh info
template <class FT, unsigned Dim, class M, class C, class PT,
          class IntOp, class CacheData>
void scatter(Field<FT,Dim,M,C>& f, const ParticleAttrib< Vektor<PT,Dim> >& pp,
	     const IntOp& intop, ParticleAttrib<CacheData>& cache,
             FT val);

// version which uses cached mesh info
template <class FT, unsigned Dim, class M, class C,
          class IntOp, class CacheData>
void scatter(Field<FT,Dim,M,C>& f, const IntOp& intop,
             const ParticleAttrib<CacheData>& cache, FT val);

// mwerks: addeded this to work around default-arg bug:
// This scatter function computes the particle number density by
// scattering the scalar value val for each particle into the Field.
template <class FT, unsigned Dim, class M, class C, class PT, class IntOp>
inline void scatter(Field<FT,Dim,M,C>& f, const ParticleAttrib< Vektor<PT,Dim> >& pp,
                    const IntOp& intop){
    scatter(f, pp, intop, FT(1));
}

// mwerks: addeded this to work around default-arg bug:
// version which also caches mesh info
template <class FT, unsigned Dim, class M, class C, class PT,
          class IntOp, class CacheData>
inline void scatter(Field<FT,Dim,M,C>& f, const ParticleAttrib< Vektor<PT,Dim> >& pp,
                    const IntOp& intop, ParticleAttrib<CacheData>& cache) {
    scatter(f, pp, intop, cache, FT(1));
}

// mwerks: addeded this to work around default-arg bug:
// version which uses cached mesh info
template <class FT, unsigned Dim, class M, class C,
          class IntOp, class CacheData>
inline void scatter(Field<FT,Dim,M,C>& f, const IntOp& intop,
                    const ParticleAttrib<CacheData>& cache) {
    scatter(f, intop, cache, FT(1));
}

#include "Particle/ParticleAttrib.hpp"

#endif // PARTICLE_ATTRIB_H

/***************************************************************************
 * $RCSfile: ParticleAttrib.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:28 $
 * IPPL_VERSION_ID: $Id: ParticleAttrib.h,v 1.1.1.1 2003/01/23 07:40:28 adelmann Exp $
 ***************************************************************************/
