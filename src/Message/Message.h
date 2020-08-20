// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef MESSAGE_H
#define MESSAGE_H

/***************************************************************************
 * Message - contains a list of items that comprise a set of data to be
 * sent, or received from, another node in a parallel architecture.  A person
 * creates a Message object, loads it with data to be sent, and gives it
 * to a Communicate object.
 *
 * The message consists of N 'MsgItem' objects, which each contain some
 * data; data is added sequentially to a new message using the 'put'
 * routine, and extracted using the 'get' routine.  The messages are
 * retrieved in a FIFO fashion - you put in messages, and get them out
 * in the order in which they were put in.  You can access MsgItem's via
 * random access with the [] operator, but it is important
 * to actually remove the data using 'get' as this will properly deal with
 * copying data and freeing up memory (if necessary).
 *
 * Note on usage: A Message is primarily intended to be created once,
 * given to a Communicate instance to transmit to another node, and then
 * to be unpacked or otherwise read by the receiver.  A received message
 * may be forwarded to another node, but it cannot be combined with another
 * or used to initialize a copy of the Message.  This is due to the problems
 * of resolving who needs to free up the storage used for the Message elements.
 ***************************************************************************/

#include "Utility/Inform.h"
#include <complex>
#include <cstddef>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>

// forward declarations
class Communicate;
class Message;
std::ostream& operator<<(std::ostream& o, const Message& m);

// Macros
//////////////////////////////////////////////////////////////////////
// simple template traits class to tell if a type is a built-in type
// or not.  The classes have an enum which is 1 if the type is build-in,
// 0 otherwise.  The default case is 0, but specialized to 1 for other types.
template <class T>
struct MessageTypeIntrinsic
{
    enum { builtin = 0 };
    enum { pointer = 0 };
};

#define DEFINE_BUILTIN_TRAIT_CLASS(T)				\
template <>							\
struct MessageTypeIntrinsic<T> {				\
  enum { builtin = 1 };						\
  enum { pointer = 0 };						\
};								\
template <>							\
struct MessageTypeIntrinsic<T *> {				\
  enum { builtin = 1 };						\
  enum { pointer = 1 };						\
};                                                              \
template <int N>						\
struct MessageTypeIntrinsic<T[N]> {				\
  enum { builtin = 1 };						\
  enum { pointer = 1 };						\
};

#define DEFINE_ALL_BUILTIN_TRAIT_CLASS(T)	\
DEFINE_BUILTIN_TRAIT_CLASS(T)			\
DEFINE_BUILTIN_TRAIT_CLASS(const T)

DEFINE_ALL_BUILTIN_TRAIT_CLASS(bool)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(char)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(unsigned char)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(short)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(unsigned short)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(int)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(unsigned int)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(long)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(unsigned long)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(long long)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(float)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(double)
DEFINE_ALL_BUILTIN_TRAIT_CLASS(std::complex<double>)

/////////////////////////////////////////////////////////////////////
// a class to put single items into a Message, which can be specialized
// to the case of built-in types and other types.

template <class T, bool builtin, bool pointer>
struct PutSingleItem { };

// specialization to a non-built-in type, which is never assumed to be a ptr
template <class T>
struct PutSingleItem<T, false, false>
{
    // put a value into a message
    static Message& put(Message&, const T&);
    // get a value out of a message
    static Message& get(Message&, T&);
    // version of put using a pair of iterators
    static Message& put(Message&, T, T);
    // version of put using a list of indices and an iterator
    static Message& put(Message&, const std::vector<size_t>&, T);
    // get_iter uses an output iterator
    static Message& get_iter(Message&, T);
};

// specialization to a built-in type that is not a pointer
template <class T>
struct PutSingleItem<T, true, false>
{
    // put a value into a message
    static Message& put(Message&, const T&);
    // get a value out of a message
    static Message& get(Message&, T&);
};

// specialization to a pointer to a built-in type. In this class, we
// know that 'T' is a pointer type.
template <class T>
struct PutSingleItem<T, true, true>
{
    // put using a pair of pointers
    static Message& put(Message&, T, T);
    // get using a pointer
    static Message& get(Message&, T);
    // put using a list of indices and a pointer
    static Message& put(Message&, const std::vector<size_t>&, T);
    // get using an output iterator
    static Message& get_iter(Message&, T);
};


class Message
{

public:
    // a class which stores a single message element.  This will either store
    // a reference to the data (if copy=false), make an internal copy (if
    // copy=true, and it is large enough), or put the data into an internal
    // fast buffer (the MsgItemBuf struct).
    class MsgItem
    {
    private:
        // a very simple struct for storing MsgItem data
        struct MsgItemBuf
        {
            unsigned long long d1, d2, d3, d4;

            MsgItemBuf() { }
            MsgItemBuf(const MsgItemBuf& m) : d1(m.d1),d2(m.d2),d3(m.d3),d4(m.d4) {}
            ~MsgItemBuf() { }
        };

    public:

        // default constructor
        MsgItem() : item(0), elements(0), bytesize(0), needDelete(false) { }

        // regular constructor, with data to store
        MsgItem(void *d, unsigned int elems, unsigned int totbytes,
                bool needcopy, bool needdel) : item(&defbuf), elements(elems),
                bytesize(totbytes), needDelete(needdel)
        {
            if (totbytes > 0 && d != 0)
            {
                if (needcopy)
                {
                    if (totbytes <= sizeof(MsgItemBuf))
                    {
                        // copy data into internal buffer
                        needDelete = false;
                    }
                    else
                    {
                        // malloc and copy over data
                        item = malloc(totbytes);
                        needDelete = true;
                    }
                    memcpy(item, d, totbytes);
                }
                else
                {
                    // we just store a reference, we do not copy
                    item = d;
                }
            }
            else
            {
                // no data in message
                item = 0;
                elements = 0;
                needDelete = false;
            }
        }

        // copy constructor
        MsgItem(const MsgItem &m) : item(&defbuf), elements(m.elements), bytesize(m.bytesize),
                defbuf(m.defbuf), needDelete(m.needDelete)
        {
            // either we just copy the 'item' pointer, or we copy the default buf
            if (m.item != &(m.defbuf))
                item = m.item;
        }

        // destructor
        ~MsgItem() { }

        // return our total byte size, number of elements, and elem size
        unsigned int numBytes() const
        {
            return bytesize;
        }
        unsigned int numElems() const
        {
            return elements;
        }
        unsigned int elemSize() const
        {
            return (elements>0?bytesize/elements:0);
        }

        // will we need to delete our data?
        bool willNeedDelete() const
        {
            return (needDelete && item != 0);
        }

        // cancel the need to delete the data
        void cancelDelete()
        {
            needDelete = false;
        }

        // return our item pointer
        void *data()
        {
            return item;
        }

        // delete our data item
        void deleteData()
        {
            if (willNeedDelete())
                free(item);
        }

    private:
        // pointer to the item; must be newed/deleted, or pointing to defbuf
        void *item;

        // number of individual elements, and total storage size in bytes
        unsigned int elements, bytesize;

        // default storage space; used for speed
        MsgItemBuf defbuf;

        // do we need to delete this item storage?
        bool needDelete;
    };

public:
    //
    // constructors and destructors
    //

    // 'default' constructor: just make an empty message, optionally requesting
    // how many items we should preallocate space for
    Message(unsigned int numelems = 8)
            : numRemoved(0), DoCopy(true), DoDelete(true), comm(0), commdata(0)
    {
        MsgItemList.reserve(numelems);
    }

    // destructor: delete all items in this message, as if 'get' had been
    // called for all the items
    ~Message();

    //
    // global Message operations
    //

    // return number of items left in this message
    size_t size() const
    {
        return (MsgItemList.size() - numRemoved);
    }
    size_t removed() const
    {
        return numRemoved;
    }
    bool empty() const
    {
        return (size() == 0);
    }

    // returns a reference to the Nth MsgItem.  Note that 'n' refers
    // to the index from te first unremoved item, and that 'get' and 'remove'
    // will remove the top item.
    MsgItem& item(size_t n)
    {
        return MsgItemList[n + numRemoved];
    }
    const MsgItem& item(size_t n) const
    {
        return MsgItemList[n+numRemoved];
    }

    // indicate that the next message item should be copied (t) or just
    // remembered via a pointer (f)
    Message& setCopy(const bool c)
    {
        DoCopy = c;
        return *this;
    }
    bool willCopy() const
    {
        return DoCopy;
    }

    // indicate that the next message item memory should be deleted by this
    // object when the Message is deleted (t)
    Message& setDelete(const bool c)
    {
        DoDelete = c;
        return *this;
    }
    bool willDelete() const
    {
        return DoDelete;
    }

    // clear the message; deletes all its items
    Message& clear();

    // return and remove the next item from this message ... if the item
    // does not exist, NULL is returned.  This is similar to get, except that
    // just a void* pointer to the data is returned (instead of having the
    // data copied into given storage), and this data is DEFINITELY a malloced
    // block of data that the user must deallocate using 'free' (NOT delete).
    // Like 'get', after this is called, the top MsgItem is removed.
    // If you wish to just access the Nth item's item  pointer, use
    // item(N).item  .
    void *remove();

    // tell this Message to call the special function 'cleanupMessage' with
    // the provided pointer, in case the message is using buffer space
    // allocated by the Communicate object.  If no Communicate object has
    // been provided, nothing is done
    void useCommunicate(Communicate *c, void *d)
    {
        comm = c;
        commdata = d;
    }

    //
    // routines to get and put data out of/into the message.
    //

    // NOTES for 'put' routines:
    // For intrinsic scalar values, there is one argument: the scalar item.
    // For arrays, there are two arguments: the begining and (one-past-end)
    // pointers (i.e. like iterators, but restricted to pointers).
    // For any other arbitrary type, calling put will in turn call the method
    // 'putMessage' in the provided object.  'putMessage' can then call put
    // to put in the proper intrinsic-type objects.
    //
    // When putting in data, you can have the data copied over, or just have
    // a pointer stored.  If a pointer is stored (copy=F), you must also
    // specify whether this Message object should be responsible for deleting
    // the data (delstor=T), or should just leave the storage alone after the
    // data has been retrieved with get.  These flags are set by calling
    // 'setCopy' and 'setDelete' with the desired setting.  After an item
    // has been 'put', the are set back to the default values 'copy=true' and
    // 'delete=true'.  You can use them in this way:
    //      Message msg;
    //      msg.setCopy(false).setDelete(false).put(data);
    // copy and delstor are used to increase performance.  They
    // should be used in the following circumstances:
    //	a. Data to be sent/rec is in a location that will not change before the
    // msg is used, but should not be affected after the Message is deleted.
    // For this case, use copy=F, delstor=F.
    //	b. Data for a message has already had space allocated for it, and so
    // does not need to be copied.  Also, since the space has been allocated
    // already, it must be freed when the msg is sent.  For this case, use
    // copy=F, delstor=T.
    //	c. For data that is in a volatile location, it must be copied to new
    // storage, so use copy=T, and don't specify any value for delstor (it will
    // be ignored if copy=T).

    // general templated version of put
    // for an arbitrary type, call the function 'putMessage' with this
    // message so that it can put in data any way it wants.  That function
    // should return a reference to this message.
    // 'putMessage' should be defined as:
    //                   Message& putMessage(Message &);
    template <class T>
    Message& put(const T& val)
    {
        return PutSingleItem<T,
               MessageTypeIntrinsic<T>::builtin,
               MessageTypeIntrinsic<T>::pointer>::put(*this, val);
    }

    // specialized versions of put for character strings
    /*
    Message &put(char *d) {	// null-terminated string
      return putmsg((void *)d, sizeof(char), strlen(d) + 1);
    }
    */
    Message &put(const char *d)  	// null-terminated string
    {
        return putmsg((void *)d, sizeof(char), strlen(d) + 1);
    }
    // specialized version for string class
    Message& put(const std::string& s)
    {
        int len = s.length() + 1;
        put(len);
        put(s.c_str());
        return *this;
    }

    // general templated version of put for two iterators
    // Template put using a pair of iterators.  This version is called by
    // the public 'put' with two iterators after getting the proper type
    // of the data pointed to by the iterators
    template <class ForwardIterator>
    Message& put(ForwardIterator beg, ForwardIterator end)
    {
        return PutSingleItem<ForwardIterator,
               MessageTypeIntrinsic<ForwardIterator>::builtin,
               MessageTypeIntrinsic<ForwardIterator>::pointer>::put(
                   *this, beg, end);
    }

    // for using an indirection list
    // Template put using an indirection list.  This version is called by
    // the public 'put' with an indirection list and a RandomAccessIterator
    // after getting the proper type of the data pointed to by the iterator
    template <class RandomAccessIterator>
    Message& put(const std::vector<size_t>& indices,
                 RandomAccessIterator beg)
    {
        return PutSingleItem<RandomAccessIterator,
               MessageTypeIntrinsic<RandomAccessIterator>::builtin,
               MessageTypeIntrinsic<RandomAccessIterator>::pointer>::put(*this,
                       indices, beg);
    }

    // general put routine; called by other cases
    // arguments are the item, its element size (in bytes),
    // and how many items total to copy (if 0, this is a scalar).
    Message &putmsg(void *, int, int = 0);


    // general templated version of get, for a ref or a pointer
    // for an arbitrary type, call the function 'getMessage' with this
    // message so that it can get data any way it wants.  That function
    // should return a reference to this message.
    // 'getMessage' should be defined as:
    //                   Message& getMessage(Message &);
    // NOTE: the argument is a const ref, but will be cast to non-const,
    // to eliminate warning messages about anachronisms.
    //////////////////////////////////////////////////////////////////////
    // general templated version of get.
    template <class T>
    Message& get(const T& cval)
    {
        T& val = const_cast<T&>(cval);
        return PutSingleItem<T,
               MessageTypeIntrinsic<T>::builtin,
               MessageTypeIntrinsic<T>::pointer>::get(*this, val);
    }

    // specialized version for string class
    Message& get(const std::string& s)
    {
        std::string& ncs = const_cast<std::string&>(s);
        int len = 0;
        get(len);
        char* cstring = new char[len];
        get(cstring);
        ncs = cstring;
        delete [] cstring;
        return *this;
    }

    // this version of get just removes the data without copying it
    Message &get()
    {
        return getmsg(0);
    }

    // an iterator-based version of get, which uses a general
    // iterator approach to copy the data out of the storage into
    // the space pointed to by the given iterator
    //////////////////////////////////////////////////////////////////////
    // Template get using a pair of iterators.  This version is called by
    // the public 'get' with two iterators after getting the proper type
    // of the data pointed to by the iterators
    template <class OutputIterator>
    Message& get_iter(OutputIterator o)
    {
        return PutSingleItem<OutputIterator,
               MessageTypeIntrinsic<OutputIterator>::builtin,
               MessageTypeIntrinsic<OutputIterator>::pointer>::get_iter(*this, o);
    }

    // general get routine; called by other cases
    // arguments are the location where to write the data, and the type
    // of the data.
    Message &getmsg(void *);

private:
    // MsgItem's that are in this message
    std::vector<MsgItem> MsgItemList;

    // number of elements that have been removed (via get or remove)
    size_t numRemoved;

    // should we copy the next added item?  if not, just store pointer
    bool DoCopy;

    // should we be responsible for deleting the next item?
    bool DoDelete;

    // a Communicate object which should be informed when this object is
    // deleted, and a comm-supplied object that it might need
    Communicate *comm;
    void *commdata;

    // delete the data in the top MsgItem, and remove that item from the top
    // (by incrementing numRemoved)
    void deleteMsgItem();
};


// General template for the put routine:
template<class T>
inline void putMessage(Message &m, const T &t)
{
    m.put(t);
}

// for using a pair of iterators
template<class ForwardIterator>
inline void putMessage(Message &m, ForwardIterator beg, ForwardIterator end)
{
    m.put(beg, end);
}

// for using an indirection list
template <class RandomAccessIterator>
inline void putMessage(Message &m, const std::vector<size_t> &v,
                       RandomAccessIterator r)
{
    m.put(v, r);
}


// General template for the get routine:
template<class T>
inline void getMessage(Message &m, T &t)
{
    m.get(t);
}

// this templated version of getMessage is for arbitrary pointers
template<class T>
inline void getMessage(Message &m, T *t)
{
    m.get_iter(t);
}

// this templated version of getMessage is for arbitrary pointers
template<class T>
inline void getMessage(Message &m, T *t, T *)
{
    m.get_iter(t);
}

// an iterator-based version of get, which uses a general
// iterator approach to copy the data out of the storage into
// the space pointed to by the given iterator
template<class OutputIterator>
inline void getMessage_iter(Message &m, OutputIterator o)
{
    m.get_iter(o);
}

#include "Message/Message.hpp"

#endif // MESSAGE_H
