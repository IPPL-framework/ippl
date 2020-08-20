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
#include "Message/Message.h"
#include "Message/Communicate.h"
#include "Utility/Inform.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"


#include <new>
#include <memory>
using namespace std;

#include <cstdlib>
#include <cstddef>

//////////////////////////////////////////////////////////////////////
// destructor: delete all items in this message, as if 'get' had been
// called for all the items
Message::~Message()
{
    

    // remove all unused items
    clear();

    // delete the storage in the existing items
    int n = MsgItemList.size();
    for (int i=0; i < n; ++i)
        MsgItemList[i].deleteData();

    // tell the communicate item to cleanup the provided data,
    // if necessary
    if (comm != 0)
        comm->cleanupMessage(commdata);
}


//////////////////////////////////////////////////////////////////////
// remove the top MsgItem
void Message::deleteMsgItem()
{
    

    if (!empty())
        numRemoved++;
}


//////////////////////////////////////////////////////////////////////
// general put routine; called by other cases
// arguments are the item, its element size (in bytes),
// and how many items total to copy
Message& Message::putmsg(void *data, int s, int nelem)
{
    

    // if this is a scalar, we always copy, and have one element
    if (nelem < 1)
    {
        nelem = 1;
        DoCopy = true;
    }

    // initialize new MsgItem
    //Inform dbgmsg("Message::putmsg", INFORM_ALL_NODES);
    //dbgmsg << "Putting data " << MsgItemList.size() << " at " << data;
    //dbgmsg << " with " << nelem;
    //dbgmsg << " elements, totbytes=" << s * nelem << ", copy=" << DoCopy;
    //dbgmsg << ", del=" << DoDelete << endl;
    MsgItem m(data, nelem, s * nelem, DoCopy, DoDelete);

    // add MsgItem to the pool
    MsgItemList.push_back(m);

    // reset copy and delete flags to default values
    DoCopy = DoDelete = true;
    return *this;
}


//////////////////////////////////////////////////////////////////////
// General get routine; called by other cases.
// Arguments are the location where to write the data.
// If this is called with a 0 destination location, then it just
// deletes the top MsgItem.
Message& Message::getmsg(void *data)
{
    

    // check to see if there is an item
    if (empty())
    {
        ERRORMSG("Message::getmsg() no more items in Message" << endl);
        PAssert(!empty());
    }
    else
    {
        // get the next MsgItem off the top of the list
        MsgItem &m = item(0);

        // copy the data into the given location
        if (m.data() != 0 && data != 0)
        {
            //Inform dbgmsg("Message::getmsg", INFORM_ALL_NODES);
            //dbgmsg << "Getting item " << removed() << " from loc=" << m.data();
            //dbgmsg << " to loc=" << data << " with totbytes=" << m.numBytes();
            //dbgmsg << endl;
            memcpy(data, m.data(), m.numBytes());
        }

        // delete this top MsgItem
        deleteMsgItem();
    }

    return *this;
}


//////////////////////////////////////////////////////////////////////
// clear the message; remove all the MsgItems
Message& Message::clear(void)
{
    

    while (!empty())
        deleteMsgItem();

    return *this;
}


//////////////////////////////////////////////////////////////////////
// return and remove the next item from this message ... if the item
// does not exist, NULL is returned.  This is similar to get, except that
// just a void* pointer to the data is returned (instead of having the
// data copied into given storage), and this data is DEFINITELY a malloced
// block of data that the user must deallocate using 'free' (NOT delete).
// Like 'get', after this is called, the top MsgItem is removed.
// If you wish to just access the Nth item's item  pointer, use
// item(N).item  .
void *Message::remove()
{
    

    // get the data out of the item
    MsgItem &m = item(0);
    void *retdata = m.data();

    // make a copy of it, or just return it if a delete is needed
    if (m.willNeedDelete())
    {
        m.cancelDelete();
    }
    else if (retdata != 0)
    {
        retdata = malloc(m.numBytes());
        memcpy(retdata, m.data(), m.numBytes());
    }

    // delete the item
    deleteMsgItem();

    // return the data
    return retdata;
}


//////////////////////////////////////////////////////////////////////
// use the << operator to print out a summary of the message
std::ostream& operator<<(std::ostream& o, const Message& m)
{

    o << "Message contains " << m.size() << " items (" << m.removed();
    o << " removed).  Contents:\n";
    for (size_t i = 0 ; i < m.size(); ++i)
    {
        const Message::MsgItem &mi = m.item(i);
        o << "  Item " << i << ": " << mi.numElems() << " elements, ";
        o << mi.numBytes() << " bytes total, needDelete = ";
        o << mi.willNeedDelete() << endl;
    }
    return o;
}
