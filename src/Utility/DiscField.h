// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DISC_FIELD_H
#define DISC_FIELD_H

// include files
#include "Index/NDIndex.h"
#include "Field/BrickExpression.h"
#include "Field/Field.h"
#include "Utility/DiscBuffer.h"
#include "Utility/DiscConfig.h"
#include "Utility/Inform.h"
#include "Utility/vmap.h"
#include "Utility/IpplTimings.h"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <vector>
#include <iostream>

// forward declarations
template<unsigned Dim, class T> class UniformCartesian;
template<class T, unsigned Dim, class M, class C> class Field;
template<unsigned Dim>                            class FieldLayout;


// This helper class is used to represent a record for I/O of the .offset file.
// It is only used for reads and writes.  See the notes below for
// the reason the vnodedata is a set of ints instead of an NDIndex.
template <unsigned Dim, class T>
struct DFOffsetData {
  int           vnodedata[6*Dim];
  bool          isCompressed;
  long long     offset;
  T             compressedVal;
};


template <unsigned Dim>
class DiscField {

public:
  // Constructor: make a DiscField for writing only
  // fname = name of file (without extensions
  // config = name of configuration file
  // numFields = number of Fields which will be written to the file
  // typestr = string describing the 'type' of the Field to be written (this
  //           is ignored if the Field is being read).  The string should be
  //           the same as the statement used to declare the Field object
  //           e.g., for a field of the form Field<double,2> the string
  //           should be "Field<double,2>".  The string should be such that
  //           if read later into a variable F, you can use the string in
  //           a source code line as  'F A' to create an instance of the
  //           same type of data as is stored in the file.
  DiscField(const char* fname, const char* config, unsigned int numFields,
	    const char* typestr = 0);

  // Constructor: same as above, but without a config file specified.  The
  // default config file entry that will be used is "*  .", which means, for
  // each SMP machine, assume the directory to put data files in is "."
  DiscField(const char* fname, unsigned int numFields,
	    const char* typestr = 0);

  // Constructor: make a DiscField for reading only.
  // fname = name of file (without extensions
  // config = name of configuration file
  DiscField(const char* fname, const char* config);

  // Constructor: same as above, but without a config file specified.  The
  // default config file entry that will be used is "*  .", which means, for
  // each SMP machine, assume the directory to put data files in is "."
  DiscField(const char* fname);

  // Destructor.
  ~DiscField();

  //
  // accessor functions
  //

  // Obtain all the information about the file, including the number
  // of records, fields, and number of vnodes stored in each record.
  void query(int& numRecords, int& numFields, std::vector<int>& size) const;

  // Query for the number of records (e.g., timesteps) in the file.
  unsigned int get_NumRecords() const { return NumRecords; }

  // Query for the number of Fields stored in the file.
  unsigned int get_NumFields() const { return NumFields; }

  // Query for the total domain of the system.
  NDIndex<Dim> get_Domain() const { return Size; }

  // Query for the dimension of the data in the file.  This is useful
  // mainly if you are checking for dimension by contructing a DiscField
  // and then trying to check it's dimension later.  If the dimension is
  // not correctly matched with the Dim template parameter, you will get
  // an error if you try to read or write.
  unsigned int get_Dimension() const { return DataDimension; }

  // Query for the type string
  const char *get_TypeString() { return TypeString.c_str(); }

  // Query for the disctype string
  const char *get_DiscType() {
    if (DiscType.length() > 0)
      return DiscType.c_str();
    return 0;
  }

  //
  // read/write methods
  //

  // read the selected record in the file into the given Field object.
  // readDomain = the portion of the field on disk that should actually be 
  //              read in, and placed in the same location in the
  //              provided field.  All of readDomain must be contained
  //              within the data on disk, and in the field in memory,
  //              although the domain on disk and in memory do not themselves
  //              have to be the same.
  // varID = index for which field is being read ... this should be from
  //         0 ... (numFields-1), for the case where the file contains
  //         more than one Field.
  // record = which record to read.  DiscField does not keep a 'current
  //          file position' pointer, instead you explicitly request which
  //          record you wish to read.
  // Return success of operation.

  template <class T, class M, class C>
  bool read(Field<T,Dim,M,C>& f, const NDIndex<Dim> &readDomain,
	    unsigned int varID, unsigned int record) {

    // sanity checking for input arguments and state of this object
    bool canread = false;
    if (!ConfigOK) {
      ERRORMSG("Cannot read in DiscField::read - config file error." << endl);
    } else if (DataDimension != Dim) {
      ERRORMSG("Bad dimension "<< DataDimension <<" in DiscField::read"<<endl);
      ERRORMSG("(" << DataDimension << " != " << Dim << ")" << endl);
    } else if (WritingFile) {
      ERRORMSG("DiscField::read called for DiscField opened for write."<<endl);
    } else if (varID >= NumFields) {
      ERRORMSG(varID << " is a bad Field ID in DiscField::read." << endl);
      ERRORMSG("(" << varID << " is >= " << NumFields << ")" << endl);
    } else if (record >= NumRecords) {
      ERRORMSG(record << " is a bad record number in DiscField::read."<<endl);
      ERRORMSG("(" << record << " is >= " << NumRecords << ")" << endl);
    } else if (!(f.getLayout().getDomain().contains(readDomain))) {
      ERRORMSG("DiscField::read - the total field domain ");
      ERRORMSG(f.getLayout().getDomain() << " must contain the requested ");
      ERRORMSG("read domain " << readDomain << endl);
    } else if (!(get_Domain().contains(readDomain))) {
      ERRORMSG("DiscField::read - the DiscField domain ");
      ERRORMSG(get_Domain() << " must contain the requested ");
      ERRORMSG("read domain " << readDomain << endl);
    } else {
      canread = true;
    }

    // If there was an error, we will abort
    if (!canread) {
      Ippl::abort("Exiting due to DiscField error.");
      return false;
    }

    // A typedef used later
    typedef typename LField<T,Dim>::iterator LFI;
    typedef BrickExpression<Dim,LFI,LFI,OpAssign> Expr_t;

    // Start timer for just the read portion
    static IpplTimings::TimerRef readtimer =
      IpplTimings::getTimer("DiscField read");
    IpplTimings::startTimer(readtimer);

    // Get a new tag value for this read operation, used for all sends
    // to other nodes with data.
    int tag = Ippl::Comm->next_tag(DF_READ_TAG, DF_TAG_CYCLE);

    // At the start of a new record, determine how many elements of the
    // Field should be stored into this node's vnodes.
    int expected = compute_expected(f.getLayout(), readDomain);

    // On all nodes, loop through all the file sets, and:
    //   1. box0: Get the number of vnodes stored there, from the layout file
    //   2. box0: Get offset information for all the vnodes, and
    //      assign other nodes on the same SMP selected vnodes to read.
    //   2. For each vnode assigned to a processor to read:
    //       - read data (if necessary)
    //       - distribute data to interested parties, or yourself
    // On all nodes, when you get some data from another node:
    //       - copy it into the relevant vnode
    //       - decrement your expected value.
    // When expected hits zero, we're done with reading on that node.

    for (unsigned int sf=0; sf < numFiles(); ++sf) {

      // Create the data file handle, but don't yet open it ... only open
      // it if we need to later (if we run into any uncompressed blocks).
      int outputDatafd = (-1);

      // offset data read in from file or obtained from the box0 node
      std::vector<DFOffsetData<Dim,T> > offdata;

      // the number of vnodes we'll be working on on this node
      int vnodes = 0;

      // the maximum number of elements we'll need to have buffer space for
      int maxsize = 0;

      // on box0 nodes, read in the layout and offest info
      if ((unsigned int) Ippl::myNode() == myBox0()) {

	// Get the number of vnodes in this file.
	vnodes = read_layout(record, sf);

	// Get the offset data for this field and record.
	read_offset(varID, record, sf, offdata, vnodes);
      }

      // On all nodes, either send out or receive in offset information.
      // Some nodes will not get any, and will not have to do any reading.
      // But those that do, will read in data for the vnodes they are
      // assigned.  'vnodes' will be set to the number of vnodes assigned
      // for reading from this node, and  'maxsize' will be set
      // to the maximum size of the vnodes in this file, for use in
      // preparing the buffer that will be used to read those vnodes.
      distribute_offsets(offdata, vnodes, maxsize, readDomain);

      // Loop through all the vnodes now; they will appear in any
      // order, which is fine, we just read them and and see where they
      // go.  The info in the offset struct includes the domain for that
      // block and whether it was written compressed or not.

      for (int vn=0; vn < vnodes; ++vn) {
	// Create an NDIndex object storing the vnode domain for this vnode.
	NDIndex<Dim> vnodeblock;
	offset_data_to_domain(offdata[vn], vnodeblock);

	// If there is no intersection of this vnode and the read-domain,
	// we can just skip it entirely.
	if (! vnodeblock.touches(readDomain)) {
	  continue;
	}

	// Compute the size of a block to add to the base of this block,
	// based on the chunk size.  If the data is compressed, this won't
	// matter.
	int msdim = (Dim-1);	// this will be zero-based
	int chunkelems = Ippl::chunkSize() / sizeof(T);
	NDIndex<Dim> chunkblock = chunk_domain(vnodeblock, chunkelems, msdim,
					       offdata[vn].isCompressed);
 
	// Initialize the NDIndex we'll use to indicate what portion of the
	// domain we're reading and processing.
	NDIndex<Dim> currblock = vnodeblock;
	currblock[msdim] = Index(vnodeblock[msdim].first() - 1,
				 vnodeblock[msdim].first() - 1);
	for (unsigned int md = (msdim+1); md < Dim; ++md)
	  currblock[md] = Index(vnodeblock[md].first(),vnodeblock[md].first());

	// Initialize the offset value for this vnode.  The seek position
	// is stored as a byte offset, although it is read from disk as
	// a number of elements offset from the beginning.
	Offset_t seekpos = (-1);

	// Loop through the chunks, reading and processing each one.
	int unread = vnodeblock.size();
	while (unread > 0) {
	  // Compute the domain of the chunk we'll work on now, and store
	  // this in currblock.

	  // First determine if we're at the end of our current incr dimension,
	  // and determine new bounds
	  bool incrhigher=(currblock[msdim].last()==vnodeblock[msdim].last());
	  int a = (incrhigher ?
		   vnodeblock[msdim].first() :
		   currblock[msdim].last() + 1);
	  int b = a + chunkblock[msdim].length() - 1;
	  if (b > vnodeblock[msdim].last())
	    b = vnodeblock[msdim].last();

	  // Increment this dimension
	  currblock[msdim] = Index(a, b);

	  // Increment higher dimensions, if necessary
	  if (incrhigher) {
	    for (unsigned int cd = (msdim+1); cd < Dim; ++cd) {
	      if (currblock[cd].last() < vnodeblock[cd].last()) {
		// This dim is not at end, so just inc by 1
		currblock[cd] = Index(currblock[cd].first() + 1,
				      currblock[cd].last() + 1);
		break;
	      } else {
		// Move this dimension back to start, and go on to next one
		currblock[cd] = Index(vnodeblock[cd].first(),
				      vnodeblock[cd].first());
	      }
	    }
	  }

	  // Decrement our unread count, since we'll process this block
	  // either by actually reading it or getting its compressed value
	  // from the offset file, if we have to read it at all.
	  int nelems = currblock.size();
	  unread -= nelems;

	  // Set the seek position now, if necessary
	  if (!offdata[vn].isCompressed && seekpos < 0) {
	    seekpos = offdata[vn].offset * sizeof(T);
	  }

	  // At this point, we might be able to skip a lot of work if this
	  // particular chunk does not intersect with our read domain any.
	  if (! currblock.touches(readDomain)) {
	    // Before we skip the rest, we must update the offset
	    Offset_t readbytes  = nelems * sizeof(T);
	    seekpos += readbytes;

	    // Then, we're done with this chunk, move on to the next.
	    continue;
	  }

	  // Put the intersecting domain in readDomainSection.
	  NDIndex<Dim> readDomainSection = currblock.intersect(readDomain);

	  // if it is not compressed, read in the data.  If it is,
	  // just keep the buffer pointer at zero.
	  T *buffer = 0;
	  if (!offdata[vn].isCompressed) {
	    // If we have not yet done so, open the data file.
	    if (outputDatafd < 0) {
	      outputDatafd = open_df_file_fd(Config->getFilename(sf), ".data",
					     O_RDONLY);
	    }

	    // Resize the read buffer in case it is not large enough.
	    // We use the max size for all the vnodes here, to avoid doing
	    // this more than once per file set.  This also returns the
	    // pointer to the buffer to use, as a void *, which we cast
	    // to the proper type.  For direct-io, we might need to make
	    // this a little bigger to match the device block size.

	    long nbytes = maxsize*sizeof(T);
	    buffer = static_cast<T *>(DiscBuffer::resize(nbytes));

	    // Create some initial values for what and where to read.
	    // We might adjust these if we're doing direct-io.
	    T *      readbuffer = buffer;
	    Offset_t readbytes  = nelems * sizeof(T);
	    Offset_t readoffset = seekpos;

	    // seekpos was only used to set readoffset, so we can update
	    // seekpos now.  Add in the extra amount we'll be reading.
	    seekpos += readbytes;

	    // Read data in a way that might do direct-io
	    read_data(outputDatafd, readbuffer, readbytes, readoffset);
	  }

	  // we have the data block now; find out where the data should
	  // go, and either send the destination node a message, or copy
	  // the data into the destination lfield.

	  // Set up to loop over the touching remote vnodes, and send out
	  // messages
	  typename FieldLayout<Dim>::touch_iterator_dv rv_i;
	  //	  int remaining = nelems;
	  int remaining = readDomainSection.size();

	  // compute what remote vnodes touch this block's domain, and
	  // iterate over them.
	  //	  typename FieldLayout<Dim>::touch_range_dv
	  //	    range(f.getLayout().touch_range_rdv(currblock));
	  typename FieldLayout<Dim>::touch_range_dv
	    range(f.getLayout().touch_range_rdv(readDomainSection));
	  for (rv_i = range.first; rv_i != range.second; ++rv_i) {
	    // Compute the intersection of our domain and the remote vnode
	    //	    NDIndex<Dim> ri = currblock.intersect((*rv_i).first);
	    NDIndex<Dim> ri = readDomainSection.intersect((*rv_i).first);
	    
	    // Find out who will be sending this data
	    int rnode = (*rv_i).second->getNode();

	    // Send this data to that remote node, by preparing a
	    // CompressedBrickIterator and putting in the proper data.
	    Message *msg = new Message;
	    ri.putMessage(*msg);
	    LFI cbi(buffer, ri, currblock, offdata[vn].compressedVal);
	    cbi.TryCompress();
	    cbi.putMessage(*msg, false);  // 'false' = avoid copy if possible
	    Ippl::Comm->send(msg, rnode, tag);

	    // Decrement the remaining count
	    remaining -= ri.size();
	  }

	  // loop over touching local vnodes, and copy in data, if there
	  // is anything left
	  typename BareField<T,Dim>::iterator_if lf_i = f.begin_if();
	  for (; remaining > 0 && lf_i != f.end_if(); ++lf_i) {
	    // Get the current LField and LField domain, and make an alias
	    // for the domain of the block we've read from disk
	    LField<T,Dim> &lf = *(*lf_i).second;
	    const NDIndex<Dim>& lo = lf.getOwned();
	    //	    const NDIndex<Dim>& ro = currblock;
	    const NDIndex<Dim>& ro = readDomainSection;

	    // See if it touches the domain of the recently read block.
	    if (lo.touches(ro)) {
	      // Find the intersection.
	      NDIndex<Dim> ri = lo.intersect(ro);

	      // If these are compressed we might not have to do any work.
	      if (lf.IsCompressed() &&
		  offdata[vn].isCompressed &&
		  ro.contains(lo)) {
		PETE_apply(OpAssign(),*lf.begin(),offdata[vn].compressedVal);
	      } else {
		// Build an iterator for the read-data block
		// LFI rhs_i(buffer, ri, ro, offdata[vn].compressedVal);
		LFI rhs_i(buffer, ri, currblock, offdata[vn].compressedVal);

		// Could we compress that rhs iterator?
		if (rhs_i.CanCompress(*rhs_i) && f.compressible() &&
		    ri.contains(lf.getAllocated())) {
		  // Compress the whole LField to the value on the right
		  lf.Compress(*rhs_i);
		} else { // Assigning only part of LField on the left
		  // Must uncompress lhs, if not already uncompressed
		  lf.Uncompress(true);

		  // Get the iterator for it.
		  LFI lhs_i = lf.begin(ri);

		  // And do the assignment.
		  Expr_t(lhs_i,rhs_i).apply();
		}
	      }

	      // Decrement the expected count and the remaining count.
	      // Remaining is how many cells are left of the current block.
	      // Expected is how many cells this node expects to get copied
	      // into its blocks.
	      int bsize = ri.size();
	      remaining -= bsize;
	      expected -= bsize;
	    }
	  }

	  // If we're here and still have remaining elements, we're screwed.
	  if (remaining > 0)
	    Ippl::abort("remaining > 0 at end of box0 vnode read!!!");
	}
      }

      // Close the data file now
      
      if (outputDatafd >= 0)
	close(outputDatafd);
    }

    // On all nodes, now, keep receiving messages until our expected count
    // goes to zero.
    while (expected > 0) {
      // Receive the next message from any node with the current read tag
      int node = COMM_ANY_TAG;
      Message *msg = Ippl::Comm->receive_block(node, tag);

      // Extract the domain from the message
      NDIndex<Dim> ro;
      ro.getMessage(*msg);

      // Extract the data from the message
      T rhs_compressed_data;
      LFI rhs_i(rhs_compressed_data);
      rhs_i.getMessage(*msg);

      // Find what local LField contains this domain
      typename BareField<T,Dim>::iterator_if lf_i = f.begin_if();
      bool foundlf = false;
      for (; lf_i != f.end_if(); ++lf_i) {
	// Get the current LField and LField domain
	LField<T,Dim> &lf = *(*lf_i).second;
	const NDIndex<Dim>& lo = lf.getOwned();

	// See if it contains the domain of the recently received block.
	// If so, assign the block to this LField
	if (lo.contains(ro)) {

	  // Check and see if we really have to do this.
	  if ( !(rhs_i.IsCompressed() && lf.IsCompressed() &&
		 (*rhs_i == *lf.begin())) ) {
	    // Yep. gotta do it, since something is uncompressed or
	    // the values are different.

	    // Uncompress the LField first, if necessary.  It's necessary
	    // if the received block size is smaller than the LField's.
	    lf.Uncompress(!ro.contains(lo));

	    // Make an iterator over the received block's portion of the
	    // LField
	    LFI lhs_i = lf.begin(ro);

	    // Do the assignment.
	    Expr_t(lhs_i,rhs_i).apply();
	  }

	  // Update our expected value
	  expected -= ro.size();

	  // Indicate we're done, since the block we received is
	  // guaranteed to be within only one of our LFields.
	  foundlf = true;
	  break;
	}
      }

      // Make sure we found what vnode this message is for; if we don't
      // we're screwed
      if (!foundlf) {
	ERRORMSG("Did not find destination local vnode for received domain ");
	ERRORMSG(ro << " from node " << node << endl);
	Ippl::abort("DID NOT FIND DESINATION LOCAL VNODE IN DISCFIELD::READ");
      }

      // Now we are done with the message
      delete msg;
    }

    // We're all done reading, so clean up
    IpplTimings::stopTimer(readtimer);

    // This is just like an assign, so set dirty flags, fill guard cells,
    // and try to compress the result.

    f.setDirtyFlag();
    f.fillGuardCellsIfNotDirty();
    f.Compress();

    // Let everything catch up
    Ippl::Comm->barrier();

    // print out malloc info at end of read
    //     Inform memmsg("DiscField::read::mallinfo");
    //     struct mallinfo mdata;
    //     mdata = mallinfo();
    //     memmsg << "After read, new malloc info:" << endl;
    //     memmsg << "----------------------------" << endl;
    //     memmsg << "  total arena space = " << mdata.arena << endl;
    //     memmsg << "    ordinary blocks = " << mdata.ordblks << endl;
    //     memmsg << "       small blocks = " << mdata.smblks << endl;
    //     memmsg << "    user-held space = " << mdata.usmblks+mdata.uordblks;
    //     memmsg << endl;
    //     memmsg << "         free space = " << mdata.fsmblks+mdata.fordblks;
    //     memmsg << endl;

    return true;
  }

  // versions of read that provide default values for the arguments
  template <class T, class M, class C>
  bool read(Field<T,Dim,M,C>& f, unsigned int varID, unsigned int record) {
    return read(f, f.getLayout().getDomain(), varID, record);
  }

  template <class T, class M, class C>
  bool read(Field<T,Dim,M,C>& f, const NDIndex<Dim> &readDomain,
	    unsigned int varID) {
    return read(f, readDomain, varID, 0);
  }

  template <class T, class M, class C>
  bool read(Field<T,Dim,M,C>& f, unsigned int varID) {
    return read(f, f.getLayout().getDomain(), varID, 0);
  }

  template <class T, class M, class C>
  bool read(Field<T,Dim,M,C>& f, const NDIndex<Dim> &readDomain) {
    return read(f, readDomain, 0, 0);
  }

  template <class T, class M, class C>
  bool read(Field<T,Dim,M,C>& f) {
    return read(f, f.getLayout().getDomain(), 0, 0);
  }


  ///////////////////////////////////////////////////////////////////////////
  // write the data from the given Field into the file.  This can be used
  // to either append new records, or to overwrite an existing record.
  // varID = index for which field is being written ... this should be from
  //         0 ... (numFields-1), for the case where the file contains
  //         more than one Field.  Writing continues for the current record
  //         until all numField fields have been written; then during the
  //         next write, the record number is incremented.
  //--------------------------------------------------------------------
  // notes for documentation:
  // - for now, you can not overwrite Field data on succesive writes
  //   (this can easily be added when needed - some restrictions will apply)
  // - when writing, all the Fields in a record must have the same layout
  // - layouts (and vnodes) can vary between records, however
  // - separate DiscField objects need to be opened for reading and writing
  //--------------------------------------------------------------------
  template<class T, class M, class C>
  bool write(Field<T,Dim,M,C>& f, unsigned int varID) {

    // sanity checking for input arguments and state of this object
    if (!ConfigOK) {
      ERRORMSG("Cannot write in DiscField::write - config file error."<<endl);
      Ippl::abort("Exiting due to DiscField error.");
      return false;
    }
    else if (!WritingFile) {
      ERRORMSG("DiscField::write called for DiscField opened for read."<<endl);
      Ippl::abort("Exiting due to DiscField error.");
      return false;
    }
    else if (varID >= NumFields) {
      ERRORMSG(varID << " is a bad variable ID in DiscField::write." << endl);
      Ippl::abort("Exiting due to DiscField error.");
      return false;
    }
    else if (NeedStartRecord == 0 && ValidField[varID]) {
      ERRORMSG("DiscField:write - attempt to overwrite Field " << varID);
      ERRORMSG(" at record " << NumRecords - 1 << endl);
      Ippl::abort("Exiting due to DiscField error.");
      return false;
    }

    //    INCIPPLSTAT(incDiscWrites);

    // useful typedefs for later
    typedef typename LField<T,Dim>::iterator LFI;

    // Get a new tag value for this write operation, used for all sends
    // to other nodes with data.
    int tag = Ippl::Comm->next_tag(FB_WRITE_TAG, FB_TAG_CYCLE);

    // Get the layout reference, and set up an iterator over lfields
    FieldLayout<Dim>& layout = f.getLayout();
    typename Field<T,Dim,M,C>::iterator_if local;

    // do we need to start a new record?  If so, extend data structures and
    // file storage
    if (NeedStartRecord != 0) {
      // convert the layout information for the field into internal storage,
      // represented as a map from NDIndex --> owner node
      if (!make_globalID(layout)) {
	ERRORMSG("DiscField::write - all Field's must have the same ");
	ERRORMSG("global domain in a single DiscField.\n");
	ERRORMSG("The original domain is " << get_Domain() << ", ");
	ERRORMSG("the attempted new domain is " << layout.getDomain() << endl);
	Ippl::abort("Exiting due to DiscField error.");
      }

      // update vnode and valid field information for new record
      if (numFiles() > 0 && myBox0() == (unsigned int) Ippl::myNode()) {
	int nvtally = 0;
	NumVnodes[0].push_back(globalID.size());
	if (NumRecords > 0)
	  nvtally = VnodeTally[0][NumRecords-1] + NumVnodes[0][NumRecords-1];
	VnodeTally[0].push_back(nvtally);
      }

      // indicate we have not written out data for any fields for this record
      for (unsigned int i=0; i < NumFields; ++i)
	ValidField[i] = false;

      // increment total record number
      NumRecords++;
      NumWritten = 0;

      // update necessary data files at the start of a new record
      if ((unsigned int) Ippl::myNode() == myBox0()) {
	// Update the meta information ... this can be changed to be only
	// written out during destruction.
	if (!write_meta()) {
	  ERRORMSG("Could not write .meta file on node " << Ippl::myNode());
	  ERRORMSG(endl);
	  Ippl::abort("Exiting due to DiscField error.");
	  return false;
	}

	// write out the NDIndex objects from the FieldLayout to the
	// .layout file
	if (!write_layout()) {
	  ERRORMSG("Could not update .layout file on node "<<Ippl::myNode());
	  ERRORMSG(endl);
	  Ippl::abort("Exiting due to DiscField error.");
	  return false;
	}
      }
    }

    // On box0 nodes, do most of the work ... other nodes forward data
    // to box0 nodes.
    if ((unsigned int) Ippl::myNode() == myBox0()) {

      // Open the offset data file, and write the Field number.
      // This precedes all the OffsetData structs for vnodes in the Field,
      // which are in random order for the given Field.  The OffsetData
      // structs contains a field 'vnode' which tells which vnode the data is
      // for (0 ... numVnodes - 1).
      FILE *outputOffset = open_df_file(Config->getFilename(0),
					".offset", std::string("a"));
      int wVarID = (int)varID;
      if (fwrite(&wVarID, sizeof(int), 1, outputOffset) != 1) {
	ERRORMSG("DiscField::write - cannot write field number to .offset ");
	ERRORMSG("file" << endl);
	Ippl::abort("Exiting due to DiscField error.");
	fclose(outputOffset);
	return false;
      }

      // Initialize output file handle ... we might never write anything to
      // it if the field is completely compressed.
      int outputDatafd = open_df_file_fd(Config->getFilename(0), ".data",
					 O_RDWR|O_CREAT);
      // Later we will receive message from other nodes.  This is the
      // number of blocks we should receive.  We'll decrease this by
      // the number of vnodes we already have on this processor, however.
      int unreceived = globalID.size();
      int fromothers = unreceived - layout.size_iv();

      // Now we start processing blocks.  We have 'unreceived' total to
      // write, either from ourselves or others.  We first check for
      // messages, if there are any to receive, get one and process it,
      // otherwise write out one of our local blocks.

      local = f.begin_if();
      while (unreceived > 0) {
	// Keep processing remote blocks until we don't see one available
	// or we've received them all
	bool checkremote = (fromothers > 0);
	while (checkremote) {
	  // Get a message
	  int any_node = COMM_ANY_NODE;
	  Message *msg = Ippl::Comm->receive(any_node, tag);

	  // If we found one, process it
	  if (msg != 0) {
	    // Extract the domain from the message
	    NDIndex<Dim> ro;
	    ro.getMessage(*msg);
	    
	    // Extract the data from the message
	    T rhs_compressed_data;
	    LFI cbi(rhs_compressed_data);
	    cbi.getMessage(*msg);

	    // Write this data out
	    write_offset_and_data(outputOffset, outputDatafd, cbi, ro);

	    // finish with this message
	    delete msg;

	    // Update counters
	    unreceived -= 1;
	    fromothers -= 1;
	  } else {
	    // We didn't see one, so stop checking for now
	    checkremote = false;
	  }
	}

	// Process a local block if any are left
	if (local != f.end_if()) {
	  // Cache some information about this local field.
	  LField<T,Dim>& l = *(*local).second.get();
	  LFI cbi = l.begin();
	  
	  // Write this data out
	  write_offset_and_data(outputOffset, outputDatafd, cbi, l.getOwned());

	  // Update counters
	  ++local;
	  unreceived -= 1;
	}
      }

      // Close the output data file
      if (outputDatafd >= 0)
	close(outputDatafd);

      // Close the output offset file
      if (outputOffset != 0)
	fclose(outputOffset);

    } else {
      // On other nodes, just send out our LField blocks.
      for (local = f.begin_if(); local != f.end_if(); ++local) {
	// Cache some information about this local field.
	LField<T,Dim>& l = *(*local).second.get();
	const NDIndex<Dim> &ro = l.getOwned();
	LFI cbi = l.begin();

	// Create a message to send to box0
	Message *msg = new Message;

	// Put in the domain and data
	ro.putMessage(*msg);
	cbi.putMessage(*msg, false);  // 'false' = avoid copy if possible

	// Send this message to the box0 node.
	int node = myBox0();
	Ippl::Comm->send(msg, node, tag);
      }
    }

    // indicate we've written one more field
    ValidField[varID] = true;
    NeedStartRecord = (++NumWritten == NumFields);

    // Let everything catch up
    Ippl::Comm->barrier();

    // if we actually make it down to here, we were successful in writing
    return true;
  }

  // version of write that provides default value for varID
  template<class T, class M, class C>
  bool write(Field<T,Dim,M,C>& f) {
    return write(f, 0);
  }


  //
  // console printing methods
  //

  // print out debugging info to the given stream
  void printDebug(std::ostream&);
  void printDebug();

private:
  // private typedefs
  typedef vmap<NDIndex<Dim>, int>  GlobalIDList_t;
  typedef long long  Offset_t; 

  //
  // meta data (static info for the file which does not change)
  //

  // the configuration file mechanism
  DiscConfig *Config;
  bool ConfigOK;

  // flag which is true if we're writing, false if reading
  bool WritingFile;

  // the base name for the output file, and the descriptive type string
  std::string BaseFile;
  std::string TypeString;
  std::string DiscType;

  // dimension of data in file ... this may not match the template dimension.
  unsigned int DataDimension;

  //
  // dynamic data (varies as records are written or read)
  //

  // do we need to start a new record during the next write?  Or, if reading,
  // which record have we current read into our Size and globalID variables?
  // If this is < 0, we have not read in any yet.
  int NeedStartRecord;

  // the number of fields and records in the file, and the number of Fields
  // written to the current record
  unsigned int NumFields;
  unsigned int NumRecords;
  unsigned int NumWritten;

  // this keeps track of where in the .data file writing is occuring
  // it is correlated with a given Field and record through the .offset file
  Offset_t CurrentOffset;

  // the global domain of the Fields in this DiscField object
  NDIndex<Dim> Size;

  // keep track of which Fields have been written to the current record
  std::vector<bool> ValidField;

  // the running tally of vnodes ON THIS SMP for each record, for each file.
  // VnodeTally[n] = number of vnodes written out total in prev records.
  // NOTE: this is not the number of vnodes TOTAL, it is the number of
  // vnodes on all the processors which are on this same SMP machine.
  std::vector<int> *VnodeTally;

  // the number of vnodes ON THIS SMP in each record, for each file.
  // NumVnodes[n] = number of vnodes in record n.
  // NOTE: this is not the number of vnodes TOTAL, it is the number of
  // vnodes on all the processors which are on this same SMP machine.
  std::vector<int> *NumVnodes;

  // store a mapping from an NDIndex to the physical node it resides on.
  // These values are stored only for those vnodes on processors which are
  // on our same SMP.  This must be remade when the layout changes
  // (e.g., every time we write a record for a Field,
  // since each Field can have a different layout and each Field's layout
  // can change from record to record).
  // key: local NDIndex, value: node
  GlobalIDList_t globalID;

  //
  // functions used to build/query information about the processors, etc.
  //

  // perform initialization based on the constuctor arguments
  void initialize(const char* base, const char* config,
		  const char* typestr, unsigned int numFields);

  // open a file in the given mode.  If an error occurs, print a message (but
  // only if the last argument is true).
  // fnm = complete name of file (can include a path)
  // mode = open method ("r" == read, "rw" == read/write, etc.
  FILE *open_df_file(const std::string& fnm, const std::string& mode);
  FILE *open_df_file(const std::string& fnm, const std::string& suffix,
		     const std::string& mode);

  // Open a file using direct IO, if possible, otherwise without.  This
  // returns a file descriptor, not a FILE handle.  If the file was opened
  // using direct-io, also initialize the dioinfo member data.
  // The last argument indicates whether to init (create)
  // the file or just open for general read-write.
  int open_df_file_fd(const std::string& fnm, const std::string& suf, int flags);

  // create the data files used to store Field data.  Return success.
  bool create_files();

  // return the total number of SMP's in the system
  unsigned int numSMPs() const {
    return Config->numSMPs();
  }

  // return the total number of files which are being read or written
  unsigned int fileSMPs() const {
    return Config->fileSMPs();
  }

  // return the index of our SMP
  unsigned int mySMP() const {
    return Config->mySMP();
  }

  // return the Box0 node for this SMP
  unsigned int myBox0() const {
    return Config->getSMPBox0();
  }

  // return the number of files on our SMP (if no arg given) or the
  // given SMP
  unsigned int numFiles() const {
    return Config->getNumFiles();
  }
  unsigned int numFiles(unsigned int s) const {
    return Config->getNumFiles(s);
  }

  // compute how many physical nodes there are on the same SMP as the given
  // pnode.
  unsigned int pNodesPerSMP(unsigned int node) const {
    return Config->pNodesPerSMP(node);
  }

  // parse the IO configuration file and store the information.
  // arguments = name of config file, and if we're writing a file (true) or
  // reading (false)
  bool parse_config(const char *, bool);

  // Compute how many elements we should expect to store into the local
  // node for the given FieldLayout.  Just loop through the local vnodes
  // and accumulate the sizes of the owned domains.  This is modified
  // by the second "read domain" argument, which might be a subset of
  // the total domain.
  int compute_expected(const FieldLayout<Dim> &, const NDIndex<Dim> &);

  // Since the layout can be different every time write
  // is called, the globalID container needs to be recalculated.  The total
  // domain of the Field should not change, though, just the layout.  Return
  // success.
  bool make_globalID(FieldLayout<Dim> &);

  // Compute the size of a domain, zero-based, that has a total
  // size <= chunkelems and has evenly chunked slices.
  NDIndex<Dim> chunk_domain(const NDIndex<Dim> &currblock,
			    int chunkelems,
			    int &msdim,
			    bool iscompressed);

  //
  //
  // read/write functions for individual components
  //

  // read or write .meta data file information.  Return success.
  bool write_meta();
  bool read_meta();

  // read or write NDIndex values for a file.  Return success.
  bool read_NDIndex(FILE *, NDIndex<Dim> &);
  bool write_NDIndex(FILE *, const NDIndex<Dim> &);

  // Write .layout data file information.  Return success.
  bool write_layout();

  // Read layout info for one file set in the given record.
  int read_layout(int record, int sf);

  ///////////////////////////////////////////////////////////////////////////
  // Write out the data in a provided brick iterator with given owned
  // domain, updating both the offset file and the data file.  The .offset file
  // contains info  on where in the main data file each vnode's data can be
  // found.  The .offset file's structure looks like this:
  //   |--Record n----------------------------|
  //   | Field ID a                           |
  //   | VNa | VNb | VNc | VNd | .... | VNx   |
  //   | Field ID b                           |
  //   | VNa | VNb | VNc | VNd | .... | VNx   |
  //   | Field ID c                           |
  //   | VNa | VNb | VNc | VNd | .... | VNx   |
  //   |--------------------------------------|
  // where
  //   VNn is the data for a single Offset struct.  The sets of Offset structs
  //   for the Fields can appear in any order in a record, and the Offsets
  //   structs within a specific Field can also appear in any order.
  template<class T>
  void write_offset_and_data(FILE *outputOffset, int outputDatafd,
			     CompressedBrickIterator<T,Dim> &cbi,
			     const NDIndex<Dim> &owned) {

    // Create an offset output file struct, and initialize what we can.
    // We must take care to first zero out the offset struct.
    DFOffsetData<Dim,T> offset;
    memset(static_cast<void *>(&offset), 0, sizeof(DFOffsetData<Dim,T>));

    domain_to_offset_data(owned, offset);
    offset.isCompressed = cbi.IsCompressed();
    offset.offset       = CurrentOffset;

    // Set the compressed or uncompressed data in the offset struct
    if (offset.isCompressed) {
      // For compressed data, we just need to write out the entry to the
      // offset file ... that will contain the single compressed value.
      offset.compressedVal = *cbi;
    } else {
      // This is not compressed, so we must write to the data file.  The
      // main question now is whether we can use existing buffers, or write
      // things out in chunks, or what.

      // First, calculate how many elements to write out at a time.  The
      // 'chunkbytes' might be adjusted to match the maximum amount of data
      // that can be written out in a single direct-io call.  This is
      // true only if we are actually using direct-io, of course.

      long elems = owned.size();
      long chunkbytes = Ippl::chunkSize();
      long chunksize = chunkbytes / sizeof(T);
      if (chunksize < 1 || chunksize > elems)
	chunksize = elems;

      // If cbi is iterating over its whole domain, we can just use the block
      // there as-is to write out data.  So if cbiptr is set to an non-zero
      // address, we'll just use that for the data, otherwise we'll have
      // to copy to a buffer.

      T *cbiptr = 0;
      if (cbi.whole())
	cbiptr = &(*cbi);

      // Loop through the data, writing out chunks.
      int needwrite = elems;
      while (needwrite > 0) {
	// Find out how many elements we'll write this time.
	int amount = chunksize;
	if (amount > needwrite)
	  amount = needwrite;

	// Find the size of a buffer of at least large enough size.  We
	// might need a slighly larger buffer if we are using direct-io,
	// where data must be written out in blocks with sizes that
	// match the device block size.
	size_t nbytes = amount*sizeof(T);

	// Get a pointer to the next data, or copy more data into a buffer
	// Initially start with the vnode pointer
	T *buffer = cbiptr;

	// If necessary, make a copy of the data
	if (buffer == 0) {
	  buffer = static_cast<T *>(DiscBuffer::resize(nbytes));

	  // Copy data into this buffer from the iterator.
	  T *bufptr = buffer;
	  T *bufend = buffer + amount;
	  for ( ; bufptr != bufend; ++bufptr, ++cbi)
	    new (bufptr) T(*cbi);
	}

	// Write the data now
	off_t seekoffset = CurrentOffset * sizeof(T);
	bool seekok = true;
	Timer wtimer;
	wtimer.clear();
	wtimer.start();

	size_t nout = 0;
	if (::lseek(outputDatafd, seekoffset, SEEK_SET) == seekoffset) {
          char *wbuf = (char *)buffer;
	  nout = ::write(outputDatafd, wbuf, nbytes);
	} else {
          seekok = false;
        }

	wtimer.stop();
	DiscBuffer::writetime += wtimer.clock_time();
	DiscBuffer::writebytes += nbytes;

	if (!seekok) {
	  ERRORMSG("Seek error in DiscField::write_offset_and_data" << endl);
	  ERRORMSG("Could not seek to position " << seekoffset << endl);
	  Ippl::abort("Exiting due to DiscField error.");
	}

	if (nout != nbytes) {
	  ERRORMSG("Write error in DiscField::write_offset_and_data" << endl);
	  ERRORMSG("Could not write " << nbytes << " bytes." << endl);
	  Ippl::abort("Exiting due to DiscField error.");
	}

	// Update pointers and counts
	needwrite -= amount;
	if (cbiptr != 0)
	  cbiptr += amount;

	// update the offset and stats

	CurrentOffset += (nbytes / sizeof(T));
      }
    }

    // write to offset file now
    if (fwrite(&offset, sizeof(DFOffsetData<Dim,T>), 1, outputOffset) != 1) {
      ERRORMSG("Write error in DiscField::write_offset_and_data" << endl);
      Ippl::abort("Exiting due to DiscField error.");
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // seek to the beginning of the vnode data for field 'varID' in record
  // 'record', for file 'sf'.  If not found, close the file and return false.
  // If it is found, read in all the offset records, and return them
  // in the provided vector.
  template <class T>
  bool read_offset(unsigned int varID,
		   unsigned int record,
		   unsigned int sf,
		   std::vector<DFOffsetData<Dim,T> > &offdata,
		   int vnodes) {

    // Open the offset file
    FILE *outputOffset = open_df_file(Config->getFilename(sf),
				      ".offset", std::string("r"));

    // seek to the start of this record
    Offset_t seekpos = NumFields * (record * sizeof(int) +
				    VnodeTally[sf][record] *
				    sizeof(DFOffsetData<Dim,T>));
    if (fseek(outputOffset, seekpos, SEEK_SET) != 0) {
      ERRORMSG("Error seeking to position " << static_cast<long>(seekpos));
      ERRORMSG(" in .offset file " << endl);
      Ippl::abort("Exiting due to DiscField error.");
      fclose(outputOffset);
      return false;
    }

    // now keep looking at the Field ID in this record until we find the one
    // we want
    unsigned int checked = 0;
    while (checked < NumFields) {
      // read the next field ID number
      int rVarID;
      if (fread(&rVarID, sizeof(int), 1, outputOffset) != 1) {
	ERRORMSG("Error reading field ID from .offset file" << endl);
	Ippl::abort("Exiting due to DiscField error.");
	fclose(outputOffset);
	return false;
      }

      // is it what we want?
      if ((unsigned int) rVarID == varID) {
        // Yes it is, so read in the offset record data.  First resize
	// the offset data vector.
	offdata.resize(vnodes);
	size_t result = fread(&(offdata[0]), sizeof(DFOffsetData<Dim,T>),
                              offdata.size(), outputOffset);
	if (result != offdata.size()) {
	  ERRORMSG("Read error in DiscField::find_file_in_offset" << endl);
	  ERRORMSG("Results is " << result << ", should be ");
	  ERRORMSG(offdata.size() << endl);
	  ERRORMSG("outputOffset is " << (void *)outputOffset << endl);
	  Ippl::abort("Exiting due to DiscField error.");
	  fclose(outputOffset);
	  return false;
	}

        // And return success.
	fclose(outputOffset);
	return true;
      }

      // it was not, so move on to the next one
      checked++;
      seekpos += (NumVnodes[sf][record] * sizeof(DFOffsetData<Dim,T>) +
		  sizeof(int));
      if (fseek(outputOffset, seekpos, SEEK_SET) != 0) {
	ERRORMSG("Error seeking to position " << static_cast<long>(seekpos));
	ERRORMSG(" in .offset file " << endl);
	Ippl::abort("Exiting due to DiscField error.");
	fclose(outputOffset);
	return false;
      }
    }

    // if we're here, we did not find the Field ID anywhere in the .offset file
    ERRORMSG("Could not find data for field " << varID << " of record ");
    ERRORMSG(record << " in .offset file." << endl);
    Ippl::abort("Exiting due to DiscField error.");
    fclose(outputOffset);
    return false;
  }

  ///////////////////////////////////////////////////////////////////////////
  // On all nodes, either send out or receive in offset information.
  // Some nodes will not get any, and will not have to do any reading.
  // But those that do, will read in data for the vnodes they are
  // assigned.  'vnodes' will be set to the number of vnodes assigned
  // for reading from this node, and  'maxsize' will be set
  // to the maximum size of the vnodes in this file, for use in
  // preparing the buffer that will be used to read those vnodes.
  template <class T>
  void distribute_offsets(std::vector<DFOffsetData<Dim,T> > &offdata,
			  int &vnodes, int &maxsize,
			  const NDIndex<Dim> &readDomain) {

    // Initialize the vnode and maxsize values.
    vnodes = 0;
    maxsize = 0;

    // If parallel reads are turned off, just box0 nodes will read
    if (!Ippl::perSMPParallelIO()) {
      if ((unsigned int) Ippl::myNode() == myBox0())
	vnodes = offdata.size();

    } else {

      // Generate the tag to use
      int tag = Ippl::Comm->next_tag(DF_OFFSET_TAG, DF_TAG_CYCLE);

      // Nodes that do not have their box0 process on the same SMP should
      // not receive anything
      if (Config->getNodeSMPIndex(myBox0()) != mySMP()) {
	return;
      }

      // All box0 nodes will (possibly) send out offset data.  Others will
      // receive it, even if it just says "you don't get any vnodes."
      if ((unsigned int) Ippl::myNode() == myBox0()) {
	// How many offset blocks per processor
	int pernode = offdata.size() / pNodesPerSMP(myBox0());

	// Extra vnodes we might have to give to others
	int extra = offdata.size() % pNodesPerSMP(myBox0());

	// The next vnode to assign; box0 will always get an extra one if
	// necessary.
	int nextvnode = pernode;
	if (extra > 0) {
	  nextvnode += 1;
	  extra -= 1;
	}

	// box0 nodes get the first 'nextvnode' vnodes.
	vnodes = nextvnode;

	// Loop through the nodes on this vnode; nodes other than box0 will
	// get sent a message.
	for (unsigned int n=0; n < Config->getNumSMPNodes(); ++n) {
	  int node = Config->getSMPNode(mySMP(), n);
	  if (node != Ippl::myNode()) {
	    // How many vnodes to assign?
	    int numvnodes = pernode;
	    if (extra > 0) {
	      numvnodes += 1;
	      extra -= 1;
	    }

	    // Create a message for this other node, storing:
	    //   - number of vnodes (int)
	    //   - list of vnode offset structs (list)
	    Message *msg = new Message;
	    msg->put(numvnodes);
	    if (numvnodes > 0) {
	      msg->setCopy(false);
	      msg->setDelete(false);
	      msg->putmsg(static_cast<void *>(&(offdata[nextvnode])),
			  sizeof(DFOffsetData<Dim,T>),
			  numvnodes);
	    }

	    // Send this message to the other node
	    Ippl::Comm->send(msg, node, tag);

	    // Update what the next vnode info to send is
	    nextvnode += numvnodes;
	  }
	}

	// At the end, we should have no vnodes left to send
	if ((unsigned int) nextvnode != offdata.size())
	  Ippl::abort("ERROR: Could not give away all my vnodes!");

      } else {
	// On non-box0 nodes, receive offset info
	int node = myBox0();
	Message *msg = Ippl::Comm->receive_block(node, tag);

	// Get the number of vnodes to store here
	msg->get(vnodes);

	// If this is > 0, copy out vnode info
	if (vnodes > 0) {
	  // resize the vector to make some storage
	  offdata.resize(vnodes);

	  // get offset data from the message
	  ::getMessage_iter(*msg, &(offdata[0]));
	}

	// Done with the message now.
	delete msg;
      }
    }

    // Now, finally, on all nodes we scan the vnodes to find out the maximum
    // size of the buffer needed to read this data in.
    for (int v=0; v < vnodes; ++v) {
      // Convert data to NDIndex
      NDIndex<Dim> dom;
      offset_data_to_domain(offdata[v], dom);
      if (dom.touches(readDomain)) {
	// Compute chunk block size
	int msdim = (Dim-1);	// this will be zero-based
	int chunkelems = Ippl::chunkSize() / sizeof(T);
	NDIndex<Dim> chunkblock = chunk_domain(dom, chunkelems, msdim,
					       offdata[v].isCompressed);

	// Now compare the size
	int dsize = chunkblock.size();
	if (dsize > maxsize) {
	  maxsize = dsize;
	}
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // read the data for a block of values of type T from the given data file.
  // Return success of read.
  // The size and seekpos values are in bytes.
  template <class T>
  bool read_data(int outputDatafd, T* buffer, Offset_t readsize,
		 Offset_t seekpos) {

    PAssert_GE(seekpos, 0);
    PAssert_GT(readsize, 0);
    PAssert(buffer);
    PAssert_GE(outputDatafd, 0);
    PAssert_EQ(readsize % sizeof(T), 0);

    // Now read the block of data
    off_t seekoffset = seekpos;
    size_t nbytes = readsize;
    bool seekok = true;

    Timer rtimer;
    rtimer.clear();
    rtimer.start();

    size_t nout = 0;
    if (::lseek(outputDatafd, seekoffset, SEEK_SET) == seekoffset) {
      char *rbuf = (char *)buffer;
      nout = ::read(outputDatafd, rbuf, nbytes);
    } else {
      seekok = false;
    }

    rtimer.stop();
    DiscBuffer::readtime += rtimer.clock_time();
    DiscBuffer::readbytes += readsize;

    if (!seekok) {
      ERRORMSG("Seek error in DiscField::read_data" << endl);
      ERRORMSG("Could not seek to position " << seekoffset << endl);
      Ippl::abort("Exiting due to DiscField error.");
    }

    if (nout != nbytes) {
      ERRORMSG("Read error in DiscField::read_data" << endl);
      ERRORMSG("Could not read " << nbytes << " bytes." << endl);
      Ippl::abort("Exiting due to DiscField error.");
    }

    return true;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Convert data in an offset data struct to an NDIndex, and return it
  // in the second argument.
  template<class T>
  void offset_data_to_domain(DFOffsetData<Dim,T> &offdata,
			     NDIndex<Dim> &domain) {
    int *dptr = offdata.vnodedata + 1;
    for (unsigned int i=0; i < Dim; ++i) {
      int first = *dptr;
      int stride = *(dptr + 1);
      int length = *(dptr + 2);
      domain[i] = Index(first, first + (length - 1)*stride, stride);
      dptr += 6;
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Convert domain data to offset I/O struct data
  template<class T>
  void domain_to_offset_data(const NDIndex<Dim> &domain,
			     DFOffsetData<Dim,T> &offdata) {
    int *dptr = offdata.vnodedata;
    for (unsigned int i=0; i < Dim; ++i) {
      *dptr++ = 0;
      *dptr++ = domain[i].first();
      *dptr++ = domain[i].stride();
      *dptr++ = domain[i].length();
      *dptr++ = 0;
      *dptr++ = 0;
    }
  }

  //
  // don't allow copy or assign ... these are declared but never defined,
  // if something tries to use them it will generate a missing symbol error
  //

  DiscField(const DiscField<Dim>&);
  DiscField& operator=(const DiscField<Dim>&);
};

#include "Utility/DiscField.hpp"

#endif // DISC_FIELD_H

/***************************************************************************
 * $RCSfile: DiscField.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscField.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
