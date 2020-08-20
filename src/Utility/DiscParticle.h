// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DISC_PARTICLE_H
#define DISC_PARTICLE_H

// include files
#include "Utility/DiscConfig.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Message/Message.h"



#include <vector>
#include <iostream>
#include <cstdio>

// forward declarations
template<class T> class IpplParticleBase;
template<class T> class ParticleAttrib;
class Message;

class DiscParticle {

public:
  // an enumeration used to indicate whether the file is for reading (INPUT)
  // or writing (OUTPUT), or for appending (APPEND)
  enum DPMode1 { INPUT, OUTPUT, APPEND };

  // an enumeration used to indicate whether we should read/write just a
  // single attribute, or the whole set of particle attributes
  enum DPMode2 { ALL, ATTRIB };

public:
  // Constructor: make a DiscParticle for reading or writing.
  // fname = name of file (without extensions)
  // config = name of configuration file
  // I/O mode (INPUT, OUTPUT, or APPEND)
  // typestr = string describing the 'type' of the Particles to be written
  //           (this is ignored if the data is being read).  The string can
  //           be anything the user prefers.
  DiscParticle(const char *fname, const char *config, int iomode,
	       const char *typestr = 0);

  // Constructor: same as above, but without a config file specified.  The
  // default config file entry that will be used is "*  .", which means, for
  // each SMP machine, assume the directory to put data files in is "."
  DiscParticle(const char *fname, int iomode, const char * = 0);

  // Destructor.
  ~DiscParticle();

  //
  // accessor functions
  //

  // Query for whether everything is OK so far.
  bool get_OK() const { return ConfigOK; }

  // Query for the number of records in the file.  If all attributes in a
  // Particle object are being written out, this is the number of sets of
  // attribute collections; if only single attributes are being written out,
  // this is the number of attributes which were written.
  unsigned int get_NumRecords() const { return RecordList.size(); }

  // Query for the mode of the Nth record, which can be either ALL (meaning
  // an entire IpplParticleBase's list of attributes was written out), or
  // ATTRIB (meaning only one specific attribute was written out).
  int get_DataMode(unsigned int record=0) const {
    return (RecordList[record]->attributes > 0 ? ALL : ATTRIB);
  }

  // Query for how this DiscParticle is operationg, either INPUT, OUTPUT, or
  // APPEND.
  int get_IOMode() const { return IOMode; }

  // Query for how many individual particles are stored in the Nth record,
  // for just the local filesets
  unsigned int get_NumLocalParticles(unsigned int record=0) const;

  // Query for how many individual particles are stored in the Nth record,
  // for the entire particle object (sum of all local particle counts)
  unsigned int get_NumGlobalParticles(unsigned int record=0) const {
    return RecordList[record]->globalparticles;
  }

  // Query for the number of attributes in the Nth record
  unsigned int get_NumAttributes(unsigned int record=0) const {
    return (get_DataMode(record)==ALL ? RecordList[record]->attributes : 1);
  }

  // Query for the number of bytes/elem in the Mth attribute in the Nth record
  unsigned int get_ElemByteSize(unsigned int record=0,
                                unsigned int attrib=0) const {
    return RecordList[record]->bytesize[attrib];
  }

  // Query for the user-specified type string
  const char *get_TypeString() const { return TypeString.c_str(); }

  // Query for the DiscType string for the Mth attribute in the Nth record.
  // If there not one available, return 0.
  const char *get_DiscType(unsigned int record=0,
			   unsigned int attrib=0) const;

  //
  // read methods
  //
  // read the specifed record in the file into the given IpplParticleBase or
  // ParticleAttrib object.
  // If the method is to read all the IpplParticleBase, this will delete all the
  // existing particles in the given object, create new ones and store the
  // values, and then do an update.  If an attribute is being read, this
  // will only work if the number of particles in the attribute already
  // matches the number in the file.

  // a templated read for IpplParticleBase objects.  This should only be called
  // if the object was opened with iomode == INPUT
  //   pbase = IpplParticleBase object to read into
  //   record = which record to read.  DiscParticle does not keep a 'current
  //          file position' pointer, instead you explicitly request which
  //          record you wish to read.
  // Return success of operation.
  //mwerks  template<class T>
  //mwerks  bool read(IpplParticleBase<T> &pbase, unsigned int record=0);
  ///////////////////////////////////////////////////////////////////////////
  // a templated read for IpplParticleBase objects.  This should only be called
  // if the object was opened with iomode == INPUT, datamode == ALL.
  //   pbase = IpplParticleBase object to read into
  //   record = which record to read.  DiscParticle does not keep a 'current
  //          file position' pointer, instead you explicitly request which
  //          record you wish to read.
  // Return success of operation.
  template<class T>
  bool read(IpplParticleBase<T> &pbase, unsigned int record) {

    // re-read the meta file since it might have changed
    ConfigOK = read_meta();

    // do some sanity checking first
    if (!ConfigOK) {
      ERRORMSG("Bad config or meta file in DiscParticle::read." << endl);
      return false;
    } else if (IOMode != INPUT) {
      ERRORMSG("Trying to read for DiscParticle created for output." << endl);
      return false;
    } else if (record >= get_NumRecords()) {
      ERRORMSG("Illegal record number in DiscParticle::read." << endl);
      return false;
    } else if (get_DataMode(record) != ALL) {
      ERRORMSG("Record " << record << " does not contain information for an ");
      ERRORMSG("entire IpplParticleBase." << endl);
      return false;
    } else if (get_NumAttributes(record) != pbase.numAttributes()) {
      ERRORMSG("Record " << record <<" has a different number of attributes ");
      ERRORMSG("than in the given IpplParticleBase." << endl);
      return false;
    }

// ada:  incDiscReads is not found  INCIPPLSTAT(incDiscReads);

    // make sure all the attribute sizes match
    for (unsigned int ca=0; ca < get_NumAttributes(record); ++ca) {
      if (get_ElemByteSize(record, ca)!=pbase.getAttribute(ca).elementSize()) {
	ERRORMSG("Mismatched data type size for attribute " << ca << " in ");
	ERRORMSG("DiscParticle::read." << endl);
	return false;
      }
    }

    // since we're reading in data for the entire IpplParticleBase, delete all the
    // existing particles, and do an update, before reading in new particles
    pbase.destroy(pbase.getLocalNum(), 0);
    pbase.update();

    // if we're on a box0 node, read in the data for the attributes
    if ((unsigned int) Ippl::myNode() == Config->getSMPBox0()) {

      // loop over all the files on this Box0 node
      for (unsigned int sf=0; sf < Config->getNumFiles(); ++sf) {

	// only need to process this file if there are particles in the file
	int localnum = RecordList[record]->localparticles[sf];
	if (localnum > 0) {

	  // open the data file
	  std::string filename = Config->getFilename(sf) + ".data";
	  FILE *datafile = open_file(filename, std::string("r"));

	  // if the file is available, read it
	  if (datafile != 0) {
	    // read the data for each attribute into a buffer, then put
	    // these buffers in a message.
	    Message *msg = new Message;
	    msg->put(localnum);

	    // read in the data for all the attributes
	    for (unsigned int a=0; a < get_NumAttributes(record); ++a) {
	      // read the data
	      void *buf = read_data(datafile, a, record, sf);
	      PAssert(buf);

	      // put it in the Message
	      msg->setCopy(false).setDelete(true);
	      msg->putmsg(buf, get_ElemByteSize(record, a), localnum);
	    }

	    // create new particles, by getting them from the message
	    pbase.getMessageAndCreate(*msg);

	    // we're done with the message now
	    delete msg;
	  }
	}
      }
    }

    // at the end, do an update to get everything where it is supposed to be
    pbase.update();
    return true;
  }

  // a templated read for ParticleAttrib objects.  This should only be
  // called if the object was opened with iomode == INPUT
  //   pattr = ParticleAttrib object to read into
  //   record = which record to read.  DiscParticle does not keep a 'current
  //          file position' pointer, instead you explicitly request which
  //          record you wish to read.
  // Return success of operation.
  //mwerks  template<class T>
  //mwerks  bool read(ParticleAttrib<T> &pattr, unsigned int record=0);
  ///////////////////////////////////////////////////////////////////////////
  // a templated read for ParticleAttrib objects.  This should only be called
  // if the object was opened with iomode == INPUT, datamode == ATTRIB.  When
  // reading just a single attribute, the particles simply replace the existing
  // particles in the attribute.
  //   pbase = ParticleAttrib object to read into
  //   record = which record to read.  DiscParticle does not keep a 'current
  //          file position' pointer, instead you explicitly request which
  //          record you wish to read.
  // Return success of operation.
  template<class T>
  bool read(ParticleAttrib<T> &pattr, unsigned int record) {

    // re-read the meta file since it might have changed
    ConfigOK = read_meta();

    // do some sanity checking first
    if (!ConfigOK) {
      ERRORMSG("Bad config or meta file in DiscParticle::read." << endl);
      return false;
    } else if (IOMode != INPUT) {
      ERRORMSG("Trying to read for a DiscParticle created for output."<<endl);
      return false;
    } else if (record >= get_NumRecords()) {
      ERRORMSG("Illegal record number in DiscParticle::read." << endl);
      return false;
    } else if (get_DataMode(record) != ATTRIB) {
      ERRORMSG("Record " << record << " does not contain information for a ");
      ERRORMSG("single ParticleAttrib." << endl);
      return false;
    } else if (get_ElemByteSize(record, 0) != pattr.elementSize()) {
      ERRORMSG("Mismatched attribute data type size in ");
      ERRORMSG("DiscParticle::read." << endl);
      return false;
    }

    pattr.destroy(pattr.size(), 0);

    // if we're on a box0 node, read in the data for the attributes
    if ((unsigned int) Ippl::myNode() == Config->getSMPBox0()) {

      // loop over all the files on this Box0 node
      for (unsigned int sf=0; sf < Config->getNumFiles(); ++sf) {

	// only need to process this file if there are particles in the file
	int localnum = RecordList[record]->localparticles[sf];
	if (localnum > 0) {

	  // open the data file
	  std::string filename = Config->getFilename(sf) + ".data";
	  FILE *datafile = open_file(filename, std::string("r"));

	  // if the file is available, read it
	  if (datafile != 0) {
	    // read the data for each attribute into a buffer, then put
	    // these buffers in a message.
	    Message *msg = new Message;

	    // read in the data for the attribute
	    void *buf = read_data(datafile, 0, record, sf);
	    PAssert(buf);

	    // put it in the Message
	    msg->setCopy(false).setDelete(true);
	    msg->putmsg(buf, get_ElemByteSize(record, 0), localnum);

	    // create new particles, by getting them from the message
	    pattr.getMessage(*msg, localnum);

	    // we're done with the message now
	    delete msg;
	  }
	}
      }
    }

    return true;
  }

  //
  // write methods
  //
  // write the data from the given IpplParticleBase or ParticleAttrib into the
  // file.  Data is appended as a new record.

  // a templated write for IpplParticleBase objects.  All attributes in the
  // IpplParticleBase are written as a single record.  This should only be
  // called if the object was opened with iomode == OUTPUT or APPEND.
  //   pbase = IpplParticleBase object to read from.
  // Return success of operation.
  //mwerks  template<class T>
  //mwerks  bool write(IpplParticleBase<T> &pbase);
  ///////////////////////////////////////////////////////////////////////////
  // a templated write for IpplParticleBase objects.  All attributes in the
  // IpplParticleBase are written as a single record.  This should only be
  // called if the object was opened with iomode == OUTPUT or APPEND.
  //   pbase = IpplParticleBase object to read into
  // Return success of operation.
  template<class T>
  bool write(IpplParticleBase<T> &pbase) {

    // generate a tag to use for communication
    int tag = Ippl::Comm->next_tag(FB_WRITE_TAG, FB_TAG_CYCLE);

    // if the file already has some records, re-read the meta file in case it
    // has changed.  If we do not have any record info, and we've opened
    // for OUTPUT, we do NOT read any possible meta file since we want to start
    // a new file.
    if (get_NumRecords() > 0 || IOMode == APPEND)
      ConfigOK = read_meta();

    // do some sanity checking first
    if (!ConfigOK) {
      ERRORMSG("Bad config or meta file in DiscParticle::write." << endl);
      return false;
    } else if (IOMode == INPUT) {
      ERRORMSG("Trying to write for a DiscParticle created for input."<<endl);
      return false;
    }

    // create a new record entry, and set it to the proper values
    RecordInfo *info = new RecordInfo;
    info->attributes = pbase.numAttributes();
    info->globalparticles = pbase.getTotalNum();
    for (int a=0; a < info->attributes; ++a) {
      info->bytesize.push_back(pbase.getAttribute(a).elementSize());
      info->disctypes.push_back(pbase.getAttribute(a).typeString());
    }

    // Create a message with our local particles, which will then be used
    // to write out the data
    Message *msg = new Message;
    pbase.putMessage(*msg, pbase.getLocalNum(), 0);

    // on Box0 nodes, first write out your own particles, then write out
    // all the other node's particles
    if ((unsigned int) Ippl::myNode() == Config->getSMPBox0()) {
      // create the data file, if it does not yet exist; otherwise, open
      // it for append
      std::string openmode = "a";
      if (get_NumRecords() == 0)
	openmode = "w";
      std::string filename = Config->getFilename(0) + ".data";
      FILE *datafile = open_file(filename, openmode);
      if (datafile == 0) {
	delete info;
	delete msg;
	return false;
      }

      // create a vector of Messages, which will hold the info to be written
      // out.  We do not write until we have all the messages from the
      // different nodes writing to the Box0 file.
      std::vector<Message *> msgvec;
      msgvec.push_back(msg);

      // write out our local attribute data, saving where we started to write
      // determine how many other SMP nodes we expect to receive data from
      int notreceived = (Config->getNumSMPNodes() - 1);
      for (unsigned int s=0; s < Config->getNumOtherSMP(); ++s)
	notreceived += Config->getNumSMPNodes(Config->getOtherSMP(s));

      // now wait for messages from all the other nodes with their particle
      // data, and save the messages
      while (notreceived > 0) {
	// receive the message
	int any_node = COMM_ANY_NODE;
	Message *recmsg = Ippl::Comm->receive_block(any_node, tag);
	PAssert(recmsg);
	notreceived--;

	// get the number of particles and save the info
	// write the info out to disk
	msgvec.push_back(recmsg);
      }

      // we have all the messages, so write the data to disk.  This will
      // delete all the messages and save the offset information, as well as
      // the particle count.
      if (!write_data(datafile, msgvec, info)) {
	delete info;
	return false;
      }

      // done writing; close the file and save the particle count
      fclose(datafile);

    } else {
      // just send out the message with our local particles now
      Ippl::Comm->send(msg, Config->getSMPBox0(), tag);

      // and save extra necessary info into RecordInfo struct
      info->localparticles.push_back(0);
      info->offset.push_back(std::vector<Offset_t>());
    }

    // add this new record information to our list
    RecordList.push_back(info);

    // rewrite the meta file, if we're on a box0 node
    if ((unsigned int) Ippl::myNode() == Config->getSMPBox0()) {
      if (!write_meta())
	return false;
    }

    // to be safe, do a barrier here, since some nodes could have had very
    // little to do
    Ippl::Comm->barrier();

    // return success
    return true;
  }

  // a templated write for ParticleAttrib objects.  The single attribute
  // data is written as a single record.  This should only be
  // called if the object was opened with iomode == OUTPUT or APPEND.
  //   pattr = ParticleAttrib object to read from.
  // Return success of operation.
  //mwerks  template<class T>
  //mwerks  bool write(ParticleAttrib<T> &pattr);
  ///////////////////////////////////////////////////////////////////////////
  // a templated write for ParticleAttrib objects.  This should only be
  // called if the object was opened with iomode == OUTPUT or APPEND.
  //   pattr = ParticleAttrib object to read from
  // Return success of operation.
  template<class T>
  bool write(ParticleAttrib<T> &pattr) {

    // generate a tag to use for communication
    int tag = Ippl::Comm->next_tag(FB_WRITE_TAG, FB_TAG_CYCLE);

    // if the file already has some records, re-read the meta file in case it
    // has changed.  If we do not have any record info, and we've opened
    // for OUTPUT, we do NOT ready any possible meta file since we want to
    // start a new file
    if (get_NumRecords() > 0 || IOMode == APPEND)
      ConfigOK = read_meta();

    // do some sanity checking first
    if (!ConfigOK) {
      ERRORMSG("Bad config or meta file in DiscParticle::write." << endl);
      return false;
    } else if (IOMode == INPUT) {
      ERRORMSG("Trying to write for a DiscParticle created for input."<<endl);
      return false;
    }

    // create a new record entry, and set it to the proper values
    RecordInfo *info = new RecordInfo;
    info->attributes = 0;
    info->globalparticles = 0;
    info->bytesize.push_back(pattr.elementSize());
    info->disctypes.push_back(pattr.typeString());

    // Create a message with our local particles, which will then be used
    // to write out the data
    Message *msg = new Message;
    msg->put(pattr.size());
    pattr.putMessage(*msg, pattr.size(), 0);

    // on Box0 nodes, first write out your own particles, then write out
    // all the other node's particles
    if ((unsigned int) Ippl::myNode() == Config->getSMPBox0()) {
      // create the data file, if it does not yet exist; otherwise, open
      // it for append
      std::string openmode = "a";
      if (get_NumRecords() == 0)
	openmode = "w";
      std::string filename = Config->getFilename(0) + ".data";
      FILE *datafile = open_file(filename, openmode);
      if (datafile == 0) {
	delete info;
	delete msg;
	return false;
      }

      // create a vector of Messages, which will hold the info to be written
      // out.  We do not write until we have all the messages from the
      // different nodes writing to the Box0 file.
      std::vector<Message *> msgvec;
      msgvec.push_back(msg);

      // write out our local attribute data, saving where we started to write
      // determine how many other SMP nodes we expect to receive data from
      int notreceived = (Config->getNumSMPNodes() - 1);
      for (unsigned int s=0; s < Config->getNumOtherSMP(); ++s)
	notreceived += Config->getNumSMPNodes(Config->getOtherSMP(s));

      // now wait for messages from all the other nodes with their particle
      // data, and save the messages
      while (notreceived > 0) {
	// receive the message
	int any_node = COMM_ANY_NODE;
	Message *recmsg = Ippl::Comm->receive_block(any_node, tag);
	PAssert(recmsg);
	notreceived--;

	// get the number of particles and save the info
	// write the info out to disk
	msgvec.push_back(recmsg);
      }

      // we have all the messages, so write the data to disk.  This will
      // delete all the messages and save the offset information, as well as
      // the particle count.
      if (!write_data(datafile, msgvec, info)) {
	delete info;
	return false;
      }

      // done writing; close the file and save the particle count
      fclose(datafile);

    } else {
      // just send out the message with our local particles now
      Ippl::Comm->send(msg, Config->getSMPBox0(), tag);

      // and save extra necessary info into RecordInfo struct
      info->localparticles.push_back(0);
      info->offset.push_back(std::vector<Offset_t>());
    }

    // add this new record information to our list
    RecordList.push_back(info);

    // rewrite the meta file, if we're on a box0 node
    if ((unsigned int) Ippl::myNode() == Config->getSMPBox0()) {
      if (!write_meta())
	return false;
    }

    // to be safe, do a barrier here, since some nodes could have had very
    // little to do
    Ippl::Comm->barrier();

    // return success
    return true;
  }

  //
  // console printing methods
  //

  // print out debugging info to the given stream
  void printDebug(std::ostream&);
  void printDebug();

private:
  // a typedef used to select the data type for file offsets
  typedef long  Offset_t;

  // the configuration file mechanism
  DiscConfig *Config;
  bool ConfigOK;

  // I/O mode (INPUT or OUTPUT)
  int IOMode;

  // the base name for the output file, and the descriptive type string
  std::string BaseFile;
  std::string TypeString;

  // a simple struct which stores information about each record, for each
  // file set
  struct RecordInfo {
    // a typedef used to select the data type for file offsets
    typedef long  Offset_t;

    int attributes;		 // number of attributes; 0 == just writing
				 // one attribute, > 0 == writing a whole
				 // IpplParticleBase's worth of particles
    int globalparticles;	 // total number of particles in whole system
    std::vector<int> localparticles;  // number of particles in this fileset's files
    std::vector<int> bytesize;	 // how many bytes/attrib elem in each attrib
    std::vector<std::vector<Offset_t> > offset; // starting offset for attrib in .data
    std::vector<std::string> disctypes;	 // attribute types determined by DiscType
    RecordInfo() : attributes(0), globalparticles(0) { }
  };

  // the list of information for each record
  std::vector<RecordInfo *> RecordList;

  // this keeps track of where in the .data file writing is occuring
  Offset_t CurrentOffset;

  //
  // functions used to build/query information about the processors, etc.
  //

  // perform initialization based on the constuctor arguments
  void initialize(const char *base, const char *config,
		  const char *typestr, int iomode);

  // open a file in the given mode.  If an error occurs, print a message (but
  // only if the last argument is true).
  // fnm = complete name of file (can include a path)
  // mode = open method ("r" == read, "rw" == read/write, etc.
  FILE *open_file(const std::string& fnm, const std::string& mode,
	          bool reporterr = true);

  //
  // read/write functions for individual components
  //

  // read or write .meta data file information.  Return success.
  bool read_meta();
  bool write_meta();

  // read the data for the Nth attribute of record R, in the Fth fileset,
  // and return the newly allocated buffer (or 0 if an error occurs).
  void *read_data(FILE *outputData, unsigned int attrib,
		  unsigned int record, unsigned int fileset);

  // write the data for a block of particles to the given file.  The
  // data is just appended to the end.  Return success of write.
  bool write_data(FILE *outputData, std::vector<Message *> &, RecordInfo *);

  //
  // don't allow copy or assign ... this are declared but never defined,
  // if something tries to use them it will generate a missing symbol error
  //

  DiscParticle(const DiscParticle&);
  DiscParticle& operator=(const DiscParticle&);
};

#endif // DISC_PARTICLE_H

/***************************************************************************
 * $RCSfile: DiscParticle.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscParticle.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
