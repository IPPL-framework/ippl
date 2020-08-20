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
#include "Utility/DiscParticle.h"
#include "Utility/DiscConfig.h"
#include "Utility/DiscMeta.h"
#include "Message/Communicate.h"
#include "Message/Message.h"

#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"

///////////////////////////////////////////////////////////////////////////
// Constructor: make a DiscParticle for reading or writing.
// fname = name of file (without extensions)
// config = name of configuration file
// I/O mode
// typestr = string describing the 'type' of the Particles to be written
//           (this is ignored if the data is being read).  The string can
//           be anything the user prefers.
DiscParticle::DiscParticle(const char *base, const char *config,
			   int iomode, const char *typestr) {

  initialize(base, config, typestr, iomode);
}


///////////////////////////////////////////////////////////////////////////
// Constructor: same as above, but without a config file specified.  The
// default config file entry that will be used is "*  .", which means, for
// each SMP machine, assume the directory to put data files in is "."
DiscParticle::DiscParticle(const char *base,
			   int iomode, const char *typestr) {

  initialize(base, 0, typestr, iomode);
}


///////////////////////////////////////////////////////////////////////////
// perform initialization based on the constuctor arguments
void DiscParticle::initialize(const char *base, const char *config,
			      const char *typestr, int iomode) {


  // save which IO mode we're running, and make sure it is OK
  if (iomode != INPUT && iomode != OUTPUT && iomode != APPEND)
    Ippl::abort("IO mode for DiscParticle must be INPUT, OUTPUT, or APPEND.");
  IOMode = iomode;
  CurrentOffset = 0;

  // save the name information
  BaseFile = base;
  if (typestr != 0)
    TypeString = typestr;
  else
    TypeString = "unknown";

  // parse the configuration file to find our SMP's, file directories, etc.
  // create a DiscConfig instance, which will parse the file
  Config = new DiscConfig(config, BaseFile.c_str(), IOMode != INPUT);
  ConfigOK = Config->ok();

  // indicate how many filesets we're reading or writing
  /*
  if (iomode != INPUT) {
    ADDIPPLSTAT(incDiscFilesetWrites,Config->getNumFiles());
  } else {
    ADDIPPLSTAT(incDiscFilesetReads,Config->getNumFiles());
  }
  */
  // read the meta information about numbers of particles, etc. in each record
  if (ConfigOK && IOMode != OUTPUT)
    ConfigOK = read_meta();
}


///////////////////////////////////////////////////////////////////////////
// Destructor
DiscParticle::~DiscParticle() {

  // delete per-record information
  std::vector<RecordInfo *>::iterator rec = RecordList.begin();
  for ( ; rec != RecordList.end(); ++rec)
    delete (*rec);

  // delete the configuration file info
  if (Config != 0)
    delete Config;
}


///////////////////////////////////////////////////////////////////////////
// Query for the DiscType string for the Mth attribute in the Nth record.
// If there not one available, return 0.
const char *DiscParticle::get_DiscType(unsigned int record,
				       unsigned int attrib) const {
  // return the string, if it is available
  if (record < get_NumRecords() &&
      attrib < RecordList[record]->disctypes.size())
    return (RecordList[record]->disctypes[attrib].c_str());
  else
    return 0;
}


///////////////////////////////////////////////////////////////////////////
// Query for how many individual particles are stored in the Nth record,
// for just the local filesets
unsigned int DiscParticle::get_NumLocalParticles(unsigned int record) const {
  unsigned int retval = 0;
  for (unsigned int i=0; i < RecordList[record]->localparticles.size(); ++i)
    retval += RecordList[record]->localparticles[i];
  return retval;
}


///////////////////////////////////////////////////////////////////////////
// open a file in the given mode.  If an error occurs, print a message
// and return 0.
FILE *DiscParticle::open_file(const std::string& fnm, const std::string& mode,
			      bool reporterr) {

  // open the file, with the given name and mode
  FILE *f = fopen(fnm.c_str(), mode.c_str());

  // check for an error
  if (f == 0 && reporterr) {
    ERRORMSG("DiscParticle: Could not open file '" << fnm.c_str());
    ERRORMSG("' on node " << Ippl::myNode() << "." << endl);
  }

  // return the file pointer; if an error occurred, this will be zero
  return f;
}


///////////////////////////////////////////////////////////////////////////
// write out a new .meta file.  The .meta file contains the following
// information, on each line:
//   String with type of data stored here, supplied by user
//   Number of SMP filesets
//   Number of Records
//   For each record:
//     Number of attributes (if 0, this is just for one attrib, not all)
//     Number of local particles
//     Number of particles total
//     For each attribute in the record:
//       Size of attribute elements, in bytes
//       Offset location of start of attribute data in .data file
// return success of operation.
bool DiscParticle::write_meta() {

  // Sanity checking
  if (!ConfigOK) {
    ERRORMSG("Bad config file in DiscParticle::write_meta." << endl);
    return false;
  } else if (IOMode == INPUT) {
    ERRORMSG("Trying to write a DiscParticle .meta file for a DiscParticle");
    ERRORMSG(" created for reading." << endl);
    return false;
  } else if (Config->getNumFiles() > 1) {
    ERRORMSG("Only one fileset/SMP max allowed when writing" << endl);
    return false;
  }

  // no need to write anything if we have no files for this SMP
  if ((unsigned int)Ippl::myNode() != Config->getSMPBox0() || Config->getNumFiles() == 0)
    return true;

  // open the meta data file
  std::string filename = Config->getFilename(0) + ".meta";
  FILE *outputMeta = open_file(filename, std::string("w"));
  if (outputMeta == 0)
    return false;

  // write the initial header info
  long rnum = RecordList.size();
  fprintf(outputMeta, "Type =           %s\n", TypeString.c_str());
  fprintf(outputMeta, "SMPs =           %u\n", Config->fileSMPs());
  fprintf(outputMeta, "Records =        %ld\n", rnum);
  fprintf(outputMeta, "NextOffset =     %ld\n", CurrentOffset);

  // write information for each record
  for (unsigned int r=0; r < RecordList.size(); ++r) {
    // general record information
    fprintf(outputMeta, "RecordAttribs =  %u  %d\n", r,
	    RecordList[r]->attributes);
    fprintf(outputMeta, "RecordLocPtcl =  %u  %d\n", r,
	    RecordList[r]->localparticles[0]);
    fprintf(outputMeta, "RecordGlobPtcl = %u  %d\n", r,
	    RecordList[r]->globalparticles);

    // information about each attribute
    fprintf(outputMeta, "RecordElemByte = %u ", r);
    int b, bmax;
    bmax = RecordList[r]->bytesize.size();
    for (b=0; b<bmax; ++b)
      fprintf(outputMeta, " %d", RecordList[r]->bytesize[b]);
    fprintf(outputMeta, "\n");
    fprintf(outputMeta, "RecordDiscType = %u ", r);
    for (b=0; b<bmax; ++b)
      fprintf(outputMeta, " %s", RecordList[r]->disctypes[b].c_str());
    fprintf(outputMeta, "\n");
    fprintf(outputMeta, "RecordOffset =   %u ", r);
    bmax = RecordList[r]->offset[0].size();
    for (b=0; b<bmax; ++b)
      fprintf(outputMeta, " %ld", RecordList[r]->offset[0][b]);
    fprintf(outputMeta, "\n");
  }

  // close data file and return
  fclose(outputMeta);
  return true;
}


///////////////////////////////////////////////////////////////////////////
// read in data from .meta file, and replace current storage values.
// The format for a .meta file is described in the write_meta routine.
// return success of operation.
bool DiscParticle::read_meta() {

  // Sanity checking
  if (!ConfigOK) {
    ERRORMSG("Bad config file in DiscParticle::read_meta." << endl);
    return false;
  }

  bool iserror = false;
  int tag = Ippl::Comm->next_tag(DF_READ_META_TAG, DF_TAG_CYCLE);

  // initialize data before reading .meta file
  TypeString = "unknown";
  {
      std::vector<RecordInfo *>::iterator rec = RecordList.begin();
      for ( ; rec != RecordList.end(); ++rec)
          delete (*rec);
  }
  RecordList.erase(RecordList.begin(), RecordList.end());

  // on Box0 nodes, read in the meta data ... on others, wait for
  // Box0 nodes to send info to them
  if ((unsigned int)Ippl::myNode() == Config->getSMPBox0()) {
    // loop over all the files on this Box0 node
    for (unsigned int sf=0; sf < Config->getNumFiles(); ++sf) {
      // open and parse the meta data file
      std::string filename = Config->getFilename(sf) + ".meta";
      DiscMeta outputMeta(filename.c_str());
      if (outputMeta.size() == 0)
	return false;

      // keep reading until we have all data
      DiscMeta::iterator metaline = outputMeta.begin();
      for ( ; metaline != outputMeta.end(); ++metaline) {
	// get number of tokens and list of tokens in the line
	int linesread  = (*metaline).first;
	int numtokens  = (*metaline).second.first;
	std::string *tokens = (*metaline).second.second;

        // action is based on first keyword
        if (tokens[0] == "Type") {
	  if (numtokens > 1 && sf == 0)
	    TypeString = tokens[1];
        } else if (tokens[0] == "SMPs" && numtokens == 2) {
          unsigned int checkfileSMPs = atoi(tokens[1].c_str());
	  if (Config->fileSMPs() != checkfileSMPs) {
	    ERRORMSG("SMP number mismatch in file " << filename);
	    ERRORMSG(" (" << checkfileSMPs << "!=" << Config->fileSMPs());
	    ERRORMSG(")" << endl);
	    iserror = true;
	  }
        } else if (tokens[0] == "NextOffset" && numtokens == 2) {
	  CurrentOffset = atol(tokens[1].c_str());
        } else if (tokens[0] == "Records" && numtokens == 2) {
          unsigned int numrec = atoi(tokens[1].c_str());
	  if (RecordList.size() != 0 && RecordList.size() != numrec) {
	    ERRORMSG("Illegal Records value in file '" << filename << "' ");
	    ERRORMSG("(" << numrec << " != " << RecordList.size() << ")");
	    ERRORMSG(endl);
	    iserror = true;
	  } else {
	    // create N empty record storage objects
	    for (int r=0; r < atoi(tokens[1].c_str()); ++r)
	      RecordList.push_back(new RecordInfo());
	  }
        } else if (tokens[0] == "RecordAttribs" && numtokens == 3) {
	  unsigned int recnum = atoi(tokens[1].c_str());
	  int attribs = atoi(tokens[2].c_str());
	  if (recnum >= RecordList.size()) {
	    ERRORMSG("Illegal record number '" << recnum << "'" << endl);
	    iserror = true;
	  } else if (sf > 0 && attribs != RecordList[recnum]->attributes) {
	    ERRORMSG("Illegal number of attributes in file " << filename);
	    ERRORMSG(" (" << attribs << " != ");
	    ERRORMSG(RecordList[recnum]->attributes << ")" << endl);
	    iserror = true;
	  } else {
	    RecordList[recnum]->attributes = attribs;
	  }
        } else if (tokens[0] == "RecordLocPtcl" && numtokens == 3) {
	  unsigned int recnum = atoi(tokens[1].c_str());
	  int nump = atoi(tokens[2].c_str());
	  if (recnum >= RecordList.size()) {
	    ERRORMSG("Illegal record number '" << recnum << "'" << endl);
	    iserror = true;
	  } else {
	    RecordList[recnum]->localparticles.push_back(nump);
	  }
        } else if (tokens[0] == "RecordGlobPtcl" && numtokens == 3) {
	  unsigned int recnum = atoi(tokens[1].c_str());
	  int nump = atoi(tokens[2].c_str());
	  if (recnum >= RecordList.size()) {
	    ERRORMSG("Illegal record number '" << recnum << "'" << endl);
	    iserror = true;
	  } else if (sf > 0 && nump != RecordList[recnum]->globalparticles) {
	    ERRORMSG("Illegal number of global particles in file "<<filename);
	    ERRORMSG(" (" << nump << " != ");
	    ERRORMSG(RecordList[recnum]->globalparticles << ")" << endl);
	    iserror = true;
	  } else {
	    RecordList[recnum]->globalparticles = nump;
	  }
        } else if (tokens[0] == "RecordElemByte" && numtokens >= 2) {
          unsigned int recnum = atoi(tokens[1].c_str());
	  if (recnum >= RecordList.size()) {
	    ERRORMSG("Illegal record number '" << recnum << "'" << endl);
	    iserror = true;
	  } else {
	    for (int b=2; b < numtokens; ++b) {
	      int bytesize = atoi(tokens[b].c_str());
	      if (sf > 0 && bytesize != RecordList[recnum]->bytesize[b-2]) {
		ERRORMSG("Illegal byte size for attribute " << b-2);
		ERRORMSG(" in file " << filename << "(" << bytesize);
		ERRORMSG(" != " << RecordList[recnum]->bytesize[b-2] << ")");
		ERRORMSG(endl);
		iserror = true;
	      } else if (sf == 0) {
		RecordList[recnum]->bytesize.push_back(bytesize);
	      }
	    }
	    if (RecordList[recnum]->bytesize.size() !=
		(RecordList[recnum]->attributes == 0 ? 1u :
		 static_cast<unsigned int>(RecordList[recnum]->attributes))) {
	      ERRORMSG("Incorrect number of byte size values in file ");
	      ERRORMSG(filename << endl);
	      iserror = true;
	    }
	  }
        } else if (tokens[0] == "RecordDiscType" && numtokens >= 2) {
	  unsigned int recnum = atoi(tokens[1].c_str());
	  if (recnum >= RecordList.size()) {
	    ERRORMSG("Illegal record number '" << recnum << "'" << endl);
	    iserror = true;
	  } else {
	    for (int b=2; b < numtokens; ++b) {
	      if (sf > 0 && tokens[b] != RecordList[recnum]->disctypes[b-2]) {
		ERRORMSG("Inconsistent DiscType for attribute " << b-2);
		ERRORMSG(" in file " << filename << "(" << tokens[b]);
		ERRORMSG(" != " << RecordList[recnum]->disctypes[b-2] << ")");
		ERRORMSG(endl);
		iserror = true;
	      } else if (sf == 0) {
		RecordList[recnum]->disctypes.push_back(tokens[b]);
	      }
	    }
	    if (RecordList[recnum]->disctypes.size() !=
		(RecordList[recnum]->attributes == 0 ? 1u :
		 static_cast<unsigned int>(RecordList[recnum]->attributes))) {
	      ERRORMSG("Incorrect number of DiscType values in file ");
	      ERRORMSG(filename << endl);
	      iserror = true;
	    }
	  }
        } else if (tokens[0] == "RecordOffset" && numtokens >= 2) {
	  unsigned int recnum = atoi(tokens[1].c_str());
	  if (recnum >= RecordList.size()) {
	    ERRORMSG("Illegal record number '" << recnum << "'" << endl);
	    iserror = true;
	  } else {
            std::vector<Offset_t> offsetvec;
	    for (int b=2; b < numtokens; ++b) {
	      Offset_t offset = atol(tokens[b].c_str());
	      if (sf > 0 && offset != RecordList[recnum]->offset[sf][b-2]) {
		ERRORMSG("Illegal offset for attribute " << b-2);
		ERRORMSG(" in file " << filename << "(");
                ERRORMSG(static_cast<long>(offset) << " != ");
		ERRORMSG(static_cast<long>(
                         RecordList[recnum]->offset[sf][b-2]) << ")" << endl);
		iserror = true;
	      } else if (sf == 0) {
		offsetvec.push_back(offset);
	      }
	    }
	    if (sf == 0) {
                if (offsetvec.size() != (RecordList[recnum]->attributes==0 ? 1u :
                                         static_cast<unsigned int>(RecordList[recnum]->attributes))) {
		ERRORMSG("Incorrect number of offset values in file ");
		ERRORMSG(filename << endl);
		iserror = true;
	      } else {
		RecordList[recnum]->offset.push_back(offsetvec);
	      }
	    }
	  }
        } else {
	  // error in line: unknown keyword
	  iserror = true;
        }

	// print if there was an error
	if (iserror) {
	  ERRORMSG("Format error on line " << linesread << " in file ");
	  ERRORMSG(filename << endl);
          break;
	}
      }

      // stop processing meta files is there was an error
      if (iserror)
        break;
    }

    // now send meta info to all nodes which expect it
    int numinform = Config->getNumOtherSMP();
    for (int s=0; s <= numinform; ++s) {
      int smp = Config->mySMP();
      if (s != numinform)
	smp = Config->getOtherSMP(s);
      for (unsigned int n=0; n < Config->getNumSMPNodes(smp); ++n) {
        int node = Config->getSMPNode(smp, n);
        if (node != Ippl::myNode()) {
          // create a message with meta info
          Message *msg = new Message;
          ::putMessage(*msg, TypeString);
	  int errint = iserror;
          msg->put(errint);
	  if (!iserror) {
            long curroff = CurrentOffset;
	    msg->put(curroff);
	    int recnum = RecordList.size();
	    msg->put(recnum);
	    for (unsigned int r=0; r < RecordList.size(); ++r) {
	      int filesnum = RecordList[r]->localparticles.size();
	      msg->put(filesnum);
	      msg->put(RecordList[r]->attributes);
	      msg->put(RecordList[r]->globalparticles);
	      for (unsigned int p=0; p < RecordList[r]->localparticles.size(); ++p)
		msg->put(RecordList[r]->localparticles[p]);
	      for (unsigned int b=0; b < RecordList[r]->bytesize.size(); ++b)
		msg->put(RecordList[r]->bytesize[b]);

	      // make sure we have disctypes ... add if necessary
	      std::string unknowndtype = "u";
	      for (unsigned int dr = RecordList[r]->disctypes.size();
		   dr < RecordList[r]->bytesize.size(); ++dr)
		RecordList[r]->disctypes.push_back(unknowndtype);

	      for (unsigned int t=0; t < RecordList[r]->bytesize.size(); ++t)
		::putMessage(*msg, RecordList[r]->disctypes[t]);
	      for (unsigned int z=0; z < RecordList[r]->offset.size(); ++z) {
		for (unsigned int o=0; o < RecordList[r]->offset[z].size(); ++o) {
		  long value = RecordList[r]->offset[z][o];
		  msg->put(value);
		}
	      }
	    }
	  }

          // send the message to the intended node
          Ippl::Comm->send(msg, node, tag);
        }
      }
    }

  } else {
    // all other nodes (which are not Box0 nodes) should get a message
    // telling them the meta info
    int node = Config->getSMPBox0();
    Message *msg = Ippl::Comm->receive_block(node, tag);
    PAssert(msg);

    // get info out of message
    ::getMessage(*msg, TypeString);
    int errint = 0, numrec = 0, val = 0;
    msg->get(errint);
    iserror = (errint != 0);
    if (!iserror) {
      long curroff = 0;
      msg->get(curroff);
      CurrentOffset = curroff;
      msg->get(numrec);
      for (int r=0; r < numrec; ++r) {
	RecordList.push_back(new RecordInfo());
	int filesnum = RecordList[r]->localparticles.size();
	msg->get(filesnum);
	msg->get(RecordList[r]->attributes);
	int numattr = RecordList[r]->attributes;
	if (RecordList[r]->attributes == 0)
	  numattr = 1;
	msg->get(RecordList[r]->globalparticles);
	for (int p=0; p < filesnum; ++p) {
	  msg->get(val);
	  RecordList[r]->localparticles.push_back(val);
	}
	for (int b=0; b < numattr; ++b) {
	  msg->get(val);
	  RecordList[r]->bytesize.push_back(val);
	}
	for (int t=0; t < numattr; ++t) {
	  std::string dtype;
	  ::getMessage(*msg, dtype);
	  RecordList[r]->disctypes.push_back(dtype);
	}
	for (int z=0; z < filesnum; ++z) {
          std::vector<Offset_t> offsetvec;
	  Offset_t offset;
	  long value = 0;
	  for (int o=0; o < numattr; ++o) {
	    msg->get(value);
	    offset = value;
	    offsetvec.push_back(offset);
	  }
	  RecordList[r]->offset.push_back(offsetvec);
	}
      }
    }

    // we're done with this message
    delete msg;
  }

  // at the end, if there was an error, get rid of all the existing record
  // info
  if (iserror) {
    std::vector<RecordInfo *>::iterator rec = RecordList.begin();
    for ( ; rec != RecordList.end(); ++rec)
      delete (*rec);
    RecordList.erase(RecordList.begin(), RecordList.end());
  }

  // return whether there was an error
  return (!iserror);
}


///////////////////////////////////////////////////////////////////////////
bool DiscParticle::write_data(FILE *outputData, std::vector<Message *> &msgvec,
			      RecordInfo *info) {
        
  // a vector for storing information about how many particles are in each msg
  std::vector<int> numvec;
  std::vector<Offset_t> offsetvec;
  int totalnum = 0;

  // seek to the current offset, since the file may have changed
  if (fseek(outputData, CurrentOffset, SEEK_SET) != 0) {
    ERRORMSG("Error seeking to position " << static_cast<long>(CurrentOffset));
    ERRORMSG(" in .data file for DiscParticle::write_data." << endl);
    fclose(outputData);
    return false;
  }

  // loop over all the attributes in the messages
  int numattr = info->attributes;
  if (numattr == 0)
    numattr = 1;
  for (int a=0; a < numattr; ++a) {
    offsetvec.push_back(CurrentOffset);

    // loop over all the nodes that provided data
    for (unsigned int n=0; n < msgvec.size(); ++n) {
      // for each node, get out the number of particles, get the data, and
      // write it out
      int nump = 0;
      if (a == 0) {
	msgvec[n]->get(nump);
	numvec.push_back(nump);
	totalnum += nump;
      } else {
	nump = numvec[n];
      }

      if (nump > 0) {
	// get the data for this attribute
	void *msgdata = msgvec[n]->remove();

	// write it out now ...
	unsigned int totalbytes = nump * info->bytesize[a];
	if (fwrite(msgdata, 1, totalbytes, outputData) != totalbytes) {
	  ERRORMSG("Write error in DiscParticle::write_data" << endl);
	  fclose(outputData);
	  return false;
	}

	// update the current offset position
	CurrentOffset += totalbytes;
	//	ADDIPPLSTAT(incDiscBytesWritten, totalbytes);

	// free the message data storage
	free(msgdata);
      }

      // delete the message, if we can
      if ((a+1) == numattr)
	delete (msgvec[n]);
    }
  }

  // save the offset information for each attribute
  info->offset.push_back(offsetvec);
  info->localparticles.push_back(totalnum);

  // we're done writing; return success
  return true;
}


///////////////////////////////////////////////////////////////////////////
// read the data for the Nth attribute of record R, in the Fth fileset,
// and return the newly allocated buffer (or 0 if an error occurs).
void *DiscParticle::read_data(FILE *outputData, unsigned int attrib,
			      unsigned int record, unsigned int fileset) {


  // determine the byte size of the attribute, and the offset
  int nump = RecordList[record]->localparticles[fileset];
  int bytesize = RecordList[record]->bytesize[attrib];

  // if there are no particles to read, just return
  if (nump < 1 || bytesize < 1)
    return 0;

  // seek to proper position first, for the first attribute
  if (attrib == 0) {
    Offset_t seekpos = RecordList[record]->offset[fileset][attrib];
    if (fseek(outputData, seekpos, SEEK_SET) != 0) {
      ERRORMSG("Error seeking to position " << static_cast<long>(seekpos));
      ERRORMSG(" in .data file for DiscParticle::read_data." << endl);
      fclose(outputData);
      return 0;
    }
  }

  // allocate a block of memory to store the data now
  unsigned int totalbytes = nump * bytesize;
  void *buffer = malloc(totalbytes);

  // read the data into the provided buffer
  if (fread(buffer, 1, totalbytes, outputData) != totalbytes) {
    ERRORMSG("Read error in DiscParticle::read_data" << endl);
    free(buffer);
    fclose(outputData);
    return 0;
  }

  return buffer;
}


///////////////////////////////////////////////////////////////////////////
// print out debugging information for this DiscParticle
void DiscParticle::printDebug() { printDebug(std::cout); }

void DiscParticle::printDebug(std::ostream& outmsg) {

  Inform msg("DiscParticle", outmsg, INFORM_ALL_NODES);

  msg << "BaseFile   = " << BaseFile << endl;
  msg << "NumRecords = " << get_NumRecords() << endl;
  msg << "NextOffset = " << static_cast<long>(CurrentOffset) << endl;

  for (unsigned int r=0; r < get_NumRecords(); ++r) {
    msg << "For record " << r << ":" << endl;
    msg << "  Number of attributes = " << get_NumAttributes(r) << endl;
    msg << "  Total number of particles = " << get_NumGlobalParticles(r)<<endl;
    msg << "  Byte size of attributes =";
    for (unsigned int b=0; b < get_NumAttributes(r); ++b)
      msg << " " << get_ElemByteSize(r, b);
    msg << endl;
    msg << "  DiscTypes of attributes =";
    for (unsigned int t=0; t < get_NumAttributes(r); ++t) {
      if (get_DiscType(r, t) != 0)
	msg << " " << get_DiscType(r, t);
    }
    msg << endl;
    for (unsigned int f=0; f < Config->getNumFiles(); ++f) {
      msg << "  Local number of particles in fileset " << f << " = ";
      msg << RecordList[r]->localparticles[f] << endl;
      for (unsigned int a=0; a < get_NumAttributes(r); ++a) {
	msg << "  Offset of attribute " << a << " in fileset " << f << " = ";
	msg << static_cast<long>(RecordList[r]->offset[f][a]) << endl;
      }
    }
  }

  msg << "Configuration file information:" << endl;
  Config->printDebug(msg);

  msg << endl;
}


/***************************************************************************
 * $RCSfile: DiscParticleFunctions.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscParticleFunctions.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
