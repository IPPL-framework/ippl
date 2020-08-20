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
#include "Utility/DiscField.h"
#include "Utility/DiscConfig.h"
#include "Utility/DiscMeta.h"
#include "Field/BrickIterator.h"
#include "Message/Tags.h"

#include "Utility/PAssert.h"
#include "Utility/IpplStats.h"
#include <cstring>
#include <cerrno>


///////////////////////////////////////////////////////////////////////////
// Constructor: make a DiscField for writing
// fname = name of file (without extensions)
// config = name of configuration file
// numFields = number of Fields in the file (for writing)
// typestr = string describing the 'type' of the Field to be written (this
//           is ignored if the Field is being read).  The string should be
//           the same as the statement used to declare the Field object
//           e.g., for a field of the form Field<double,2> the string
//           should be "Field<double,2>".  The string should be such that
//           if read later into a variable F, you can use the string in
//           a source code line as  'F A' to create an instance of the
//           same type of data as is stored in the file.
template <unsigned Dim>
DiscField<Dim>::DiscField(const char* base, const char* config,
			  unsigned int numFields, const char* typestr) {

  initialize(base, config, typestr, numFields);
}


///////////////////////////////////////////////////////////////////////////
// Constructor: same as above, but without a config file specified.  The
// default config file entry that will be used is "*  .", which means, for
// each SMP machine, assume the directory to put data files in is "."
template <unsigned Dim>
DiscField<Dim>::DiscField(const char* base, unsigned int numFields,
			  const char* typestr) {

  initialize(base, 0, typestr, numFields);
}


///////////////////////////////////////////////////////////////////////////
// Constructor: make a DiscField for reading only.
// fname = name of file (without extensions
// config = name of configuration file
template <unsigned Dim>
DiscField<Dim>::DiscField(const char* base, const char* config) {

  initialize(base, config, 0, 0);
}


///////////////////////////////////////////////////////////////////////////
// Constructor: same as above, but without a config file specified.  The
// default config file entry that will be used is "*  .", which means, for
// each SMP machine, assume the directory to put data files in is "."
template <unsigned Dim>
DiscField<Dim>::DiscField(const char* base) {

  initialize(base, 0, 0, 0);
}


///////////////////////////////////////////////////////////////////////////
// perform initialization based on the constuctor arguments
template <unsigned Dim>
void DiscField<Dim>::initialize(const char *base, const char *config,
				const char *typestr, unsigned int numFields) {

  // save string data
  BaseFile = base;
  DiscType = "";
  if (typestr != 0)
    TypeString = typestr;
  else
    TypeString = "unknown";

  // initialize member data
  DataDimension = Dim;
  CurrentOffset = 0;
  NumRecords = 0;
  NumWritten = 0;
  NumVnodes = 0;
  VnodeTally = 0;

  // save the number of fields, which indicates if this object is being
  // opened for reading or writing
  NumFields = numFields;
  WritingFile = (NumFields > 0);
  if (WritingFile)
    NeedStartRecord = 1;
  else
    NeedStartRecord = -1;

  // initialize valid field flags
  for (unsigned int i=0; i < NumFields; ++i)
    ValidField.push_back(false);

  // parse the configuration file to find our SMP's, file directories, etc.
  ConfigOK = parse_config(config, WritingFile);

  // figure out the number of fields, number of records, size, and
  // typestring from the .meta file, and store it here.  This is only done
  // if we're reading the file
  if (ConfigOK && !WritingFile)
    ConfigOK = read_meta();
}


///////////////////////////////////////////////////////////////////////////
// Destructor
template <unsigned Dim>
DiscField<Dim>::~DiscField() {

  // delete per-record vnode information
  if (NumVnodes != 0)
    delete [] NumVnodes;
  if (VnodeTally != 0)
    delete [] VnodeTally;

  // delete the configuration file info
  if (Config != 0)
    delete Config;
}


///////////////////////////////////////////////////////////////////////////
// Obtain all the information about the file, including the number
// of records, fields, and number of vnodes stored in each record.
template <unsigned Dim>
void DiscField<Dim>::query(int& numRecords, int& numFields,
			   std::vector<int>& size) const {

  numRecords = NumRecords;
  numFields = NumFields;
  if (numFiles() > 0 && myBox0() == Ippl::myNode()) {
    size = NumVnodes[0];
    for (int i=1; i < numFiles(); ++i) {
      for (int j=0; j < NumVnodes[i].size(); ++j)
	size[j] += NumVnodes[i][j];
    }
  }
}


///////////////////////////////////////////////////////////////////////////
// open a file in the given mode.  If an error occurs, print a message
// and return 0.
template <unsigned Dim>
FILE* DiscField<Dim>::open_df_file(const std::string& fnm, const std::string& mode) {

  FILE *f = fopen(fnm.c_str(), mode.c_str());
  if (f == 0) {
    ERRORMSG("DiscField: Could not open file '" << fnm.c_str());
    ERRORMSG("' for mode '" << mode.c_str() << "' on node ");
    ERRORMSG(Ippl::myNode() << "." << endl);
    Ippl::abort("Exiting due to DiscField error.");
  }
  return f;
}


///////////////////////////////////////////////////////////////////////////
// open a file in the given mode.  If an error occurs, print a message
// and return 0.  This version is used for data files.
template <unsigned Dim>
int DiscField<Dim>::open_df_file_fd(const std::string& fnm, const std::string& suf,
				    int origflags) {

  // Form a string with the total filename
  std::string fnamebuf("");
  if (fnm.length() > 0)
    fnamebuf += fnm;
  if (suf.length() > 0)
    fnamebuf += suf;

  // Form the open flags
  int flags = origflags;

  // Try to open the file
  int f = ::open(fnamebuf.c_str(), flags, 0644);
  if (f < 0) {
    ERRORMSG("DiscField: Could not open file '" << fnamebuf.c_str());
    ERRORMSG("' on node " << Ippl::myNode() << ", f = " << f << "."<<endl);
    return (-1);
  }
  return f;
}


///////////////////////////////////////////////////////////////////////////
// same as above, but also specifying a file suffix
template <unsigned Dim>
FILE* DiscField<Dim>::open_df_file(const std::string& fnm, const std::string& suf,
				   const std::string& mode) {

  std::string fnamebuf("");
  if (fnm.length() > 0)
    fnamebuf += fnm;
  if (suf.length() > 0)
    fnamebuf += suf;

  /*
  char fnamebuf[1024];
  fnamebuf[0] = '\0';
  if (fnm.length() > 0)
    strcat(fnamebuf, fnm.c_str());
  if (suf.length() > 0)
    strcat(fnamebuf, suf.c_str());
  FILE *f = fopen(fnamebuf, mode.c_str());
  */

  FILE *f = fopen(fnamebuf.c_str(), mode.c_str());

  if (f == 0) {
    ERRORMSG("DiscField: Could not open file '" << fnamebuf);
    ERRORMSG("' for mode '" << mode.c_str() << "' on node ");
    ERRORMSG(Ippl::myNode() << "." << endl);
    Ippl::abort("Exiting due to DiscField error.");
  }
  return f;
}


///////////////////////////////////////////////////////////////////////////
// create the data files used to store Field data.  Return success.
template <unsigned Dim>
bool DiscField<Dim>::create_files() {


  FILE *f;
  std::string om("w");
  std::string suff[4];
  suff[0] = ".meta";
  suff[1] = ".layout";
  suff[2] = ".offset";
  suff[3] = ".data";

  unsigned int nfiles = 3;

  // create the non-data files
  for (unsigned int i=0; i < nfiles; ++i) {
    std::string fname(Config->getFilename(0) + suff[i]);
    if ((f = open_df_file(fname, om)) == 0) {
      ERRORMSG("DiscField: Could not create file '" << fname.c_str());
      ERRORMSG("'." << endl);
      Ippl::abort("Exiting due to DiscField error.");
    }
    fclose(f);
  }

  // create the data file
  int fd = open_df_file_fd(Config->getFilename(0), suff[3],
			   O_RDWR|O_CREAT|O_TRUNC);
  if (fd < 0) {
    std::string fname(Config->getFilename(0) + suff[3]);
    ERRORMSG("DiscField: Could not create data file '"<<fname.c_str());
    ERRORMSG("'. errno = " << errno << endl);
    Ippl::abort("Exiting due to DiscField error.");
  } else {
    close(fd);
  }

  return true;
}


///////////////////////////////////////////////////////////////////////////
// Since the layout can be different every time write
// is called, the globalID container needs to be recalculated.  The total
// domain of the Field should not change, though, just the layout.  Return
// success.
template <unsigned Dim>
bool DiscField<Dim>::make_globalID(FieldLayout<Dim>& layout) {

  // check layout to make sure it's valid
  if (Size.size() != 0 && !(Size == layout.getDomain()))
    return false;
  else
    Size = layout.getDomain();

  // get rid of the existing mapping
  globalID.erase(globalID.begin(), globalID.end());

  // for each local vnode, get the NDIndex it has and store it along with
  // the node that owns it.
  typedef typename GlobalIDList_t::value_type vtype;
  typename FieldLayout<Dim>::iterator_iv local;
  for (local = layout.begin_iv() ; local != layout.end_iv(); ++local) {
    // get the domain, and the node and SMP that holds that domain
    NDIndex<Dim>& domain = (NDIndex<Dim>&) (*local).second.get()->getDomain();
    int node = (*local).second.get()->getNode();
    unsigned int nodesmp = Config->getNodeSMPIndex(node);

    // find out of any of our SMP's contain that node
    bool foundsmp = (nodesmp == mySMP());
    unsigned int checksmp = 0;
    while (!foundsmp && checksmp < Config->getNumOtherSMP()) {
      foundsmp = (nodesmp == Config->getOtherSMP(checksmp));
      checksmp++;
    }

    // if we are responsible for this vnode, save it
    if (foundsmp) {
      // WARNMSG(" ==> Found vnode " << domain << " on this SMP, from node ");
      // WARNMSG(node << endl);
      globalID.insert(vtype(domain, node));
    }
  }

  // for each remote vnode, get the NDIndex it has and store it along with
  // the node that owns it.
  typename FieldLayout<Dim>::iterator_dv remote;
  for (remote = layout.begin_rdv() ; remote != layout.end_rdv(); ++remote) {
    // get the domain, and the node and SMP that holds that domain
    NDIndex<Dim>& domain = (NDIndex<Dim>&) (*remote).first;
    int node = (*remote).second->getNode();
    unsigned int nodesmp = Config->getNodeSMPIndex(node);

    // find out of any of our SMP's contain that node
    bool foundsmp = (nodesmp == mySMP());
    unsigned int checksmp = 0;
    while (!foundsmp && checksmp < Config->getNumOtherSMP()) {
      foundsmp = (nodesmp == Config->getOtherSMP(checksmp));
      checksmp++;
    }

    // if we are responsible for this vnode, save it
    if (foundsmp) {
      // WARNMSG(" ==> Found vnode " << domain << " on this SMP, from node ");
      // WARNMSG(node << endl);
      globalID.insert(vtype(domain, node));
    }
  }

  return true;
}


///////////////////////////////////////////////////////////////////////////
// read in from configuration file - an ascii file of token pairs.
// This is mostly handled by the DiscConfig class.  Return success.
template <unsigned Dim>
bool DiscField<Dim>::parse_config(const char *fname, bool writing) {

  // create a DiscConfig instance, which will parse the file on some of
  // the nodes and distribute the information to all the other nodes.
  Config = new DiscConfig(fname, BaseFile.c_str(), writing);

  // need to set up a few things if the config file checked out OK
  if (Config->ok()) {
    // create vnode information storage for Box0 nodes
    if (numFiles() > 0 && myBox0() == (unsigned int) Ippl::myNode()) {
      NumVnodes  = new std::vector<int>[numFiles()];
      VnodeTally = new std::vector<int>[numFiles()];

      // if we need to, create the files
      if (writing) {
        if (!create_files())
	  return false;
      }
    }

    // indicate how many filesets we're reading or writing
    /*
    if (writing) {
      ADDIPPLSTAT(incDiscFilesetWrites,numFiles());
    } else {
      ADDIPPLSTAT(incDiscFilesetReads,numFiles());
    }
    */
  } else {
    ERRORMSG("DiscField: A problem occurred reading the config file '");
    ERRORMSG(fname << "'." << endl);
    Ippl::abort("Exiting due to DiscField error.");
  }

  return Config->ok();
}


///////////////////////////////////////////////////////////////////////////
// print out debugging information for this DiscField
template <unsigned Dim>
void DiscField<Dim>::printDebug() { printDebug(std::cout); }

template <unsigned Dim>
void DiscField<Dim>::printDebug(std::ostream& outmsg) {

  Inform msg("DiscField", outmsg, INFORM_ALL_NODES);

  msg << "BaseFile = " << BaseFile << endl;
  msg << "Field Type = " << TypeString << endl;
  msg << "NumRecords = " << NumRecords << endl;
  msg << "NumFields = " << NumFields << endl;

  msg << "Configuration file information:" << endl;
  Config->printDebug(msg);

  msg << endl;
}


///////////////////////////////////////////////////////////////////////////
// write out a new .meta file.  The .meta file contains the following
// information, on each line:
//   String with type of data stored here, supplied by user
//   Dimension
//   For each dimension:
//     Total domain of the Fields, as   first  last  stride
//   Number of Fields
//   Number of Records
//   Number of SMPs
//   Vnodes/SMP for each record, on one line
//   Vnodes/SMP tally for each record, on one line
// return success of operation.
template <unsigned Dim>
bool DiscField<Dim>::write_meta() {

  unsigned int r, d;

  // no need to write anything if we have no files for this SMP
  if (numFiles() == 0)
    return true;

  // open the meta data file
  FILE *outputMeta = open_df_file(Config->getFilename(0),".meta",std::string("w"));
  if (outputMeta == 0)
    return false;

  // write the initial header info
  fprintf(outputMeta, "Type =           %s\n", TypeString.c_str());
  fprintf(outputMeta, "Dim =            %u\n", Dim);
  for (d=0; d < Dim; ++d)
    fprintf(outputMeta, "Domain =         %d %d %d\n",
	    Size[d].first(), Size[d].last(), Size[d].stride());
  fprintf(outputMeta, "Fields =         %u\n", NumFields);
  fprintf(outputMeta, "Records =        %u\n", NumRecords);
  fprintf(outputMeta, "SMPs =           %u\n", fileSMPs());

  // write information for each record.  When writing, we will only
  // write one file set per box, so we use '0' for the fileset number.
  fprintf(outputMeta, "VnodesInRecord = ");
  for (r=0; r < NumRecords; ++r)
    fprintf(outputMeta, " %d", NumVnodes[0][r]);
  fprintf(outputMeta, "\n");

  fprintf(outputMeta, "VnodeTally=    ");
  for (r=0; r < NumRecords; ++r)
    fprintf(outputMeta, " %d", VnodeTally[0][r]);
  fprintf(outputMeta, "\n");

  // close data file and return
  fclose(outputMeta);
  return true;
}


///////////////////////////////////////////////////////////////////////////
// read in data from .meta file, and replace current storage values.
// The format for a .meta file is described in the write_meta routine.
// return success of operation.
template <unsigned Dim>
bool DiscField<Dim>::read_meta() {

  bool iserror = false;
  int tag = Ippl::Comm->next_tag(DF_READ_META_TAG, DF_TAG_CYCLE);

  // on Box0 nodes, read in the meta data ... on others, wait for
  // Box0 nodes to send info to them
  if ((unsigned int) Ippl::myNode() == myBox0()) {
    // loop over all the files on this Box0 node
    for (unsigned int sf=0; sf < numFiles(); ++sf) {
      // open and parse the meta data file
      std::string filename = Config->getFilename(sf) + ".meta";
      DiscMeta outputMeta(filename.c_str());
      if (outputMeta.size() == 0) {
	ERRORMSG("DiscField: The meta file '" << filename << "' is empty ");
	ERRORMSG("or does not exist." << endl);
	Ippl::abort("Exiting due to DiscField error.");
	return false;
      }

      // initialize data before reading .meta file
      unsigned int dimread = 0;
      TypeString = "unknown";
      DataDimension = Dim;
      NumFields = 0;
      NumRecords = 0;
      NumVnodes[sf].erase(NumVnodes[sf].begin(), NumVnodes[sf].end());
      VnodeTally[sf].erase(VnodeTally[sf].begin(), VnodeTally[sf].end());

      // keep reading until we have all data
      DiscMeta::iterator metaline, metaend = outputMeta.end();
      for (metaline = outputMeta.begin(); metaline != metaend; ++metaline) {
	// get number of tokens and list of tokens in the line
	int linesread  = (*metaline).first;
	int numtokens  = (*metaline).second.first;
	std::string *tokens = (*metaline).second.second;

        // action is based on first keyword
        if (tokens[0] == "Type") {
	  if (numtokens > 1)
	    TypeString = tokens[1];
        }
        else if (tokens[0] == "Dim" && numtokens == 2) {
	  DataDimension = atoi(tokens[1].c_str());
	  if (DataDimension < 1) {
	    ERRORMSG("DiscField: The meta file '" << filename << "' ");
	    ERRORMSG("contains a value for dimension < 1, '");
	    ERRORMSG(tokens[1] << "'." << endl);
	    Ippl::abort("Exiting due to DiscField error.");
	    iserror = true;
	  }
        }
        else if (tokens[0] == "Fields" && numtokens == 2) {
	  NumFields = atoi(tokens[1].c_str());
	  if (NumFields < 1) {
	    ERRORMSG("DiscField: The meta file '" << filename << "' ");
	    ERRORMSG("contains a value for Fields < 1, '");
	    ERRORMSG(tokens[1] << "'." << endl);
	    Ippl::abort("Exiting due to DiscField error.");
	    iserror = true;
	  }
        }
        else if (tokens[0] == "Records" && numtokens == 2) {
	  NumRecords = atoi(tokens[1].c_str());
        }
        else if (tokens[0] == "SMPs" && numtokens == 2) {
          unsigned int checkfileSMPs = atoi(tokens[1].c_str());
	  if (fileSMPs() != checkfileSMPs) {
	    ERRORMSG("DiscField: The meta file '" << filename << "' ");
	    ERRORMSG("contains a value for the number of filesets that\n");
	    ERRORMSG("does not match the number of filesets in the config\n");
	    ERRORMSG("file: metafile filesets = " << tokens[1] << ", ");
	    ERRORMSG("config filesets = " << fileSMPs() << "." << endl);
	    Ippl::abort("Exiting due to DiscField error.");
	    iserror = true;
	  }
        }
        else if (tokens[0] == "Domain" && numtokens == 4) {
	  if (dimread < Dim) {
	    Size[dimread] = Index(atoi(tokens[1].c_str()),
				  atoi(tokens[2].c_str()),
				  atoi(tokens[3].c_str()));
	  }
	  dimread++;
        }
        else if (tokens[0] == "VnodesInRecord") {
	  for (int r=1; r < numtokens; ++r)
	    NumVnodes[sf].push_back(atoi(tokens[r].c_str()));
        }
        else if (tokens[0] == "VnodeTally") {
	  for (int r=1; r < numtokens; ++r)
	    VnodeTally[sf].push_back(atoi(tokens[r].c_str()));
        }
        else {
	  // error in line
	  ERRORMSG("DiscField: Format error on line " << linesread);
	  ERRORMSG(" in meta file '" << filename << "'." << endl);
	  Ippl::abort("Exiting due to DiscField error.");
	  iserror = true;
        }

	if (iserror)
          break;
      }

      // do a little sanity checking
      if (DataDimension != dimread) {
        ERRORMSG("DiscField: Dim != # Domain lines in meta file '");
	ERRORMSG(filename << "'. (" << DataDimension << " != " << dimread);
	ERRORMSG(")" << endl);
	Ippl::abort("Exiting due to DiscField error.");
        iserror = true;
      }
      if (NumRecords != NumVnodes[sf].size()) {
        ERRORMSG("DiscField: Records != VnodesInRecord items in meta file '");
	ERRORMSG(filename << "'. (" << NumRecords << " != ");
	ERRORMSG(NumVnodes[sf].size() << ")" << endl);
	Ippl::abort("Exiting due to DiscField error.");
        iserror = true;
      }
      if (NumRecords != VnodeTally[sf].size()) {
        ERRORMSG("DiscField: Records != VnodeTally items in meta file '");
	ERRORMSG(filename << "'. (" << NumRecords << " != ");
	ERRORMSG(VnodeTally[sf].size() << ")" << endl);
	Ippl::abort("Exiting due to DiscField error.");
        iserror = true;
      }

      // stop processing meta files is there was an error
      if (iserror)
        break;
    }

    // now send meta info to all nodes which expect it
    int numinform = Config->getNumOtherSMP();
    for (int s=0; s <= numinform; ++s) {
      int smp = mySMP();
      if (s != numinform)
	smp = Config->getOtherSMP(s);
      for (unsigned int n=0; n < Config->getNumSMPNodes(smp); ++n) {
        int node = Config->getSMPNode(smp, n);
        if (node != Ippl::myNode()) {
          // create a message with meta info
          Message *msg = new Message;
	  int errint = iserror;
          msg->put(errint);
          msg->put(DataDimension);
          msg->put(NumFields);
          msg->put(NumRecords);
          ::putMessage(*msg, Size);
          ::putMessage(*msg, TypeString);

          // send the message to the intended node
          Ippl::Comm->send(msg, node, tag);
	}
      }
    }

  } else {
    // all other nodes (which are not Box0 nodes) should get a message
    // telling them the meta info
    int node = myBox0();
    Message *msg = Ippl::Comm->receive_block(node, tag);
    PAssert(msg);

    // get info out of message
    int errint = 0;
    msg->get(errint);
    iserror = (errint != 0);
    msg->get(DataDimension);
    msg->get(NumFields);
    msg->get(NumRecords);
    ::getMessage(*msg, Size);
    ::getMessage(*msg, TypeString);

    // we're done with this message
    delete msg;
  }

  return (!iserror);
}


///////////////////////////////////////////////////////////////////////////
// Read the data for a single NDIndex from the given file.  Return success.
template <unsigned Dim>
bool DiscField<Dim>::read_NDIndex(FILE *f, NDIndex<Dim> &ndi) {
  // an array of ints used to read the data
  int ndidata[6*Dim];

  // OK, this is a mess, for the reasons described in write_NDIndex.  This
  // reads in 6 ints for each dimension, of which three are used for
  // the first, stride, and length parameters.  These are put in to the given
  // NDIndex.

  // first read the data into the int array
  if (fread(ndidata, sizeof(int), 6*Dim, f) != 6*Dim) {
    ERRORMSG("DiscField: Error reading NDIndex line from data file." << endl);
    Ippl::abort("Exiting due to DiscField error.");
    return false;
  }

  // now copy data int the NDIndex
  int *dptr = ndidata + 1;
  for (unsigned int d=0; d < Dim; ++d) {
    int first = *dptr;
    int stride = *(dptr + 1);
    int length = *(dptr + 2);
    ndi[d] = Index(first, first + (length - 1)*stride, stride);
    dptr += 6;
  }

  return true;
}


///////////////////////////////////////////////////////////////////////////
// Write the data for a single NDIndex to the given file.  Return success.
template <unsigned Dim>
bool DiscField<Dim>::write_NDIndex(FILE *f, const NDIndex<Dim> &ndi) {
  // OK, this is a mess ... originally, data was just written
  // out from an NDIndex directly, which is just an array of Index objs.
  // However, the contents of an Index are compiler-specific, since some
  // compilers use the empty-base-class optimization, some put in an
  // extra int in a base class, etc.  So, this has been switched from
  // writing out the NDIndex data using
  //   fwrite(&((*id).first), sizeof(NDIndex<Dim>), 1, outputLayout)
  // to
  //   fwrite(ndidata, sizeof(int), 6*Dim, outputLayout)
  // since the original, most-used version of DiscField did so with
  // a compiler that constructed an Index object with this data:
  //   int = 0    (empty-base-class storage)
  //   int = first
  //   int = stride
  //   int = length
  //   int = base
  //   int = id
  // With some compilers, the initial int = 0 is not included, since it
  // is not necessary.  But for backwards compatibility, we'll use this
  // format of 6 ints, with the first, fifth, and sixth set to zero

  // first copy the data into the int array
  int ndidata[6*Dim];
  int *dptr = ndidata;
  for (unsigned int d=0; d < Dim; ++d) {
    *dptr++ = 0;
    *dptr++ = ndi[d].first();
    *dptr++ = ndi[d].stride();
    *dptr++ = ndi[d].length();
    *dptr++ = 0;
    *dptr++ = 0;
  }

  // now write the data, and report whether the result is OK
  return (fwrite(ndidata, sizeof(int), 6*Dim, f) == 6*Dim);
}


///////////////////////////////////////////////////////////////////////////
// update the .layout file, which contains information on the layout
// of the Fields for each record.  This file contains a set of NDIndex
// objects for each record.  The format, for each record, is:
//   Number of vnodes on this SMP box
//   For each vnode on this SMP box:
//     Vnode object, containing   NDIndex(first  last  stride)
// return success of update
template <unsigned Dim>
bool DiscField<Dim>::write_layout() {

  // no need to write anything if we have no files for this SMP
  if (numFiles() == 0)
    return true;

  // open the layout data file
  FILE *outputLayout = open_df_file(Config->getFilename(0), ".layout",
				    std::string("a"));

  // write out the number of vnodes in this record for this file set
  int numvnodes = globalID.size();
  if (fwrite(&numvnodes, sizeof(int), 1, outputLayout) != 1) {
    ERRORMSG("Error writing .layout file in DiscField::write_layout." << endl);
    Ippl::abort("Exiting due to DiscField error.");
    fclose(outputLayout);
    return false;
  }

  // write out the current vnode sizes from the provided FieldLayout
  typename GlobalIDList_t::iterator id, idend = globalID.end();
  for (id = globalID.begin(); id != idend; ++id) {
    if (!write_NDIndex(outputLayout, (*id).first)) {
      ERRORMSG("Error writing record " << NumRecords-1 << " to .layout file.");
      Ippl::abort("Exiting due to DiscField error.");
      fclose(outputLayout);
      return false;
    }
  }

  // close the file, we're done
  fclose(outputLayout);
  return true;
}


///////////////////////////////////////////////////////////////////////////
// Read layout info for one file set in the given record.
template <unsigned Dim>
int DiscField<Dim>::read_layout(int record, int sf) {

  // open the layout data file
  std::string filename = Config->getFilename(sf) + ".layout";
  FILE *outputLayout = open_df_file(Config->getFilename(sf),
				    ".layout", std::string("r"));
  if (outputLayout == 0)
    return (-1);

  // seek to the proper location
  Offset_t seekpos = record*sizeof(int) +
    VnodeTally[sf][record]*6*Dim*sizeof(int);

  if (fseek(outputLayout, seekpos, SEEK_SET) != 0) {
    ERRORMSG("Error seeking to position " << static_cast<long>(seekpos));
    ERRORMSG(" in file '" << filename << "' for DiscField::read_layout.");
    ERRORMSG(endl);
    Ippl::abort("Exiting due to DiscField error.");
    fclose(outputLayout);
    return (-1);
  }

  // read the number of vnodes stored here
  int numvnodes = 0;
  if (fread(&numvnodes, sizeof(int), 1, outputLayout) != 1) {
    ERRORMSG("Error reading file '" << filename);
    ERRORMSG("' in DiscField::read_layout.");
    ERRORMSG(endl);
    Ippl::abort("Exiting due to DiscField error.");
    fclose(outputLayout);
    return (-1);
  }

  // we used to have to read the vnode domains, but now it is
  // not necessary, we get that from the offset file.  layout files
  // will continue to store correct data for backwards compatibility,
  // but we don't need it for reading any more.  So, just return.

  fclose(outputLayout);
  return numvnodes;
}


///////////////////////////////////////////////////////////////////////////
// Compute how many elements we should expect to store into the local
// node for the given FieldLayout.  Just loop through the local vnodes
// and accumulate the sizes of the owned domains.  This is modified
// by the second "read domain" argument, which might be a subset of
// the total domain.
template <unsigned Dim>
int DiscField<Dim>::compute_expected(const FieldLayout<Dim> &f,
				     const NDIndex<Dim> &readDomain) {
  int expected = 0;

  typename FieldLayout<Dim>::const_iterator_iv local = f.begin_iv();
  for (; local != f.end_iv(); ++local) {
    // Compute the size of the domain of this vnode, and add it to the
    // amount we expect for this node.
    NDIndex<Dim>& domain = (NDIndex<Dim>&)(*local).second.get()->getDomain();
    if (domain.touches(readDomain)) {
      NDIndex<Dim> newdomain = domain.intersect(readDomain);
      // expected += domain.size();
      expected += newdomain.size();
    }
  }

  return expected;
}


///////////////////////////////////////////////////////////////////////////
// Compute the size of a domain, zero-based, that has a total
// size <= chunkelems and has evenly chunked slices.
template <unsigned Dim>
NDIndex<Dim> DiscField<Dim>::chunk_domain(const NDIndex<Dim> &currblock,
					  int chunkelems,
					  int &msdim,
					  bool iscompressed)
{

  // Initialize result to the total block sizes
  NDIndex<Dim> sliceblock;
  for (unsigned int i=0; i < Dim; ++i)
    sliceblock[i] = Index(currblock[i].length());

  // If this is compressed, or we are not chunking, just use the whole block
  int currsize = currblock.size();
  if (chunkelems < 1 || iscompressed)
    chunkelems = currsize;

  // Find out actual size for each dimension, generally smaller than the
  // total size
  for (int d=(Dim-1); d >= 0; --d) {
    // Size of this domain slice
    int axislen = currblock[d].length();
    currsize /= axislen;

    // Compute the number of lower-dim slices
    int numslices = chunkelems / currsize;
    if (numslices > axislen)
      numslices = axislen;

    // If there are some slices, just use those and we're done
    if (numslices > 0) {
      sliceblock[d] = Index(numslices);
      msdim = d;
      break;
    } else {
      // No slices here, so change the length of this dimension to 1
      sliceblock[d] = Index(1);
    }
  }

  // Return the newly computed block size; msdim is set to the maximum
  // dimension of the chunk block
  return sliceblock;
}


/***************************************************************************
 * $RCSfile: DiscField.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscField.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
