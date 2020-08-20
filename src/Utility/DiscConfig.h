// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DISC_CONFIG_H
#define DISC_CONFIG_H

/***************************************************************************
 * DiscConfig is a utility class which will read in a configuration file
 * for use by classes such as DiscField that read or write parallel data
 * files.  It should be given a config file name, which will be parsed by
 * the constructor; the config file should have the format:
 *
 *   # comment line(s), starting with '#'
 *   machine1   directory1-1
 *   machine2   directory2-1
 *   machine2   directory2-2
 *
 * Instead of an SMP machine name, you can specify '*', which acts as a
 * default line for all machines which are not listed elsewhere in the config
 * file.  There are also some allowed variables in the directory names,
 * which have the form '$(varname)'.  Allowed variables are
 *   $(n)   ... the node ID, from 0 ... (total nodes - 1)
 *   $(*)   ... the SMP machine name
 *   $(ENV) ... the value of environment variable 'ENV'
 * Examples of using these kinds of wildcards
 *
 *   ######################################################################
 *   # a sample config file with wildcards and variables
 *
 *   # first, a wildcard, for the directory /scratch/<username>/<hostname>
 *   *  /scratch/$(USER)/$(*)
 *
 *   # specific host commands
 *   host3  /usr/local/data/host3/dir1
 *   host3  /usr/local/data/host3/dir2
 *   host4  /tmp/dir3/$(n)
 *   #
 *   ######################################################################
 *
 * Machines may be listed more than once; they may also be omitted (that is,
 * the config file may be read by a job running on several different SMP's,
 * where the config file contains listings for only some of the SMP's).
 *
 ***************************************************************************/

// include files
#include "Utility/vmap.h"
#include "Utility/Inform.h"


class DiscConfig {

public:
  // Constructor: read in and parse the given config file.  We must know
  // whether the configuration file is being used to read or write data.
  DiscConfig(const char *, const char *, bool);

  // Destructor.
  ~DiscConfig();

  //
  // general accessor functions
  //

  // did everything in the config file parse OK?
  bool ok() const { return ConfigOK; }

  // return the name of our config file
  const std::string &getConfigFile() const { return ConfigFile; }

  // return the number of SMP's there are, in total, and the number
  // of SMP 'filesets' which will be read or written.
  unsigned int numSMPs() const { return NumSMPs; }
  unsigned int fileSMPs() const { return FileSMPs; }
  unsigned int mySMP() const { return MySMP; }

  //
  // SMP accessor functions
  //

  // return the SMP index for the SMP with the given name, or for my own
  unsigned int getSMPIndex() const { return MySMP; }
  unsigned int getSMPIndex(const std::string &smpname) const {
    return SMPMap[smpname]->SMPIndex;
  }

  // return the host name of the Nth SMP
  const std::string &getSMPHost() const { return getSMPHost(MySMP); }
  const std::string &getSMPHost(unsigned int smp) const {
    return SMPList[smp]->HostName;
  }

  // return the number of nodes in the Nth SMP
  unsigned int getNumSMPNodes() const { return getNumSMPNodes(MySMP); }
  unsigned int getNumSMPNodes(unsigned int smp) const {
    return SMPList[smp]->NodeList.size();
  }

  // return the mth node ID for the Nth SMP
  unsigned int getSMPNode(unsigned int n) const { return getSMPNode(MySMP,n);}
  unsigned int getSMPNode(unsigned int smp, unsigned int n) const {
    return SMPList[smp]->NodeList[n];
  }

  // return the Box0 node of the Nth SMP
  unsigned int getSMPBox0() const { return getSMPBox0(MySMP); }
  unsigned int getSMPBox0(unsigned int smp) const {
    return SMPList[smp]->Box0Node;
  }

  // return the number of filesets being read/written on the Nth smp
  unsigned int getNumFiles() const { return getNumFiles(MySMP); }
  unsigned int getNumFiles(unsigned int smp) const {
    return SMPList[smp]->BaseFileNum;
  }

  // return the base name of the mth fileset on the Nth smp
  const std::string &getFilename(unsigned int fn) const {
    return getFilename(MySMP, fn);
  }
  const std::string &getFilename(unsigned int smp, unsigned int fn) const {
    return SMPList[smp]->BaseFileName[fn];
  }

  // return the number of SMP's which depend on getting info from the Nth SMP
  unsigned int getNumOtherSMP() const { return getNumOtherSMP(MySMP); }
  unsigned int getNumOtherSMP(unsigned int smp) const {
    return SMPList[smp]->InformSMPList.size();
  }

  // return the index of the mth SMP which depends on info from the Nth SMP
  unsigned int getOtherSMP(unsigned int sn) const {
    return getOtherSMP(MySMP, sn);
  }
  unsigned int getOtherSMP(unsigned int smp, unsigned int sn) const {
    return SMPList[smp]->InformSMPList[sn];
  }

  //
  // Node accessor functions
  //

  // compute how many physical nodes there are on the same SMP as that w/node
  unsigned int pNodesPerSMP(unsigned int node) const;

  // return the total number of nodes
  unsigned int getNumNodes() const { return NodeList.size(); }

  // return the SMP index for the Nth node
  unsigned int getNodeSMPIndex(unsigned int n) const {
    return NodeList[n]->SMPIndex;
  }

  // return the hostname for the Nth node
  const std::string &getNodeHost(unsigned int n) const {
    return NodeList[n]->HostName;
  }

  
  //
  // utility functions
  //

  // a simple routine to take an input string and a list of token separators,
  // and return the number of tokens plus fill in a new array of strings
  // with the words.  We had a nicer way to do this with a vector of strings,
  // but a #*#^@($ bug in KCC requires this workaround.
  static int dc_tokenize_string(const char *s, const char *tok, std::string *&);

  // print out debugging information to the given Inform stream
  void printDebug(Inform &);

private:
  // Data needed for each SMP being used to read/write data.
  // Note that BaseFileNameStringList is being used here to store
  // a list of names only for use in constructing the BaseFileName list;
  // due to a bug in the use of vector<string> with KCC (debug only),
  // we cannot use the much more useful vector<string> to store the
  // fileset directory names.  Sigh.
  struct SMPData {
    std::string         HostName;
    std::string *       BaseFileName;
    std::string         BaseFileNameStringList;
    std::vector<int>    NodeList;
    std::vector<int>    InformSMPList;
    unsigned int   Box0Node;
    unsigned int   SMPIndex;
    unsigned int   BaseFileNum;
  };

  // Data needed for each pnode
  struct NodeData {
    std::string       HostName;
    unsigned int SMPIndex;
  };

  // The name of the configuration file
  std::string ConfigFile;

  // the number of SMP boxes found during setup and in the files,
  // and which SMP this node is on (0 ... NumSMPs - 1)
  unsigned int NumSMPs;
  unsigned int FileSMPs;
  unsigned int MySMP;

  // was the configuration file read successfully?  If not, we cannot
  // do any read/write operations.
  bool ConfigOK;

  // data for each SMP, and for each physical node.  These are built
  // when the config file is parsed and the system is analyzed.
  // vmap: key = hostname for SMP, value = SMPData structure
  vmap<std::string,SMPData *>   SMPMap;
  std::vector<SMPData *>        SMPList;
  std::vector<NodeData *>       NodeList;

  // take a string with configuration filename wildcards, and substitute
  // in the specific values.
  // The first argument is the original string with wildcards (listed below),
  // and the second is the machine name to use when substituting in the
  // machine name.
  // Return a new string with the changes in place.
  std::string replace_wildcards(const std::string& s, const std::string& machine);

  // take the information about the directory and hostname for a given
  // SMP, and add it to the list of directories for that SMP.  Make sure
  // the directory is not repeated.  If it is, issue a warning and continue.
  // Must be told if we're writing a file (otherwise, we're reading)
  void add_SMP_directory(SMPData *&, const std::string& s, const std::string& m, bool);

  // parse the IO configuration file and store the information.  Must be
  // told if we're writing a file (otherwise, we're reading)
  bool parse_config(const char *, bool);
};

#endif // DISC_CONFIG_H

/***************************************************************************
 * $RCSfile: DiscConfig.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscConfig.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
