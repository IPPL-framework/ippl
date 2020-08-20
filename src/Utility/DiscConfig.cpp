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
#include "Utility/DiscConfig.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "Message/Communicate.h"
#include "Message/Message.h"


#include <algorithm>
using namespace std;
#include <cstring>
#include <unistd.h>
#include <cstdio>


///////////////////////////////////////////////////////////////////////////
// a simple routine to take an input string and a list of token separators,
// and return the number of tokens plus fill in a new array of strings
// with the words.  We had a nicer way to do this with a vector of strings,
// but a #*#^@($ bug in KCC requires this workaround.
int DiscConfig::dc_tokenize_string(const char *s, const char *tok,
				   string *&slist) {
  // first determine how many words there are
  int num = 0;
  char *tempstring = new char[strlen(s) + 1];
  strcpy(tempstring, s);
  slist = 0;
  char* tokenp = strtok(tempstring, tok);
  while (tokenp) {
    num++;
    tokenp = strtok(0, tok);
  }

  // create an array of strings, and find the words and copy them into strings
  // but only if we have any words at all
  if (num > 0) {
    slist = new string[num];
    num = 0;
    strcpy(tempstring, s);
    tokenp = strtok(tempstring, tok);
    while (tokenp) {
      slist[num++] = tokenp;
      tokenp = strtok(0, tok);
    }
  }

  delete [] tempstring;
  return num;
}


///////////////////////////////////////////////////////////////////////////
// Constructor: read in and parse the given config file.  We must know
// whether the configuration file is being used to read or write data,
// and the base filename for the input/output file.
DiscConfig::DiscConfig(const char *config, const char *BaseFile,
		       bool WritingFile)
  : NumSMPs(0), FileSMPs(0), MySMP(0), ConfigOK(false) {

  if (config != 0)
    ConfigFile = config;

  if (BaseFile == 0) {
    ERRORMSG("Null base filename in DiscConfig constructor." << endl);
    Ippl::abort("Exiting due to DiscConfig error.");
    ConfigOK = false;
  } else {
    ConfigOK = parse_config(BaseFile, WritingFile);
  }
}


///////////////////////////////////////////////////////////////////////////
// Destructor
DiscConfig::~DiscConfig() {

  // delete the SMP structures
  vector<SMPData *>::iterator smpiter = SMPList.begin();
  vector<SMPData *>::iterator smpend  = SMPList.end();
  while (smpiter != smpend) {
    if ((*smpiter)->BaseFileNum > 0)
      delete [] (*smpiter)->BaseFileName;
    delete (*smpiter++);
  }

  // delete the node structures
  vector<NodeData *>::iterator nodeiter = NodeList.begin();
  vector<NodeData *>::iterator nodeend  = NodeList.end();
  while (nodeiter != nodeend)
    delete (*nodeiter++);
}


///////////////////////////////////////////////////////////////////////////
// compute how many physical nodes there are on the same SMP as the given
// pnode.  This returns the total number of nodes which are writing data
// to this SMP.
unsigned int DiscConfig::pNodesPerSMP(unsigned int node) const {
  unsigned int nodesmp = NodeList[node]->SMPIndex;
  if (getNumFiles(nodesmp) == 0)
    return 0;

  unsigned int numnodes = SMPList[nodesmp]->NodeList.size();
  //unsigned int extrasmps = SMPList[nodesmp]->InformSMPList.size();
  //for (unsigned int i=0; i < extrasmps; ++i)
  //  numnodes += SMPList[SMPList[nodesmp]->InformSMPList[i]]->NodeList.size();
  return numnodes;
}


///////////////////////////////////////////////////////////////////////////
// take a string with configuration filename wildcards, and substitute
// in the specific values.
// The first argument is the original string with wildcards (listed below),
// and the second is the machine name to use when substituting in the
// machine name.
//
// Possible wildcards (can be upper or lower case, must start with $ and
// have the name enclosed in () ):
//   $(*) ... use the machine name given in the second argument
//   $(n) ... our node number
//   $(env var name) ... environment variable, if not one of the above names
//
// Return a new string with the changes in place.
string DiscConfig::replace_wildcards(const string& s,
				     const string& machine) {

  // the return string
  string retval;

  // make sure we have non-null input
  if (s.length() == 0 || machine.length() == 0)
    return retval;

  // copy of input string
  string scpy(s);
  char *sptrbase = (char*) scpy.c_str();

  // skip leading "./" if necessary
  if (s.length() > 2 && s[0] == '.' && s[1] == '/')
    sptrbase += 2;

  // start moving along the string until we get to a $
  char *sptr = sptrbase;
  while (*sptr != '\0') {
    if (*sptr != '$') {
      ++sptr;
    } else {
      // append previous text to return value
      if (sptr != sptrbase) {
	*sptr = '\0';
	retval += sptrbase;
	sptrbase = sptr + 1;
      }
      // find name of wildcard, enclosed by ()'s
      ++sptr;
      char *tok1 = sptr;
      char *tok2 = sptr;
      while (*tok1 != '(' && *tok1 != '\0') ++tok1;
      while (*tok2 != ')' && *tok2 != '\0') ++tok2;
      if (*tok1 == '\0' || *tok2 == '\0' || *tok2 <= *tok1) {
	ERRORMSG("Unbalanced parenthesis in DiscConfig config file in line ");
	ERRORMSG(s.c_str() << endl);
	Ippl::abort("Exiting due to DiscConfig error.");
	break;
      }
      // make string object with the wildcard name, and look for name.
      // replace token with new name
      *tok2 = '\0';
      string token(tok1 + 1);
      if (token == "*") {
	token = machine;
      } else if (token == "node" || token == "n" || token == "N") {
	char buf[32];
	sprintf(buf, "%d", Ippl::myNode());
	token = buf;
      } else {
	// look for an env var with this name
	char *env = getenv(token.c_str());
	if (env != 0) {
	  token = env;
	} else {
	  ERRORMSG("Unknown wildcard name '" << token.c_str()<<"' in line ");
	  ERRORMSG(s.c_str() << endl);
	  Ippl::abort("Exiting due to DiscConfig error.");
	  break;
	}
      }

      // add this token to the return string, and move on past wildcard
      retval += token;
      sptr = sptrbase = tok2 + 1;
    }
  }

  // append the final word to the return string
  if (sptr != sptrbase)
    retval += sptrbase;

  // done with substitution; return the string
  return retval;
}


///////////////////////////////////////////////////////////////////////////
// take the information about the directory and hostname for a given
// SMP, and add it to the list of directories for that SMP.  Make sure
// the directory is not repeated.  If it is, issue a warning and continue.
void DiscConfig::add_SMP_directory(SMPData *&smpd,
				   const string& s,
				   const string& machine,
				   bool WritingFile) {

  // create a new smpd if necessary, and add it to the list
  if (smpd == 0) {
    smpd = new SMPData;
    smpd->Box0Node = Ippl::getNodes();
    smpd->HostName = machine;
    smpd->BaseFileNum = 0;
    smpd->BaseFileName = 0;
    SMPMap.insert(vmap<string,SMPData *>::value_type(machine, smpd));
  }

  // if necessary, try to add a new directory.  Buf if no directory
  // was specified, we're done
  if (s.length() == 0)
    return;

  // create a string with the wildcards replaced, etc.
  string basename = replace_wildcards(s, machine);

  // check to make sure it does already occur in the list of BaseFileName's
  for (unsigned int sptr=0; sptr < smpd->BaseFileNum; ++sptr) {
    if (strcmp(basename.c_str(), (smpd->BaseFileName[sptr]).c_str()) == 0) {
      WARNMSG("DiscConfig: Duplicate configuration file entry '" << basename);
      WARNMSG("' for host " << machine << " ... second one ignored." << endl);
      return;
    }
  }

  // check to make sure we're not trying to write to more than one output
  // file on an SMP (multipple read files are OK, but we can only write
  // to one file, since we cannot determine how to partition items among
  // the files)
  if (WritingFile && smpd->BaseFileNum > 0) {
    WARNMSG("DiscConfig: Cannot write to more than one file per");
    WARNMSG(" SMP.  Only the first file listed for host '");
    WARNMSG(smpd->HostName << "', " << smpd->BaseFileName[0]);
    WARNMSG(", will be used." << endl);
    return;
  }

  // if we're here, the entry is not duplicated, so add the name.  We'll
  // need to add the name, then regenerate our tokenized list
  smpd->BaseFileNameStringList += " ";
  smpd->BaseFileNameStringList += basename;
  if (smpd->BaseFileNum > 0)
    delete [] smpd->BaseFileName;
  smpd->BaseFileNum = dc_tokenize_string(smpd->BaseFileNameStringList.c_str(),
					 " ", smpd->BaseFileName);
}


///////////////////////////////////////////////////////////////////////////
// read in from configuration file - an ascii file of token pairs.
// On each line, the first token is the hostname and the second
// token is the directory where the file is to be placed on this
// host.
// This will construct the vector with SMP data and set how many SMP's
// we expect to find.  If this does not match later, an error will be
// reported then.
bool DiscConfig::parse_config(const char *BaseFile, bool WritingFile) {

  const int bufferSize = 1024;
  char bufferstore[bufferSize];
  char *buffer;
  FILE *inC;
  string WildCard;
  string ConfigItems;
  string NodeNameItems;

  // create a tag for use in sending info to/from other nodes
  int tag = Ippl::Comm->next_tag(DF_MAKE_HOST_MAP_TAG, DF_TAG_CYCLE);

  // save the number of nodes and which node we're on
  NumSMPs = 0;
  FileSMPs = 0;
  MySMP = 0;

  // initialize the list of node information
  for (int i=0; i < Ippl::getNodes(); ++i)
    NodeList.push_back(new NodeData);

  // obtain the hostname and processor ID to send out
  char name[1024];
  if (gethostname(name, 1023) != 0) {
    WARNMSG("DiscConfig: Could not get hostname.  Using localhost." << endl);
    strcpy(name, "localhost");
  }
  NodeNameItems = name;

  // all other nodes send their hostname to node 0; node 0 gets the names,
  // reads the config file, then broadcasts all the necessary info to all
  // other nodes
  if (Ippl::myNode() != 0) {
    // other nodes send their node name to node 0
    Message *msg = new Message;
    ::putMessage(*msg,NodeNameItems);
    Ippl::Comm->send(msg, 0, tag);

    // receive back the config file and node name info
    int node = 0;
    msg = Ippl::Comm->receive_block(node, tag);
    PAssert(msg);
    PAssert_EQ(node, 0);
    ::getMessage(*msg,ConfigItems);
    ::getMessage(*msg,NodeNameItems);
    delete msg;
  } else {
    // only node 0 reads config file - others get a broadcast message
    // open the configuration file
    if ((inC = fopen(ConfigFile.c_str(), "r")) != 0) {
      // read in each line, and append it to end of broadcast string
      while (fgets(bufferstore, bufferSize, inC) != 0) {
	// skip leading spaces, and any comment lines starting with '#'
	buffer = bufferstore;
	while (*buffer == ' ' || *buffer == '\t' || *buffer == '\n')
	  ++buffer;
	if (*buffer == '#' || *buffer == '\0')
	  continue;
	ConfigItems += buffer;
	ConfigItems += "\n";
      }
      fclose(inC);
    }

    // see if there was an error, or no config file was specified ...
    // if so, use default
    if (ConfigItems.length() == 0) {
      ConfigItems = "* .";
      ConfigItems += "\n";
    }

    // collect node names from everyone else, and then retransmit the collected
    // list.  The first name should be the node 0 name.
    NodeNameItems += " 0";
    int unreceived = Ippl::getNodes() - 1;
    while (unreceived-- > 0) {
      // get the hostname from the remote node, and append to a list
      int node = COMM_ANY_NODE;
      Message *msg = Ippl::Comm->receive_block(node, tag);
      PAssert(msg);
      string nodename;
      ::getMessage(*msg,nodename);
      sprintf(name, " %s %d", nodename.c_str(), node);
      NodeNameItems += name;
      delete msg;
    }

    // broadcast string to all other nodes
    if (Ippl::getNodes() > 1) {
      Message *msg = new Message;
      ::putMessage(*msg,ConfigItems);
      ::putMessage(*msg,NodeNameItems);
      Ippl::Comm->broadcast_others(msg, tag);
    }
  }

  // from the configuration string, break it up into single lines and parse.
  // This sets up the SMP information list.
  string *conflines;
  int conflinenum = dc_tokenize_string(ConfigItems.c_str(), "\n", conflines);
  for (int is=0; is < conflinenum; ++is) {

    // tokenize string, and store values
    string *tokens;
    int ntok = dc_tokenize_string(conflines[is].c_str(), " \t,\n", tokens);
    if (ntok != 2) {
      ERRORMSG("Wrong number of parameters in DiscConfig config file ");
      ERRORMSG("'" << ConfigFile << "' (" << ntok << " != 2)" << endl);
      Ippl::abort("Exiting due to DiscConfig error.");
    } else {
      // append / to directory name if necessary, and also the base filename
      if (tokens[1].c_str()[tokens[1].length() - 1] != '/')
	tokens[1] += "/";
      tokens[1] += BaseFile;
      if (tokens[0] == "*") {
	// save the wildcard string
	WildCard = tokens[1];
      } else {
	// line was good ... store the values found there.  If a line is
	// repeated, we just replace the value.
        SMPData *smpd = 0;
        vmap<string,SMPData *>::iterator smpiter = SMPMap.find(tokens[0]);
        if (smpiter != SMPMap.end())
          smpd = (*smpiter).second;
	add_SMP_directory(smpd, tokens[1], tokens[0], WritingFile);
      }
    }

    // delete the tokens
    if (tokens != 0)
      delete [] tokens;
  }

  // delete the conf lines
  if (conflines != 0)
    delete [] conflines;

  // make sure we found SOMETHING ...
  if (SMPMap.size() < 1 && WildCard.length() == 0) {
    ERRORMSG("No hostname/directory pairs found in DiscConfig config file ");
    ERRORMSG("'" << ConfigFile << "' " << endl);
    Ippl::abort("Exiting due to DiscConfig error.");
    return false;
  }

  // set up the node information list
  string *nodenames;
  dc_tokenize_string(NodeNameItems.c_str(), " ", nodenames);
  for (int in=0; in < Ippl::getNodes(); ++in) {
    // get node number and node name from list of node information
    int node = atoi(nodenames[2*in + 1].c_str());
    string machine = nodenames[2*in];

    // find the host name in our list of SMP's
    SMPData *smpdata = 0;
    vmap<string,SMPData *>::iterator smpiter = SMPMap.find(machine);
    if (smpiter != SMPMap.end()) {
      // this SMP has already been set up earlier
      smpdata = (*smpiter).second;
    } else {
      // we must make a new info structure for this SMP, since it was
      // not mentioned in the configuration file.  The routine
      // sets the value of smpdata to a newly allocated pointer
      add_SMP_directory(smpdata, WildCard, machine, WritingFile);
    }

    // fill in the SMP info and node info
    NodeList[node]->HostName = machine;
    smpdata->NodeList.push_back(node);
  }

  // delete the node names
  if (nodenames != 0)
    delete [] nodenames;

  // go through the SMP list, assign them numbers, and sort the node data
  int firstSMPWithFiles = (-1);
  vmap<string,SMPData *>::iterator smpa;
  for (smpa = SMPMap.begin() ; smpa != SMPMap.end(); ++smpa) {
    // add this SMP info to our SMPList array (for fast access)
    SMPData *smpdata   = (*smpa).second;
    smpdata->SMPIndex  = NumSMPs++;
    SMPList.push_back(smpdata);

    // find out if this SMP is the first one we find with files.  There
    // must be at least one, since by this point we know the config
    // file was not empty, or we used a wildcard.
    if (firstSMPWithFiles < 0 && smpdata->BaseFileNum > 0)
      firstSMPWithFiles = smpdata->SMPIndex;

    // sort the list of nodes so that all nodes have them in the same order
    // (this is needed in find_processors).  But if an SMP has 0 nodes,
    // it is an error since the configuration file lists an SMP on which
    // we are not running.
    if (smpdata->NodeList.size() > 0) {
      sort(smpdata->NodeList.begin(), smpdata->NodeList.end());
      smpdata->Box0Node = smpdata->NodeList[0];
    } else {
      ERRORMSG("DiscConfig: The SMP '" << smpdata->HostName);
      ERRORMSG("' was listed in the config file\n");
      ERRORMSG("'" << ConfigFile << "' but you are not running on that SMP.");
      ERRORMSG(endl);
      Ippl::abort("Exiting due to DiscConfig error.");
    }

    // tell the proper nodes which SMP they're on
    vector<int>::iterator nodea = smpdata->NodeList.begin();
    for ( ; nodea != smpdata->NodeList.end(); ++nodea)
      NodeList[*nodea]->SMPIndex = smpdata->SMPIndex;


    // increment how many file sets we're dealing with
    FileSMPs += smpdata->BaseFileNum;
  }

  // determine our parent SMP
  MySMP = NodeList[Ippl::myNode()]->SMPIndex;

  // determine Box0 nodes, and whether we need to make sure to send
  // layout and other info to other SMP's
  for (smpa = SMPMap.begin() ; smpa != SMPMap.end(); ++smpa) {
    SMPData *smpdata = (*smpa).second;
    if (smpdata->BaseFileNum == 0) {
      // we'll need to send data to another SMP's Node 0
      smpdata->Box0Node = SMPList[firstSMPWithFiles]->Box0Node;
      SMPList[firstSMPWithFiles]->InformSMPList.push_back(smpdata->SMPIndex);
    }
  }

  return true;
}


///////////////////////////////////////////////////////////////////////////
// print out debugging information for this DiscConfig
void DiscConfig::printDebug(Inform& msg) {
  msg << "ConfigFile   = " << getConfigFile() << endl;
  msg << "Num Filesets = " << fileSMPs() << endl;
  msg << "NumSMPs = " << numSMPs() << endl;
  msg << "MySMP   = " << mySMP() << " (" << getSMPHost(mySMP()) << ")" << endl;
  msg << "MyBox0  = " << getSMPBox0() << endl;
  msg << "MyNode  = " << Ippl::myNode() << endl;

  // print out summary of SMP info
  msg << "DiscConfig SMP Summary:" << endl;
  for (unsigned int smp=0; smp < numSMPs(); ++smp) {
    msg << "  SMP host=" << getSMPHost(smp);
    msg << ", numnodes=" << getNumSMPNodes(smp);
    msg << ", box0=" << getSMPBox0(smp) << endl;
    msg << "    FileList =";
    for (unsigned int fl=0; fl < getNumFiles(smp); ++fl)
      msg << " " << getFilename(smp, fl);
    msg << endl;
    msg << "    OtherSMPList =";
    for (unsigned int sl=0; sl < getNumOtherSMP(smp); ++sl)
      msg << " " << getOtherSMP(smp, sl);
    msg << endl;
    msg << "    NodeList =";
    for (unsigned int nl=0; nl < getNumSMPNodes(smp); ++nl)
      msg << " " << getSMPNode(smp,nl);
    msg << endl;
  }

  // print out summary of node info
  msg << "DiscConfig Node Summary:" << endl;
  for (unsigned int n=0; n < getNumNodes(); ++n) {
    msg << "  Node " << n << " on SMP " << getNodeSMPIndex(n);
    msg << " (" << getNodeHost(n) << ")" << endl;
  }
}


/***************************************************************************
 * $RCSfile: DiscConfig.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscConfig.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
