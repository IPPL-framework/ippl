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
#include "Utility/DiscMeta.h"
#include "Utility/DiscConfig.h"
#include "Utility/PAssert.h"
#include "Utility/IpplInfo.h"
#include "Utility/Inform.h"


///////////////////////////////////////////////////////////////////////////
// Constructor: read and parse the given meta file
DiscMeta::DiscMeta(const char *fname) {



  if (fname != 0)
    MetaFile = fname;

  // open the input file
  FILE *f = fopen(fname, "r");

  // make sure it was OK ...
  if (f == 0) {
    ERRORMSG("DiscMeta: Could not open file '" << fname << "' on node ");
    ERRORMSG(Ippl::myNode() << "." << endl);
    Ippl::abort("Exiting due to DiscConfig error.");
  } else {
    // ... it was, parse it's contents, adding new valid lines to Lines list
    int numtokens = 0;
    std::string *words = 0;
    int linenum = 0;
    while (read_meta_line(f, words, numtokens)) {
      // indicate we've read another line
      ++linenum;

      // if the line actually contains some keywords and values, add it
      if (words != 0 && numtokens > 0) {
	Lines.insert(value_type(linenum, element_t(numtokens, words)));
      }
    }

    // close the file
    fclose(f);

  }
}


///////////////////////////////////////////////////////////////////////////
// Destructor
DiscMeta::~DiscMeta() {
  // delete all the lists of strings
  for (iterator a = begin(); a != end(); ++a) {
    if ((*a).second.second != 0)
      delete [] ((*a).second.second);
  }
}


///////////////////////////////////////////////////////////////////////////
// return the line number of the Nth valid line (from 1 ... M)
int DiscMeta::getLineNumber(unsigned int n) const {


  PAssert_LT(n, size());

  unsigned int i=0;
  const_iterator iter = begin();
  while (i != n && iter != end()) {
    ++iter;
    ++i;
  }

  PAssert(iter != end());
  return (*iter).first;
}


///////////////////////////////////////////////////////////////////////////
// return the keyword of the Nth line
const std::string &DiscMeta::getKeyword(unsigned int n) {



  return getWords(n)[0];
}


///////////////////////////////////////////////////////////////////////////
// return the number of words in the value for the Nth line
int DiscMeta::getNumWords(unsigned int n) const {


  PAssert_LT(n, size());

  unsigned int i=0;
  const_iterator iter = begin();
  while (i != n && iter != end()) {
    ++iter;
    ++i;
  }

  PAssert(iter != end());
  return (*iter).second.first;
}


///////////////////////////////////////////////////////////////////////////
// return the list of words in the Nth line
std::string *DiscMeta::getWords(unsigned int n) {



  unsigned int i=0;
  iterator iter = begin();
  while (i != n && iter != end()) {
    ++iter;
    ++i;
  }

  PAssert(iter != end());
  return (*iter).second.second;
}


///////////////////////////////////////////////////////////////////////////
// read in a single line from the meta data file and parse it.
// return success of operation.
bool DiscMeta::read_meta_line(FILE *f, std::string *&tokens, int &numtokens) {



  const int bufferSize = 1024*128;
  char bufferstore[bufferSize];
  char *buffer;

  // read next line
  tokens = 0;
  numtokens = 0;
  if (fgets(bufferstore, bufferSize, f) == 0)
    return false;
  unsigned int len = strlen(bufferstore);
  if (len > 0 && bufferstore[len-1] == '\n')
    bufferstore[len - 1] = '\0';

  // skip whitespace, and if a comment line, all the text
  buffer = bufferstore;
  while (*buffer == ' ' || *buffer == '\t')
    buffer++;
  if (*buffer == '#' || *buffer == '\n' || *buffer == '\0')
    return true;

  // not a comment or blank line ... tokenize it and return
  numtokens = DiscConfig::dc_tokenize_string(buffer, " =\t", tokens);
  return true;
}


///////////////////////////////////////////////////////////////////////////
// print out debugging information for this DiscMeta
void DiscMeta::printDebug(Inform &msg) {



  msg << "Meta file name = " << MetaFile << endl;
  msg << "Lines in file = " << size() << endl;
  for (unsigned int i=0; i < size(); ++i) {
    msg << "  Line " << i << ": '" << getKeyword(i) << "' =";
    for (int j=1; j < getNumWords(i); ++j) {
      msg << "  '" << getWords(i)[j] << "'";
    }
    msg << endl;
  }
}


/***************************************************************************
 * $RCSfile: DiscMeta.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscMeta.cpp,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/