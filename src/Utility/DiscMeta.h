// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DISC_META_H
#define DISC_META_H

// include files
#include "Utility/Inform.h"
#include "Utility/vmap.h"

#include <cstdio>

/***************************************************************************
 * DiscMeta - reads in information in a Disc meta file, which is a file in
 * the format
 *
 * # comment line
 * keyword [=] value
 * keyword [=] value
 * ...
 *
 * An optional = may separate the single-word keyword and the value.  This
 * class stores the list of lines, and can be queried about them.
 *
 ***************************************************************************/


class DiscMeta {

public:
  // useful typedefs
  typedef std::pair<int, std::string *>    element_t;
  typedef vmap<int, element_t>        container_t;
  typedef container_t::iterator       iterator;
  typedef container_t::const_iterator const_iterator;
  typedef container_t::value_type     value_type;

public:
  // Constructor: specify the meta file filename.  The constructor
  // will parse the contents and store the results.
  DiscMeta(const char *fname);

  // Destructor.
  ~DiscMeta();

  //
  // accessor functions
  //

  // return the config filename
  const std::string &getFilename() const { return MetaFile; }

  // return the number of valid lines in the file
  unsigned int size() const { return Lines.size(); }

  // return the line number of the Nth valid line (in range 1 ... M)
  int getLineNumber(unsigned int) const;

  // return the keyword of the Nth line
  const std::string &getKeyword(unsigned int);

  // return the number of words in the value for the Nth line
  int getNumWords(unsigned int) const;

  // return the list of words in the Nth line
  std::string *getWords(unsigned int);

  // return begin/end iterators for iterating over the list of lines
  iterator begin() { return Lines.begin(); }
  iterator end() { return Lines.end(); }
  const_iterator begin() const { return Lines.begin(); }
  const_iterator end() const { return Lines.end(); }

  // print out debugging information
  void printDebug(Inform &);

private:
  // the name of the meta file
  std::string MetaFile;

  // the data for the lines themselves
  container_t Lines;

  // the storage for the keywords
  // read a single meta line, and return an array of strings as tokens
  bool read_meta_line(FILE*, std::string *&, int &);
};

#endif // DISC_META_H

/***************************************************************************
 * $RCSfile: DiscMeta.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: DiscMeta.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
