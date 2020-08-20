// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef INFORM_H
#define INFORM_H

/*
 * Inform - takes messages and displays them to the given ostream.
 *	A message is sent to an Inform object by treating it as an ostream,
 *	then ending the message by sending the 'inform' manipulator.  In
 *      fact, Inform works much like an ostream, although it may actually
 *      just use stdio for I/O.
 *
 *	Each message is assigned the current 'level of interest'; the lower
 *	the level, the more important it is.  Each Inform object is also
 *      set for a current level; messages with a level <= the current level
 *	are displayed.  Levels run from 1 ... 5.  Thus, setting the level of
 *      Inform object to 0 will turn off printing of all messages.
 *
 *      By default, a new Inform object will only print out the message on
 *      node 0.  You may change the node on which this prints with the
 *      'setPrintNode(int)' method; if the argument is 'INFORM_ALL_NODES',
 *      the message will be printed on ALL nodes, not just one.  The final
 *      argument to the constructor may also be set to the node to print on.
 */

#include <iostream>
#include <iomanip>
#include <sstream>

#define INFORM_ALL_NODES        (-1)


class Inform {

public:
  // enumeration listing the ways in which a file may be opened for writing
  enum WriteMode { OVERWRITE, APPEND };

public:
  // constructor: arguments = name, print node
  Inform(const char * = 0, int = 0);

  // second constructor: this specifies the name of a file as well as
  // a prefix and a mode for opening the file (i.e. OVERWRITE or APPEND).
  // The final argument is the print node.
  Inform(const char *prefix, const char *fname, const WriteMode, int = 0);

  // third constructor: this specifies the prefix and an ostream object
  // to write to, as well as as the print node
  Inform(const char *, std::ostream&, int = 0);

  // fourth constructor: this specifies the prefix and an Inform instance
  // from which the ostream object is copied, as well as as the print node
  Inform(const char *myname, const Inform& os, int pnode = 0);

  // destructor
  ~Inform();

  // turn messages on/off
  void on(const bool o) { On = o; }
  bool isOn() const { return On; }

  // change output destination
  void setDestination(std::ostream &dest);
  std::ostream& getDestination() { return *MsgDest; }

  // get/set the current output level
  Inform& setOutputLevel(const int);
  int getOutputLevel(void) const { return OutputLevel; }

  // get/set the current message level
  Inform& setMessageLevel(const int);
  int getMessageLevel(void) const { return MsgLevel; }

  // get/set the printing node.  If set to a value < 0, all nodes print.
  int getPrintNode() const { return PrintNode; }
  void setPrintNode(int n = (-1)) { PrintNode = n; }

  // return a reference to the internal ostream used to print messages
  std::ostream& getStream() { return FormatBuf; }

  // Was the stream opened successfully on construction?
  bool openedSuccessfully() { return OpenedSuccessfully; }

  // the signal has been given, print out the message.  Return ref to object.
  Inform& outputMessage(void);

  // functions used to change format state; used just as for iostreams

  typedef std::ios_base::fmtflags FmtFlags_t;

  FmtFlags_t setf(FmtFlags_t setbits, FmtFlags_t field)
  { return FormatBuf.setf(setbits,field); }

  FmtFlags_t setf(FmtFlags_t f) { return FormatBuf.setf(f); }
  void /*long*/ unsetf(FmtFlags_t f) { FormatBuf.unsetf(f); }
  FmtFlags_t flags() const { return FormatBuf.flags(); }
  FmtFlags_t flags(FmtFlags_t f) { return FormatBuf.flags(f); }
  int width() const { return FormatBuf.width(); }
  int width(int w) { return FormatBuf.width(w); }
  char fill() const { return FormatBuf.fill(); }
  char fill(char c) { return FormatBuf.fill(c); }
  int precision() const { return FormatBuf.precision(); }
  int precision(int p) { return FormatBuf.precision(p); }
  void flush() { MsgDest->flush();}
private:
  // name of this object; put at the start of each message.
  char *Name;

  // storage for the message text
  std::string MsgBuf;
  // an ostringstream used to format the messages
  std::ostringstream FormatBuf;

  // where to put the messages; can be changed, by default = cout
  std::ostream *MsgDest;

  // do we need to close the destination stream?
  bool NeedClose;

  // Was the stream opened successfully on construction?
  bool OpenedSuccessfully;

  // do we output the message?
  bool On;

  // limit printing only to this node (if < 0, all nodes print)
  int PrintNode;

  // output level of this Inform object; messages with a level <= the output
  // level are printed.  Setting this to < 1 turns off messages.
  int OutputLevel;

  // current message level; this is set by the 'levelN' manipulators, or
  // by the routine setMsgLevel(int).  After a message is printed, the current
  // message level is reset to the minimum.
  int MsgLevel;

  // print out the message in the given buffer.  Will modify the string,
  // so beware.  Arguments: string
  void display_message(char *);

  // print out just a single line of the message.
  void display_single_line(char *);

  // perform initialization for this object; called by the constructors.
  // arguments = prefix string, print node
  void setup(const char *, int);
};


// manipulator for signaling we want to send the message.
extern Inform& endl(Inform&);

// manipulators for setting the current msg level
extern Inform& level1(Inform&);
extern Inform& level2(Inform&);
extern Inform& level3(Inform&);
extern Inform& level4(Inform&);
extern Inform& level5(Inform&);


// templated version of operator<< for Inform objects
template<class T>
inline
Inform& operator<<(Inform& o, const T& val) {
  o.getStream() << val;
  return o;
}


// specialized version of operator<< to handle Inform-specific manipulators
inline
Inform& operator<<(Inform& o, Inform& (*d)(Inform&)) {
  return d(o);
}


// specialized version of operator<< to handle void * arguments
inline
Inform& operator<<(Inform& o, const void *val) {
  Inform::FmtFlags_t oldformat = o.setf(std::ios::hex, std::ios::basefield);
  o.getStream() << "0x" << (long)val;
  o.setf(oldformat, std::ios::basefield);
  return o;
}

// specialized version of operator<< to handle long long type (KCC workaround)
inline
Inform& operator<<(Inform& o, const long long& val) {
  o.getStream() << val;
  return o;
}

// specialized function for sending strings to Inform object
inline Inform& operator<<(Inform& out, const std::string& s) {
  out << s.c_str();
  return out;
}


#endif // INFORM_H

/***************************************************************************
 * $RCSfile: Inform.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: Inform.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $
 ***************************************************************************/
