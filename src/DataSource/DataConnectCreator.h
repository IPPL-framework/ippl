// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef DATA_CONNECT_CREATOR_H
#define DATA_CONNECT_CREATOR_H

/***********************************************************************
 * 
 * DataConnectCreator is a factory class which is used to create specific
 * subclasses of the base class DataConnect.  To create a new, agency-
 * specific DataConnect, the user can either instantiate such a class
 * directly, or they can call the static method
 *   DataConnectCreator::create(DataConnectCreator::CreateMethod, char *name,
 *                              DataConnectCreator::TransferMethod)
 * There is also a static method 'create' which does not take a CreateMethod
 * argument, instead it will use a 'default' value.  In the IPPL Framework,
 * this default value is set by the IpplInfo object based on the command
 * line arguments.
 *
 ***********************************************************************/


// forward declarations
class DataConnect;


// class definition
class DataConnectCreator {

public:
  // constructor and destructor
  DataConnectCreator();
  ~DataConnectCreator();

  //
  // informative methods
  //

  // return the number of connection methods available
  static int getNumMethods();

  // return the name of the Nth method
  static const char *getMethodName(int);

  // return a list of all the methods, as a single string
  static const char *getAllMethodNames();

  // return the current 'default' connection method
  static int getDefaultMethod() {
    return DefaultMethod;
  }

  // return the current 'default' connection method name
  static const char *getDefaultMethodName() {
    return getMethodName(DefaultMethod);
  }

  // is the given method available here?
  static bool supported(int);
  static bool supported(const char *nm);

  // is the given method name one we recognize?
  static bool known(int);
  static bool known(const char *nm);

  //
  // DataConnect create methods
  //

  // create a new connection.  Arguments = type, name, nodes
  static DataConnect *create(int, const char *, int = 0);

  // create a new connection, by giving the method name (nm1), the name
  // for the new connection (nm2), and the number of nodes
  static DataConnect *create(const char *nm1, const char *nm2, int n = 0) {
    return create(libindex(nm1), nm2, n);
  }

  // also create a new connection, but use the 'default' connection method.
  static DataConnect *create(const char *nm, int n = 0) {
    return create(DefaultMethod, nm, n);
  }

  // a final method for creating objects; this one provides a default name,
  // and if the default connection object has already been created, this just
  // returns that one.
  static DataConnect *create();

  //
  // DataConnectCreator manipulation methods
  //

  // change the default connection method.  Return success.
  static bool setDefaultMethod(int);

  // change the default connection method by specifying a name.  Return
  // success.
  static bool setDefaultMethod(const char *nm) {
    return setDefaultMethod(libindex(nm));
  }

  // change the default number of nodes to use for the connection
  static void setDefaultNodes(int);

  // return the default number of nodes to use in a connection
  static int getDefaultNodes();

private:
  // default connection method
  static int DefaultMethod;

  // default connection, if it has been created yet
  static DataConnect *DefaultConnection;

  // how many instances of this class exist?
  static int InstanceCount;

  // default number of nodes to use in the connection ... some connections
  // may not care about this
  static int ConnectNodes;

  // return the index of the given named method, or (-1) if not found
  static int libindex(const char *);
};

#endif // DATA_CONNECT_CREATOR_H

/***************************************************************************
 * $RCSfile: DataConnectCreator.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:24 $
 * IPPL_VERSION_ID: $Id: DataConnectCreator.h,v 1.1.1.1 2003/01/23 07:40:24 adelmann Exp $ 
 ***************************************************************************/
