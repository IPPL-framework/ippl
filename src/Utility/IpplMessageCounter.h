// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_MESSAGE_COUNTER_H
#define IPPL_MESSAGE_COUNTER_H

/*************************************************************************
 * IpplMessageCounter counts messages sent inside a region
 *************************************************************************/
#include "Message/GlobalComm.h"
#include "Message/Message.h"
#include "Utility/Inform.h"

#include <vector>
#include <string>

class IpplMessageCounterRegion {
public:
	IpplMessageCounterRegion(const std::string&);
	void begin();
	void end();
	void registerMessage(int);
	void print(Inform&);
private:
	std::string name;
	unsigned count;
	unsigned total_size;
	int index;
};

class IpplMessageCounter
{
public:
	static IpplMessageCounter& getInstance()
    {
		static IpplMessageCounter instance;
		return instance;
    }
        
	IpplMessageCounterRegion* getActiveRegion();
	
	void setActiveRegion(int);
	void unsetActiveRegion();
	
	int addRegion(IpplMessageCounterRegion*);
	
	void registerMessage(int);
	void on() { ison = true; }
	void off() { ison = false; }
	
	void print();
private:
    IpplMessageCounter();
    IpplMessageCounter(IpplMessageCounter const& copy);
    IpplMessageCounter& operator=(IpplMessageCounter const& copy);

        
    int activeRegion;
    std::vector<IpplMessageCounterRegion*> counterRegions;
    bool ison;
};

#endif

