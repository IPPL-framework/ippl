#include "Utility/IpplMessageCounter.h"

#include "PETE/IpplExpressions.h"
   
IpplMessageCounter::IpplMessageCounter() : ison(true)
{

}     
     
IpplMessageCounterRegion* IpplMessageCounter::getActiveRegion()
{ 
    if(activeRegion>=0 && (unsigned int) activeRegion<counterRegions.size())
        return counterRegions[activeRegion];
    else
        return 0;
}

void IpplMessageCounter::setActiveRegion(int ar)
{
	activeRegion = ar;
}

void IpplMessageCounter::unsetActiveRegion()
{
	activeRegion = -1;
}

int IpplMessageCounter::addRegion(IpplMessageCounterRegion *mcr)
{
	counterRegions.push_back(mcr); return counterRegions.size()-1;
}

void IpplMessageCounter::registerMessage(int size)
{
	if(ison && getActiveRegion())
		getActiveRegion()->registerMessage(size);
}

IpplMessageCounterRegion::IpplMessageCounterRegion(const std::string &n)
 : name(n), count(0), total_size(0)
{
	index = IpplMessageCounter::getInstance().addRegion(this);
}

void IpplMessageCounterRegion::begin()
{
	IpplMessageCounter::getInstance().setActiveRegion(index);
}

void IpplMessageCounterRegion::end()
{
	IpplMessageCounter::getInstance().unsetActiveRegion();
}

void IpplMessageCounterRegion::registerMessage(int size)
{
	count++;
	total_size += size;
	//std::cout << "node " << Ippl::myNode() << " sent message of size " << size << std::endl;
}


void IpplMessageCounter::print()
{
	Inform msg("MsgCounter");
	msg << "Message Counts------------------------------------------------\n";
	for(unsigned int i=0;i<counterRegions.size();++i)
	{
		counterRegions[i]->print(msg);
	}
	msg << "--------------------------------------------------------------\n";
}

void IpplMessageCounterRegion::print(Inform &msg)
{
	unsigned total_count=0, total_total_size=0;
	reduce(count, total_count, OpAddAssign());
	reduce(total_size, total_total_size, OpAddAssign());
	msg << name << " count = " << total_count << " size = "
		<< total_total_size/(1024.*1024.) << " Mb" << endl;
}
