// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef FORMATTER_H
#define FORMATTER_H

#include "Message/Message.h"
#include <cstring>
#include <vector>
#include <algorithm>

/*
 * Format and MsgBuffer class to allow serializing message objects into plain buffers
 * to send directly with mpi calls or similar means
 */ 

class Format
{
public:
    Format(Message*);
    unsigned int getItemCount()
    {
        return items;
    }
    unsigned int getSize()
    {
        return size;
    }
    unsigned int getFormatSize()
    {
        return 2*items*sizeof(int);
    }
    unsigned int getItemElems(int i)
    {
        return format_array[2*i+0];
    }
    unsigned int getItemBytes(int i)
    {
        return format_array[2*i+1];
    }

    void print();

private:
    unsigned int items, size;
    std::vector<unsigned int> format_array;
};


class MsgBuffer
{
public:
    //creates buffer with space to hold count messages of format f
    MsgBuffer(Format *f, int count, int offset = 0);
    MsgBuffer(Format *f, char* d, int size);

    bool add(Message*);
    Message* get();

	template<class T>
	void get(T &v)
	{
		std::memcpy(&v, data.data()+readpos, sizeof(T));
		readpos += sizeof(T);
	}
	
	template<class T>
	void put(T &v)
	{
		std::memcpy(data.data()+writepos, &v, sizeof(T));
		writepos += sizeof(T);
	}

    int getSize()
    {
        return writepos;
    }
    void* getBuffer()
    {
        return data.data();
    }
    
    Format* getFormat() { return format; }

    ~MsgBuffer();
private:
    Format *format;
    unsigned int datasize, writepos, readpos;
    std::vector<char> data;
};

#endif // FORMATTER_H
