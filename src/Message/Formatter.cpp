#include "Message/Formatter.h"

Format::Format(Message *msg)
{
    items = msg->size();
    size = 0;
    
    format_array.resize(2*items);
    for (unsigned int i=0; i<items; ++i)
    {
        Message::MsgItem &msgitem = msg->item(i);
        format_array[2*i+0] = msgitem.numElems();
        format_array[2*i+1] = msgitem.numBytes();
        size += format_array[2*i+1];
    }
}

void Format::print()
{
	std::cout << "size: " << size << std::endl;
	for (unsigned int i=0; i<items; ++i)
    {
		std::cout << "entry " << i << ": " << format_array[2*i+0]
			<< " elements " << format_array[2*i+1] << " bytes\n";
    }
}

MsgBuffer::MsgBuffer(Format *f, int count, int offset)
        : format(f), writepos(0), readpos(0)
{
    datasize = count*format->getSize();
    data.resize(datasize+offset);
}


MsgBuffer::MsgBuffer(Format *f, char *buf, int size)
        : format(f), writepos(0), readpos(0)
{
    datasize = size;
    data.resize(datasize);
    std::copy(buf, buf+size, data.begin());
    delete[] buf;
}

MsgBuffer::~MsgBuffer()
{
}

bool MsgBuffer::add(Message *msg)
{
    unsigned int items = msg->size();

    //check for full storage or message size mismatch
    if (writepos == datasize || items != format->getItemCount())
        return false;

    int pos = writepos;
    for (unsigned int i=0; i<items; ++i)
    {
        Message::MsgItem &msgitem = msg->item(i);

        //check for format mismatch
        if (format->getItemElems(i) != msgitem.numElems() ||
                format->getItemBytes(i) != msgitem.numBytes())
            return false;

        //actually copy to buffer
        std::memcpy(data.data()+pos, msgitem.data(), format->getItemBytes(i));
        pos += format->getItemBytes(i);
    }

    writepos = pos;
    return true;
}

Message* MsgBuffer::get()
{
    if (readpos > datasize - format->getSize())
        return 0;

    unsigned int items = format->getItemCount();
    Message *msg = new Message(items);

    //get all the items according to format and add them to the message
    for (unsigned int j = 0; j < items; j++)
    {
        unsigned int bytesize = format->getItemBytes(j);
        unsigned int elements = format->getItemElems(j);

        msg->setCopy(false);
        msg->setDelete(false);
        msg->putmsg(data.data()+readpos, bytesize/elements, elements);
        readpos += bytesize;
    }

    return msg;
}
