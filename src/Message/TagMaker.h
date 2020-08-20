/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef TAG_MAKER_H
#define TAG_MAKER_H

/*
 * TagMaker.h - creates tags from a given base tag and a cycle size.  New
 *	tags are generated each time one is requested, by adding an
 *	integer which varies from 0 ... (cycle size - 1) to the provided
 *	base tag.  Routines exist to establish a base tag and cycle size,
 *	and to get a new tag for a given base tag.
 */


// include files
#include <map>


// default cycle size, if not specified by the user
#define DEF_CYCLE_SIZE 1000


class TagMaker
{

public:
    // constructor/destructor
    TagMaker(void) { }
    virtual ~TagMaker(void) { }

    // generate a new tag given a base tag.  If the base tag has not been
    // previously established by create_base_tag, it will be done so by
    // this routine with the default cycle size.  A new tag can be established
    // at the same time by also giving a cycle size as the second argument.
    int next_tag(int t, int s = DEF_CYCLE_SIZE)
    {
        TagInfo& found = create_base_tag(t, s);
        found.current = (found.current + 1) % found.cycleSize;
        return (found.base + found.current);
    }

    // just return the `current' tag that is to be generated from the
    // given base tag, without incrementing the cycle counter.
    int current_tag(int t, int s = DEF_CYCLE_SIZE)
    {
        TagInfo& found = create_base_tag(t, s);
        return (found.base + found.current);
    }

    // reset the cycle counter for the given tag to be 0.  If the tag is
    // not in the list, it is added.  Returns the reset tag.
    int reset_tag(int t, int s = DEF_CYCLE_SIZE)
    {
        TagInfo& found = create_base_tag(t, s);
        found.current = 0;
        return found.base;
    }

private:
    // Simple struct holding info about the cycle size and current tag
    // for a base tag
    class TagInfo
    {
    public:
        int base;			// base tag value, the key for the map
        int cycleSize;		// range through which to cycle tag
        int current;		// current value of tag
        TagInfo(int b, int s) : base(b), cycleSize(s), current(0) { }
        TagInfo() : base(-1), cycleSize(-1), current(0) { }
    };

    // class used for comparisons
    class TagCompare
    {
    public:
        bool operator()(const int& x, const int& y) const
        {
            return x < y;
        }
    };

    // the list of base tags which have been established
    std::map<int, TagInfo, TagCompare> TagList;

    // Establish a new base tag and cycle size.  Returns a reference to
    // the new TagInfo structure.
    // Arguments are: base tag, cycle size.
    TagInfo& create_base_tag(int t, int s = DEF_CYCLE_SIZE)
    {
        TagInfo& found = TagList[t];
        if ( found.base < 0 )
        {
            found.base = t;
            found.cycleSize = s;
        }
        return TagList[t];
    }

};

#endif // TAG_MAKER_H

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
