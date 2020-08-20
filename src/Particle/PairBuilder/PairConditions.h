#ifndef PAIR_CONDITIONS_H
#define PAIR_CONDITIONS_H

#include "Region/NDRegion.h"

#include <limits>

template<class T, unsigned Dim>
class TrueCondition
{
public:
    TrueCondition() { }
    template<class V>
    bool operator()(const V &/*a*/, const V &/*b*/) const
    {
        return true;
    }

    T getRange(unsigned) const { return std::numeric_limits<T>::max(); }
};

template<class T, unsigned Dim>
class RadiusCondition
{
public:
    RadiusCondition(T r) : sqradius(r*r), radius(r)
    {  }

    template<class V>
    bool operator()(const V &a, const V &b) const
    {
        T sqr = 0;
        for(unsigned int d = 0;d<Dim;++d)
        {
            sqr += (a[d]-b[d])*(a[d]-b[d]);
        }
        return sqr <= sqradius;
    }

    //periodic version of radius condition
    template<class V, class Vec>
    bool operator()(const V &a, const V &b, const Vec &/*period*/) const
    {
        T sqr = 0;
        //std::cout << "checking radius condition for " << a << " and " << b << std::endl;
        for(unsigned int d = 0;d<Dim;++d)
        {
            //sqr += (std::fmod((a[d]-b[d]+period[d]),period[d])*std::fmod((a[d]-b[d]+period[d]),period[d]));
            sqr += (a[d]-b[d])*(a[d]-b[d]);
        }
        /*
         if (sqr > sqradius)
         std::cout << "rejected with dist = " << sqrt(sqr) << std::endl;
         else
         std::cout << "accepted with dist = " << sqrt(sqr) << std::endl;
         */
        return sqr <= sqradius;
    }


    T getRange(unsigned /*d*/) const { return radius; }
private:
    T sqradius, radius;
};

template<class T, unsigned Dim>
class BoxCondition
{
public:
    BoxCondition(T b[Dim]) : box(b)
    {  }

    template<class V>
    bool operator()(const V &a, const V &b) const
    {
        for(unsigned int d = 0;d<Dim;++d)
        {
            T diff = a[d]-b[d];
            if(diff > box[d] || diff < -box[d])
                return false;
        }
        return true;
    }

    T getRange(unsigned d) const { return box[d]; }
private:
    T box[Dim];
};

#endif
