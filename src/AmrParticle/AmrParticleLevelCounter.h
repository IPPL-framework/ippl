//
// Class AmrParticleLevelCounter
//   Helper class in order to keep track of particles
//   per level. It allows to iterate faster through
//   particles at a certain level.
//   The class is built on the STL map container where
//   the key represents the level and the value is the
//   the number of particles at that level.
//
// Copyright (c) 2016 - 2020, Matthias Frey Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// Implemented as part of the PhD thesis
// "Precise Simulations of Multibunches in High Intensity Cyclotrons"
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef AMR_PARTICLE_LEVEL_COUNTER_H
#define AMR_PARTICLE_LEVEL_COUNTER_H

#include <map>
#include <numeric>
#include <functional>
#include <iterator>

template <
    class Key,
    class T,
    class Compare = std::less<Key>,
    class Allocator = std::allocator<std::pair<const Key, T> >
> class AmrParticleLevelCounter
{
    
public:
    typedef typename std::map<Key, T>::value_type value_type;
    typedef typename std::map<Key, T>::size_type size_type;
    typedef typename std::map<Key, T>::iterator iterator;
    typedef typename std::map<Key, T>::const_iterator const_iterator;

public:
    
    AmrParticleLevelCounter() : count_m() { }
    
    /*!
     * Add more "particles" to that level
     * @param level where to add
     * @param nTimes to increment
     */
    void increment(const Key& level, T nTimes = T(1)) { count_m[level] += nTimes; }
    
    /*!
     * Add more "particles" to that level
     * @param level where to add
     * @param nTimes to decrement
     */
    void decrement(const Key& level, T nTimes = T(1)) { increment(level, -nTimes); }
    
    T& operator[](T level) { return count_m[level]; }
    
    const T& operator[](T level) const { return count_m[level]; }
    
    size_type size() const { return count_m.size(); }
    
    bool empty() const { return count_m.empty(); }
    
    iterator begin() { return count_m.begin(); }
    const_iterator begin() const { return count_m.begin(); }
    
    iterator end() { return count_m.end(); }
    const_iterator end() const { return count_m.end(); }
    
    
    /*!
     * Obtain the start of a level
     * @param level
     * @returns the local starting index
     */
    T begin(T level) const {
        auto end = count_m.begin();
        
        // make sure to stay within container
        T size = count_m.size();
        std::advance(end, (level > size) ? size : level);
        
        return std::accumulate(count_m.begin(), end, 0,
                               [](T sum, const value_type& value_pair) {
                                   return sum + value_pair.second;
                               });
    }
    
    
    /*!
     * Obtain the end of a level
     * @param level
     * @returns the index of the local end of that level
     */
    T end(T level) const { return begin(level + 1); }
    
    
    /*!
     * Remove particle indices from the container
     * @param num of particles that will be removed
     * @param begin of index
     */
    void remove(T num, T begin) {
        int inum = int(num);
        while ( inum > -1 ) {
            T level = which(begin + inum);
            --count_m[level];
            --inum;
        }
    }


    /*!
     * @returns the total particle count
     * (should be the same as AmrParticleBase::LocalNum)
     */
    T getLocalNumAllLevel() {
        return begin( count_m.size() );
    }


    /*!
     * @returns the total particle count up to the given level
     */
    T getLocalNumUpToLevel(T level) const {
        PAssert_GE(level, T(0));
        T sum = 0;
        for (T i = 0; i <= level; ++i) {
            sum += end(i) - begin(i);
        }
        return sum;
    }


    /*!
     * @returns the total particle count at the given level
     */
    T getLocalNumAtLevel(T level) const {
        PAssert_GE(level, T(0));
        return end(level) - begin(level);
    }

private:
    /*!
     * Find the level the particle belongs to
     * @param idx is the local index of the particle
     * @returns the level
     */
    T which(T idx) {
        T level = 0;
        
        while ( idx >= end(level) && level < size() )
            ++level;
        
        return level;
    }
        
        
private:
    /*!
     * Key represents level
     * T   represents number of particles
     */
    std::map<Key, T> count_m;
};

#endif
