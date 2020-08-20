// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 ***************************************************************************/

#ifndef RNG_XDIV_H
#define RNG_XDIV_H

/***********************************************************************
 * 
 * class RNGXDiv
 * class RNGXDivSequence : public SequenceGen<RNGXDiv>
 *
 * Simple class that implements random number generator from LANL
 * X-Division folks, in the range [0...1].  The first class may be
 * used standalone, or as a template parameter to SequenceGen.  The
 * second class is derived from SequenceGen, and makes it easier to
 * use this RNG in expressions.
 * Use RNGXDiv as a scalar or container element, and use
 * RNGXDivSequence when you need a sequence of numbers to fill a container.
 *
 ***********************************************************************/

#include "Utility/SequenceGen.h"

class RNGXDiv {

public:
  typedef double Return_t;

public:
  // default constructor
  RNGXDiv(int advance = 0) {
    // set the first random number, composed of
    // SeedUpper (top 24 bits) and SeedLower (bottom 24 bits).
    SeedUpper = static_cast<double>(long(FirstSeed * INV_SQR_RANMAX));
    SeedLower = FirstSeed - SeedUpper * SQR_RANMAX;
    AdvanceSeed(advance);  // advance the seed
  }

  // copy constructor
  RNGXDiv(const RNGXDiv& rng)
    : SeedLower(rng.SeedLower), SeedUpper(rng.SeedUpper),
      RandLower(rng.RandLower), RandUpper(rng.RandUpper) {}

  // destructor
  ~RNGXDiv(void) {}

  // advance indicates number of times to advance random number source
  inline void AdvanceSeed(int advance = 0) {
    for (int iadv=0; iadv<advance; iadv++)
      advijk();
    // set sequence to new source
    RandLower = SeedLower;
    RandUpper = SeedUpper;
  }

  // set seed to user-specified value, plus shift to ensure it is large
  inline void SetSeed(unsigned long seed) {
    Return_t rijk = Return_t(seed) + FirstSeed;
    SeedUpper = static_cast<double>(long(rijk * INV_SQR_RANMAX));
    SeedLower = rijk - SeedUpper * SQR_RANMAX;
    // set sequence to new source
    RandLower = SeedLower;
    RandUpper = SeedUpper;
  }

  // get seed value
  inline unsigned long GetSeed(void) const {
    // invert process for setting seed
    Return_t rijk = SeedLower + SeedUpper * SQR_RANMAX;
    unsigned long seed = (unsigned long) (rijk - FirstSeed);
    return seed;
  }

  // return the next pseudo-random number
  inline Return_t GetRandom(void) const {
    Return_t a = RandMultLower * RandLower;
    Return_t b = RandMultUpper * RandLower +
      RandMultLower * RandUpper + long(a * INV_SQR_RANMAX);
    RandLower = a - long(a * INV_SQR_RANMAX) * SQR_RANMAX;
    RandUpper = b - long(b * INV_SQR_RANMAX) * SQR_RANMAX;
    return ( (RandUpper * SQR_RANMAX + RandLower) * INV_RANMAX );
  }

  // pseudonym for GetRandom()
  inline Return_t operator()(void) const { return GetRandom(); }

  // conversion to Return_t, same as GetRandom()
  inline operator Return_t() const { return GetRandom(); }

  // return the period of the RNG
  static Return_t GetRandMax(void) { return Return_t(RANDOM_MAX); }

private:
  double SeedLower, SeedUpper;
  mutable double RandLower, RandUpper;

  // advance random number seed for sequence
  inline void advijk(void) {
    Return_t a = SeedMultLower * SeedLower;
    Return_t b = (SeedMultUpper * SeedLower -
      long(SeedMultUpper * SeedLower * INV_SQR_RANMAX) * SQR_RANMAX) +
      (SeedMultLower * SeedUpper -
      long(SeedMultLower * SeedUpper * INV_SQR_RANMAX) * SQR_RANMAX) +
      long(a * INV_SQR_RANMAX);
    SeedLower = a - long(a * INV_SQR_RANMAX) * SQR_RANMAX;
    SeedUpper = b - long(b * INV_SQR_RANMAX) * SQR_RANMAX;
  }

  static const double RANDOM_MAX;
  static const double SQR_RANMAX;
  static const double INV_SQR_RANMAX;
  static const double INV_RANMAX;
  static const double SeedMultUpper;
  static const double SeedMultLower;
  static const double RandMultUpper;
  static const double RandMultLower;
  static const double FirstSeed;
};

RNG_BASIC_MATH(RNGXDiv)


// A version of SequenceGen with extra constructors to make using this
// class easier.  This is the version that people should use to fill
// containers with a random number sequence in an expression.  This
// class is PETE-aware via its inheritance from SequenceGen.

class RNGXDivSequence : public SequenceGen<RNGXDiv> {

public:
  // default constructor
  RNGXDivSequence(int advance = 0)
    : SequenceGen<RNGXDiv>(RNGXDiv(advance)) {}

  // copy constructor
  RNGXDivSequence(const RNGXDivSequence& rngseq)
    : SequenceGen<RNGXDiv>(rngseq.getGenerator()) {}

  // destructor
  ~RNGXDivSequence(void) {}

  // wrappers around RNG generator functions
  inline void     AdvanceSeed(int adv = 0) { getGenerator().AdvanceSeed(adv); }
  inline void     SetSeed(unsigned long seed) { getGenerator().SetSeed(seed); }
  inline unsigned long GetSeed(void) const { return getGenerator().GetSeed(); }
  inline Return_t GetRandom(void) { return getGenerator().GetRandom(); }
  inline Return_t operator()(void) { return getGenerator().GetRandom(); }
  static Return_t GetRandMax(void) { return RNGXDiv::GetRandMax(); }
};


#endif // RNG_XDIV_H

