#ifndef IPPL_FFT_FFT_H
#define IPPL_FFT_FFT_H

// Umbrella header for the post-refactor FFT layer.
// The legacy monolithic content of this file (heFFTe backend traits,
// FFTBase, the four transform specializations, etc.) has been split into
// the three headers below — pulling them in here preserves the historical
// `#include "FFT/FFT.h"` entry point used throughout the codebase.

#include "Traits.h"
#include "Backend/Backend.h"
#include "Transform/Transform.h"

#endif
