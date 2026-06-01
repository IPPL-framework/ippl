/*!
 * @file Transform.h
 * @brief Aggregate include of all IPPL FFT transform specializations.
 *
 * Pulls in the CC (complex-to-complex), RC (real-to-complex), NUFFT,
 * pruned CC/RC, and trigonometric (sin/cos) transforms in one go.
 */
#ifndef IPPL_FFT_TRANSFORM_HPP
#define IPPL_FFT_TRANSFORM_HPP

#include "FFT/Transform/CC.h"
#include "FFT/Transform/NUFFT.h"
#include "FFT/Transform/PrunedCC.h"
#include "FFT/Transform/PrunedRC.h"
#include "FFT/Transform/RC.h"
#include "FFT/Transform/Trig.h"

#endif
