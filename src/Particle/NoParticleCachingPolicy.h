//
// Class NoParticleCachingPolicy
//   Empty caching strategy that doesn't cache anything
//
//   Please note: for the time being this class is *not* used! But since it
//   might be used in future projects, we keep this file.
//
// Copyright (c) 2003 - 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
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

#ifndef NO_PARTICLE_CACHING_POLICY
#define NO_PARTICLE_CACHING_POLICY

template <class T, unsigned Dim, class Mesh, class CachingPolicy> class ParticleSpatialLayout;

//basic policy that doesn't cache any particles
template<class T, unsigned Dim, class Mesh>
class NoParticleCachingPolicy {
public:
template<class C>
	void updateCacheInformation(
		ParticleSpatialLayout<T, Dim, Mesh, C > &/*PLayout*/
		)
	{
		//don't do anything...
	}
template<class C>
	void updateGhostParticles(
		IpplParticleBase< ParticleSpatialLayout<T,Dim,Mesh,C > > &/*PData*/,
		ParticleSpatialLayout<T, Dim, Mesh, C > &/*PLayout*/
		)
	{
		//don't do anything...
	}
protected:
	~NoParticleCachingPolicy() {}
};

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
