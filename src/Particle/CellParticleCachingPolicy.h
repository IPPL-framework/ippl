//
// Class CellParticleCachingPolicy
//
//   The Cell caching layout ensures that each node has all ghost particles
//   for each external particle that is inside a neighboring cell.
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

#ifndef CELL_PARTICLE_CACHING_POLICY
#define CELL_PARTICLE_CACHING_POLICY

#include <Particle/BoxParticleCachingPolicy.h>

template<class T, unsigned Dim, class Mesh>
class CellParticleCachingPolicy : private BoxParticleCachingPolicy<T,Dim,Mesh> {
public:
	CellParticleCachingPolicy()
	{
		std::fill(cells, cells+Dim, 0);
	}

	void setCacheCellRange(int d, int length)
	{
		cells[d] = length;
	}

	void setAllCacheCellRanges(int length)
	{
		std::fill(cells, cells+Dim, length);
	}

	template<class C>
	void updateCacheInformation(
		ParticleSpatialLayout<T, Dim, Mesh, C > &PLayout
		)
	{
		for(unsigned int d = 0;d<Dim;++d)
			BoxParticleCachingPolicy<T,Dim,Mesh>::setCacheDimension(d, cells[d]*PLayout.getLayout().getMesh().get_meshSpacing(d));

		BoxParticleCachingPolicy<T,Dim,Mesh>:: updateCacheInformation(PLayout);
	}

	template<class C>
	void updateGhostParticles(
		IpplParticleBase< ParticleSpatialLayout<T,Dim,Mesh,C > > &PData,
		ParticleSpatialLayout<T, Dim, Mesh, C > &PLayout
		)
	{
		for(unsigned int d = 0;d<Dim;++d)
			BoxParticleCachingPolicy<T,Dim,Mesh>::setCacheDimension(d, cells[d]*PLayout.getLayout().getMesh().get_meshSpacing(d));

		BoxParticleCachingPolicy<T,Dim,Mesh>::updateGhostParticles(PData, PLayout);
	}
protected:
	~CellParticleCachingPolicy() {}
private:
	int cells[Dim];
};

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End: