#ifndef BOX_PARTICLE_CACHING_POLICY
#define BOX_PARTICLE_CACHING_POLICY

/*
 *
 * The Box caching layout ensures that each node has all ghost particles
 * for each external particle that is inside an extended bounding box
 *
 */


#include <algorithm>
#include <map>

#include "Message/Message.h"
#include "Message/Communicate.h"
#include "Message/Formatter.h"

template <class T, unsigned Dim, class Mesh, class CachingPolicy> class ParticleSpatialLayout;

template<class T, unsigned Dim, class Mesh>
class BoxParticleCachingPolicy {
public:
	BoxParticleCachingPolicy()
	{
		std::fill(boxDimension, boxDimension+Dim, T());
	}

	void setCacheDimension(int d, T length)
	{
		boxDimension[d] = length;
	}

	void setAllCacheDimensions(T length)
	{
		std::fill(boxDimension, boxDimension+Dim, length);
	}
template<class C>
	void updateCacheInformation(
		ParticleSpatialLayout<T, Dim, Mesh, C > &PLayout
		)
	{
		RegionLayout<T,Dim,Mesh> &RLayout = PLayout.getLayout();
		NDRegion<T,Dim> globalDomain = RLayout.getDomain();

		typename RegionLayout<T,Dim,Mesh>::iterator_iv  localVN = RLayout.begin_iv();
		typename RegionLayout<T,Dim,Mesh>::iterator_iv localVNend = RLayout.end_iv();

		regions.clear();


		//fill in boundary conditions
		std::fill(periodic, periodic+2*Dim, false);

		if (PLayout.getUpdateFlag(ParticleLayout<T,Dim>::BCONDS))
		{
			ParticleBConds<T,Dim>& pBConds = PLayout.getBConds();
			typename ParticleBConds<T,Dim>::ParticleBCond periodicBCond=ParticlePeriodicBCond;

			for (unsigned d=0; d<2*Dim; ++d)
			{
				if(pBConds[d] == periodicBCond)
				{
					periodic[d] = true;
				}
			}
		}


		for (;localVN!=localVNend;++localVN)
		{
			//get local domain
			NDRegion<T,Dim> ldom = (*localVN).second->getDomain();

			//extrude local domain
			NDRegion<T,Dim> exdom;
			for(unsigned int d = 0;d<Dim;++d)
			{
				exdom[d] = PRegion<T>(ldom[d].first()- boxDimension[d],
									  ldom[d].last() + boxDimension[d]);
			}

			Offset_t offset;
			std::fill(offset.begin(), offset.end(), 0);

				//get all relevant offsets
				for(unsigned int d = 0;d<Dim;++d)
				{
					if(periodic[2*d] && (exdom[d].first() < globalDomain[d].first()))
					{
						offset[d] = globalDomain[d].length();
					}
					else if(periodic[2*d+1] && (exdom[d].last() > globalDomain[d].last()))
					{
						offset[d] = -globalDomain[d].length();
					}
				}
				//cycle through all combinations
				int onoff[Dim];
				std::fill(onoff, onoff+Dim, 0);
				while(true)
				{

					NDRegion<T,Dim> chckdom;
					for(unsigned int d = 0;d<Dim;++d)
					{
						chckdom[d] = PRegion<T>(ldom[d].first()- boxDimension[d]+onoff[d]*offset[d],
												ldom[d].last() + boxDimension[d]+onoff[d]*offset[d]);
					}

					//get touched external domains
					typename RegionLayout<T,Dim,Mesh>::touch_range_dv touchRange = RLayout.touch_range_rdv(chckdom);


					typename RegionLayout<T,Dim,Mesh>::touch_iterator_dv i;


					for(i = touchRange.first; i != touchRange.second; ++i)
					{
						int node = (*i).second->getNode();
						if(node == Ippl::myNode())//don't add local node
							continue;
						NDRegion<T,Dim> dom = (*i).second->getDomain();
						Offset_t tmpoffset;
						for(unsigned int d = 0;d<Dim;++d)
						{
							dom[d] = PRegion<T>(dom[d].first() - onoff[d]*offset[d],
												dom[d].last()  - onoff[d]*offset[d]);
							tmpoffset[d] = onoff[d]*offset[d];
						}

						regions[node].push_back(std::make_pair(dom,tmpoffset));
					}

					//generate next combinations. this is basically a binary incrementer
					unsigned int j = 0;
					for(;j<Dim;++j)
					{
						if(offset[j]==0)
							continue;//skip irrelevant directions
						if((onoff[j] = !onoff[j]))
							break;//flip and continue if there's a "carry"
					}

					if(j==Dim)
						break;
				}

		}
	}

template<class C>
	void updateGhostParticles(
		IpplParticleBase< ParticleSpatialLayout<T,Dim,Mesh,C > > &PData,
		ParticleSpatialLayout<T, Dim, Mesh, C > &/*PLayout*/
		)
	{

		Ippl::Comm->barrier();
		typedef typename std::map<unsigned, std::list<std::pair<NDRegion<T,Dim>, Offset_t> > >::iterator m_iterator;

		//dump the old ghost particles
		PData.ghostDestroy(PData.getGhostNum(), 0);

		//get tag
		int tag = Ippl::Comm->next_tag(P_SPATIAL_GHOST_TAG, P_LAYOUT_CYCLE);

		//these are needed to free data for nonblocking sends
		std::vector<MPI_Request> requests;
		std::vector<MsgBuffer*> buffers;

		//for each possible target node
		for(m_iterator n = regions.begin();n!=regions.end();++n)
		{
			int node = n->first;

			//find particles that need to be sent
			std::vector<size_t> sendlist;
			std::vector<Offset_t> offsetlist;
			for(typename std::list<std::pair<NDRegion<T,Dim>, Offset_t> >::iterator li = n->second.begin();li!=n->second.end();++li)
			{
				NDRegion<T, Dim> region = (*li).first;

				for (unsigned int i = 0;i < PData.getLocalNum();++i)
				{
					NDRegion<T,Dim> ploc;
					for (unsigned int d = 0;d < Dim;++d)
						ploc[d] = PRegion<T>(PData.R[i][d] - boxDimension[d],
											 PData.R[i][d] + boxDimension[d]);

					if(region.touches(ploc))
					{
						sendlist.push_back(i);
						offsetlist.push_back((*li).second);
					}
				}
			}


			//and send them
			if(sendlist.empty())
			{
				//don't bother creating an empty buffer just send an empty message
				requests.push_back(Ippl::Comm->raw_isend(0, 0, node, tag));
			}
			else
			{
				//pack and send ghost particles
				MsgBuffer *msgbuf = 0;
				PData.writeMsgBufferWithOffsets(msgbuf, sendlist,offsetlist);
				MPI_Request request = Ippl::Comm->raw_isend(msgbuf->getBuffer(), msgbuf->getSize(), node, tag);

				requests.push_back(request);
				buffers.push_back(msgbuf);
			}

		}


		//receive ghost particles
		Format *format = PData.getFormat();

		for(unsigned int n = 0;n<regions.size();++n)
		{
			int node = Communicate::COMM_ANY_NODE;
            char *buffer = 0;
            int bufsize = Ippl::Comm->raw_probe_receive(buffer, node, tag);
            if(bufsize>0)
            {
				MsgBuffer recvbuf(format, buffer, bufsize);
				PData.readGhostMsgBuffer(&recvbuf, node);
			}
		}

		//wait for communication to finish and clean up buffers
        MPI_Waitall(requests.size(), &(requests[0]), MPI_STATUSES_IGNORE);
        for (unsigned int j = 0; j<buffers.size(); ++j)
        {
			delete buffers[j]->getFormat();
            delete buffers[j];
        }

		delete format;
	}
protected:
	~BoxParticleCachingPolicy() {}
private:
	struct Offset_t
	{
		T data[Dim];
		T& operator[](int i) { return data[i]; }
		T operator[](int i) const { return data[i]; }
		T* begin() { return data; }
		T* end() { return data+Dim; }
	};

	T boxDimension[Dim];
	bool periodic[2*Dim];
	std::map<unsigned, std::list<std::pair<NDRegion<T,Dim>, Offset_t> > > regions;
};

#endif