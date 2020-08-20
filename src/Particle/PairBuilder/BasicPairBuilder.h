#ifndef BASIC_PAIR_BUILDER_H
#define BASIC_PAIR_BUILDER_H

#include <algorithm>

template<class PBase>
class BasicPairBuilder
{
public:
	enum { Dim = PBase::Dim };  
	typedef typename PBase::Position_t      Position_t;
	
	BasicPairBuilder(PBase &p) : particles(p) { }
	
	template<class Pred, class OP>
	void for_each(const Pred& pred, const OP &op)
	{
		std::size_t size = particles.getLocalNum()+particles.getGhostNum(); 
		for(std::size_t i = 0;i<size;++i)
			for(std::size_t j = i+1;j<size;++j)
				if(pred(particles.R[i], particles.R[j]))
					op(i, j, particles);
	}
private:
	PBase &particles;
};

#endif
