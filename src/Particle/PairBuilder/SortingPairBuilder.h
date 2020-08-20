#ifndef SORTING_PAIR_BUILDER_H
#define SORTING_PAIR_BUILDER_H

#include <algorithm>

//predicate used to sort index array with std::sort
template<class A>
struct ProxyPred_t
{
    ProxyPred_t(const A &a, unsigned d) : array(a), dim(d) { }

    template<class T>
    bool operator()(const T &a, const T &b)
    { return array[a][dim] < array[b][dim]; }

    unsigned dim;
    const A &array;
};

//simplifies usage by allowing the omission of the template parameter
template<class A>
ProxyPred_t<A> ProxyPred(const A &a, unsigned d)
{
    return ProxyPred_t<A>(a, d);
}

template<class PBase>
class SortingPairBuilder
{
public:
    enum { Dim = PBase::Dim };
    typedef typename PBase::Position_t      Position_t;

    SortingPairBuilder(PBase &p) : particles(p) { }

    template<class Pred, class OP>
    void for_each(const Pred& pred, const OP &op)
    {
        std::size_t size = particles.getLocalNum()+particles.getGhostNum();

        Position_t mean[Dim];
        Position_t variance[Dim];

        //calculate mean position
        std::fill(mean, mean+Dim, 0);
        for(std::size_t i = 0;i<size;++i)
            for(int d = 0;d<Dim;++d)
                mean[d] += particles.R[i][d];

        for(int d = 0;d<Dim;++d)
            mean[d] /= size;

        //calculate variance for each dimension
        std::fill(variance, variance+Dim, 0);
        for(std::size_t i = 0;i<size;++i)
            for(int d = 0;d<Dim;++d)
                variance[d] += (mean[d]-particles.R[i][d])*(mean[d]-particles.R[i][d]);

        int dimension = 0;
        int var = variance[0];
        for(int d = 1;d<Dim;++d)
            if(variance[d]>var)
            {
                dimension = d;
                var = variance[d];
            }

        //sort index array
        std::size_t *indices = new std::size_t[size];
        for(std::size_t i = 0;i<size;++i)
            indices[i] = i;

        std::sort(indices, indices+size, ProxyPred(particles.R, dimension));

        for(std::size_t i = 0;i<size;++i)
            for(std::size_t j = i+1;j<size;++j)
                if(pred(particles.R[indices[i]], particles.R[indices[j]]))
                    op(indices[i], indices[j], particles);
                else if(particles.R[indices[j]][dimension] - particles.R[indices[i]][dimension]
                        > pred.getRange(dimension))
                    break;
    }
private:
    PBase &particles;
};

#endif
