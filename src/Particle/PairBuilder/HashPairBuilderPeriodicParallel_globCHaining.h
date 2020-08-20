#ifndef HASH_PAIR_BUILDER_PERIODIC_PARALLEL_H
#define HASH_PAIR_BUILDER_PERIODIC_PARALLEL_H

#include <algorithm>
#include <limits>
#include <cmath>
#include <set>

template<class PBase>
class HashPairBuilderPeriodicParallel
{
public:
    enum { Dim = PBase::Dim };
    typedef typename PBase::Position_t      Position_t;

    HashPairBuilderPeriodicParallel(PBase &p) : particles(p) { }

    template<class Pred, class OP>
    void for_each(const Pred& pred, const OP &op,Vektor<double,3> extend_l, Vektor<double,3> extend_r )
    {
        const std::size_t END = std::numeric_limits<std::size_t>::max();
        std::size_t size = particles.getLocalNum()+particles.getGhostNum();

        Inform dmsg("debug_msg:");
        std::cout << "We use parallel hash pair builder ****************************" << std::endl;

        //compute which dimensions are really serial process neighbors itself in this direction
        Vektor<bool,3> parallel_dims(0,0,0);

        NDIndex<3> globDomain = particles.getFieldLayout().getDomain();
        NDIndex<3> locDomain = particles.getFieldLayout().getLocalNDIndex();

        parallel_dims[0] = !(globDomain[0]==locDomain[0]);
        parallel_dims[1] = !(globDomain[1]==locDomain[1]);
        parallel_dims[2] = !(globDomain[2]==locDomain[2]);

        Vektor<double,3> period;
        period=extend_r-extend_l;

        //make sure that the chaining mesh covers the whole domain and has a gridwidth > r_cut
        buckets_per_dim[0]=floor(period[0]/pred.getRange(0));
        buckets_per_dim[1]=floor(period[1]/pred.getRange(1));
        buckets_per_dim[2]=floor(period[2]/pred.getRange(2));

        for (unsigned dim = 0; dim<3; ++dim)
            h_chaining[dim] = period[dim]/buckets_per_dim[dim];

        //extend the chaining mesh by one layer of chaining cells in each dimension
        rmin_m = extend_l-h_chaining;
        rmax_m = extend_r+h_chaining;
        buckets_per_dim+=2;

        dmsg << " period = " << period << endl;
        dmsg << "buckets per dim = " << buckets_per_dim << endl;
        dmsg << "h_chaining = " << h_chaining << endl;

        std::size_t Nbucket = buckets_per_dim[0]*buckets_per_dim[1]*buckets_per_dim[2];

        std::size_t *buckets = new size_t[Nbucket]; //index of first particle in this bucket
        std::size_t *next = new size_t[size]; //index of next particle in this bucket. END indicates last particle of bucket
        std::fill(buckets, buckets+Nbucket, END);
        std::fill(next, next+size, END);

        //in 3D we interact with 14 neighboring cells (including self cell interaction)
        unsigned neigh = 14;

        int offset[14][3] = {{ 1, 1, 1}, { 0, 1, 1}, {-1, 1, 1},
            { 1, 0, 1}, { 0, 0, 1}, {-1, 0, 1},
            { 1,-1, 1}, { 0,-1, 1}, {-1,-1, 1},
            { 1, 1, 0}, { 0, 1, 0}, {-1, 1, 0},
            { 1, 0, 0}, { 0, 0, 0}};

        //assign all particles to a bucket
        for(std::size_t i = 0;i<size;++i)
        {
            unsigned bucket_id = get_bucket_id(i,pred);
            next[i] = buckets[bucket_id];
            buckets[bucket_id] = i;
        }

        double part_count = 0;
        //loop over all buckets
        for (int bx=0+int(!parallel_dims[0]); bx<buckets_per_dim[0]-int(!parallel_dims[0]); ++bx) {
            for (int by=0+int(!parallel_dims[1]); by<buckets_per_dim[1]-int(!parallel_dims[1]); ++by) {
                for (int bz=0+int(!parallel_dims[2]); bz<buckets_per_dim[2]-int(!parallel_dims[2]); ++bz) {
                    unsigned bucket_id_self = bz*buckets_per_dim[1]*buckets_per_dim[0]+by*buckets_per_dim[0]+bx;
                    //compute index of neighboring bucket to interact with
                    for (unsigned n=0; n<neigh;++n){
                        int bx_neigh, by_neigh, bz_neigh;
                        Vektor<double,3> shift(0,0,0);

                        bx_neigh = bx+offset[n][0];
                        //if we are serial in x-dimension we have no cached ghost particles. The local positions get periodically shifted
                        if (!parallel_dims[0]) {
                            if (bx_neigh == 0) {
                                //bucket in -x direction exceed domain boundary
                                bx_neigh+=(buckets_per_dim[0]-2);//consider last bucket in +x instead
                                shift[0] = -period[0];//shift particles in negative x direction by domain size
                            }
                            else if (bx_neigh == (buckets_per_dim[0]-1)) {
                                //bucket in +x direction exceeds domain boundary
                                bx_neigh -= (buckets_per_dim[0]-2);//consider first bucket in +x instead
                                shift[0] = period[0];//shift particles in positive x direction by domain size
                            }
                        }
                        //do the same for y and z direction:
                        by_neigh = by+offset[n][1];
                        if (!parallel_dims[1]) {
                            if (by_neigh == 0) {
                                by_neigh+=(buckets_per_dim[1]-2);
                                shift[1] = -period[0];
                            }
                            else if (by_neigh == (buckets_per_dim[1]-1)) {
                                by_neigh -=(buckets_per_dim[1]-2);
                                shift[1] = period[1];
                            }
                        }

                        bz_neigh = bz+offset[n][2];
                        if (!parallel_dims[2]) {
                            if (bz_neigh == 0) {
                                bz_neigh+=(buckets_per_dim[2]-2);
                                shift[2] = -period[2];
                            }
                            else if (bz_neigh == (buckets_per_dim[2]-1)) {
                                bz_neigh -=(buckets_per_dim[2]-2);
                                shift[2] = period[2];
                            }
                        }

                        if (bx_neigh >= 0 && bx_neigh<buckets_per_dim[0] &&
                            by_neigh >= 0 && by_neigh<buckets_per_dim[1] &&
                            bz_neigh >= 0 && bz_neigh<buckets_per_dim[2]) {

                            unsigned bucket_id_neigh =
                            bz_neigh*buckets_per_dim[1]*buckets_per_dim[0]+by_neigh*buckets_per_dim[0]+bx_neigh;

                            //i is index of particle considered in active cahining cell, j is index of neighbor particle considered
                            std::size_t i = buckets[bucket_id_self];
                            std::size_t j;

                            //loop over all particles in self cell
                            //self offset avoids double counting in self cell
                            int self_offset = 0;
                            part_count = 0;
                            while (i != END) {
                                part_count++;
                                j = buckets[bucket_id_neigh];
                                //increase offset by number of processed particles in self cell
                                for (int o=0;o<self_offset;o++){
                                    j = next[j];
                                }
                                //loop over all particles in neighbor cell
                                while(j != END) {
                                    if(pred(particles.R[i], particles.R[j]+shift)) {
                                        if (i!=j)
                                            op(i, j, particles, shift);
                                    }
                                    j = next[j];
                                }
                                i = next[i];
                                //adjust self_offset
                                if (bucket_id_self==bucket_id_neigh)
                                    self_offset++;
                                else
                                    self_offset=0;
                            }
                        }
                    }

                }
            }
        }

        delete[] buckets;
        delete[] next;
    }
private:

    //returns the bucket id of particle i
    template<class Pred>
    int get_bucket_id(int i, const Pred& pred)
    {
        Vektor<int,3> loc;
        for (unsigned d=0; d<3; ++d)
            loc[d] = (particles.R[i][d]-rmin_m[d])/h_chaining[d];
        int bucket_id = loc[2]*buckets_per_dim[1]*buckets_per_dim[0]+loc[1]*buckets_per_dim[0]+loc[0];
        //std::cout << "bucket id of particle " << i << "with coords " << particles.R[i] << " = [" << loc[0] << "," << loc[1] << "," << loc[2] << "] => bucket id = "  << bucket_id << std::endl;
        //std::cout << particles.R[i][0] << "," << particles.R[i][1] << "," << particles.R[i][2] << "," << bucket_id << std::endl;

        return bucket_id;
    }

    PBase &particles;
    Vektor<int,3> buckets_per_dim;
    Vektor<double,3> h_chaining;
    Vektor<double,3> rmin_m;
    Vektor<double,3> rmax_m;
};


#endif
