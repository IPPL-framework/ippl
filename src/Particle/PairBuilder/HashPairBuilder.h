#ifndef HASH_PAIR_BUILDER_H
#define HASH_PAIR_BUILDER_H

#include <algorithm>
#include <limits>
#include <cmath>
#include <set>

template<class PBase>
class HashPairBuilder
{
public:
    enum { Dim = PBase::Dim };
    typedef typename PBase::Position_t      Position_t;

    HashPairBuilder(PBase &p) : particles(p) { }

    template<class Pred, class OP>
    void for_each(const Pred& pred, const OP &op)
    {
        const std::size_t END = std::numeric_limits<std::size_t>::max();
        //	int f[3];

        std::size_t size = particles.getLocalNum()+particles.getGhostNum();
        /*
         int edge = std::pow(size/8, 1./Dim);

         std::size_t Nbucket = 1;
         for(int d = 0;d<Dim;++d)
         {
         f[d] = edge;
         Nbucket *= f[d];
         if(pred.getRange(d)==0)
         return;
         }
         */

        bounds(particles.R, rmin_m, rmax_m);
        Inform dmsg("debug_msg:");
        dmsg << "R_min = " << rmin_m << " R_max = " << rmax_m << endl;

        buckets_per_dim[0]=ceil((rmax_m[0]-rmin_m[0])/pred.getRange(0));
        buckets_per_dim[1]=ceil((rmax_m[1]-rmin_m[1])/pred.getRange(1));
        buckets_per_dim[2]=ceil((rmax_m[2]-rmin_m[2])/pred.getRange(2));

        //dmsg << "buckets per dim = " << buckets_per_dim << endl;
        std::size_t Nbucket = buckets_per_dim[0]*buckets_per_dim[1]*buckets_per_dim[2];

        std::size_t *buckets = new size_t[Nbucket];
        std::size_t *next = new size_t[size];
        std::fill(buckets, buckets+Nbucket, END);
        std::fill(next, next+size, END);

        /*
         int neigh = 1;
         for(int d = 0;d<Dim;++d)
         neigh *= 3;
         neigh /= 2;
         */

        //in 3D we interact with 14 neighboring cells (including self cell interaction)
        int neigh = 14;

        int offset[14][3] = {{ 1, 1, 1}, { 0, 1, 1}, {-1, 1, 1},
            { 1, 0, 1}, { 0, 0, 1}, {-1, 0, 1},
            { 1,-1, 1}, { 0,-1, 1}, {-1,-1, 1},
            { 1, 1, 0}, { 0, 1, 0}, {-1, 1, 0},
            { 1, 0, 0}, { 0, 0, 0}};

        //assign all particles to a bucket
        for(std::size_t i = 0;i<size;++i)
        {
            //std::size_t pos = sum(i, pred, f, offset[13]);
            unsigned bucket_id = get_bucket_id(i,pred);
            next[i] = buckets[bucket_id];
            buckets[bucket_id] = i;
        }

        for (std::size_t i=0; i< Nbucket; ++i) {
            //dmsg << "Bucket " << i << " stores particles " << endl;
		std::size_t j = buckets[i];
            while (j!= END) {
                //dmsg << j << " :: " << particles.R[j] << endl;
                j = next[j];
            }
        }

        //loop over all buckets
        for (int bx=0; bx<buckets_per_dim[0]; ++bx) {
            for (int by=0; by<buckets_per_dim[1]; ++by) {
                for (int bz=0; bz<buckets_per_dim[2]; ++bz) {
                    //dmsg << "bx = " << bx << "by = " << by << " bz = " << bz <<endl;
                    unsigned bucket_id_self = bz*buckets_per_dim[1]*buckets_per_dim[0]+by*buckets_per_dim[0]+bx;
                    //compute index of neighboring cell to interact with
                    for (int n=0; n<neigh;++n){
                        int bx_neigh = bx+offset[n][0];
                        int by_neigh = by+offset[n][1];
                        int bz_neigh = bz+offset[n][2];
                        //check if neighbor cell is within boundaries
                        //dmsg << " looking at neighbor n = " << n << endl;
                        if (bx_neigh >= 0 && bx_neigh<buckets_per_dim[0] &&
                            by_neigh >= 0 && by_neigh<buckets_per_dim[1] &&
                            bz_neigh >= 0 && bz_neigh<buckets_per_dim[2]) {

                            //compute bucket id of neighboring cell
                            unsigned bucket_id_neigh =
                            bz_neigh*buckets_per_dim[1]*buckets_per_dim[0]+by_neigh*buckets_per_dim[0]+bx_neigh;
                            //dmsg << "looking at buckets " << bucket_id_self << " and " << bucket_id_neigh << endl;

                            std::size_t i = buckets[bucket_id_self];
                            //dmsg << "head of chain is " << i << endl;
                            std::size_t j;
                            //loop over all particles in self cell

                            //self offset avoids double counting in self cell
                            int self_offset = 0;
                            while (i != END) {
                                //dmsg << "start i while loop " << endl;
                                j = buckets[bucket_id_neigh];
                                //increase offset by number of processed particles in self cell
                                for (int o=0;o<self_offset;o++){
                                    j = next[j];
                                }
                                //loop over all particles in nieghbor cell
                                while(j != END) {
                                    //dmsg << "start j while loop" << endl;
                                    if(pred(particles.R[i], particles.R[j]))
                                    {
                                        //dmsg << "processing pair (" << i << "," << j << ")"<<endl;
                                        if (i!=j)
                                            op(i, j, particles);
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
                        //dmsg << "proceed with next neighbor cell " << endl;
                    }

                }
            }
        }
        /*
         for(std::size_t i = 0;i<size;++i)
         {
         std::size_t j = next[i];
         while(j != END)
         {
         if(pred(particles.R[i], particles.R[j]))
         op(i, j, particles);
         j = next[j];
         }

         for(int k = 0;k<neigh;++k)
         {
         std::size_t tmppos = sum(i, pred, f, offset[k]);

         j = buckets[tmppos];
         while(j != END)
         {
         if(pred(particles.R[i], particles.R[j]))
         op(i, j, particles);
         j = next[j];
         }
         }

         }
         */
        delete[] buckets;
        delete[] next;

        //dmsg << "particle particle interactions DONE" << endl;
    }
private:

    template<class Pred>
    int sum(int i, const Pred& pred, int f[], int offset[])
    {
        std::cout << "SUM " << pred.getRange(1)  << "f[1]=" << f[1] << std::endl;
        int sum = 0;
        for(int d = 0;d<Dim;++d)
        {
            double scaled = particles.R[i][d]/pred.getRange(d);
            int pos = mod(int(floor(scaled+offset[d])), f[d]);
            for(int dd = 0;dd<d;++dd)
                pos*=f[dd];
            sum += pos;
        }
        std::cout << "END SUM" << std::endl;

        return sum;
    }

    //returns the bucket id of particle i
    template<class Pred>
    int get_bucket_id(int i, const Pred& pred)
    {
        Vektor<int,3> loc;
        for (unsigned d=0; d<3; ++d)
            loc[d] = (particles.R[i][d]-rmin_m[d])/pred.getRange(d);
        int bucket_id = loc[2]*buckets_per_dim[1]*buckets_per_dim[0]+loc[1]*buckets_per_dim[0]+loc[0];
        //std::cout << "bucket id of particle " << i << " = [" << loc[0] << "," << loc[1] << "," << loc[2] << "] => bucket id = "  << bucket_id << std::endl;
        return bucket_id;
    }

    int mod(int x, int m)
    {
        if(x>=0)
            return x%m;
        else
            return (m - ((-x)%m))%m;
    }

    PBase &particles;
    Vektor<int,3> buckets_per_dim;

    Vektor<double,3> rmin_m;
    Vektor<double,3> rmax_m;
};
#endif
