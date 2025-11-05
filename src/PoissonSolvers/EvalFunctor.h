// File EvalFunctor.h
// Helper header defining the EvalFunctor struct
// for the FEMPoissonSolver and the PreconditionedFEMPoissonSolver.

#ifndef IPPL_EVALFUNCTOR_H
#define IPPL_EVALFUNCTOR_H

namespace ippl {
    template <typename Tlhs, unsigned Dim, unsigned numElemDOFs>
    struct EvalFunctor {
        const Vector<Tlhs, Dim> DPhiInvT;
        const Tlhs absDetDPhi;

        EvalFunctor(Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi)
            : DPhiInvT(DPhiInvT)
            , absDetDPhi(absDetDPhi) {}

        KOKKOS_FUNCTION auto operator()(
            const size_t& i, const size_t& j,
            const Vector<Vector<Tlhs, Dim>, numElemDOFs>& grad_b_q_k) const {
            return dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply() * absDetDPhi;
        }
};
}

#endif
