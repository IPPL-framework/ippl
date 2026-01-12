// File EvalFunctor.h
// Helper header defining the EvalFunctor struct
// for the FEMPoissonSolver and the PreconditionedFEMPoissonSolver.
// This EvalFunctor represents the action of the matrix A on a
// vector x for the Poisson equation in its FEM formulation Ax=b.
// This functor is passed to the LagrangeSpace in FEMPoissonSolver
// to be used for matrix-free evaluation.

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
