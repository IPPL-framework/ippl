#ifndef IPPL_BARE_FIELD_EXPRESSIONS_H
#define IPPL_BARE_FIELD_EXPRESSIONS_H

namespace ippl {
    template <typename E>
    struct BareFieldExpr {
        auto operator()(size_t i) const {
            return static_cast<const E&>(*this)(i);
        }
    };


    #define DefineBareFieldBinaryOperation(fun, op, expr)   \
    template<typename E1, typename E2>                      \
    struct fun : public BareFieldExpr<fun<E1, E2> > {       \
        fun(const E1& u, const E2& v) : u_m(u), v_m(v) { }  \
                                                            \
        auto operator()(size_t i) const {                   \
            return expr;                                    \
        }                                                   \
                                                            \
    private:                                                \
        const E1 u_m;                                       \
        const E2 v_m;                                       \
    };                                                      \
                                                            \
    template<typename E1, typename E2>                      \
    fun<E1, E2> op(const BareFieldExpr<E1>& u,              \
                   const BareFieldExpr<E2>& v) {            \
        return fun<E1, E2>(*static_cast<const E1*>(&u),     \
                           *static_cast<const E2*>(&v));    \
    }


    DefineBareFieldBinaryOperation(BareFieldAdd,      operator+, u_m(i) + v_m(i));
    DefineBareFieldBinaryOperation(BareFieldSubtract, operator-, u_m(i) - v_m(i));
    DefineBareFieldBinaryOperation(BareFieldMultiply, operator*, u_m(i) * v_m(i));
    DefineBareFieldBinaryOperation(BareFieldDivide,   operator/, u_m(i) / v_m(i));
}

#endif