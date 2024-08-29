#ifndef IPPL_DISTRIBUTION_FUNCTION_H
#define IPPL_DISTRIBUTION_FUNCTION_H

template<typename Domain, typename Range>
class DistributionTransformationStrategy {
public:
    virtual std::function<Range(Domain)> transform(const std::function<Range(Domain)>&) const = 0;
    virtual ~DistributionTransformationStrategy() {}
};

template <typename Domain, typename Range>
class AbstractDistributionFunction {
public:
    virtual Range evaluate(Domain) const = 0;
    virtual ~AbstractDistributionFunction() {}
    virtual void applyTransformation(const DistributionTransformationStrategy<Domain, Range>& strategy) = 0;
};


template<typename Domain, typename Range>
class CompositeDistributionFunction : public AbstractDistributionFunction<Domain, Range> {
    std::function<Range(Domain)> evaluateFn;

public:
    CompositeDistributionFunction(std::function<Range(Domain)> fn) : evaluateFn(fn) {}

    Range evaluate(Domain input) const override {
        return evaluateFn(input);
    }

    CompositeDistributionFunction<Domain, Range> operator+(const CompositeDistributionFunction<Domain, Range>& other) {
        return CompositeDistributionFunction<Domain, Range>(
            [this, other](Domain input) {
                return this->evaluate(input) + other.evaluate(input);
            }
        );
    }

    CompositeDistributionFunction<Domain, Range>& operator+=(const CompositeDistributionFunction<Domain, Range>& other) {
        auto currentEvaluateFn = this->evaluateFn; 
        this->evaluateFn = [currentEvaluateFn, other](Domain input) {
            return currentEvaluateFn(input) + other.evaluate(input);
        };
        return *this;
    }

    void applyTransformation(const DistributionTransformationStrategy<Domain, Range>& strategy) override {
        evaluateFn = strategy.transform(evaluateFn);
    }
};

template<typename T>
class VortexRing : public CompositeDistributionFunction<ippl::Vector<T, 3>, ippl::Vector<T, 3> > {

public:
    VortexRing(T R, T Gamma_0, T a) : CompositeDistributionFunction<ippl::Vector<T, 3>, ippl::Vector<T, 3> >(
        [R, Gamma_0, a](ippl::Vector<T, Dim> x) -> ippl::Vector<T, 3> {

            T sigma = std::abs(std::sqrt(x[0] * x[0] + x[1] * x[1]) - R);

            T r = std::sqrt(sigma * sigma + x[2] * x[2]);
            T omega_0 = Gamma_0/(3.14159 * a * a) * std::exp(-r*r/(a*a));

            T factor = std::cos(std::atan(x[1] / x[0])) * R / std::sqrt(x[0] * x[0] + x[1]*x[1]);

            T b_x = factor * x[0];
            T b_y = factor * x[1];
            
            T s_x = x[0] - b_x;
            T s_y = x[1] - b_y;
            T s_z = x[2];

            T theta = std::acos(s_z / std::sqrt(s_x * s_x + s_y * s_y + s_z * s_z));

            T phi = std::acos(s_x / std::sqrt(s_x * s_x + s_y * s_y));
            
            if (s_y < 0) {
              phi = -phi;
            }
            

            return  ippl::Vector<T,3>({omega_0 *std::sin(theta) * std::cos(phi), omega_0 *std::sin(theta) * std::sin(phi), omega_0 *std::cos(theta)});

        }) {}
};


template<typename T>
class VortexRingScalar : public CompositeDistributionFunction<ippl::Vector<T, 3>, T > {

public:
    VortexRingScalar(T R, T Gamma_0, T a) : CompositeDistributionFunction<ippl::Vector<T, 3>, T>(
        [R, Gamma_0, a](ippl::Vector<T, Dim> x) -> T {

            T sigma = std::abs(std::sqrt(x[0] * x[0] + x[1] * x[1]) - R);

            T r = std::sqrt(sigma * sigma + x[2] * x[2]);
            T omega_0 = Gamma_0/(3.14159 * a * a) * std::exp(-r*r/(a*a));

            return omega_0;

        }) {}
};


template<typename T, unsigned Dim>
class Circle : public CompositeDistributionFunction<ippl::Vector<T, Dim>, T> {
    T radius;

public:
    Circle(T r) : CompositeDistributionFunction<ippl::Vector<T, Dim>, T>(
        [r](ippl::Vector<T, Dim> x) -> T {
            T norm = 0;
            for (size_t d = 0; d < Dim; d++) {
                norm += std::pow(x[d], 2);
            }
            return std::sqrt(norm) <= r ? 1 : 0;
        }), radius(r) {}
};

template<typename T, unsigned Dim>
class ShiftTransformation : public DistributionTransformationStrategy<ippl::Vector<T, Dim>, T> {
    ippl::Vector<T, Dim> shift;

public:
    ShiftTransformation(ippl::Vector<T, Dim> shift) : shift(shift) {}

    std::function<T(ippl::Vector<T, Dim>)> transform(const std::function<T(ippl::Vector<T, Dim>)>& original) const override {
        return [this, original](ippl::Vector<T, Dim> x) {
            ippl::Vector<T, Dim> shifted;
            for (size_t d = 0; d < Dim; d++) {
                shifted[d] = x[d] + shift[d];
            }
            return original(shifted);
        };
    }
};
#endif
