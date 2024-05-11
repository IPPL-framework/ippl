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
