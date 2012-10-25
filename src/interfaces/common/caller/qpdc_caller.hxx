#pragma once
#ifndef QPDC_CALLER_HXX_
#define QPDC_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/qpdc.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {
std::string description = std::string( "Inference algorithm for second order models. Implementation of [1] by Patrick Esser.\n" ) +
   "[1] Kumar, A., Zilberstein, S. (2011). Message-Passing Algorithms for Quadratic Programming Formulations of MAP Estimation";

template <class IO, class GM, class ACC>
class QPDCCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef QpDC<GM, ACC> QPDC;
   typedef typename QPDC::VerboseVisitorType VerboseVisitorType;
   typedef typename QPDC::EmptyVisitorType EmptyVisitorType;
   typedef typename QPDC::TimingVisitorType TimingVisitorType;
   typename QPDC::Parameter qpdcParameter_;
public:
   const static std::string name_;
   QPDCCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline QPDCCaller<IO, GM, ACC>::QPDCCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, description, ioIn) {
	addArgument( Size_TArgument< >( qpdcParameter_.maxIterations_, "mi", "maximum_iterations", "maximum number of iterations", size_t(10000) ) );
	addArgument( DoubleArgument< >( qpdcParameter_.convergenceThreshold_, "ct", "convergence_threshold",
                "stop inference if progress is below convergence threshold", 1e-20 ) );
	addArgument( IntArgument< >(    qpdcParameter_.init_method_, "in", "init_method", 
                "method to choose starting point( 0 = all label 0, 1 = uniform, x > 0 = seed for random point, x < 0 = seed for random integer point )", int(1) ) );
    addArgument( BoolArgument(      qpdcParameter_.convex_approximation_, "ca", "convex_approximation", "use a convex approximation to the problem" ) );
    addArgument( BoolArgument(      qpdcParameter_.close_gap_, "cg", "close_gap", "close gap between expectation and integer solution" ) );
    addArgument( BoolArgument(      qpdcParameter_.round_to_convergence_, "rc", "round_to_convergence", "round until convergence before starting" ) );
	addArgument( Size_TArgument< >( qpdcParameter_.max_roundings_, "mr", "max_roundings", "maximum number of roundings before start (0 to deactivate)", size_t(0) ) );
}

template <class IO, class GM, class ACC>
inline void QPDCCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running QPDC caller" << std::endl;

   this-> template infer<QPDC, TimingVisitorType, typename QPDC::Parameter>(model, outputfile, verbose, qpdcParameter_);
}

template <class IO, class GM, class ACC>
const std::string QPDCCaller<IO, GM, ACC>::name_ = "QPDC";

} // namespace interface

} // namespace opengm

#endif /* QPDC_CALLER_HXX_ */
