#pragma once
#ifndef QPDC_H
#define QPDC_H

#include <vector>
#include <type_traits>
#include <string>

#include <opengm/opengm.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitor.hxx>
#include <opengm/inference/qpdc_container.hxx>

namespace opengm {

template< class GM, class ACC >
class aFactorFunctor_Base;

template< class GM, class ACC >
class aFactorFunctor;

/** \brief Algorithm for minimization/maximization of 
 *         additive second order models.
 *
 *  Implementation of [1] by Patrick Esser. Based on a 
 *  relaxation of the original problem to a quadratic program,
 *  which in turn again gets represented as the difference of
 *  convex functions for optimization.
 *  The relaxed problem is guaranteed to decrease (resp. increase)
 *  monotonically and converges to a local optimum, while the 
 *  value of the recovered integral assignment is guaranteed to
 *  be less (resp. greater) than or equal to the value of the 
 *  relaxed problem.
 *  
 *  [1] Kumar, A., Zilberstein, S. (2011). Message-Passing Algorithms for Quadratic Programming Formulations of MAP Estimation.
 */
template< class GM, class ACC >
class QpDC : public Inference< GM, ACC > {
public:
    /// chooses suitable type for inference
    typedef typename
    std::conditional< std::is_floating_point< typename GM::ValueType >::value, typename GM::ValueType, double >::type InferValue;

    /// type of graphical model
    typedef GM GraphicalModelType;
    /// domain of modelled function
    typedef typename GraphicalModelType::LabelType LabelType;
    /// range of modelled function
    typedef typename GraphicalModelType::ValueType ValueType;
    /// type used for indexing
    typedef typename GraphicalModelType::IndexType IndexType;

    /// type used to cache values during inference
    typedef typename qpdc_container::qpdc_container< InferValue > container;

    /// iterator to modify cached values
    typedef typename container::row_iterator var_iterator;
    /// iterator to read cached values
    typedef typename container::const_row_iterator const_var_iterator;

    typedef VerboseVisitor< QpDC< GM, ACC > > VerboseVisitorType;
    typedef TimingVisitor< QpDC< GM, ACC > > TimingVisitorType;
    typedef EmptyVisitor< QpDC< GM, ACC > > EmptyVisitorType;

    /// parameter class of qpdc
    struct Parameter {
        /// constructor with default values
        Parameter(
            const std::size_t maxIterations = 10000,
            const ValueType convergenceThreshold = 0,
            const int init_method = 1,
            const bool convex_approximation = false
        ) : maxIterations_( maxIterations ),
            convergenceThreshold_( convergenceThreshold ),
            init_method_( init_method ),
            convex_approximation_( convex_approximation ) {
        }

        /// maximum number of iterations (default = 10000)
        std::size_t maxIterations_;

        /// for values greater than zero, the algorithm will stop if the progress is below this value (default = 0)
        ///
        /// if greater than zero, the progress has to be calculated at each iteration which can 
        /// decrease the performance
        ValueType convergenceThreshold_;

        /// method used for initialization of starting point (default = 1)
        /// 
        /// 0 - set all variables to first label
        ///
        /// 1 - for each variable, assume a uniform distribution acrosss the labels
        ///
        /// all other integer values will be treated as a seed for random assignment of probabilities
        int init_method_;

		/// boolean value of whether to use a convex approximation to the original problem or not
		bool convex_approximation_;
    };

    /// returns the name of the inference algorithm, QpDC
    std::string name() const;

    /// returns a const reference to the graphical model the algorithm is working with
    const GraphicalModelType& graphicalModel() const;

    /// starts the inference with empty visitor
    InferenceTermination infer();

    /// starts the inference with given visitor
    template< class VisitorType >
    InferenceTermination infer( VisitorType& );

    /// initializes algorithm 
    QpDC( const GraphicalModelType& gm, const Parameter& parameter = Parameter() );

    /// stores currently best integral assignment in given vector
    InferenceTermination arg( std::vector< LabelType >& labeling, const std::size_t N = 1 ) const;

private:
    /// returns value of quadratic program relaxation
    ValueType valueRelaxed( ) const; 

    /// main implementation of the algorithm
    template< class VisitorType >
    InferenceTermination inferRelaxed( VisitorType& );

    /// decodes cached probabilities to integer assignment
    void decodeToIntegral( std::vector< LabelType >& labeling ) const;

    /// calculates conditional expectation of objective for currently cached distribution
    InferValue calcCondExp( IndexType i, LabelType l ) const;
    /// calculates conditional expectation of objective for given distribution
    InferValue calcCondExp( IndexType i, LabelType l, const container& probs ) const;

    /// calculates some internally used values to cache them in container associated with out
    void calcNeighbourMargin( const var_iterator& out, std::size_t varLabel );
    /// calculates the lagrangian for a variable
    InferValue calcLagrange( const const_var_iterator& message, const const_var_iterator& neighbourMargin, const std::vector< char >& feasible );

    /// initializes starting distribution according to method (see Parameter)
    void initProbabilities( int method );
    /// helper method to call constructor of cache container
    std::vector< std::size_t > ctorHelper( const GraphicalModelType& );

    /// const reference to graphical model being used
    const GraphicalModelType& gm_;

    // caches
    container neighbourMargins, messages, probabilities;
    mutable container prob_tmp;

    /// functor instance for adjusted factor evaluation 
    aFactorFunctor< GM, ACC > aFactor;

    /// parameter instance used
    Parameter parameter_;
};


/** allocates space and initializes probabilities cache to feasible starting point
 *  \param gm graphical model to run inference on
 *  \param parameter optional instance of Parameter
 */
template< class GM, class ACC >
QpDC< GM, ACC >::QpDC( 
    const GM& gm, const Parameter& parameter 
) : gm_( gm ), parameter_( parameter ), 
    neighbourMargins( ctorHelper( gm ) ),
    messages( ctorHelper( gm ) ),
    probabilities( ctorHelper( gm ) ),
    prob_tmp( ctorHelper( gm ) ),
    aFactor( gm ) {

    initProbabilities( parameter.init_method_ );
}

template< class GM, class ACC >
void QpDC< GM, ACC >::initProbabilities( 
    int method 
) {
    switch( method ) {
        case 0:        /* set all variables to first label */
            for( auto pr_it = probabilities.begin(), pr_end = probabilities.end();
                 pr_it != pr_end; ++pr_it
            ) {
                auto nr_labels = pr_it.row_size();
                for( decltype( nr_labels ) label_n = 0;
                     label_n < nr_labels; ++label_n
                ) {
                    if( label_n == 0 ) {    ( *pr_it )[ label_n ] = 1.0; }
                    else {                    ( *pr_it )[ label_n ] = 0.0; }
                }
            }
            break;
        case 1:        /* assume a uniform distribution */
            for( auto pr_it = probabilities.begin(), pr_end = probabilities.end();
                 pr_it != pr_end; ++pr_it ) {
                auto nr_labels = pr_it.row_size();
                for( decltype( nr_labels ) label_n = 0;
                     label_n < nr_labels; ++label_n
                ) {
                    ( *pr_it )[ label_n ] = 1.0 / nr_labels;
                }
            }
            break;
        default:        /* assign random probabilities */
            std::srand( std::abs( method ) );
            std::vector< InferValue > interval_points;

            for( auto pr_it = probabilities.begin(), pr_end = probabilities.end();
                 pr_it != pr_end; ++pr_it
            ) {
                interval_points.clear();
                interval_points.push_back( 0.0 );
                interval_points.push_back( 1.0 );
                auto nr_labels = pr_it.row_size();
                for( decltype( nr_labels ) label_n = 0;
                     label_n < ( nr_labels - 1 ); ++label_n
                ) {
                    interval_points.push_back( InferValue( std::rand() ) / RAND_MAX );
                }

                std::sort( interval_points.begin(), interval_points.end() );
                nr_labels = pr_it.row_size();
                for( decltype( nr_labels ) label_n = 0;
                     label_n < nr_labels; ++label_n
                ) {
                    ( *pr_it )[ label_n ] = interval_points[ label_n + 1 ] - interval_points[ label_n ];
                }
            }
            break;
    }
}

/** calculates a vector such that entry i contains the number of labels for 
 *  variable i, which gets used by the constructor of the caches container
 *  \param gm graphical model on which the algorithm will run
 */
template< class GM, class ACC >
std::vector< std::size_t > QpDC< GM, ACC >::ctorHelper( 
    const GraphicalModelType& gm 
) {
    std::vector< std::size_t > nLabels_of_var( gm.numberOfVariables() );
    for( auto nVars = nLabels_of_var.size(), varN = decltype( nVars )( 0 );
         varN < nVars; ++varN 
       ) {
        nLabels_of_var[ varN ] = gm.numberOfLabels( varN );
    }

    return nLabels_of_var;
}
    
template< class GM, class ACC >
std::string QpDC< GM, ACC >::name() const {
    return "QpDC";
}

template< class GM, class ACC >
const GM& QpDC< GM, ACC >::graphicalModel() const {
    return gm_;
}


/** calls infer with empty visitor
 */
template< class GM, class ACC >
InferenceTermination QpDC< GM, ACC >::infer() {
    EmptyVisitorType visitor;
    return infer( visitor );
}

/** after running the inference, arg will give a labeling of a local optimum
 *  \param visitor instance of visitor to call
 */
template< class GM, class ACC >
template< class VisitorType >
InferenceTermination QpDC< GM, ACC >::infer( 
    VisitorType& visitor 
) {
    return inferRelaxed( visitor );
}


template< class GM, class ACC >
template< class VisitorType >
InferenceTermination QpDC<GM, ACC>::inferRelaxed( 
    VisitorType& visitor 
) {
    auto bNMItEnd = neighbourMargins.end();
    auto bMItEnd = messages.end();
    std::vector< char > feasibleUpToNow;    /* intention of vector< bool > */
    
    auto oldValue = valueRelaxed();
    decltype( oldValue ) newValue;
    auto progress = decltype( oldValue )( 1000 );

    visitor.begin( *this );

    for( auto bNMIt = neighbourMargins.begin(); bNMIt != bNMItEnd; ++ bNMIt ) {
        auto nLabels = bNMIt.row_size();
        for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
            calcNeighbourMargin( bNMIt, labelN ); 
        }
    }

    for( auto iterations = decltype( parameter_.maxIterations_ )( 0 ); 
         iterations < parameter_.maxIterations_ && parameter_.convergenceThreshold_ < progress; ++iterations ) {

        // set up messages
        for( auto bMIt = messages.begin(),
                  bPrIt = probabilities.begin(),
                  bNMIt = neighbourMargins.begin(); bMIt != bMItEnd; ++bMIt, ++bPrIt, ++bNMIt ) {
            auto varInd = bMIt.row_index();
            auto nLabels = bMIt.row_size();
            for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
                ( *bMIt )[ labelN ] = ( ( *bPrIt )[ labelN ] * ( *bNMIt )[ labelN ] ) + calcCondExp( varInd, labelN );
            }
        }

        // solve kkt greedy
        for( auto bPrIt = probabilities.begin(), 
                  bMIt = messages.begin(),
                  bNMIt = neighbourMargins.begin();
             bMIt != bMItEnd; 
             ++bPrIt, ++bMIt, ++bNMIt ) {
            bool notFeasible = true;
            feasibleUpToNow.assign( bPrIt.row_size(), true );

            auto rowSize = bPrIt.row_size(); 
            for( decltype( rowSize ) innerIt = 0; 
                 notFeasible; 
                 ++innerIt ) { 
                notFeasible = false;

                InferValue lagrangian = calcLagrange( bMIt, bNMIt, feasibleUpToNow );

                /* calculate probabilities for sub-sub-problem */
                for( decltype( rowSize ) rowIndex = 0; rowIndex < rowSize; ++rowIndex ) {
                    if( feasibleUpToNow[ rowIndex ] ) {
                        ( *bPrIt )[ rowIndex ] = ( ( *bMIt )[ rowIndex ] - lagrangian ) / ( *bNMIt )[ rowIndex ];
                        if( ( *bPrIt )[ rowIndex ] < 0 ) {
                            feasibleUpToNow[ rowIndex ] = false;
                            notFeasible = true;
                        }
                    } else {
                        ( *bPrIt )[ rowIndex ] = 0;
                    }
                } /* end for each label */
            } /* end inner loop */
        } /* end for each variable */

        visitor( *this );

        if( parameter_.convergenceThreshold_ > 0 ) {
            newValue = valueRelaxed();
            progress = std::abs( newValue - oldValue );
            oldValue = newValue;
        }

    } /* end outer loop */

    visitor.end( *this );

    return NORMAL;
}


template< class GM, class ACC >
void QpDC< GM, ACC >::calcNeighbourMargin( 
    const var_iterator& out, std::size_t varLabel 
) {
    bool varIsFirst = false;
    ( *out )[ varLabel ] = 0;

    auto varInd = out.row_index();
    auto nFact = gm_.numberOfFactors( varInd );

    decltype( varLabel ) labeling[2];
    for( decltype( nFact ) factN = 0; factN < nFact; ++factN ) {
        auto factInd = gm_.factorOfVariable( varInd, factN );

        if( gm_.numberOfVariables( factInd ) == 2 ) {
            auto other_var_index = gm_.variableOfFactor( factInd, 0 );
            if( other_var_index == varInd ) {
                varIsFirst = true;
                labeling[ 0 ] = varLabel;
                other_var_index = gm_.variableOfFactor( factInd, 1 );
            } else {
                varIsFirst = false;
                labeling[ 1 ] = varLabel;
            }
            auto nLabels = gm_.numberOfLabels( other_var_index );
            for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
                if( varIsFirst ) {
                    labeling[1] = labelN;
                } else {
                    labeling[0] = labelN;
                }
                ( *out )[ varLabel ] += aFactor( factInd, labeling );
            }
        }
    }
}


template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::calcLagrange( 
    const const_var_iterator& m, const const_var_iterator& nm, const std::vector< char >& f 
) {
    InferValue lagrangian = 0;
    InferValue normalization = 0;

    auto nLabels = m.row_size();
    for( decltype( nLabels) labelN = 0; labelN < nLabels; ++labelN ) {
        if( f[ labelN ] ) {
            lagrangian += ( *m )[ labelN ] / ( *nm )[ labelN ];
            normalization += 1.0 / ( *nm )[ labelN ];
        }
    }

    lagrangian -= 1.0;
    lagrangian /= normalization;

    return lagrangian;
}
    

/** recovers a integral solution out of the currently cached probabilities
 *  and stores them in the given vector
 *  \param[out] labeling where the integral assignment gets stored
 */
template< class GM, class ACC >
void QpDC< GM, ACC >::decodeToIntegral( std::vector< LabelType >& labeling ) const {

    probabilities.store( prob_tmp );

    InferValue maxConditionalExpectation, tmpConditionalExpectation;
    
    for( auto bPrItEnd = prob_tmp.end(), bPrIt = prob_tmp.begin(); 
         bPrIt != bPrItEnd; ++bPrIt 
    ) {
        auto nLabels = bPrIt.row_size();
        auto varInd = bPrIt.row_index();
        labeling[ varInd ] = 0;
        maxConditionalExpectation = calcCondExp( varInd, 0, prob_tmp );

        for( decltype( nLabels ) labelN = 1; labelN < nLabels; ++labelN ) {
            tmpConditionalExpectation = calcCondExp( varInd, labelN, prob_tmp );
            if( tmpConditionalExpectation >  maxConditionalExpectation ) {
                labeling[ varInd ] = labelN;
                maxConditionalExpectation = tmpConditionalExpectation;
            }
         }
        // adapt probabilities to fit assignment 
        for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
            if( labelN == labeling[ varInd ] ) {
                ( *bPrIt )[ labelN ] = 1;
            } else {
                ( *bPrIt )[ labelN ] = 0;
            }
        }
    }
}


/** calls calcCondExp with currently cached probabilities
 */
template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::calcCondExp
( 
    IndexType varInd, LabelType l 
) const {

    return calcCondExp( varInd, l, probabilities );
}
    
/** calculates the expected value of the function modelled by the graphical model
 *  given that variable i takes on label l and all the other variables are 
 *  are distributed according to the given distribution
 *  \param varInd index of variable to condition on
 *  \param l      label to condition this variable on
 *  \param probs  distribution for variables
 */
template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::calcCondExp
( 
    IndexType varInd, LabelType l, const container& probs  
) const {

    InferValue condExp = 0;
    LabelType labeling[2];
    bool varIsFirst;
    
    auto nFact = gm_.numberOfFactors( varInd );
    for( decltype( nFact ) factN = 0; factN < nFact; ++factN ) {
        auto factInd = gm_.factorOfVariable( varInd, factN );

        if( gm_.numberOfVariables( factInd ) == 2 ) {
            auto otherInd = gm_.variableOfFactor( factInd, 0 );
            varIsFirst = false;
            if( otherInd == varInd ) {
                otherInd = gm_.variableOfFactor( factInd, 1 );
                varIsFirst = true;
            }

            auto oBIt = probs.cget_row_iterator( otherInd );
            auto nLabels = oBIt.row_size();
            for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
                if( varIsFirst ) {
                    labeling[0] = l;
                    labeling[1] = labelN;
                } else {
                    labeling[0] = labelN;
                    labeling[1] = l;
                }
                
                condExp += ( aFactor( factInd, labeling ) * ( *oBIt )[ labelN ] );
            }
        } else { // first order factor 
            labeling[ 0 ] = l;
            condExp += aFactor( factInd, labeling );
            
        }
    }

    return condExp;
}

/** evaluates quadratic program formulation with currently cached probabilities 
 *  which corresponds to calculating the expectation of the modelled function
 *  when the variables are distributed according to the cached probabilities
 */
template< class GM, class ACC >
typename GM::ValueType QpDC< GM, ACC >::valueRelaxed( ) const {
    IndexType var1Index, var2Index;
    LabelType labeling[ 2 ];
    InferValue prob[ 2 ];
    auto prob1BIT = probabilities.cbegin();
    auto prob2BIT = probabilities.cbegin();

    ValueType value = 0;
    auto nFact = gm_.numberOfFactors();
    for( decltype( nFact ) factN = 0;
         factN < nFact;
         ++factN
    ) {
        var1Index = gm_.variableOfFactor( factN, 0 );
        prob1BIT.set_row( var1Index );
        auto nLabel1 = gm_.numberOfLabels( var1Index );
        for( decltype( nLabel1 ) label1N = 0;
             label1N < nLabel1;
             ++label1N
        ) {
            labeling[ 0 ] = label1N;
            prob[ 0 ] = ( *prob1BIT )[ label1N ];
            if( gm_.numberOfVariables( factN ) == 2 ) {
                var2Index = gm_.variableOfFactor( factN, 1 );
                prob2BIT.set_row( var2Index );
                auto nLabel2 = gm_.numberOfLabels( var2Index );
                for( decltype( nLabel2 ) label2N = 0;
                     label2N < nLabel2;
                     ++label2N
                ) {
                    labeling[ 1 ] = label2N;
                    prob[ 1 ] = ( *prob2BIT )[ label2N ];

                    value += gm_[ factN ]( labeling ) * prob[ 0 ] * prob[ 1 ];
                }
            } else {
                value += gm_[ factN ]( labeling ) * prob[ 0 ];
            }
        }
    }

    return value;
}
            
    


/** decodes currently cached probabilities to integral assignment and stores this
 *  assignment in given vector
 *  \param[out] labeling vector to store integral assignment
 *  \param N ignored
 */
template< class GM, class ACC >
InferenceTermination QpDC< GM, ACC >::arg
(
    std::vector<LabelType>& labeling, const std::size_t N
) const {
    labeling.resize( gm_.numberOfVariables() );
    decodeToIntegral( labeling );

    return NORMAL;
}

/** \brief Base class for wrapper around factor evaluation used by QPDC.
 *
 *  Adjusts the factor evaluation to switch between Maximization and 
 *  Minimization and to fulfill the assumption of positive factors.
 */
template< class GM, class ACC >
class aFactorFunctor_Base {
public:
    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    typedef typename GM::ValueType ValueType;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;


    aFactorFunctor_Base( const GM& gm );
protected:
    ValueType adjustment_;
    const GM& gm_;
};



template< class T, bool V >
struct set_sign {
    static constexpr T sign = T( 1 );
};

template< class T >
struct set_sign< T, false > {
    static constexpr T sign = T( -1 );
};    

template< class GM, class ACC >
aFactorFunctor_Base< GM, ACC >::aFactorFunctor_Base( const GM& gm ) : gm_( gm ) {
    typedef typename GM::ValueType ValueType;
    ValueType sign = set_sign< ValueType, std::is_same< ACC, Maximizer >::value >::sign;

    LabelType labeling[ 2 ];

    ValueType min = 0;
    ValueType tmpValue;
    auto nFact = gm.numberOfFactors();
    for( decltype( nFact ) factN = 0;
         factN < nFact;
         ++factN
    ) {
        if( gm[ factN ].numberOfVariables() == 2 ) {
            auto nLabels1 = gm[ factN ].shape( 0 );
            for( decltype( nLabels1 ) label1N = 0;
                 label1N < nLabels1;
                 ++label1N
            ) {
                labeling[ 0 ] = label1N;
                auto nLabels2 = gm[ factN ].shape( 1 );
                for( decltype( nLabels2 ) label2N = 0;
                     label2N < nLabels2;
                     ++label2N 
                ) {
                    labeling[ 1 ] = label2N;
                    tmpValue = sign * gm[ factN ]( labeling );
                    if( tmpValue < min ) {
                        min = tmpValue;
                    }
                }
            }
        }
    }

    adjustment_ = min;

}

/** \brief Generic factor adjustment class used by QPDC.
 */
template< class GM, class ACC >
class aFactorFunctor : public aFactorFunctor_Base< GM, ACC > {
public:
    typedef typename GM::ValueType ValueType;
    typedef typename GM::IndexType IndexType;

    aFactorFunctor( const GM& gm ) : aFactorFunctor_Base< GM, ACC >( gm ) {};

    template< class Iterator >
    ValueType operator() ( const IndexType factorIndex, Iterator labeling );
};

/** \brief Adjusts factor evaluation for Maximization.
 */
template< class GM >
class aFactorFunctor< GM, Maximizer > : public aFactorFunctor_Base< GM, Maximizer > {
public:
    typedef typename GM::ValueType ValueType;
    typedef typename GM::IndexType IndexType;
    using aFactorFunctor_Base< GM, Maximizer >::gm_;
    using aFactorFunctor_Base< GM, Maximizer >::adjustment_;

    aFactorFunctor( const GM& gm ) : aFactorFunctor_Base< GM, Maximizer >( gm ) {};

    template< class Iterator >
    ValueType operator() ( const IndexType factorIndex, Iterator labeling ) const {
        return gm_[ factorIndex ]( labeling ) - adjustment_ + ( gm_.numberOfVariables( factorIndex ) == 2 ) * 0.01; 
    }
};

/** \brief Adjusts factor evaluation for Minimization.
 */
template< class GM >
class aFactorFunctor< GM, Minimizer > : public aFactorFunctor_Base< GM, Minimizer > {
public:
    typedef typename GM::ValueType ValueType;
    typedef typename GM::IndexType IndexType;
    using aFactorFunctor_Base< GM, Minimizer >::gm_;
    using aFactorFunctor_Base< GM, Minimizer >::adjustment_;

    aFactorFunctor( const GM& gm ) : aFactorFunctor_Base< GM, Minimizer >( gm ) {};

    template< class Iterator >
    ValueType operator() ( const IndexType factorIndex, Iterator labeling ) const {
        return (-1) * gm_[ factorIndex ]( labeling ) - adjustment_ + ( gm_.numberOfVariables( factorIndex ) == 2 ) * 0.01;; 
    }

};


} //namespace opengm
#endif
