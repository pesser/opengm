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


template< class GM, class ACC >
class QpDC : public Inference< GM, ACC > {
public:
    /* choose suitable type for inference */
    typedef typename
    std::conditional< std::is_floating_point< typename GM::ValueType >::value, typename GM::ValueType, double >::type InferValue;

    // typedefs, see inference.hxx
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS

    typedef typename qpdc_container::qpdc_container< InferValue > container;

    typedef typename container::row_iterator var_iterator;
    typedef typename container::const_row_iterator const_var_iterator;

    typedef VerboseVisitor< QpDC< GM, ACC > > VerboseVisitorType;
    typedef TimingVisitor< QpDC< GM, ACC > > TimingVisitorType;
    typedef EmptyVisitor< QpDC< GM, ACC > > EmptyVisitorType;

    struct Parameter {
        Parameter(
            const std::size_t maxIterations = 1000,
            const ValueType convergenceThreshold = 0,
            const int init_method = 42
        ) : maxIterations_( maxIterations ),
            convergenceThreshold_( convergenceThreshold ),
            init_method_( init_method ) {
        }

        std::size_t maxIterations_;
        ValueType convergenceThreshold_;
        int init_method_;
    };

    // minimal interface required by opengm
    std::string name() const;
    const GraphicalModelType& graphicalModel() const;
    InferenceTermination infer();

    template< class VisitorType >
    InferenceTermination infer( VisitorType& );

    //constructor
    QpDC( const GraphicalModelType& gm, const Parameter& parameter = Parameter() );

    // custom implementation of interface
    InferenceTermination arg( std::vector< LabelType >& labeling, const std::size_t N = 1 ) const;
    ValueType value() const;

private:
    // subroutines
    ValueType valueRelaxed( ) const; 

    template< class VisitorType >
    InferenceTermination inferRelaxed( VisitorType& );

    void decodeToIntegral();

    InferValue calcCondExp( IndexType, LabelType );

    void calcNeighbourMargin( const var_iterator& out, std::size_t varLabel );
    InferValue calcLagrange( const const_var_iterator& message, const const_var_iterator& neighbourMargin, const std::vector< char >& feasible );

    void initProbabilities( int method );
    std::vector< std::size_t > ctorHelper( const GraphicalModelType& );

    // state
    const GraphicalModelType& gm_;
    std::vector< LabelType > labeling_;

    /* caches */
    container neighbourMargins, messages, probabilities;

    /* functor class for adjusted factor evaluation */
    aFactorFunctor< GM, ACC > aFactor;

    Parameter parameter_;
};

template< class GM, class ACC >
QpDC< GM, ACC >::QpDC( 
    const GM& gm, const Parameter& parameter 
) : gm_( gm ), labeling_( gm.numberOfVariables(), 0), parameter_( parameter ), 
    neighbourMargins( ctorHelper( gm ) ),
    messages( ctorHelper( gm ) ),
    probabilities( ctorHelper( gm ) ),
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
                for( auto nr_labels = pr_it.row_size(), label_n = decltype( nr_labels )( 0 );
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
                for( auto nr_labels = pr_it.row_size(), label_n = decltype( nr_labels )( 0 );
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
                for( auto nr_labels = pr_it.row_size(), label_n = decltype( nr_labels )( 0 );
                     label_n < ( nr_labels - 1 ); ++label_n
                ) {
                    interval_points.push_back( InferValue( std::rand() ) / RAND_MAX );
                }

                std::sort( interval_points.begin(), interval_points.end() );
                for( auto nr_labels = pr_it.row_size(), label_n = decltype( nr_labels )( 0 );
                     label_n < nr_labels; ++label_n
                ) {
                    ( *pr_it )[ label_n ] = interval_points[ label_n + 1 ] - interval_points[ label_n ];
                }
            }
            break;
    }
}

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


template< class GM, class ACC >
InferenceTermination QpDC< GM, ACC >::infer() {
    EmptyVisitorType visitor;
    return infer( visitor );
}

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
        for( auto nLabels = bNMIt.row_size(), labelN = decltype( nLabels )( 0 ); labelN < nLabels; ++labelN ) {
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
                ( *bMIt )[ labelN ] = ( *bPrIt )[ labelN ] + ( calcCondExp( varInd, labelN ) / ( *bNMIt )[ labelN ] );
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
                        ( *bPrIt )[ rowIndex ] = ( *bMIt )[ rowIndex ] - ( lagrangian / ( *bNMIt )[ rowIndex ] );
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

    decodeToIntegral();

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
            for( auto nLabels = gm_.numberOfLabels( other_var_index ), labelN =  decltype(nLabels)( 0 ); labelN < nLabels; ++labelN ) {
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
            lagrangian += ( *m )[ labelN ];
            normalization += ( 1.0 / ( *nm )[ labelN ] );
        }
    }

    lagrangian -= 1.0;
    lagrangian /= normalization;

    return lagrangian;
}
    


template< class GM, class ACC >
void QpDC< GM, ACC >::decodeToIntegral() {
    auto bPrItEnd = probabilities.end();

    InferValue maxConditionalExpectation, tmpConditionalExpectation;
    
    for( auto bPrIt = probabilities.begin(); bPrIt != bPrItEnd; ++bPrIt ) {
        auto nLabels = bPrIt.row_size();
        auto varInd = bPrIt.row_index();
        labeling_[ varInd ] = 0;
        maxConditionalExpectation = calcCondExp( varInd, 0 );
        for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
            tmpConditionalExpectation = calcCondExp( varInd, labelN );
            if( tmpConditionalExpectation >  maxConditionalExpectation ) {
                labeling_[ varInd ] = labelN;
                maxConditionalExpectation = tmpConditionalExpectation;
            }
         }
        // adapt probabilities to fit assignment 
        for( decltype( nLabels ) labelN = 0; labelN < nLabels; ++labelN ) {
            if( labelN == labeling_[ varInd ] ) {
                ( *bPrIt )[ labelN ] = 1;
            } else {
                ( *bPrIt )[ labelN ] = 0;
            }
        }
    }
}


template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::calcCondExp
( 
    IndexType varInd, LabelType l  
) {
    InferValue condExp = 0;
    LabelType labeling[2];
    auto nFact = gm_.numberOfFactors( varInd );
    bool varIsFirst;
    
    for( decltype( nFact ) factN = 0; factN < nFact; ++factN ) {
        auto factInd = gm_.factorOfVariable( varInd, factN );

        if( gm_.numberOfVariables( factInd ) == 2 ) {
            auto otherInd = gm_.variableOfFactor( factInd, 0 );
            varIsFirst = false;
            if( otherInd == varInd ) {
                otherInd = gm_.variableOfFactor( factInd, 1 );
                varIsFirst = true;
            }

            auto oBIt = probabilities.get_row_iterator( otherInd );
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

template< class GM, class ACC >
typename GM::ValueType QpDC< GM, ACC >::value() const {
    return valueRelaxed( );
}

template< class GM, class ACC >
typename GM::ValueType QpDC< GM, ACC >::valueRelaxed( ) const {
    IndexType var1Index, var2Index;
    LabelType labeling[ 2 ];
    InferValue prob[ 2 ];
    auto prob1BIT = probabilities.cbegin();
    auto prob2BIT = probabilities.cbegin();

    ValueType value = 0;
    for( auto nFact = gm_.numberOfFactors(),
              factN = decltype( nFact )( 0 );
         factN < nFact;
         ++factN
    ) {
        var1Index = gm_.variableOfFactor( factN, 0 );
        prob1BIT.set_row( var1Index );
        for( auto nLabel1 = gm_.numberOfLabels( var1Index ),
                  label1N = decltype( nLabel1 )( 0 );
             label1N < nLabel1;
             ++label1N
        ) {
            labeling[ 0 ] = label1N;
            prob[ 0 ] = ( *prob1BIT )[ label1N ];
            if( gm_.numberOfVariables( factN ) == 2 ) {
                var2Index = gm_.variableOfFactor( factN, 1 );
                prob2BIT.set_row( var2Index );
                for( auto nLabel2 = gm_.numberOfLabels( var2Index ),
                          label2N = decltype( nLabel2 )( 0 );
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
            
    


template< class GM, class ACC >
InferenceTermination QpDC< GM, ACC >::arg
(
    std::vector<LabelType>& labeling, const std::size_t N
) const {
    labeling = labeling_;

    return NORMAL;
}

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
    for( auto nFact = gm.numberOfFactors(), factN = decltype( nFact )( 0 );
         factN < nFact;
         ++factN
    ) {
        if( gm[ factN ].numberOfVariables() == 2 ) {
            for( auto nLabels1 = gm[ factN ].shape( 0 ), label1N = decltype( nLabels1 )( 0 );
                 label1N < nLabels1;
                 ++label1N
            ) {
                labeling[ 0 ] = label1N;
                for( auto nLabels2 = gm[ factN ].shape( 1 ), label2N = decltype( nLabels2 )( 0 );
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

template< class GM, class ACC >
class aFactorFunctor : public aFactorFunctor_Base< GM, ACC > {
public:
    typedef typename GM::ValueType ValueType;
    typedef typename GM::IndexType IndexType;

    aFactorFunctor( const GM& gm ) : aFactorFunctor_Base< GM, ACC >( gm ) {};

    template< class Iterator >
    ValueType operator() ( const IndexType factorIndex, Iterator labeling );
};

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
