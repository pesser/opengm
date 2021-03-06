#pragma once
#ifndef QPDC_H
#define QPDC_H

#include <vector>
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
 *  Implementation of [1] by Patrick Esser, supervised by 
 *  Joerg Kappes. Based on a relaxation
 *  of the original problem to a quadratic program,
 *  which in turn again gets represented as the difference of
 *  convex functions for optimization.
 *  
 *  [1] Kumar, A., Zilberstein, S. (2011). Message-Passing Algorithms for Quadratic Programming Formulations of MAP Estimation.
 */
template< class GM, class ACC >
class QpDC : public Inference< GM, ACC > {
public:
    typedef typename
    GM::ValueType InferValue;

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
            const ValueType convergenceThreshold = 1e-20,
            const int init_method = 1,
            const bool convex_approximation = false,
            const bool close_gap = false,
            const bool round_to_convergence = false,
            const std::size_t max_roundings = 0,
            const std::vector< LabelType > start_state = std::vector< LabelType >()
        ) : maxIterations_( maxIterations ),
            convergenceThreshold_( convergenceThreshold ),
            init_method_( init_method ),
            convex_approximation_( convex_approximation ),
            close_gap_( close_gap ),
            round_to_convergence_( round_to_convergence ),
            max_roundings_( max_roundings ),
            start_state_( start_state ) {
        }

        /// maximum number of iterations (default = 10000)
        std::size_t maxIterations_;

        /// stop if the progress is below this value (default = 1e-20)
        ValueType convergenceThreshold_;

        /// method used for initialization of starting point (default = 1)
        /// 
        /// 0 - set all variables to first label
        ///
        /// 1 - for each variable, assume a uniform distribution acrosss the labels
        ///
        /// all other positive integer values will be treated as a seed for random assignment of probabilities
        ///
        /// all other negative integer values will be treated as a seed for random assignment of an integer solution
        int init_method_;

		/// use convex approximation to the original problem if true (default = false)
		bool convex_approximation_;

        /// close gap between expectation and value (default = false)
        bool close_gap_;

        /// round solution until convergence before starting qpdc (default = false)
        bool round_to_convergence_;

        /// maximum number of roundings before start (0 to deactivate). (default = 0)
        ///
        /// Only has an effect with round_to_convergence = true 
        std::size_t max_roundings_;

        /// starting state to use
        std::vector< LabelType > start_state_;
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
    /// returns value of quadratic program relaxation, i.e. the expectation
    ValueType valueRelaxed( ) const; 

    /// main implementation of the algorithm
    template< class VisitorType >
    InferenceTermination inferRelaxed( VisitorType& );

    /// decodes cached probabilities to integer assignment
    void decodeToIntegral( std::vector< LabelType >& labeling ) const;

    /// non const rounding; same as decodeToIntegral but sets cached probabilities to integer solution
    void round();

    /// calculates something like the conditional expectation of objective for currently cached distribution
    InferValue calcCondExp( IndexType i, LabelType l ) const;
    /// calculates something like the conditional expectation of objective for given distribution
    InferValue calcCondExp( IndexType i, LabelType l, const container& probs ) const;

    /// calculates some internally used values 
    InferValue calcNeighbourMargin( std::size_t varInd, std::size_t varLabel );
    /// calculates the lagrangian for a variable
    InferValue calcLagrange( const const_var_iterator& gradv, 
        const const_var_iterator& neighbourMargin, 
        const std::vector< char >& feasible );

    template< bool V > class typeWrap { };

    /// calculates the diagonal entries needed to make the factor matrix definite
    void calcDiagonals( container& diagonals, typeWrap< true > dummy );
    /// makes sure to set diagonal entries to zero if convex approximation is deactivated
    void calcDiagonals( container& diagonals, typeWrap< false > dummy );

    /// set state
    void setState( const std::vector< LabelType >& state );
    /// initializes starting distribution according to method (see Parameter)
    void initProbabilities( int method );
    /// helper method to call constructor of cache container
    std::vector< std::size_t > ctorHelper( const GraphicalModelType& );

    /// used for convergence criterium
    InferValue l2metric( const container prob1, const container prob2 );

    /// const reference to graphical model being used
    const GraphicalModelType& gm_;

    // caches
    container neighbourMargins, gradv, probabilities, diagonals;
    container prob_before;
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
    gradv( ctorHelper( gm ) ),
    probabilities( ctorHelper( gm ) ),
    diagonals( ctorHelper( gm ) ),
    prob_before( ctorHelper( gm ) ),
    prob_tmp( ctorHelper( gm ) ),
    aFactor( gm ) {

    if( parameter.start_state_.size() == gm_.numberOfVariables() ) {
        setState( parameter.start_state_ );
    } else {
        initProbabilities( parameter.init_method_ );
    }
}

template< class GM, class ACC >
void QpDC< GM, ACC >::setState( 
    const std::vector< LabelType >& state
) {
    typename std::vector< LabelType >::const_iterator state_it = state.begin();
    for( var_iterator pr_it = probabilities.begin(), pr_end = probabilities.end();
         pr_it != pr_end;
         ++pr_it, ++state_it
       ) {
        for( std::size_t label_n = 0, nr_labels = pr_it.row_size();
             label_n < nr_labels;
             ++label_n
           ) {
            if( label_n == *state_it ) { ( *pr_it )[ label_n ] = 1.0; }
            else {                       ( *pr_it )[ label_n ] = 0.0; }
        }
    }
}

template< class GM, class ACC >
void QpDC< GM, ACC >::initProbabilities( 
    int method 
) {
    switch( method ) {
        case 0:        /* set all variables to first label */
            for( var_iterator pr_it = probabilities.begin(), pr_end = probabilities.end();
                 pr_it != pr_end; ++pr_it
            ) {
                std::size_t nr_labels = pr_it.row_size();
                for( std::size_t label_n = 0;
                     label_n < nr_labels; ++label_n
                ) {
                    if( label_n == 0 ) {    ( *pr_it )[ label_n ] = 1.0; }
                    else {                    ( *pr_it )[ label_n ] = 0.0; }
                }
            }
            break;
        case 1:        /* assume a uniform distribution */
            for( var_iterator pr_it = probabilities.begin(), pr_end = probabilities.end();
                 pr_it != pr_end; ++pr_it ) {
                std::size_t nr_labels = pr_it.row_size();
                for( std::size_t label_n = 0;
                     label_n < nr_labels; ++label_n
                ) {
                    ( *pr_it )[ label_n ] = 1.0 / nr_labels;
                }
            }
            break;
        default:        /* assign random probabilities */
            std::srand( std::abs( method ) );

            if( method >= 0 ) { // assign fractal solution
                std::vector< InferValue > interval_points;

                for( var_iterator pr_it = probabilities.begin(), pr_end = probabilities.end();
                     pr_it != pr_end; ++pr_it
                ) {
                    interval_points.clear();
                    interval_points.push_back( 0.0 );
                    interval_points.push_back( 1.0 );
                    std::size_t nr_labels = pr_it.row_size();
                    for( std::size_t label_n = 0;
                         label_n < ( nr_labels - 1 ); ++label_n
                    ) {
                        interval_points.push_back( InferValue( std::rand() ) / RAND_MAX );
                    }

                    std::sort( interval_points.begin(), interval_points.end() );
                    nr_labels = pr_it.row_size();
                    for( std::size_t label_n = 0;
                         label_n < nr_labels; ++label_n
                    ) {
                        ( *pr_it )[ label_n ] = interval_points[ label_n + 1 ] - interval_points[ label_n ];
                    }
                }
            } else {    // assign integer solution

                for( var_iterator pr_it = probabilities.begin(), 
                        pr_end = probabilities.end();
                        pr_it != pr_end;
                        ++pr_it
                   ) {
                    std::size_t nr_labels = pr_it.row_size();
                    ValueType rnd = ValueType( std::rand() ) / RAND_MAX;
                    bool found = false;

                    for( std::size_t labelN = 0;
                            labelN < nr_labels;
                            ++labelN
                       ) {
                        if( found ) {
                            ( *pr_it )[ labelN ] = 0.0;
                        } else {
                            if( rnd <= ( labelN + 1 )/nr_labels ) {
                                ( *pr_it )[ labelN ] = 1.0;
                                found = true;
                            } else {
                                ( *pr_it )[ labelN ] = 0.0;
                            }
                        }
                    }
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
    for( std::size_t nVars = nLabels_of_var.size(), varN = 0;
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

/** main algorithm
 */
template< class GM, class ACC >
template< class VisitorType >
InferenceTermination QpDC<GM, ACC>::inferRelaxed( 
    VisitorType& visitor 
) {
    std::vector< char > feasibleUpToNow;    /* intention of vector< bool > */
    ValueType progress = 1;

    visitor.begin( *this, std::string( "expectation" ), valueRelaxed(),  std::string( "progress" ), 0.0 );

    for( std::size_t iterations = 1; 
            parameter_.round_to_convergence_ && ( progress > 1e-12 ); 
            ++iterations 
       ) {
        ValueType before = this->value();
        round();
        progress = ( before - this->value() ) / ( std::abs( before ) + 1 );

        if( parameter_.max_roundings_ > 0 && iterations > parameter_.max_roundings_ ) {
            parameter_.round_to_convergence_ = false;
        }
    }
    
    probabilities.store( prob_before );
    progress = 1000;

    if( parameter_.convex_approximation_ ) {
        calcDiagonals( diagonals, typeWrap< true >() );
    } else {
        calcDiagonals( diagonals, typeWrap< false >() );
    }

    for( var_iterator nm_varEnd = neighbourMargins.end(), 
            nm_varIt = neighbourMargins.begin(),
            diag_varIt = diagonals.begin();
            nm_varIt != nm_varEnd;
            ++nm_varIt, ++diag_varIt
       ) {
        std::size_t varInd = nm_varIt.row_index();

        for( std::size_t nrLabels = nm_varIt.row_size(), labelN = 0; 
                labelN < nrLabels; 
                ++labelN 
           ) {
            ( *nm_varIt )[ labelN ] = calcNeighbourMargin( varInd, labelN ) + 2 * ( *diag_varIt )[ labelN ]; 
        }
    }

    for( std::size_t iterations = 0; 
            iterations < std::size_t( parameter_.maxIterations_ ) && parameter_.convergenceThreshold_ < progress; 
            ++iterations 
       ) {
        // calculate gradient of v
        for( var_iterator prob_varEnd = probabilities.end(),
                prob_varIt = probabilities.begin(),
                gradv_varIt = gradv.begin(),
                nm_varIt = neighbourMargins.begin(),
                diag_varIt = diagonals.begin();
                prob_varIt != prob_varEnd;
                ++prob_varIt, ++gradv_varIt, ++nm_varIt, ++diag_varIt
           ) {
            std::size_t varInd = gradv_varIt.row_index();

            for( std::size_t nrLabels = gradv_varIt.row_size(), labelN = 0; 
                    labelN < nrLabels; 
                    ++labelN 
               ) {
                ( *gradv_varIt )[ labelN ] = (  ( *prob_varIt )[ labelN ] * ( *nm_varIt )[ labelN ] + 
                                                calcCondExp( varInd, labelN ) + 
                                                ( *diag_varIt )[ labelN ] -
                                                2 * ( *prob_varIt )[ labelN ] * ( *diag_varIt )[ labelN ]
                                             );
            }
        }

        // solve kkt greedy
        for( var_iterator prob_varEnd = probabilities.end(), 
                prob_varIt = probabilities.begin(), 
                gradv_varIt = gradv.begin(),
                nm_varIt = neighbourMargins.begin();
                prob_varIt != prob_varEnd; 
                ++prob_varIt, ++gradv_varIt, ++nm_varIt 
           ) {

            bool notFeasible = true;
            feasibleUpToNow.assign( prob_varIt.row_size(), true );

            std::size_t nrLabels = prob_varIt.row_size(); 
            while( notFeasible ) {
                notFeasible = false;

                InferValue lagrangian = calcLagrange( gradv_varIt, nm_varIt, feasibleUpToNow );

                /* calculate probabilities for sub-sub-problem */
                for( std::size_t labelN = 0; labelN < nrLabels; ++labelN ) {
                    if( feasibleUpToNow[ labelN ] ) {
                        ( *prob_varIt )[ labelN ] = ( ( *gradv_varIt )[ labelN ] - lagrangian ) / 
                            ( ( *nm_varIt )[ labelN ] );
                        if( ( *prob_varIt )[ labelN ] < 0 ) {
                            feasibleUpToNow[ labelN ] = false;
                            notFeasible = true;
                        }
                    } else {
                        ( *prob_varIt )[ labelN ] = 0;
                    }
                } /* end for each label */
            } /* end inner loop */
        } /* end for each variable */


        progress = l2metric( prob_before, probabilities );
        probabilities.store( prob_before );

        visitor( *this, std::string( "expectation" ), valueRelaxed(),  std::string( "progress" ), progress );

        if( parameter_.close_gap_ ) {
            ValueType vrel = valueRelaxed();
            ValueType val = this->value();
            ValueType mVal = std::max( std::abs(vrel), std::abs(val) );
            if( ( vrel - val > mVal * 1e-20 ) && mVal > 0 ) {
                round();
            }
        }

    } /* end outer loop */

    visitor.end( *this, std::string( "expectation" ), valueRelaxed(),  std::string( "progress" ), progress );

    return NORMAL;
}


template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::calcNeighbourMargin( 
    std::size_t varInd, std::size_t varLabel 
) {
    bool varIsFirst = false;
    std::size_t labeling[2];
    InferValue out = 0;

    std::size_t nFact = gm_.numberOfFactors( varInd );

    for( std::size_t factN = 0; factN < nFact; ++factN ) {

        std::size_t factInd = gm_.factorOfVariable( varInd, factN );

        if( gm_.numberOfVariables( factInd ) == 2 ) {
            std::size_t other_var_index = gm_.variableOfFactor( factInd, 0 );
            if( other_var_index == varInd ) {
                varIsFirst = true;
                labeling[ 0 ] = varLabel;
                other_var_index = gm_.variableOfFactor( factInd, 1 );
            } else {
                varIsFirst = false;
                labeling[ 1 ] = varLabel;
            }
            std::size_t nLabels = gm_.numberOfLabels( other_var_index );
            for( std::size_t labelN = 0; labelN < nLabels; ++labelN ) {
                if( varIsFirst ) {
                    labeling[1] = labelN;
                } else {
                    labeling[0] = labelN;
                }
                out += aFactor( factInd, labeling );
            }
        } else {
            labeling[0] = varLabel;
            out += aFactor( factInd, labeling );
        }
    }

    return out;
}


template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::calcLagrange( 
    const const_var_iterator& gv, 
    const const_var_iterator& nm, 
    const std::vector< char >& f 
) {
    InferValue lagrangian = 0;
    InferValue normalization = 0;

    std::size_t nLabels = gv.row_size();
    for( std::size_t labelN = 0; labelN < nLabels; ++labelN ) {
        if( f[ labelN ] ) {
            lagrangian += ( *gv )[ labelN ] / ( *nm )[ labelN ];
            normalization += 1.0 / ( *nm )[ labelN ];
        }
    }

    lagrangian -= 1.0;
    lagrangian /= normalization;

    return lagrangian;
}


template< class GM, class ACC >
void QpDC< GM, ACC >::calcDiagonals( container& diagonals, typeWrap< true > dummy ) {
    LabelType labeling[ 2 ];
    bool var_i_first = true;

    var_iterator diag_it_end = diagonals.end();
    for( var_iterator diag_i = diagonals.begin(); diag_i != diag_it_end; ++diag_i ) {
        std::size_t var_i = diag_i.row_index();
        std::size_t n_labels = diag_i.row_size();
        for( std::size_t label_n = 0; label_n < n_labels; ++label_n ) {
            InferValue d = 0;
            std::size_t n_factors = gm_.numberOfFactors( var_i );
            for( std::size_t factor_n = 0; factor_n < n_factors; ++factor_n ) {
                std::size_t fact_i = gm_.factorOfVariable( var_i, factor_n );
                if( gm_.numberOfVariables( fact_i ) == 2 ) {
                    std::size_t var_j = gm_.variableOfFactor( fact_i, 0 );
                    if( var_j == var_i ) {
                        var_i_first = true;
                        var_j = gm_.variableOfFactor( fact_i, 1 );
                    } else {
                        var_i_first = false;
                    }
                    std::size_t n_labels_j = gm_.numberOfLabels( var_j );
                    for( std::size_t label_n_j = 0; 
                         label_n_j < n_labels_j; ++label_n_j 
                       ) {
                        if( var_i_first ) {
                            labeling[ 0 ] = label_n;
                            labeling[ 1 ] = label_n_j;
                        } else {
                            labeling[ 0 ] = label_n_j;
                            labeling[ 1 ] = label_n;
                        }
                        d += aFactor( fact_i, labeling );
                    }
                }         
            }
            ( *diag_i )[ label_n ] = d / 2;
        }
    }
}

template< class GM, class ACC >
void QpDC< GM, ACC >::calcDiagonals( container& diagonals, typeWrap< false > dummy ) {
    var_iterator diag_it_end = diagonals.end();

    for( var_iterator diag_i = diagonals.begin(); diag_i != diag_it_end; ++diag_i ) {
        std::size_t n_labels = diag_i.row_size();
        for( std::size_t label_n = 0; label_n < n_labels; ++label_n ) {
            ( *diag_i )[ label_n ] = 0;
        }
    }
}

/** recovers a integral solution out of the currently cached probabilities
 *  and stores them in the given vector
 *  \param[out] labeling where the integral assignment gets stored
 */
template< class GM, class ACC >
void QpDC< GM, ACC >::decodeToIntegral( std::vector< LabelType >& labeling ) const {

    // copy currently cached probabilities to prob_tmp
    probabilities.store( prob_tmp );

    InferValue maxConditionalExpectation, tmpConditionalExpectation;
    
    for( var_iterator bPrItEnd = prob_tmp.end(), bPrIt = prob_tmp.begin(); 
         bPrIt != bPrItEnd; ++bPrIt 
    ) {
        std::size_t nLabels = bPrIt.row_size();
        std::size_t varInd = bPrIt.row_index();
        labeling[ varInd ] = 0;
        maxConditionalExpectation = calcCondExp( varInd, 0, prob_tmp );

        for( std::size_t labelN = 1; labelN < nLabels; ++labelN ) {
            tmpConditionalExpectation = calcCondExp( varInd, labelN, prob_tmp );
            if( tmpConditionalExpectation >  maxConditionalExpectation ) {
                labeling[ varInd ] = labelN;
                maxConditionalExpectation = tmpConditionalExpectation;
            }
         }
        // adapt probabilities to fit assignment 
        for( std::size_t labelN = 0; labelN < nLabels; ++labelN ) {
            if( labelN == labeling[ varInd ] ) {
                ( *bPrIt )[ labelN ] = 1;
            } else {
                ( *bPrIt )[ labelN ] = 0;
            }
        }
    }
}

/** non-const version of decodeToIntegral - sets cached probabilities
 *  to integer solution
 */
template< class GM, class ACC >
void QpDC< GM, ACC >::round()  {
    std::vector< LabelType > labeling( gm_.numberOfVariables() ); 

    InferValue maxConditionalExpectation, tmpConditionalExpectation;
    
    for( var_iterator bPrItEnd = probabilities.end(), bPrIt = probabilities.begin(); 
         bPrIt != bPrItEnd; ++bPrIt 
    ) {
        std::size_t nLabels = bPrIt.row_size();
        std::size_t varInd = bPrIt.row_index();
        labeling[ varInd ] = 0;
        maxConditionalExpectation = calcCondExp( varInd, 0, probabilities );

        for( std::size_t labelN = 1; labelN < nLabels; ++labelN ) {
            tmpConditionalExpectation = calcCondExp( varInd, labelN, probabilities );
            if( tmpConditionalExpectation >  maxConditionalExpectation ) {
                labeling[ varInd ] = labelN;
                maxConditionalExpectation = tmpConditionalExpectation;
            }
         }
        // adapt probabilities to fit assignment 
        for( std::size_t labelN = 0; labelN < nLabels; ++labelN ) {
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
    
/** calculates something related to the expected value of the function 
 *  modelled by the graphical model
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
    
    std::size_t nFact = gm_.numberOfFactors( varInd );
    for( std::size_t factN = 0; factN < nFact; ++factN ) {
        std::size_t factInd = gm_.factorOfVariable( varInd, factN );

        if( gm_.numberOfVariables( factInd ) == 2 ) {
            std::size_t otherInd = gm_.variableOfFactor( factInd, 0 );
            varIsFirst = false;
            if( otherInd == varInd ) {
                otherInd = gm_.variableOfFactor( factInd, 1 );
                varIsFirst = true;
            }

            const_var_iterator oBIt = probs.cget_row_iterator( otherInd );
            std::size_t nLabels = oBIt.row_size();
            for( std::size_t labelN = 0; labelN < nLabels; ++labelN ) {
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
    const_var_iterator prob1BIT = probabilities.cbegin();
    const_var_iterator prob2BIT = probabilities.cbegin();

    ValueType value = 0;
    std::size_t nFact = gm_.numberOfFactors();
    for( std::size_t factN = 0;
         factN < nFact;
         ++factN
    ) {
        var1Index = gm_.variableOfFactor( factN, 0 );
        prob1BIT.set_row( var1Index );
        std::size_t nLabel1 = gm_.numberOfLabels( var1Index );
        for( std::size_t label1N = 0;
             label1N < nLabel1;
             ++label1N
        ) {
            labeling[ 0 ] = label1N;
            prob[ 0 ] = ( *prob1BIT )[ label1N ];
            if( gm_.numberOfVariables( factN ) == 2 ) {
                var2Index = gm_.variableOfFactor( factN, 1 );
                prob2BIT.set_row( var2Index );
                std::size_t nLabel2 = gm_.numberOfLabels( var2Index );
                for( std::size_t label2N = 0;
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

    /* //uncomment to model the actual objective that gets optimized by the convex approximation
    for( const_var_iterator prob_varEnd = probabilities.cend(),
            prob_varIt = probabilities.cbegin(),
            diag_varIt = diagonals.cbegin();
            prob_varIt != prob_varEnd;
            ++prob_varIt, ++diag_varIt
       ) {
        for( std::size_t nrLabels = prob_varIt.row_size(),
                labelN = 0;
                labelN < nrLabels;
                ++labelN
           ) {
            value += ( ( *diag_varIt )[ labelN ] * 
                ( std::pow( ( *prob_varIt )[ labelN ], 2 ) - ( *prob_varIt )[ labelN ] )
                );
        }
    }
    */

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

/** calculates the l2metric between two vectors given in containers
 */
template< class GM, class ACC >
typename QpDC< GM, ACC >::InferValue QpDC< GM, ACC >::l2metric
( 
    const container prob1, const container prob2 
) {
    InferValue diff = 0;
    for( const_var_iterator prob1_end = prob1.cend(), 
         prob1_varIt = prob1.cbegin(), prob2_varIt = prob2.cbegin();
         prob1_varIt != prob1_end; ++prob1_varIt, ++prob2_varIt
       ) {
        for( std::size_t nrStates = prob1_varIt.row_size(), stateN = 0;
             stateN < nrStates; ++stateN
           ) {
            diff += std::pow( ( *prob1_varIt )[ stateN ] - 
                              ( *prob2_varIt )[ stateN ], 2
                    );
        }
    }

    return diff;
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
    static const T sign = T( 1 );
};

template< class T >
struct set_sign< T, false > {
    static const T sign = T( -1 );
};    

template< class GM, class ACC >
aFactorFunctor_Base< GM, ACC >::aFactorFunctor_Base( const GM& gm ) : gm_( gm ) {
    typedef typename GM::ValueType ValueType;
    ValueType sign = set_sign< ValueType, qpdc_container::is_same< ACC, Maximizer >::value >::sign;

    LabelType labeling[ 2 ];

    ValueType min = 0;
    ValueType tmpValue;
    std::size_t nFact = gm.numberOfFactors();

    for( std::size_t factN = 0;
         factN < nFact;
         ++factN
    ) {
        if( gm[ factN ].numberOfVariables() == 2 ) {
            std::size_t nLabels1 = gm[ factN ].shape( 0 );
            for( std::size_t label1N = 0;
                 label1N < nLabels1;
                 ++label1N
            ) {
                labeling[ 0 ] = label1N;
                std::size_t nLabels2 = gm[ factN ].shape( 1 );
                for( std::size_t label2N = 0;
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
        } else {
            std::size_t nLabels = gm[ factN ].shape( 0 );
            for( std::size_t labelN = 0;
                    labelN < nLabels;
                    ++labelN
               ) {
                labeling[ 0 ] = labelN;
                tmpValue = sign * gm[ factN ]( labeling );
                if( tmpValue < min ) {
                    min = tmpValue;
                }
            }
        }
    }

    // make sure factors are strictly positive
    adjustment_ = min - 0.0001;

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
        return gm_[ factorIndex ]( labeling ) - adjustment_; 
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
        return (-1) * gm_[ factorIndex ]( labeling ) - adjustment_; 
    }

};


} //namespace opengm
#endif
