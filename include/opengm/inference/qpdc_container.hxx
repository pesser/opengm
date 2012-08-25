#pragma once
#ifndef QPDC_CONTAINER_H
#define QPDC_CONTAINER_H

#include <cstddef>
#include <type_traits>

namespace opengm {
namespace qpdc_container {

template< class Iterator, class Container >
class qpdc_iterator { 
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;
    typedef typename Container::row_iterator_container row_iterator_container;
    typedef std::size_t size_t;

    typedef typename Container::reference reference;

    friend class qpdc_iterator< const_iterator, Container >;

    qpdc_iterator( const row_iterator_container& row_iterators, const std::vector< size_t >& columns, size_t row_index ) :
        row_iterators_( row_iterators ), columns_( columns ), row_index_( row_index ) { }
    

    /* implicit conversion to const_iterator */
    template< class Iter >
    qpdc_iterator( const qpdc_iterator< Iter, 
                                        typename std::enable_if< std::is_same< Iter, iterator >::value,
                                                                Container >::type >& nonconst_qpdc_iterator ) :
        row_iterators_( nonconst_qpdc_iterator.row_iterators_ ), columns_( nonconst_qpdc_iterator.columns_ ), row_index_( nonconst_qpdc_iterator.row_index_ ) {}

    qpdc_iterator& operator ++() {
        ++row_index_;
        return *this;
    }

    Iterator operator *() const {
        return row_iterators_[ row_index_ ];
    }

    Iterator operator []( size_t row_index ) const {
        return row_iterators_[ row_index ];
    }

    bool operator ==( const qpdc_iterator< Iterator, Container >& other ) const {
        return row_iterators_[ row_index_ ] == other.row_iterators_[ other.row_index_ ];
    }

    bool operator !=( const qpdc_iterator< Iterator, Container >& other ) const {
        return row_iterators_[ row_index_ ] != other.row_iterators_[ other.row_index_ ];
    }

    size_t row_index() const {
        return row_index_;
    }

    size_t row_size() const {
        return columns_[ row_index_ ];
    }
    
    void set_row( size_t row_index ) {
        row_index_ = row_index;
    }

protected:
    const row_iterator_container& row_iterators_;
    const std::vector< size_t >& columns_;
    size_t row_index_;
};
        



template< class Value_t >
class qpdc_container {
public:
    typedef Value_t value_type;
    typedef std::vector< Value_t > data_container;
    typedef std::vector< typename data_container::iterator > row_iterator_container;

    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;

    typedef qpdc_iterator< iterator, qpdc_container< Value_t > > row_iterator;
    typedef qpdc_iterator< const_iterator, qpdc_container< Value_t > > const_row_iterator;

    typedef Value_t& reference;

private:
    std::vector< typename std::size_t > columns_;
    row_iterator_container row_iterators_;
    data_container data_;

public:
    qpdc_container( const std::vector< size_t >& columns ) : 
        columns_( columns ),
        row_iterators_( columns.size() + 1 ),
        data_( std::accumulate( columns.begin(), columns.end(), 0 ) ) {
        
        row_iterators_[ 0 ] = data_.begin();
        for( auto nr_of_rows = columns.size(), i = decltype( nr_of_rows )( 0 );
             i < nr_of_rows; ++i 
           ) {
            row_iterators_[ i + 1 ] = row_iterators_[ i ] + columns[ i ];
        }
    }

    row_iterator begin() {
        return row_iterator( row_iterators_, columns_, 0 );
    }

    const_row_iterator cbegin() const {
        return const_row_iterator( row_iterators_, columns_, 0 );
    }

    row_iterator end() {
        return row_iterator( row_iterators_, columns_, columns_.size() );
    }

    const_row_iterator cend() const {
        return const_row_iterator( row_iterators_, columns_, columns_.size() );
    }

    row_iterator get_row_iterator( std::size_t row_index ) {
        return row_iterator( row_iterators_, columns_, row_index );
    }

    const_row_iterator cget_row_iterator( std::size_t row_index ) const {
        return const_row_iterator( row_iterators_, columns_, row_index );
    }

    void store( qpdc_container< Value_t >& storage ) const {
        storage.data() = data_;
    }    

    void restore( qpdc_container< Value_t >& storage ) {
        data_ = storage.data();
    }

    data_container& data() {
        return data_;
    }    

};

}
}
#endif
