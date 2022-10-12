/*
COBYLA---Constrained Optimization BY Linear Approximation.
Copyright (C) 1992 M. J. D. Powell (University of Cambridge)
Copyright (C) 2022 L. T. Meredith

This package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License
 https://www.gnu.org/copyleft/lesser.html
for more details.
*/

#ifndef KOKKOS_COBYLA_IMPL_HPP
#define KOKKOS_COBYLA_IMPL_HPP

#include "kokkos_cobyla.hpp"

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_INLINE_FUNCTION
void cobyla(
    IntegralType n,
    IntegralType m,
    SolutionViewType x,
    ScalarType rhobeg,
    ScalarType rhoend,
    // IntegralType iprint,
    IntegralType maxfun,
    ScalarWorkViewType w,
    IntegralWorkViewType iact
) {

}

#endif // KOKKOS_COBYLA_IMPL_HPP
