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
    using SizeType = typename ScalarWorkViewType::size_type;
    SizeType mpp = m+2;
    SizeType isim = mpp;
    SizeType isimi = isim + n*n + n;
    SizeType idatm = isimi + n*n;
    SizeType ia = idatm + n*mpp + mpp;
    SizeType ivsig = ia + m*n + n;
    SizeType iveta = ivsig + n;
    SizeType isigb = iveta + n;
    SizeType idx = isigb + n;
    SizeType iwork = idx + n;
    SizeType total_size = n*(3*n + 2*m + 11) + 4*m + 6;

    cobylb(
        n,
        m,
        mpp,
        x,
        rhobeg,
        rhoend,
        // iprint,
        maxfun,
        Kokkos::subview(w, Kokkos::make_pair(0, isim)),
        Kokkos::subview(w, Kokkos::make_pair(isim, isimi)),
        Kokkos::subview(w, Kokkos::make_pair(isimi, idatm)),
        Kokkos::subview(w, Kokkos::make_pair(idatm, ia)),
        Kokkos::subview(w, Kokkos::make_pair(ia, ivsig)),
        Kokkos::subview(w, Kokkos::make_pair(ivsig, iveta)),
        Kokkos::subview(w, Kokkos::make_pair(iveta, isigb)),
        Kokkos::subview(w, Kokkos::make_pair(isigb, idx)),
        Kokkos::subview(w, Kokkos::make_pair(idx, iwork)),
        Kokkos::subview(w, Kokkos::make_pair(iwork, total_size)),
        iact
    );
}

#endif // KOKKOS_COBYLA_IMPL_HPP
