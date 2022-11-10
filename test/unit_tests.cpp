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

#include "kokkos_cobyla.hpp"
#include <gtest/gtest.h>

using namespace kokkos_cobyla;

namespace math =
#if KOKKOS_VERSION < 30700
    Kokkos::Experimental;
#else
    Kokkos;
#endif

static constexpr double rhobeg = 0.5;
static constexpr double rhoend = 0.001;
static constexpr int maxfun = 2000;

template<
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType
>
KOKKOS_INLINE_FUNCTION
void SimpleQuadratic(
    IntegralType,
    IntegralType,
    SolutionViewType x,
    ScalarType &f,
    ScalarWorkViewType
) {
    f = 10.0 * math::pow(x(0) + 1.0, 2.0) + math::pow(x(1), 2.0);
}

TEST(unit_tests, SimpleQuadratic) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=0;
    Kokkos::View<double*> x("SimpleQuadratic::x", n);
    Kokkos::View<double*> w("SimpleQuadratic::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("SimpleQuadratic::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "SimpleQuadratic::callCobyla",
        1,
        KOKKOS_LAMBDA (const char)
    {
        cobyla(
            n,
            m,
            x,
            rhobeg,
            rhoend,
            maxfun,
            w,
            iact,
            SimpleQuadratic
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    EXPECT_DOUBLE_EQ(-1.0, h_x(0));
    EXPECT_DOUBLE_EQ( 0.0, h_x(1));
}
