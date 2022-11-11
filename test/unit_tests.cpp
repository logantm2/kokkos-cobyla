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
static constexpr double rhoend = 0.0001;
static constexpr int maxfun = 2000;

KOKKOS_INLINE_FUNCTION
void SimpleQuadratic(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1)))
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

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - (-1.0), 2.0);
    l2_error += std::pow(h_x(1) -   0.0 , 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

KOKKOS_INLINE_FUNCTION
void TwoDUnitCircleCalculation(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = x(0) * x(1);
    con(0) = 1.0 - math::pow(x(0), 2.0) - math::pow(x(1), 2.0);
}

TEST(unit_tests, TwoDUnitCircleCalculation) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=1;
    Kokkos::View<double*> x("TwoDUnitCircleCalculation::x", n);
    Kokkos::View<double*> w("TwoDUnitCircleCalculation::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("TwoDUnitCircleCalculation::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "TwoDUnitCircleCalculation::callCobyla",
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
            &TwoDUnitCircleCalculation
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - std::sqrt(0.5), 2.0);
    l2_error += std::pow(h_x(1) + std::sqrt(0.5), 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

KOKKOS_INLINE_FUNCTION
void ThreeDEllipsoidCalculation(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = x(0) * x(1) * x(2);
    con(0) = 1.0 - math::pow(x(0), 2.0) - 2.0*math::pow(x(1), 2.0) - 3.0*math::pow(x(2), 2.0);
}

TEST(unit_tests, ThreeDEllipsoidCalculation) {
    Kokkos::ScopeGuard kokkos;

    int n=3;
    int m=1;
    Kokkos::View<double*> x("ThreeDEllipsoidCalculation::x", n);
    Kokkos::View<double*> w("ThreeDEllipsoidCalculation::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("ThreeDEllipsoidCalculation::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "ThreeDEllipsoidCalculation::callCobyla",
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
            &ThreeDEllipsoidCalculation
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - std::sqrt(1.0/3.0), 2.0);
    l2_error += std::pow(h_x(1) - std::sqrt(1.0/6.0), 2.0);
    l2_error += std::pow(h_x(2) +           1.0/3.0 , 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

KOKKOS_INLINE_FUNCTION
void WeakRosenbrock(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1)))
) {
    f = math::pow(math::pow(x(0), 2.0) - x(1), 2.0) + math::pow(1.0 + x(0), 2.0);
}

TEST(unit_tests, WeakRosenbrock) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=0;
    Kokkos::View<double*> x("WeakRosenbrock::x", n);
    Kokkos::View<double*> w("WeakRosenbrock::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("WeakRosenbrock::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "WeakRosenbrock::callCobyla",
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
            WeakRosenbrock
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) + 1.0, 2.0);
    l2_error += std::pow(h_x(1) - 1.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

KOKKOS_INLINE_FUNCTION
void IntermediateRosenbrock(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1)))
) {
    f = 10.0 * math::pow(math::pow(x(0), 2.0) - x(1), 2.0) + math::pow(1.0 + x(0), 2.0);
}

TEST(unit_tests, IntermediateRosenbrock) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=0;
    Kokkos::View<double*> x("IntermediateRosenbrock::x", n);
    Kokkos::View<double*> w("IntermediateRosenbrock::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("IntermediateRosenbrock::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "IntermediateRosenbrock::callCobyla",
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
            IntermediateRosenbrock
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) + 1.0, 2.0);
    l2_error += std::pow(h_x(1) - 1.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

KOKKOS_INLINE_FUNCTION
void Fletcher9(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = - x(0) - x(1);
    con(0) = x(1) - math::pow(x(0), 2.0);
    con(1) = 1.0 - math::pow(x(0), 2.0) - math::pow(x(1), 2.0);
}

TEST(unit_tests, Fletcher9) {
    Kokkos::ScopeGuard kokkos;

    int n=2;
    int m=2;
    Kokkos::View<double*> x("Fletcher9::x", n);
    Kokkos::View<double*> w("Fletcher9::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("Fletcher9::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "Fletcher9::callCobyla",
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
            Fletcher9
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0) - std::sqrt(0.5), 2.0);
    l2_error += std::pow(h_x(1) - std::sqrt(0.5), 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}

KOKKOS_INLINE_FUNCTION
void Fletcher14(
    int,
    int,
    Kokkos::View<double*> x,
    double &f,
    decltype(Kokkos::subview(x, Kokkos::make_pair(0, 1))) con
) {
    f = x(2);
    con(0) = 5.0 * x(0) - x(1) + x(2);
    con(1) = x(2) - math::pow(x(0), 2.0) - math::pow(x(1), 2.0) - 4.0 * x(1);
    con(2) = x(2) - 5.0 * x(0) - x(1);
}

TEST(unit_tests, Fletcher14) {
    Kokkos::ScopeGuard kokkos;

    int n=3;
    int m=3;
    Kokkos::View<double*> x("Fletcher14::x", n);
    Kokkos::View<double*> w("Fletcher14::w", requiredScalarWorkViewSize(n, m));
    Kokkos::View<int*> iact("Fletcher14::iact", requiredIntegralWorkViewSize(m));

    Kokkos::deep_copy(x, 1.0);

    Kokkos::parallel_for(
        "Fletcher14::callCobyla",
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
            Fletcher14
        );
    });

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);

    double l2_error = 0.0;
    l2_error += std::pow(h_x(0)      , 2.0);
    l2_error += std::pow(h_x(1) + 3.0, 2.0);
    l2_error += std::pow(h_x(1) + 3.0, 2.0);

    const double abs_tol = 1.e-2;

    EXPECT_NEAR(0.0, l2_error, abs_tol);
}
