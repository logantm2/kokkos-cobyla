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

#ifndef KOKKOS_COBYLA_HPP
#define KOKKOS_COBYLA_HPP

#if __cplusplus == 199711L || __cplusplus == 201103L || __cplusplus == 201402L
#define KOKKOS_COBYLA_CONSTEXPR_IF if
#else
#define KOKKOS_COBYLA_CONSTEXPR_IF if constexpr
#endif

#include <Kokkos_Core.hpp>

namespace kokkos_cobyla {

// LTM since the original F77 code indexes from 1,
// encapsulate subtracting one from array indices here.
// I don't wanna bother rewriting the code to not assume indexing from 1.
// The right way to do this is probably to use Kokkos OffsetViews,
// but I'm not confident how well those play with SubViews.
template<typename IntegralType>
KOKKOS_INLINE_FUNCTION
IntegralType m1(IntegralType i) {
    return i - 1;
}

template <
    typename IntegralType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType
>
KOKKOS_INLINE_FUNCTION
void trstlp(
    IntegralType n,
    IntegralType m,
    ScalarWorkViewType a_flat,
    ScalarWorkViewType b,
    ScalarType rho,
    ScalarWorkViewType dx,
    IntegralType &ifull,
    IntegralWorkViewType iact,
    ScalarWorkViewType z_flat,
    ScalarWorkViewType zdota,
    ScalarWorkViewType vmultc,
    ScalarWorkViewType sdirn,
    ScalarWorkViewType dxnew,
    ScalarWorkViewType vmultd
) {
    // This subroutine calculates an N-component vector DX by applying the
    // following two stages. In the first stage, DX is set to the shortest
    // vector that minimizes the greatest violation of the constraints
    // A(1,K)*DX(1)+A(2,K)*DX(2)+...+A(N,K)*DX(N) .GE. B(K), K=2,3,...,M,
    // subject to the Euclidean length of DX being at most RHO. If its length is
    // strictly less than RHO, then we use the resultant freedom in DX to
    // minimize the objective function
    //         -A(1,M+1)*DX(1)-A(2,M+1)*DX(2)-...-A(N,M+1)*DX(N)
    // subject to no increase in any greatest constraint violation. This
    // notation allows the gradient of the objective function to be regarded as
    // the gradient of a constraint. Therefore the two stages are distinguished
    // by MCON .EQ. M and MCON .GT. M respectively. It is possible that a
    // degeneracy may prevent DX from attaining the target length RHO. Then the
    // value IFULL=0 would be set, but usually IFULL=1 on return.

    // In general NACT is the number of constraints in the active set and
    // IACT(1),...,IACT(NACT) are their indices, while the remainder of IACT
    // contains a permutation of the remaining constraint indices. Further, Z is
    // an orthogonal matrix whose first NACT columns can be regarded as the
    // result of Gram-Schmidt applied to the active constraint gradients. For
    // J=1,2,...,NACT, the number ZDOTA(J) is the scalar product of the J-th
    // column of Z with the gradient of the J-th active constraint. DX is the
    // current vector of variables and here the residuals of the active
    // constraints should be zero. Further, the active constraints have
    // nonnegative Lagrange multipliers that are held at the beginning of
    // VMULTC. The remainder of this vector holds the residuals of the inactive
    // constraints at DX, the ordering of the components of VMULTC being in
    // agreement with the permutation of the indices of the constraints that is
    // in IACT. All these residuals are nonnegative, which is achieved by the
    // shift RESMAX that makes the least residual zero.all the active
    // constraint violations by one simultaneously.

    namespace math =
#if KOKKOS_VERSION < 30700
        Kokkos::Experimental;
#else
        Kokkos;
#endif

    // LTM Wrap unmanaged Views around the flattened Views
    // so that we can do 2D indexing.
    Kokkos::View<
        typename ScalarWorkViewType::value_type**,
        typename ScalarWorkViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    >
    a(a_flat.data(), n, m+1),
    z(z_flat.data(), n, n);

    // LTM F77 implicitly declares all of these variables.
    // We gotta explicitly declare them.
    IntegralType
        mcon,
        nact,
        icon,
        i,
        j,
        k,
        nactx,
        isave,
        kk,
        kw,
        kp,
        kl,
        // iout,
        icount;
    ScalarType
        resmax,
        optold,
        optnew,
        tot,
        temp,
        alpha,
        beta,
        sp,
        spabs,
        acca,
        accb,
        ratio,
        zdotv,
        zdvabs,
        vsave,
        dd,
        ss,
        sd,
        stpful,
        step,
        zdotw,
        zdwabs,
        resold,
        sumabs,
        sum,
        tempa;

    // Initialize Z and some other variables. The value of RESMAX will be
    // appropriate to DX=0, while ICON will be the index of a most violated
    // constraint if RESMAX is positive. Usually during the first stage the
    // vector SDIRN gives a search direction that reduces
    ifull = 1;
    mcon = m;
    nact = 0;
    resmax = 0.0;
    icon = 0;
    for (i=1; i<=n; i++) {
        for (j=1; j<=n; j++) {
            z(m1(i),m1(j)) = 0.0;
        }
        z(m1(i),m1(i)) = 1.0;
        dx(m1(i)) = 0.0;
    }
    if (m >= 1) {
        for (k=1; k<=m; k++) {
            if (b(m1(k)) > resmax) {
                resmax = b(m1(k));
                icon = k;
            }
        }
        for (k=1; k<=m; k++) {
            iact(m1(k)) = k;
            vmultc(m1(k)) = resmax - b(m1(k));
        }
    }
    if (resmax == 0.0) goto line_480;
    for (i=1; i<=n; i++) {
        sdirn(m1(i)) = 0.0;
    }

    // End the current stage of the calculation if 3 consecutive iterations
    // have either failed to reduce the best calculated value of the objective
    // function or to increase the number of active constraints since the best
    // value was calculated. This strategy prevents cycling, but there is a
    // remote possibility that it will cause premature termination.
line_60:
    optold = 0.0;
    icount = 0;
line_70:
    if (mcon == m) {
        optnew = resmax;
    }
    else {
        optnew = 0.0;
        for (i=1; i<=n; i++) {
            optnew = optnew - dx(m1(i))*a(m1(i),m1(mcon));
        }
    }
    nactx = 0;
    if (icount == 0 or optnew < optold) {
        optold = optnew;
        nactx = nact;
        icount = 3;
    }
    else if (nact > nactx) {
        nactx = nact;
        icount = 3;
    }
    else {
        icount = icount-1;
        if (icount == 0) goto line_490;
    }

    // If ICON exceeds NACT, then we add the constraint with index IACT(ICON) to
    // the active set. Apply Givens rotations so that the last N-NACT-1 columns
    // of Z are orthogonal to the gradient of the new constraint, a scalar
    // product being set to zero if its nonzero value could be due to computer
    // rounding errors. The array DXNEW is used for working space.
    if (icon <= nact) goto line_260;
    kk = iact(m1(icon));
    for (i=1; i<=n; i++) {
        dxnew(m1(i)) = a(m1(i),m1(kk));
    }
    tot = 0.0;
    k = n;
line_100:
    if (k > nact) {
        sp = 0.0;
        spabs = 0.0;
        for (i=1; i<=n; i++) {
            temp = z(m1(i),m1(k))*dxnew(m1(i));
            sp = sp + temp;
            spabs = spabs + math::fabs(temp);
        }
        acca = spabs + 0.1 * math::fabs(sp);
        accb = spabs + 0.2 * math::fabs(sp);
        if (spabs >= acca or acca >= accb) sp = 0.0;
        if (tot == 0.0) {
            tot = sp;
        }
        else {
            kp = k+1;
            temp = math::sqrt(sp*sp + tot*tot);
            alpha = sp/temp;
            beta = tot/temp;
            tot = temp;
            for (i=1; i<=n; i++) {
                temp = alpha * z(m1(i),m1(k)) + beta * z(m1(i),m1(kp));
                z(m1(i),m1(kp)) = alpha * z(m1(i),m1(kp)) - beta * z(m1(i),m1(k));
                z(m1(i),m1(k)) = temp;
            }
        }
        k = k-1;
        goto line_100;
    }

    // Add the new constraint if this can be done without a deletion from the
    // active set.
    if (tot != 0.0) {
        nact = nact+1;
        zdota(m1(nact)) = tot;
        vmultc(m1(icon)) = vmultc(m1(nact));
        vmultc(m1(nact)) = 0.0;
        goto line_210;
    }

    // The next instruction is reached if a deletion has to be made from the
    // active set in order to make room for the new active constraint, because
    // the new constraint gradient is a linear combination of the gradients of
    // the old active constraints. Set the elements of VMULTD to the multipliers
    // of the linear combination. Further, set IOUT to the index of the
    // constraint to be deleted, but branch if no suitable index can be found.
    ratio = -1.0;
    k = nact;
line_130:
    zdotv = 0.0;
    zdvabs = 0.0;
    for (i=1; i<=n; i++) {
        temp = z(m1(i),m1(k))*dxnew(m1(i));
        zdotv = zdotv + temp;
        zdvabs = zdvabs + math::fabs(temp);
    }
    acca = zdvabs + 0.1*math::fabs(zdotv);
    accb = zdvabs + 0.2*math::fabs(zdotv);
    if (zdvabs < acca and acca < accb) {
        temp = zdotv/zdota(m1(k));
        if (temp > 0.0 and iact(m1(k)) <= m) {
            tempa = vmultc(m1(k))/temp;
            if (ratio < 0.0 or tempa < ratio) {
                ratio = tempa;
                // iout = k;
            }
        }
        if (k >= 2) {
            kw = iact(m1(k));
            for (i=1; i<=n; i++) {
                dxnew(m1(i)) = dxnew(m1(i)) - temp*a(m1(i),m1(kw));
            }
        }
        vmultd(m1(k)) = temp;
    }
    else {
        vmultd(m1(k)) = 0.0;
    }
    k = k-1;
    if (k > 0) goto line_130;
    if (ratio < 0.0) goto line_490;

    // Revise the Lagrange multipliers and reorder the active constraints so
    // that the one to be replaced is at the end of the list. Also calculate the
    // new value of ZDOTA(NACT) and branch if it is not acceptable.
    for (k=1; k<=nact; k++) {
        vmultc(m1(k)) = math::fmax(0.0, vmultc(m1(k)) - ratio*vmultd(m1(k)));
    }
    if (icon < nact) {
        isave = iact(m1(icon));
        vsave = vmultc(m1(icon));
        k = icon;
line_170:
        kp = k + 1;
        kw = iact(m1(kp));
        sp = 0.0;
        for (i=1; i<=n; i++) {
            sp = sp + z(m1(i),m1(k)) * a(m1(i),m1(kw));
        }
        temp = math::sqrt(sp*sp + math::pow(zdota(m1(kp)), 2.0));
        alpha = zdota(m1(kp))/temp;
        beta = sp/temp;
        zdota(m1(kp)) = alpha*zdota(m1(k));
        zdota(m1(k)) = temp;
        for (i=1; i<=n; i++) {
            temp = alpha*z(m1(i),m1(kp)) + beta*z(m1(i),m1(k));
            z(m1(i),m1(kp)) = alpha*z(m1(i),m1(k)) - beta*z(m1(i),m1(kp));
            z(m1(i),m1(k)) = temp;
        }
        iact(m1(k)) = kw;
        vmultc(m1(k)) = vmultc(m1(kp));
        k = kp;
        if (k < nact) goto line_170;
        iact(m1(k)) = isave;
        vmultc(m1(k)) = vsave;
    }
    temp = 0.0;
    for (i=1; i<=n; i++) {
        temp = temp + z(m1(i),m1(nact))*a(m1(i),m1(kk));
    }
    if (temp == 0.0) goto line_490;
    zdota(m1(nact)) = temp;
    vmultc(m1(icon)) = 0.0;
    vmultc(m1(nact)) = ratio;

    // Update IACT and ensure that the objective function continues to be
    // treated as the last active constraint when MCON>M.
line_210:
    iact(m1(icon)) = iact(m1(nact));
    iact(m1(nact)) = kk;
    if (mcon > m and kk != mcon) {
        k = nact-1;
        sp = 0.0;
        for (i=1; i<=n; i++) {
            sp = sp + z(m1(i),m1(k)) * a(m1(i),m1(kk));
        }
        temp = math::sqrt(sp*sp + zdota(m1(nact))*zdota(m1(nact)));
        alpha = zdota(m1(nact)) / temp;
        beta = sp / temp;
        zdota(m1(nact)) = alpha*zdota(m1(k));
        zdota(m1(k)) = temp;
        for (i=1; i<=n; i++) {
            temp = alpha * z(m1(i),m1(nact)) + beta*z(m1(i),m1(k));
            z(m1(i),m1(nact)) = alpha*z(m1(i),m1(k)) - beta*z(m1(i),m1(nact));
            z(m1(i),m1(k)) = temp;
        }
        iact(m1(nact)) = iact(m1(k));
        iact(m1(k)) = kk;
        temp = vmultc(m1(k));
        vmultc(m1(k)) = vmultc(m1(nact));
        vmultc(m1(nact)) = temp;
    }

    // If stage one is in progress, then set SDIRN to the direction of the next
    // change to the current vector of variables.
    if (mcon > m) goto line_320;
    kk = iact(m1(nact));
    temp = 0.0;
    for (i=1; i<=n; i++) {
        temp = temp + sdirn(m1(i))*a(m1(i),m1(kk));
    }
    temp = temp - 1.0;
    temp = temp / zdota(m1(nact));
    for (i=1; i<=n; i++) {
        sdirn(m1(i)) = sdirn(m1(i)) - temp*z(m1(i),m1(nact));
    }
    goto line_340;

    // Delete the constraint that has the index IACT(ICON) from the active set.
line_260:
    if (icon < nact) {
        isave = iact(m1(icon));
        vsave = vmultc(m1(icon));
        k = icon;
line_270:
        kp = k + 1;
        kk = iact(m1(kp));
        sp = 0.0;
        for (i=1; i<=n; i++) {
            sp = sp + z(m1(i),m1(k))*a(m1(i),m1(kk));
        }
        temp = math::sqrt(sp*sp + math::pow(zdota(m1(kp)), 2.0));
        alpha = zdota(m1(kp)) / temp;
        beta = sp / temp;
        zdota(m1(kp)) = alpha*zdota(m1(k));
        zdota(m1(k)) = temp;
        for (i=1; i<=n; i++) {
            temp = alpha*z(m1(i),m1(kp)) + beta*z(m1(i),m1(k));
            z(m1(i),m1(kp)) = alpha*z(m1(i),m1(k)) - beta*z(m1(i),m1(kp));
            z(m1(i),m1(k)) = temp;
        }
        iact(m1(k)) = kk;
        vmultc(m1(k)) = vmultc(m1(kp));
        k = kp;
        if (k < nact) goto line_270;
        iact(m1(k)) = isave;
        vmultc(m1(k)) = vsave;
    }
    nact = nact - 1;

    // If stage one is in progress, then set SDIRN to the direction of the next
    // change to the current vector of variables.
    if (mcon > m) goto line_320;
    temp = 0.0;
    for (i=1; i<=n; i++) {
        temp = temp + sdirn(m1(i))*z(m1(i), m1(nact+1));
    }
    for (i=1; i<=n; i++) {
        sdirn(m1(i)) = sdirn(m1(i)) - temp*z(m1(i), m1(nact+1));
    }
    goto line_340;

    // Pick the next search direction of stage two.
line_320:
    temp = 1.0 / zdota(m1(nact));
    for (i=1; i<=n; i++) {
        sdirn(m1(i)) = temp*z(m1(i), m1(nact));
    }

    // Calculate the step to the boundary of the trust region or take the step
    // that reduces RESMAX to zero. The two statements below that include the
    // factor 1.0E-6 prevent some harmless underflows that occurred in a test
    // calculation. Further, we skip the step if it could be zero within a
    // reasonable tolerance for computer rounding errors.
line_340:
    dd = rho*rho;
    sd = 0.0;
    ss = 0.0;
    for (i=1; i<=n; i++) {
        if (math::fabs(dx(m1(i))) >= 1.0e-6*rho) dd = dd - math::pow(dx(m1(i)), 2.0);
        sd = sd + dx(m1(i))*sdirn(m1(i));
        ss = ss + math::pow(sdirn(m1(i)), 2.0);
    }
    if (dd <= 0.0) goto line_490;
    temp = math::sqrt(ss*dd);
    if (math::fabs(sd) >= 1.0e-6*temp) temp = math::sqrt(ss*dd + sd*sd);
    stpful = dd / (temp + sd);
    step = stpful;
    if (mcon == m) {
        acca = step + 0.1*resmax;
        accb = step + 0.2*resmax;
        if (step >= acca or acca >= accb) goto line_480;
        step = math::fmin(step, resmax);
    }

    // Set DXNEW to the new variables if STEP is the steplength, and reduce
    // RESMAX to the corresponding maximum residual if stage one is being done.
    // Because DXNEW will be changed during the calculation of some Lagrange
    // multipliers, it will be restored to the following value later.
    for (i=1; i<=n; i++) {
        dxnew(m1(i)) = dx(m1(i)) + step*sdirn(m1(i));
    }
    if (mcon == m) {
        resold = resmax;
        resmax = 0.0;
        for (k=1; k<=nact; k++) {
            kk = iact(m1(k));
            temp = b(m1(kk));
            for (i=1; i<=n; i++) {
                temp = temp - a(m1(i),m1(kk))*dxnew(m1(i));
            }
            resmax = math::fmax(resmax, temp);
        }
    }

    // Set VMULTD to the VMULTC vector that would occur if DX became DXNEW. A
    // device is included to force VMULTD(K)=0.0 if deviations from this value
    // can be attributed to computer rounding errors. First calculate the new
    // Lagrange multipliers.
    k = nact;
line_390:
    zdotw = 0.0;
    zdwabs = 0.0;
    for (i=1; i<=n; i++) {
        temp = z(m1(i),m1(k))*dxnew(m1(i));
        zdotw = zdotw + temp;
        zdwabs = zdwabs + math::fabs(temp);
    }
    acca = zdwabs + 0.1*math::fabs(zdotw);
    accb = zdwabs + 0.2*math::fabs(zdotw);
    if (zdwabs >= acca or acca >= accb) zdotw = 0.0;
    vmultd(m1(k)) = zdotw/zdota(m1(k));
    if (k >= 2) {
        kk = iact(m1(k));
        for (i=1; i<=n; i++) {
            dxnew(m1(i)) = dxnew(m1(i)) - vmultd(m1(k))*a(m1(i),m1(kk));
        }
        k = k-1;
        goto line_390;
    }
    if (mcon > m) vmultd(m1(nact)) = math::fmax(0.0, vmultd(m1(nact)));

    // Complete VMULTC by finding the new constraint residuals.
    for (i=1; i<=n; i++) {
        dxnew(m1(i)) = dx(m1(i)) + step*sdirn(m1(i));
    }
    if (mcon > nact) {
        kl = nact + 1;
        for (k=kl; k<=mcon; k++) {
            kk = iact(m1(k));
            sum = resmax-b(m1(kk));
            sumabs = resmax + math::fabs(b(m1(kk)));
            for (i=1; i<=n; i++) {
                temp = a(m1(i),m1(kk))*dxnew(m1(i));
                sum = sum + temp;
                sumabs = sumabs + math::fabs(temp);
            }
            acca = sumabs + 0.1*math::fabs(sum);
            accb = sumabs + 0.2*math::fabs(sum);
            if (sumabs >= acca or acca >= accb) sum = 0.0;
            vmultd(m1(k)) = sum;
        }
    }

    // Calculate the fraction of the step from DX to DXNEW that will be taken.
    ratio = 1.0;
    icon = 0;
    for (k=1; k<=mcon; k++) {
        if (vmultd(m1(k)) < 0.0) {
            temp = vmultc(m1(k))/(vmultc(m1(k)) - vmultd(m1(k)));
            if (temp < ratio) {
                ratio = temp;
                icon = k;
            }
        }
    }

    // Update DX, VMULTC and RESMAX.
    temp = 1.0 - ratio;
    for (i=1; i<=n; i++) {
        dx(m1(i)) = temp*dx(m1(i)) + ratio*dxnew(m1(i));
    }
    for (k=1; k<=mcon; k++) {
        vmultc(m1(k)) = math::fmax(0.0, temp*vmultc(m1(k)) + ratio*vmultd(m1(k)));
    }
    if (mcon == m) resmax = resold + ratio*(resmax - resold);

    // If the full step is not acceptable then begin another iteration.
    // Otherwise switch to stage two or end the calculation.
    if (icon > 0) goto line_70;
    if (step == stpful) goto line_500;
line_480:
    mcon = m + 1;
    icon = mcon;
    iact(m1(mcon)) = mcon;
    vmultc(m1(mcon)) = 0.0;
    goto line_60;

    // We employ any freedom that may be available to reduce the objective
    // function before returning a DX whose length is less than RHO.
line_490:
    if (mcon == m) goto line_480;
    ifull = 0;
line_500:
    return;
}

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarSubViewType,
    typename IntegralWorkViewType,
    int iprint
>
KOKKOS_INLINE_FUNCTION
void cobylb(
    IntegralType n,
    IntegralType m,
    IntegralType mpp,
    SolutionViewType x,
    ScalarType rhobeg,
    ScalarType rhoend,
    IntegralType maxfun,
    ScalarSubViewType con,
    ScalarSubViewType sim_flat,
    ScalarSubViewType simi_flat,
    ScalarSubViewType datmat_flat,
    ScalarSubViewType a_flat,
    ScalarSubViewType vsig,
    ScalarSubViewType veta,
    ScalarSubViewType sigbar,
    ScalarSubViewType dx,
    ScalarSubViewType w,
    IntegralWorkViewType iact,
    void (*calcfc) (
        IntegralType n_in,
        IntegralType m_in,
        SolutionViewType x_in,
        ScalarType &f_in,
        ScalarSubViewType con_in
    )
) {
    // Wrap unmanaged Views around the flattened Views
    // so that we can do 2D indexing.
    Kokkos::View<
        typename ScalarSubViewType::value_type**,
        typename ScalarSubViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    > sim(sim_flat.data(), n, n+1);
    Kokkos::View<
        typename ScalarSubViewType::value_type**,
        typename ScalarSubViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    > simi(simi_flat.data(), n, n);
    Kokkos::View<
        typename ScalarSubViewType::value_type**,
        typename ScalarSubViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    > datmat(datmat_flat.data(), mpp, n+1);
    Kokkos::View<
        typename ScalarSubViewType::value_type**,
        typename ScalarSubViewType::memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    > a(a_flat.data(), n, m+1);

    // LTM F77 implicitly declares all of these.
    // We gotta explicitly declare them.
    IntegralType
        i,
        j,
        k,
        nbest,
        l,
        iflag,
        izdota,
        ivmc,
        isdirn,
        idxnew,
        ivmd,
        ifull;
    ScalarType
        resmax,
        phimin,
        tempa,
        error,
        parsig,
        pareta,
        wsig,
        weta,
        cvmaxp,
        cvmaxm,
        sum,
        dxsign,
        resnew,
        barmu,
        phi,
        prerec,
        prerem,
        vmold,
        vmnew,
        trured,
        ratio,
        edgmax,
        denom,
        cmin,
        cmax,
        f;

    namespace math =
#if KOKKOS_VERSION < 30700
        Kokkos::Experimental;
#else
        Kokkos;
#endif

    // Set the initial values of some parameters. The last column of SIM holds
    // the optimal vertex of the current simplex, and the preceding N columns
    // hold the displacements from the optimal vertex to the other vertices.
    // Further, SIMI holds the inverse of the matrix that is contained in the
    // first N columns of SIM.
    // IntegralType iptem = n < 5 ? n : 5;
    // IntegralType iptemp = iptem + 1;
    IntegralType np = n + 1;
    IntegralType mp = m + 1;
    ScalarType alpha = 0.25;
    ScalarType beta  = 2.1;
    ScalarType gamma = 0.5;
    ScalarType delta = 1.1;
    ScalarType rho = rhobeg;
    ScalarType parmu = 0.0;
    KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 2) {
        printf(
            "\n   The initial value of RHO is%13.6e  and PARMU is set to zero.",
            rho
        );
    }
    IntegralType iptem = n < 5 ? n : 5;
    IntegralType iptemp = iptem+1;

    IntegralType nfvals = 0;
    ScalarType temp = 1.0/rho;
    for (i=1; i<=n; i++) {
        sim(m1(i), m1(np)) = x(m1(i));
        for (j=1; j<=n; j++) {
            simi(m1(i),m1(j)) = 0.0;
        }
        sim(m1(i),m1(i)) = rho;
        simi(m1(i),m1(i)) = temp;
    }
    IntegralType jdrop = np;
    IntegralType ibrnch = 0;

    // Make the next call of the user-supplied subroutine CALCFC. These
    // instructions are also used for calling CALCFC during the iterations of
    // the algorithm.
line_40:
    if (nfvals >= maxfun and nfvals > 0) {
        KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 1) {
            printf("\n   Return from subroutine COBYLA because the MAXFUN limit has been reached.");
        }
        goto line_600;
    }
    nfvals = nfvals + 1;
    (*calcfc)(n, m, x, f, con);
    resmax = 0.0;
    if (m > 0) {
        for (k=1; k<=m; k++) {
            resmax = math::fmax(resmax, -con(m1(k)));
        }
    }
    KOKKOS_COBYLA_CONSTEXPR_IF ((nfvals == iprint-1 and iprint >=1) or iprint == 3) {
        printf(
            "\n   NFVALS =%5d   F =%13.6e    MAXCV =%13.6e\n   X =",
            nfvals,
            f,
            resmax
        );
        for (i=1; i<=iptem; i++) {
            printf("%13.6e  ", x(m1(i)));
        }
        if (iptem < n) {
            printf("\n      ");
            for (i=iptemp; i<=n; i++) {
                printf("%13.6e  ", x(m1(i)));
            }
        }
    }
    con(m1(mp)) = f;
    con(m1(mpp)) = resmax;
    if (ibrnch == 1) {
        goto line_440;
    }

    // Set the recently calculated function values in a column of DATMAT. This
    // array has a column for each vertex of the current simplex, the entries of
    // each column being the values of the constraint functions (if any)
    // followed by the objective function and the greatest constraint violation
    // at the vertex.
    for (k=1; k<=mpp; k++) {
        datmat(m1(k), m1(jdrop)) = con(m1(k));
    }
    if (nfvals > np) {
        goto line_130;
    }

    // Exchange the new vertex of the initial simplex with the optimal vertex if
    // necessary. Then, if the initial simplex is not complete, pick its next
    // vertex and calculate the function values there.
    if (jdrop <= n) {
        if (datmat(m1(mp),m1(np)) <= f) {
            x(m1(jdrop)) = sim(m1(jdrop), m1(np));
        }
        else {
            sim(m1(jdrop), m1(np)) = x(m1(jdrop));
            for (k=1; k<=mpp; k++) {
                datmat(m1(k), m1(jdrop)) = datmat(m1(k), m1(np));
                datmat(m1(k), m1(np)) = con(m1(k));
            }
            for (k=1; k<=jdrop; k++) {
                sim(m1(jdrop), m1(k)) = -rho;
                temp = 0.0;
                for (i=k; i<=jdrop; i++) {
                    temp = temp - simi(m1(i),m1(k));
                }
                simi(m1(jdrop), m1(k)) = temp;
            }
        }
    }
    if (nfvals <= n) {
        jdrop = nfvals;
        x(m1(jdrop)) = x(m1(jdrop)) + rho;
        goto line_40;
    }
line_130:
    ibrnch=1;

    // Identify the optimal vertex of the current simplex.
line_140:
    phimin = datmat(m1(mp), m1(np)) + parmu * datmat(m1(mpp),m1(np));
    nbest = np;
    for (j=1; j<=n; j++) {
        temp = datmat(m1(mp),m1(j)) + parmu * datmat(m1(mpp), m1(j));
        if (temp < phimin) {
            nbest = j;
            phimin = temp;
        }
        else if (temp == phimin and parmu == 0.0) {
            if (datmat(m1(mpp), m1(j)) < datmat(m1(mpp), m1(nbest))) {
                nbest = j;
            }
        }
    }

    // Switch the best vertex into pole position if it is not there already,
    // and also update SIM, SIMI and DATMAT.
    if (nbest <= n) {
        for (i=1; i<=mpp; i++) {
            temp = datmat(m1(i), m1(np));
            datmat(m1(i), m1(np)) = datmat(m1(i), m1(nbest));
            datmat(m1(i), m1(nbest)) = temp;
        }
        for (i=1; i<=n; i++) {
            temp = sim(m1(i), m1(nbest));
            sim(m1(i), m1(nbest)) = 0.0;
            sim(m1(i), m1(np)) = sim(m1(i), m1(np)) + temp;
            tempa = 0.0;
            for (k=1; k<=n; k++) {
                sim(m1(i), m1(k)) = sim(m1(i), m1(k)) - temp;
                tempa = tempa - simi(m1(k), m1(i));
            }
            simi(m1(nbest), m1(i)) = tempa;
        }
    }

    // Make an error return if SIGI is a poor approximation to the inverse of
    // the leading N by N submatrix of SIG.
    error = 0.0;
    for (i=1; i<=n; i++) {
        for (j=1; j<=n; j++) {
            temp = 0.0;
            if (i == j) {
                temp = temp - 1.0;
            }
            for (k=1; k<=n; k++) {
                temp = temp + simi(m1(i),m1(k))*sim(m1(k),m1(j));
            }
            error = math::fmax(error, math::fabs(temp));
        }
    }
    if (error > 0.1) {
        KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 1) printf(
            "\n   Return from subroutine COBYLA because rounding errors are becoming damaging."
        );
        goto line_600;
    }

    // Calculate the coefficients of the linear approximations to the objective
    // and constraint functions, placing minus the objective function gradient
    // after the constraint gradients in the array A. The vector W is used for
    // working space.
    for (k=1; k<=mp; k++) {
        con(m1(k)) = -datmat(m1(k), m1(np));
        for (j=1; j<=n; j++) {
            w(m1(j)) = datmat(m1(k),m1(j)) + con(m1(k));
        }
        for (i=1; i<=n; i++) {
            temp = 0.0;
            for (j=1; j<=n; j++) {
                temp = temp + w(m1(j))*simi(m1(j), m1(i));
            }
            if (k == mp) {
                temp = -temp;
            }
            a(m1(i),m1(k)) = temp;
        }
    }

    // Calculate the values of sigma and eta, and set IFLAG=0 if the current
    // simplex is not acceptable.
    iflag = 1;
    parsig = alpha*rho;
    pareta = beta*rho;
    for (j=1; j<=n; j++) {
        wsig = 0.0;
        weta = 0.0;
        for (i=1; i<=n; i++) {
            wsig = wsig + simi(m1(j),m1(i))*simi(m1(j),m1(i));
            weta = weta + sim(m1(i),m1(j)) *sim(m1(i),m1(j));
        }
        vsig(m1(j)) = 1.0 / math::sqrt(wsig);
        veta(m1(j)) = math::sqrt(weta);
        if (vsig(m1(j)) < parsig or veta(m1(j)) > pareta) {
            iflag = 0;
        }
    }

    // If a new vertex is needed to improve acceptability, then decide which
    // vertex to drop from the simplex.
    if (ibrnch == 1 or iflag == 1) {
        goto line_370;
    }
    jdrop = 0;
    temp = pareta;
    for (j=1; j<=n; j++) {
        if (veta(m1(j)) > temp) {
            jdrop = j;
            temp = veta(m1(j));
        }
    }
    if (jdrop == 0) {
        for (j=1; j<=n; j++) {
            if (vsig(m1(j)) < temp) {
                jdrop = j;
                temp = vsig(j);
            }
        }
    }

    // Calculate the step to the new vertex and its sign.
    temp = gamma*rho*vsig(m1(jdrop));
    for (i=1; i<=n; i++) {
        dx(m1(i)) = temp*simi(m1(jdrop), m1(i));
    }
    cvmaxp = 0.0;
    cvmaxm = 0.0;
    sum = 0.0;
    for (k=1; k<=mp; k++) {
        sum = 0.0;
        for (i=1; i<=n; i++) {
            sum = sum + a(m1(i),m1(k))*dx(m1(i));
        }
        if (k < mp) {
            temp = datmat(m1(k), m1(np));
            cvmaxp = math::fmax(cvmaxp, -sum-temp);
            cvmaxm = math::fmax(cvmaxm,  sum-temp);
        }
    }
    dxsign = 1.0;
    if (parmu*(cvmaxp-cvmaxm) > sum+sum) {
        dxsign = -1.0;
    }

    // Update the elements of SIM and SIMI, and set the next X.
    temp = 0.0;
    for (i=1; i<=n; i++) {
        dx(m1(i)) = dxsign*dx(m1(i));
        sim(m1(i),m1(jdrop)) = dx(m1(i));
        temp = temp + simi(m1(jdrop),m1(i))*dx(m1(i));
    }
    for (i=1; i<=n; i++) {
        simi(m1(jdrop),m1(i)) = simi(m1(jdrop),m1(i))/temp;
    }
    for (j=1; j<=n; j++) {
        if (j != jdrop) {
            temp = 0.0;
            for (i=1; i<=n; i++) {
                temp = temp + simi(m1(j),m1(i))*dx(m1(i));
            }
            for (i=1; i<=n; i++) {
                simi(m1(j),m1(i)) = simi(m1(j),m1(i)) - temp*simi(m1(jdrop),m1(i));
            }
        }
        x(m1(j)) = sim(m1(j),m1(np)) + dx(m1(j));
    }
    goto line_40;

    // Calculate DX=x(*)-x(0). Branch if the length of DX is less than 0.5*RHO.
line_370:
    izdota = n*n;
    ivmc = izdota + n;
    isdirn = ivmc + mp;
    idxnew = isdirn + n;
    ivmd = idxnew + n;
    ifull = 0;
    trstlp(
        n,
        m,
        a_flat,
        con,
        rho,
        dx,
        ifull,
        iact,
        Kokkos::subview(w, Kokkos::make_pair(0, izdota)),
        Kokkos::subview(w, Kokkos::make_pair(izdota, ivmc)),
        Kokkos::subview(w, Kokkos::make_pair(ivmc, isdirn)),
        Kokkos::subview(w, Kokkos::make_pair(isdirn, idxnew)),
        Kokkos::subview(w, Kokkos::make_pair(idxnew, ivmd)),
        Kokkos::subview(w, Kokkos::make_pair(ivmd, ivmd+m))
    );
    if (ifull == 0) {
        temp = 0.0;
        for (i=1; i<=n; i++) {
            temp = temp + math::pow(dx(m1(i)), 2.0);
        }
        if (temp < 0.25 * rho * rho) {
            ibrnch=1;
            goto line_550;
        }
    }

    // Predict the change to F and the new maximum constraint violation if the
    // variables are altered from x(0) to x(0)+DX.
    resnew = 0.0;
    con(m1(mp)) = 0.0;
    for (k=1; k<=mp; k++) {
        sum = con(m1(k));
        for (i=1; i<=n; i++) {
            sum = sum - a(m1(i),m1(k))*dx(m1(i));
        }
        if (k < mp) {
            resnew = math::fmax(resnew, sum);
        }
    }

    // Increase PARMU if necessary and branch back if this change alters the
    // optimal vertex. Otherwise PREREM and PREREC will be set to the predicted
    // reductions in the merit function and the maximum constraint violation
    // respectively.
    barmu = 0.0;
    prerec = datmat(m1(mpp), m1(np)) - resnew;
    if (prerec > 0.0) {
        barmu = sum/prerec;
    }
    if (parmu < 1.5 * barmu) {
        parmu = 2.0*barmu;
        KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 2) printf(
            "\n   Increase in PARMU to%13.6e",
            parmu
        );
        phi = datmat(m1(mp),m1(np)) + parmu*datmat(m1(mpp),m1(np));
        for (j=1; j<=n; j++) {
            temp = datmat(m1(mp),m1(j)) + parmu*datmat(m1(mpp),m1(j));
            if (temp < phi) {
                goto line_140;
            }
            if (temp == phi and parmu == 0.0) {
                if (datmat(m1(mpp),m1(j)) < datmat(m1(mpp),m1(np))) {
                    goto line_140;
                }
            }
        }
    }
    prerem = parmu*prerec - sum;

    // Calculate the constraint and objective functions at x(*). Then find the
    // actual reduction in the merit function.
    for (i=1; i<=n; i++) {
        x(m1(i)) = sim(m1(i),m1(np)) + dx(m1(i));
    }
    ibrnch=1;
    goto line_40;
line_440:
    vmold = datmat(m1(mp),m1(np)) + parmu*datmat(m1(mpp),m1(np));
    vmnew = f + parmu*resmax;
    trured = vmold - vmnew;
    if (parmu == 0.0 and f == datmat(m1(mp),m1(np))) {
        prerem = prerec;
        trured = datmat(m1(mpp),m1(np)) - resmax;
    }

    // Begin the operations that decide whether x(*) should replace one of the
    // vertices of the current simplex, the change being mandatory if TRURED is
    // positive. Firstly, JDROP is set to the index of the vertex that is to be
    // replaced.
    ratio = 0.0;
    if (trured <= 0.0) {
        ratio = 1.0;
    }
    jdrop = 0;
    for (j=1; j<=n; j++) {
        temp = 0.0;
        for (i=1; i<=n; i++) {
            temp = temp + simi(m1(j),m1(i))*dx(m1(i));
        }
        temp = math::fabs(temp);
        if (temp > ratio) {
            jdrop = j;
            ratio = temp;
        }
        sigbar(m1(j)) = temp*vsig(m1(j));
    }

    // Calculate the value of ell.
    edgmax = delta * rho;
    l=0;
    for (j=1; j<=n; j++) {
        if (sigbar(m1(j)) >= parsig or sigbar(m1(j)) >= vsig(m1(j))) {
            temp = veta(m1(j));
            if (trured > 0.0) {
                temp = 0.0;
                for (i=1; i<=n; i++) {
                    temp = temp + math::pow(dx(m1(i)) - sim(m1(i),m1(j)), 2.0);
                }
                temp = math::sqrt(temp);
            }
            if (temp > edgmax) {
                l = j;
                edgmax = temp;
            }
        }
    }
    if (l > 0) {
        jdrop = l;
    }
    if (jdrop == 0) {
        goto line_550;
    }

    // Revise the simplex by updating the elements of SIM, SIMI and DATMAT.
    temp = 0.0;
    for (i=1; i<=n; i++) {
        sim(m1(i), m1(jdrop)) = dx(m1(i));
        temp = temp + simi(m1(jdrop),m1(i)) * dx(m1(i));
    }
    for (i=1; i<=n; i++) {
        simi(m1(jdrop),m1(i)) = simi(m1(jdrop),m1(i)) / temp;
    }
    for (j=1; j<=n; j++) {
        if (j != jdrop) {
            temp = 0.0;
            for (i=1; i<=n; i++) {
                temp = temp + simi(m1(j),m1(i)) * dx(m1(i));
            }
            for (i=1; i<=n; i++) {
                simi(m1(j),m1(i)) = simi(m1(j),m1(i)) - temp * simi(m1(jdrop),m1(i));
            }
        }
    }
    for (k=1; k<=mpp; k++) {
        datmat(m1(k),m1(jdrop)) = con(m1(k));
    }

    // Branch back for further iterations with the current RHO.
    if (trured > 0.0 and trured >= 0.1*prerem) {
        goto line_140;
    }
line_550:
    if (iflag == 0) {
        ibrnch = 0;
        goto line_140;
    }

    // Otherwise reduce RHO if it is not at its least value and reset PARMU.
    if (rho > rhoend) {
        rho = 0.5*rho;
        if (rho <= 1.5*rhoend) {
            rho = rhoend;
        }
        if (parmu > 0.0) {
            denom = 0.0;
            for (k=1; k<=mp; k++) {
                cmin = datmat(m1(k),m1(np));
                cmax = cmin;
                for (i=1; i<=n; i++) {
                    cmin = math::fmin(cmin, datmat(m1(k),m1(i)));
                    cmax = math::fmax(cmax, datmat(m1(k),m1(i)));
                }
                if (k <= m and cmin < 0.5*cmax) {
                    temp = math::fmax(cmax, 0.0) - cmin;
                    if (denom <= 0.0) {
                        denom = temp;
                    }
                    else {
                        denom = math::fmin(denom, temp);
                    }
                }
            }
            if (denom == 0.0) {
                parmu = 0.0;
            }
            else if (cmax-cmin < parmu*denom) {
                parmu = (cmax-cmin)/denom;
            }
        }
        KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 2) printf(
            "\n   Reduction in RHO to%13.6e  and PARMU =%13.6e",
            rho,
            parmu
        );
        KOKKOS_COBYLA_CONSTEXPR_IF (iprint == 2) {
            printf(
                "\n   NFVALS =%5d   F =%13.6e    MAXCV =%13.6e\n   X =",
                nfvals,
                datmat(m1(mp), m1(np)),
                datmat(m1(mpp), m1(np))
            );
            for (i=1; i<=iptem; i++) {
                printf("%13.6e  ", sim(m1(i), m1(np)));
            }
            if (iptem < n) {
                printf("\n      ");
                for (i=iptemp; i<=n; i++) {
                    printf("%13.6e  ", x(m1(i)));
                }
            }
        }
        goto line_140;
    }

    // Return the best calculated values of the variables.
    KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 1) printf(
        "\n   Normal return from subroutine COBYLA"
    );
    if (ifull == 1) goto line_620;
line_600:
    for (i=1; i<=n; i++) {
        x(m1(i)) = sim(m1(i), m1(np));
    }
    f = datmat(m1(mp),m1(np));
    resmax = datmat(m1(mpp),m1(np));
line_620:
    KOKKOS_COBYLA_CONSTEXPR_IF (iprint >= 1) {
        printf(
            "\n   NFVALS =%5d   F =%13.6e    MAXCV =%13.6e\n   X =",
            nfvals,
            f,
            resmax
        );
        for (i=1; i<=iptem; i++) {
            printf("%13.6e  ", x(m1(i)));
        }
        if (iptem < n) {
            printf("\n      ");
            for (i=iptemp; i<=n; i++) {
                printf("%13.6e  ", x(m1(i)));
            }
        }
    }
    maxfun = nfvals;
    return;
}

/*
This subroutine minimizes an objective function F(X) subject to M
inequality constraints on X, where X is a vector of variables that has
N components. The algorithm employs linear approximations to the
objective and constraint functions, the approximations being formed by
linear interpolation at N+1 points in the space of the variables.
We regard these interpolation points as vertices of a simplex. The
parameter RHO controls the size of the simplex and it is reduced
automatically from RHOBEG to RHOEND. For each RHO the subroutine tries
to achieve a good vector of variables for the current size, and then
RHO is reduced until the value RHOEND is reached. Therefore RHOBEG and
RHOEND should be set to reasonable initial changes to and the required
accuracy in the variables respectively, but this accuracy should be
viewed as a subject for experimentation because it is not guaranteed.
The subroutine has an advantage over many of its competitors, however,
which is that it treats each constraint individually when calculating
a change to the variables, instead of lumping the constraints together
into a single penalty function. The name of the subroutine is derived
from the phrase Constrained Optimization BY Linear Approximations.

The user must set the values of N, M, RHOBEG and RHOEND, and must
provide an initial vector of variables in X. Further, the value of
IPRINT should be set to 0, 1, 2 or 3, which controls the amount of
printing during the calculation. Specifically, there is no output if
IPRINT=0 and there is output only at the end of the calculation if
IPRINT=1. Otherwise each new value of RHO and SIGMA is printed.
Further, the vector of variables and some function information are
given either when RHO is reduced or when each new value of F(X) is
computed in the cases IPRINT=2 or IPRINT=3 respectively. Here SIGMA
is a penalty parameter, it being assumed that a change to X is an
improvement if it reduces the merit function
           F(X)+SIGMA*MAX(0.0,-C1(X),-C2(X),...,-CM(X)),
where C1,C2,...,CM denote the constraint functions that should become
nonnegative eventually, at least to the precision of RHOEND. In the
printed output the displayed term that is multiplied by SIGMA is
called MAXCV, which stands for 'MAXimum Constraint Violation'. The
argument MAXFUN is an integer variable that must be set by the user to a
limit on the number of calls of CALCFC, the purpose of this routine being
given below. The value of MAXFUN will be altered to the number of calls
of CALCFC that are made. The arguments W and IACT provide real and
integer arrays that are used as working space. Their lengths must be at
least N*(3*N+2*M+11)+4*M+6 and M+1 respectively.

In order to define the objective and constraint functions, we require
a subroutine that has the name and arguments
           SUBROUTINE CALCFC (N,M,X,F,CON)
           DIMENSION X(*),CON(*)  .
The values of N and M are fixed and have been defined already, while
X is now the current vector of variables. The subroutine should return
the objective and constraint functions at X in F and CON(1),CON(2),
...,CON(M). Note that we are trying to adjust X so that F(X) is as
small as possible subject to the constraint functions being nonnegative.

Partition the working space array W to provide the storage that is needed
for the main calculation.
*/

template <
    typename IntegralType,
    typename SolutionViewType,
    typename ScalarType,
    typename ScalarWorkViewType,
    typename IntegralWorkViewType,
    int iprint = 0
>
KOKKOS_INLINE_FUNCTION
void cobyla(
    IntegralType n,
    IntegralType m,
    SolutionViewType x,
    ScalarType rhobeg,
    ScalarType rhoend,
    IntegralType maxfun,
    ScalarWorkViewType w,
    IntegralWorkViewType iact,
    void (*calcfc) (
        IntegralType n_in,
        IntegralType m_in,
        SolutionViewType x_in,
        ScalarType &f_in,
        decltype(Kokkos::subview(w, Kokkos::make_pair(0, 1))) con_in
    )
) {
    IntegralType mpp = m+2;
    IntegralType isim = mpp;
    IntegralType isimi = isim + n*n + n;
    IntegralType idatm = isimi + n*n;
    IntegralType ia = idatm + n*mpp + mpp;
    IntegralType ivsig = ia + m*n + n;
    IntegralType iveta = ivsig + n;
    IntegralType isigb = iveta + n;
    IntegralType idx = isigb + n;
    IntegralType iwork = idx + n;
    IntegralType total_size = n*(3*n + 2*m + 11) + 4*m + 6;

    cobylb<
        IntegralType,
        SolutionViewType,
        ScalarType,
        decltype(Kokkos::subview(w, Kokkos::make_pair(0, 1))),
        IntegralWorkViewType,
        iprint
    > (
        n,
        m,
        mpp,
        x,
        rhobeg,
        rhoend,
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
        iact,
        calcfc
    );
}

template<typename IntegralType>
IntegralType requiredScalarWorkViewSize(
    IntegralType n,
    IntegralType m
) {
    return n*(3*n + 2*m + 11) + 4*m + 6;
}

template<typename IntegralType>
IntegralType requiredIntegralWorkViewSize(
    IntegralType m
) {
    return m+1;
}

} // namespace kokkos_cobyla

#endif // KOKKOS_COBYLA_HPP
