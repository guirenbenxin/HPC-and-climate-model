
module buoysort

    implicit none
    private
    save

    public :: cal_buoysort, conden, cal_qsat, conden_new, cal_fracmix, cal_entdet

    integer,parameter :: r8 = selected_real_kind(12)

    real(r8), parameter :: criqc  = 0.7e-3
    real(r8), parameter :: latvap = 2.501e6
    real(r8), parameter :: latice = 3.34e5
    real(r8), parameter :: latall = latvap+latice

    real(r8), parameter :: mwh2o = 34.
    real(r8), parameter :: mwdry = 28.966
    real(r8), parameter :: ep2 = mwh2o/mwdry

    real(r8), parameter :: g = 9.8

    real(r8), parameter :: p00   = 1.e5
    real(r8), parameter :: rair   = 287.042311365
    real(r8), parameter :: cpair  = 1004.64
    real(r8), parameter :: rovcp=rair/cpair

    real(r8), parameter :: epsilo = 0.6219705862
    real(r8), parameter :: tveps  = 1.0/epsilo - 1


contains


subroutine conden_new(T0, qt, p, T, qv, ql)
    real(r8), intent(in) :: T0, qt, p
    real(r8), intent(out) :: T, qv, ql
    real(r8) :: qs, mindT, ft0, ft, ft1, ft2, T1, T2, qs1, qs2
    integer :: iter, maxiter

    maxiter = 20

    ft0 = T0 + latvap*qt/cpair
    call cal_qsat(T0, p, qs)
    if (qs >= qt) then
        T = T0
        qv = qt
        ql = 0.0
    else
        T1 = T0
        T2 = 350.0
        call cal_qsat(T1, p, qs1)
        ft1 = T1 + latvap * qs1/cpair
        call cal_qsat(T2, p, qs2)
        ft2 = T2 + latvap * qs2/cpair
        do iter = 1, maxiter 
            T = (T1+T2)/2.0
            call cal_qsat(T, p, qs)
            ft = T + latvap*qs/cpair
            if (ft-ft0 <= 0) then
                T1 = T
            else
                T2 = T
            end if
            qv = qs
            ql = qt-qv
            
            if (abs(T1-T2) < 0.001) then
                exit
            end if

        end do

    end if

end subroutine conden_new


subroutine cal_fracmix(Te, qte, Tc, qtc, p, w, cridis, xc)
    real(r8), intent(in) :: Te, qte, Tc, qtc, p, w, cridis
    real(r8), intent(out) :: xc
    real(r8) :: xsat, x0, qse, qve, qle, qsc, qvc, qlc, qsx, qvx, qlx, qtx, buoyc, buoysat
    real(r8) :: Tle, Tlc, Tlx, Tx, x, x1, x2, xc1, xc2, Tve, Tvc, Tvx, fa,fb,fc,r1,r2
    integer :: iter, maxiter, status

    maxiter = 10

    call cal_qsat(Te, p, qse)
    qve = min(qte, qse)
    qle = max(0.0, qte-qse)
    Tle = Te - latvap*qle/cpair
    Tve = Te * (1 + tveps * qve - qle)

    call cal_qsat(Tc, p, qsc)
    qvc = min(qtc, qsc)
    qlc = max(0.0, qtc-qsc)
    Tlc = Tc - latvap*qlc/cpair
    Tvc = Tc * (1 + tveps * qvc - qlc)
    buoyc = g*(Tvc - Tve)/Tve

    if (qtc >= qsc) then
        if (qte >= qse) then
            xsat = 1.0
            x0 = 1.0
            xc = 1.0
        else
            x1 = 0.0
            x2 = 1.0
            do iter = 1,maxiter
                x = (x1+x2)/2.0
                Tlx = x*Tle + (1-x)*Tlc
                qtx = x*qte + (1-x)*qtc
                call conden_new(Tlx, qtx, p, Tx, qvx, qlx)
                call cal_qsat(Tx, p, qsx)
                if (qtx >= qsx) then
                    x1 = x
                else
                    x2 = x
                end if
                if (abs(x1-x2) < 0.001) then
                    exit
                end if
            end do
            xsat = x
            Tvx = Tx * (1+tveps*qvx - qlx)
            buoysat = g*(Tvx-Tve)/Tve
            
            !write(*, *) buoyc, buoysat
            
            if (buoysat > 0) then
                x0 = 1.0
                xc = 1.0
            else
                if (buoyc < 0) then
                    x0 = 0.0
                    if (w*w/2/max(1e-6, -buoyc) <= cridis ) then
                        xc = 0.0
                    else
                        fa = w*w
                        fb = 2*(buoysat-buoyc)*cridis/max(1e-5,xsat) - 2*w*w
                        fc = w*w - 2*x0*(buoysat-buoyc)*cridis/max(1e-5,xsat)
                        call roots(fa,fb,fc,r1,r2,status)
                        xc1 = min(r1,r2)
                        if (xc1 <= xsat) then
                            xc = xc1
                        else
                            fa = w*w
                            fb = -2*buoysat*cridis/max(1e-5,1-xsat) + 2*w*w
                            fc = w*w + 2*buoysat*cridis/max(1e-5,1-xsat)
                            call roots(fa,fb,fc,r1,r2,status)
                            xc2 = min(r1,r2)
                            xc = max(xsat, xc2)
                        end if
                    end if
                else
                    x0 = max(0.0, min(xsat, (Tvc-Tve)/max(1e-5, Tvc-Tvx)*xsat ))
                    fa = w*w
                    fb = 2*buoysat*cridis/max(1e-5,xsat-x0) - 2*w*w
                    fc = w*w - 2*x0*buoysat*cridis/max(1e-5,xsat-x0)
                    call roots(fa,fb,fc,r1,r2,status)
                    xc1 =  min(r1,r2)
                    if (xc1 <= xsat) then
                        xc = xc1
                    else
                        fa = w*w
                        fb = -2*buoysat*cridis/max(1e-5,1-xsat) + 2*w*w
                        fc = w*w + 2*buoysat*cridis/max(1e-5,1-xsat)
                        call roots(fa,fb,fc,r1,r2,status)
                        xc2 = min(r1,r2)
                        xc = max(xsat, xc2)
                    end if

                end if
            end if  ! buoysat v.s. buoyc

        end if  ! qte v.s. qse
    else
        xsat = 0.0
        x0 = 1.0
        xc = 1.0
    end if  ! qtc v.s. qsc

end subroutine cal_fracmix

subroutine cal_entdet(flagbspdf, xc, ent, det)
    integer, intent(in) :: flagbspdf
    real(r8), intent(in) :: xc
    real(r8), intent(out) :: ent, det

    if (flagbspdf == 1) then
        ent    = xc**2
        det    = 1.0 - 2.0*xc + xc**2
    end if

    if (flagbspdf == 2) then
        if (xc < 0.5) then
            ent = 2.0/3 *xc*xc
            det = 2.0/3 *(1-xc)*(1-xc) - 1.0/18
        else
            ! check
            !ent = 2.0/3 *xc*xc *(1-xc)*(1-xc) + 1.0/18
            ent = 4.0/3 *xc*xc *(1-2.0/3*xc) - 1.0/18
            det = 8.0/9 *(1-xc)*(1-xc)*(1-xc)
        end if
    end if

end subroutine cal_entdet


subroutine cal_buoysort(flagbspdf, cridis, z, p, rho, thle, qte, thlue0, qtue0, wue, xc, ent_rate, det_rate)

    integer, intent(in) :: flagbspdf
    real(r8), intent(in) :: cridis
    real(r8), intent(in) :: z, p, rho
    real(r8), intent(in) :: thle, qte, thlue0, qtue0
    real(r8), intent(in) :: wue
    real(r8), intent(out) :: xc
    real(r8), intent(out) :: ent_rate, det_rate
!    real(r8), intent(out) :: fer, fdr
    real(r8) :: fer, fdr

    real(r8), parameter :: rbuoy = 1.0
    real(r8), parameter :: rkm = 14.0

    real(r8) :: exne, exql, exqi, thlue, qtue

    real(r8) :: tj, thvj, thv0j
    real(r8) :: excessu, excess0

    real(r8) :: qsat_arg, thj, qvj, qlj, qij, qse
    real(r8) :: qs

    real(r8) :: aquad, bquad, cquad
    real(r8) :: xsat, thlxsat, qtxsat
    real(r8) :: thv_x0, thv_x1, x_en, x_cu, thvxsat
    real(r8) :: xs1, xs2

    real(r8) :: ee2,ud2,rei

    integer :: id_check, exit_conden
    logical :: id_exit

    integer :: kk, stat

    qtue = qtue0
    thlue = thlue0

    exne = (p/p00)**rovcp

    call conden(p,thle,qte,thj,qvj,qlj,qij,qse,id_check)
    thv0j = thj * ( 1.0 + tveps * qvj - qlj - qij )
    tj   = thj * exne
!thle = thle + (latvap/cpair/exne)*exql + (latall/cpair/exne)*exqi
    qsat_arg = thle*exne
    call cal_qsat(qsat_arg, p, qs)
    excess0 = qte-qs

!    write(*,"(5a20)") "thle", "qte", "qs", "excess0"
!    write(*,"(5f20.10)") thle, qte, qs, excess0


    call conden(p,thlue,qtue,thj,qvj,qlj,qij,qse,id_check)
      if( (qlj + qij) > criqc ) then
           exql  = ( ( qlj + qij ) - criqc ) * qlj / ( qlj + qij )
           exqi  = ( ( qlj + qij ) - criqc ) * qij / ( qlj + qij )
           qtue  = qtue - exql - exqi
           thlue = thlue + (latvap/cpair/exne)*exql + (latall/cpair/exne)*exqi 
      endif
    thvj = thj * ( 1.0 + tveps * qvj - qlj - qij )
    tj   = thj * exne
!thlue = thlue + (latvap/cpair/exne)*exql + (latall/cpair/exne)*exqi
    qsat_arg = thlue*exne
    call cal_qsat(qsat_arg, p, qs)
    excessu = qtue-qs


!    write(*,"(5a20)") "thlue", "qtue", "qs", "excessu"
!    write(*,"(5f20.10)") thlue, qtue, qs, excessu


    if ( ( excessu .le. 0.0 .and. excess0 .le. 0.0 ) .or. ( excessu .ge. 0.0 .and. excess0 .ge. 0.0 ) ) then
        xc = min(1.0,max(0.0,1.0-2.0*rbuoy*g*cridis/wue**2.0*(1.0-thvj/thv0j)))
              ! Below 3 lines are diagnostic output not influencing
              ! numerical calculations.
        aquad = 0.0
        bquad = 0.0
        cquad = 0.0
    else
          ! -------------------------------------------------- !
          ! Case 2 : When either cumulus or env. is saturated. !
          ! -------------------------------------------------- !
        xsat    = excessu / ( excessu - excess0 )
        thlxsat = thlue + xsat * ( thle - thlue )
        qtxsat  = qtue  + xsat * ( qte - qtue )
        call conden(p,thlxsat,qtxsat,thj,qvj,qlj,qij,qse,id_check)
        if( id_check .eq. 1 ) then
            exit_conden = 1.0
            id_exit = .true.
            write(*,*) 'error'
        end if
        thvxsat = thj * ( 1.0 + tveps * qvj - qlj - qij )
              ! -------------------------------------------------- !
              ! kk=1 : Cumulus Segment, kk=2 : Environment Segment !
              ! -------------------------------------------------- !
        do kk = 1, 2
!            write(*,*) "kk=", kk
            if( kk .eq. 1 ) then
                thv_x0 = thvj
                thv_x1 = ( 1.0 - 1.0/xsat ) * thvj + ( 1.0/xsat ) * thvxsat
            else
                thv_x1 = thv0j
                thv_x0 = ( xsat / ( xsat - 1.0 ) ) * thv0j + ( 1.0/( 1.0 - xsat ) ) * thvxsat
            endif
            aquad =  wue**2
            bquad =  2.0*rbuoy*g*cridis*(thv_x1 - thv_x0)/thv0j - 2.0*wue**2
            cquad =  2.0*rbuoy*g*cridis*(thv_x0 -  thv0j)/thv0j +       wue**2

            !write(*,*)
            !write(*,"(5a20)") "a", "b", "c", "b2-4ab"
            !write(*,"(5f20.10)") aquad, bquad, cquad, bquad**2-4._r8*aquad*cquad

            if( kk .eq. 1 ) then
                if( ( bquad**2-4.0*aquad*cquad ) .ge. 0.0 ) then
                    call roots(aquad,bquad,cquad,xs1,xs2,stat)
                    x_cu = min(1.0,max(0.0,min(xsat,min(xs1,xs2))))
!                    write(*,*) "solve x_cu", x_cu, xs1, xs2
                else
                    x_cu = xsat
                endif
            else
                if( ( bquad**2-4.0*aquad*cquad) .ge. 0.0 ) then
                    call roots(aquad,bquad,cquad,xs1,xs2,stat)
                    x_en = min(1.0,max(0.0,max(xsat,min(xs1,xs2))))
!                    write(*,*) "solve x_en", x_en, xs1, xs2
                else
                    x_en = 1.0
                endif
            endif

            !write(*,"(5a20)") "xsat", "x_cu", "x_en"

        enddo

        !write(*,"(5f10.3)") xsat, x_cu, x_en
        if( abs(x_cu - xsat) < 1.0e-6 ) then
            xc = max(x_cu, x_en)
        else
            xc = x_cu
        endif

    endif

    if (flagbspdf == 1) then
        ee2    = xc**2
        ud2    = 1.0 - 2.0*xc + xc**2

        rei = ( 0.5 * rkm / z / g /rho )

    !    if( xc .gt. 0.5_r8 ) rei = min(rei,0.9_r8*log(dp0(k)/g/dt/umf(km1) + 1._r8)/dpe/(2._r8*xc-1._r8))
        fer = rei * ee2
        fdr = rei * ud2

        !write(*,"(6a20)") "xc", "ee2", "ud2", "fer", "fdr"
        !write(*,"(6f20.10)") xc, ee2, ud2, fer, fdr
        !write(*,*)

        ent_rate = fer
        det_rate = fdr
    end if

    if (flagbspdf == 2) then
        if (xc < 0.5) then
            ent_rate = 2.0/3 *xc*xc
            det_rate = 2.0/3 *(1-xc)*(1-xc) - 1.0/18
        else
            ent_rate = 2.0/3 *xc*xc *(1-xc)*(1-xc) + 1.0/18
            det_rate = 8.0/9 *(1-xc)*(1-xc)*(1-xc)
        end if
    end if

end subroutine cal_buoysort

real(r8) function exnf(pressure)
    real(r8), intent(in)              :: pressure
    exnf = (pressure/p00)**rovcp
    return
end function exnf

subroutine roots(a,b,c,r1,r2,status)
  ! --------------------------------------------------------- !
  ! Subroutine to solve the second order polynomial equation. !
  ! I should check this subroutine later.                     !
  ! --------------------------------------------------------- !
    real(r8), intent(in)  :: a
    real(r8), intent(in)  :: b
    real(r8), intent(in)  :: c
    real(r8), intent(out) :: r1
    real(r8), intent(out) :: r2
    integer , intent(out) :: status
    real(r8)              :: q

    status = 0

    if( a .eq. 0.0 ) then                            ! Form b*x + c = 0
        if( b .eq. 0.0 ) then                        ! Failure: c = 0
            status = 1
        else                                           ! b*x + c = 0
            r1 = -c/b
        endif
        r2 = r1
    else
        if( b .eq. 0.0 ) then                        ! Form a*x**2 + c = 0
            if( a*c .gt. 0.0 ) then                  ! Failure: x**2 = -c/a < 0
                status = 2
            else                                       ! x**2 = -c/a 
                r1 = sqrt(-c/a)
            endif
            r2 = -r1
        else                                            ! Form a*x**2 + b*x + c = 0
            if( (b**2 - 4.0*a*c) .lt. 0.0 ) then   ! Failure, no real roots
                status = 3
            else
                if (b < 0) then
                    q  = -0.5*(b - sqrt(b**2 - 4.0*a*c))
                else
                    q  = -0.5*(b + sqrt(b**2 - 4.0*a*c))
                end if
                r1 =  q/a
                r2 =  c/q
            endif
        endif
    endif

    return
end subroutine roots

subroutine cal_qsat( t, p, qsat)
    real(r8) :: t, p, qsat
    qsat = epsilo * 611.2*exp(17.67*(t-273.15)/(t-273.15+243.5)) / p
end subroutine cal_qsat

subroutine conden(p,thl,qt,th,qv,ql,qi,rvls,id_check)
  ! --------------------------------------------------------------------- !
  ! Calculate thermodynamic properties from a given set of ( p, thl, qt ) !
  ! --------------------------------------------------------------------- !
    implicit none
    real(r8), intent(in)  :: p
    real(r8), intent(in)  :: thl
    real(r8), intent(in)  :: qt
    real(r8), intent(out) :: th
    real(r8), intent(out) :: qv
    real(r8), intent(out) :: ql
    real(r8), intent(out) :: qi
    real(r8), intent(out) :: rvls
    integer , intent(out) :: id_check
    real(r8)              :: tc,temps,t
    real(r8)              :: leff, nu, qc
    integer               :: iteration
    real(r8)              :: es              ! Saturation vapor pressure
    real(r8)              :: qs              ! Saturation spec. humidity


    tc   = thl*exnf(p)
  ! Modification : In order to be compatible with the dlf treatment in stratiform.F90,
  !                we may use ( 268.15, 238.15 ) with 30K ramping instead of 20 K,
  !                in computing ice fraction below. 
  !                Note that 'cldfrc_fice' uses ( 243.15, 263.15 ) with 20K ramping for stratus.
    nu   = max(min((268.0 - tc)/20.0,1.0),0.0)  ! Fraction of ice in the condensate. 
    leff = (1.0 - nu)*latvap + nu*latall                      ! This is an estimate that hopefully speeds convergence

    ! --------------------------------------------------------------------------- !
    ! Below "temps" and "rvls" are just initial guesses for iteration loop below. !
    ! Note that the output "temps" from the below iteration loop is "temperature" !
    ! NOT "liquid temperature".                                                   !
    ! --------------------------------------------------------------------------- !
    temps  = tc
    call cal_qsat(temps, p, qs)
    rvls   = qs

    if( qs .ge. qt ) then
        id_check = 0
        qv = qt
        qc = 0.0
        ql = 0.0
        qi = 0.0
        th = tc/exnf(p)
    else
        do iteration = 1, 10
            temps  = temps + ( (tc-temps)*cpair/leff + qt - rvls )/( cpair/leff + ep2*leff*rvls/rair/temps/temps )
            call cal_qsat(temps, p, qs)
            rvls   = qs
        end do
        qc = max(qt - qs,0.0)
        qv = qt - qc
        ql = qc*(1.0 - nu)
        qi = nu*qc
        th = temps/exnf(p)
        if( abs((temps-(leff/cpair)*qc)-tc) .ge. 1.0 ) then
            id_check = 1
        else
            id_check = 0
        end if
    end if

    return
end subroutine conden



end module buoysort

