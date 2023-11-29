
module subcolphys

!---------------------------------------------------------------------------------
! Purpose:
! Generic sub column generator
!
! Contributor: Xin Xie
!---------------------------------------------------------------------------------

use gamma_mod
!  use spmd_utils,      only: masterproc
!  use abortutils,      only: endrun
use shr_kind_mod,    only: r8 => shr_kind_r8

use netcdf

implicit none
private
save

real(r8), allocatable, target :: sc_cld(:, :, :)
real(r8), allocatable, target :: cld(:, :)
real(r8), allocatable, target :: sc_wgt(:, :)
real(r8), allocatable, target :: cldcount(:, :)
real(r8), allocatable, target :: cldw(:, :)
real(r8), allocatable :: prec_state(:, :, :)

real(r8), allocatable, target :: sc_cldlmr(:, :, :)
real(r8), allocatable, target :: sc_cldimr(:, :, :)

real(r8), allocatable, target :: sc_cldlwp(:, :, :)
real(r8), allocatable, target :: sc_cldiwp(:, :, :)

integer  :: overlap_opt ! 1 maximum-random
integer  :: nsubcol, ncol, nlev

!real(r8), parameter :: cldlmrvar = 2._r8
!real(r8), parameter :: cldimrvar = 2._r8
!public :: cldlmrvar, cldimrvar

!real(r8), parameter :: cldlwpvar = 2._r8
!real(r8), parameter :: cldiwpvar = 2._r8

public :: nsubcol

!public subcolumn data
public :: sc_cld
public :: sc_cldlmr, sc_cldimr
public :: sc_cldlwp, sc_cldiwp

!public subroutines or functions
public :: subcol_init, subcol_gencld, &
          subcol_diag,            &
          subcol_inverse_gamma_cdf, subcol_gamma
public :: subcol_invgamma
public :: subcol_calv
!public :: subcol_cesm_rand, subcol_cesm_invgamma
public :: subcol_gengamma, subcol_get, subcol_gengammafast
public :: subcol_genuni
public :: subcol_ptr

public :: subcol_netcdf_init, netcdf_check
public :: subcol_netcdf_addfld
public :: subcol_netcdf_putfld
public :: subcol_netcdf_putclm
public :: subcol_netcdf_nextstep
public :: subcol_netcdf_end

integer, parameter :: nvarmax = 200

!private data for subcol_netcdf
type cdf_diag

   integer :: ncid

   integer :: subcol_dimid, lev_dimid, rec_dimid
   integer :: nsubcol, nlev, nrec

   integer :: lev_varid, subcol_varid

   integer :: nvar
   integer       :: varidlist(nvarmax)
   character(50) :: varnamelist(nvarmax)
   character(50) :: vartypelist(nvarmax)
   integer :: varstartlist(nvarmax)
   integer :: curtimestep

end type cdf_diag

logical :: netcdf_enddef = .false.
type (cdf_diag) output

contains

subroutine subcol_netcdf_init( outfile )
   character(*), intent(in) :: outfile
   
   character(*), parameter :: units = "units"

   character(*), parameter :: subcol_name = "subcol"
   character(*), parameter :: lev_name    = "lev"
   character(*), parameter :: rec_name    = "time"

   call netcdf_check( nf90_create( trim(outfile), nf90_clobber, output%ncid) )

   call netcdf_check( nf90_def_dim(output%ncid, subcol_name, &
      nsubcol, output%subcol_dimid) )
   call netcdf_check( nf90_def_dim(output%ncid, lev_name,    &
      nlev   , output%lev_dimid)    )
   call netcdf_check( nf90_def_dim(output%ncid, rec_name,    &
      NF90_UNLIMITED, output%rec_dimid)    )

   output%nvar = 0
   output%varidlist = 0
   output%varstartlist = 0
   output%curtimestep = 0
end subroutine subcol_netcdf_init



subroutine subcol_netcdf_addfld(varname, varunit, vartype)
   character(*), intent(in) :: varname
   character(*), intent(in) :: varunit
   character(*), intent(in) :: vartype

   integer :: dimids(3)
   integer :: ndim

   if ( netcdf_enddef .eqv. .true.) then
      write(*,*) "enddef marked, can not add variable."
      return
   end if
   write(*,*) "add ", varname

   if (vartype == "slev") then
      ndim = 2
      dimids(1:ndim) = (/ output%subcol_dimid, output%rec_dimid /)
   else if (vartype == "mlev") then
      ndim = 3
      dimids(1:ndim) = (/ output%subcol_dimid, output%lev_dimid, output%rec_dimid /)
   else if (vartype == "stdmlev") then
      ndim = 2
      dimids(1:ndim) = (/ output%lev_dimid, output%rec_dimid /)
   end if

   output%nvar = output%nvar + 1

   output%vartypelist(output%nvar) = vartype

   call netcdf_check( nf90_def_var(output%ncid, trim(varname), NF90_REAL,&
      dimids(1:ndim), output%varidlist(output%nvar)) )

   call netcdf_check( nf90_put_att(output%ncid, &
      output%varidlist(output%nvar), "units", trim(varunit) ) )
   call netcdf_check( nf90_put_att(output%ncid, &
      output%varidlist(output%nvar), "_FillValue", nf90_fill_real ) )


   output%varnamelist(output%nvar) = trim(varname)
end subroutine subcol_netcdf_addfld



!subroutine used to output the field over all subcolumns.
subroutine subcol_netcdf_putfld(varname, input)
   character(*), intent(in) :: varname
   real(r8),     intent(in) :: input(nsubcol,*)
   integer :: i, varind
   integer :: starts(3), counts(3)
   integer :: ndim

!  write(*,*) "put ", varname
   if ( netcdf_enddef .eqv. .false. ) then
      netcdf_enddef = .true.
      call netcdf_check( nf90_enddef(output%ncid) )
   end if

!locate the var position in the list.
   varind = 0
   do i = 1, output%nvar
      if ( output%varnamelist(i) == varname) then
         varind = i
      end if
   end do

   output%varstartlist(varind) = output%varstartlist(varind) + 1

!put the var
   if ( output%vartypelist(varind) == "mlev" ) then
      ndim = 3
!     starts(1:ndim) = (/ 1, 1, output%varstartlist(varind) /)
      starts(1:ndim) = (/ 1, 1, output%curtimestep /)
      counts(1:ndim) = (/ nsubcol, nlev, 1 /)
      call netcdf_check( nf90_put_var(output%ncid, &
         output%varidlist(varind), input(:, 1:nlev), &
         start=starts(1:ndim), &
         count=counts(1:ndim)) )
   else if ( output%vartypelist(varind) == "slev" ) then
      ndim = 2
!     starts(1:ndim) = (/ 1, output%varstartlist(varind) /)
      starts(1:ndim) = (/ 1, output%curtimestep /)
      counts(1:ndim) = (/ nsubcol, 1 /)
      call netcdf_check( nf90_put_var(output%ncid, &
         output%varidlist(varind), input(:, 1), &
         start=starts(1:ndim), &
         count=counts(1:ndim)) )
   else if ( output%vartypelist(varind) == "stdmlev" ) then
      ndim = 2
!     starts(1:ndim) = (/ 1, output%varstartlist(varind) /)
      starts(1:ndim) = (/ 1, output%curtimestep /)
      counts(1:ndim) = (/ nlev, 1 /)
      call netcdf_check( nf90_put_var(output%ncid, &
         output%varidlist(varind), input(:, 1), &
         start=starts(1:ndim), &
         count=counts(1:ndim)) )
   end if

!  write(*,*) starts(1:ndim)
!  write(*,*) counts(1:ndim)
end subroutine subcol_netcdf_putfld



!subroutine used to output the field over all subcolumns.
subroutine subcol_netcdf_putclm(varname, input, isubcol)
   character(*), intent(in) :: varname
   real(r8),     intent(in) :: input(nlev)
   integer ,     intent(in) :: isubcol
   integer :: i, varind
   integer :: starts(3), counts(3)
   integer :: ndim

!  write(*,*) "put ", varname, " isub:", isubcol
   if ( netcdf_enddef .eqv. .false. ) then
      netcdf_enddef = .true.
      call netcdf_check( nf90_enddef(output%ncid) )
   end if

   varind = 0
   do i = 1, output%nvar
      if ( output%varnamelist(i) == varname) then
         varind = i
      end if
   end do
   if ( varind == 0 ) then
      write(*,*) "CANNOT FIND OUTPUT FIELD", varname
      return
   end if
   if ( output%vartypelist(varind) == "mlev" ) then
      ndim = 3
!     starts(1:ndim) = (/ isubcol, 1, output%varstartlist(varind) /)
      starts(1:ndim) = (/ isubcol, 1, output%curtimestep /)
      counts(1:ndim) = (/ 1, nlev, 1 /)
      call netcdf_check( nf90_put_var(output%ncid, &
         output%varidlist(varind), input(1:nlev), &
         start=starts(1:ndim), &
         count=counts(1:ndim)) )
   else if ( output%vartypelist(varind) == "slev" ) then
      ndim = 2
!     starts(1:ndim) = (/ isubcol, output%varstartlist(varind) /)
      starts(1:ndim) = (/ isubcol, output%curtimestep /)
      counts(1:ndim) = (/ 1, 1 /)
      call netcdf_check( nf90_put_var(output%ncid, &
         output%varidlist(varind), input(1:1), &
         start=starts(1:ndim), &
         count=counts(1:ndim)) )
   end if

!  write(*,*) starts(1:ndim)
!  write(*,*) counts(1:ndim)
end subroutine subcol_netcdf_putclm



subroutine subcol_netcdf_nextstep
   output%curtimestep = output%curtimestep + 1
end subroutine subcol_netcdf_nextstep



subroutine subcol_netcdf_end
   write(*,*) "end netcdf"
   call netcdf_check( nf90_close(output%ncid) )
end subroutine subcol_netcdf_end



subroutine netcdf_check( status )
   integer, intent ( in) :: status

   if(status /= nf90_noerr) then 
      print *, trim(nf90_strerror(status))
      stop "Stopped"
   end if
end subroutine netcdf_check



subroutine subcol_init( nsubcol_in, ncol_in, nlev_in)
   integer, intent(in) :: nsubcol_in, ncol_in, nlev_in
   integer :: istat

   nsubcol = nsubcol_in
   ncol    = ncol_in
   nlev    = nlev_in

   if (nsubcol_in .le. 0) then
      write(*,*) 'ERROR: can not init with 0 sub columns.'
      call exit
   end if
   if (allocated(sc_cld)) deallocate(sc_cld)
   if (allocated(cld)) deallocate(cld)
   if (allocated(cldcount)) deallocate(cldcount)
   if (allocated(cldw)) deallocate(cldw)
   if (allocated(prec_state)) deallocate(prec_state)

   if (allocated(sc_cldlwp)) deallocate(sc_cldlwp)
   if (allocated(sc_cldiwp)) deallocate(sc_cldiwp)

   if (allocated(sc_cldlmr)) deallocate(sc_cldlmr)
   if (allocated(sc_cldimr)) deallocate(sc_cldimr)

   allocate( sc_cld(nsubcol, ncol, nlev), stat=istat )
   allocate( cld(ncol, nlev), stat=istat )
   allocate( cldcount(ncol, nlev), stat=istat )
   allocate( cldw(ncol, nlev), stat=istat )
   allocate( prec_state(nsubcol, ncol, nlev), stat=istat )

   allocate( sc_cldlwp(nsubcol, ncol, nlev), stat=istat )
   allocate( sc_cldiwp(nsubcol, ncol, nlev), stat=istat )

   allocate( sc_cldlmr(nsubcol, ncol, nlev), stat=istat )
   allocate( sc_cldimr(nsubcol, ncol, nlev), stat=istat )

   sc_cld  = 0._r8
   cld  = 0._r8
   cldcount = 0._r8
   cldw = 0._r8
   prec_state = 0._r8

   sc_cldlwp  = 0._r8
   sc_cldiwp  = 0._r8

   sc_cldlmr  = 0._r8
   sc_cldimr  = 0._r8
end subroutine subcol_init



subroutine subcol_calv(lcnlev, hscale, z, p, t, q, cld, scv, stab)
   integer , intent(in) :: lcnlev
   real(r8), intent(in) :: hscale
   real(r8), intent(in) :: z(lcnlev) !(nlev) geopotential height
   real(r8), intent(in) :: p(lcnlev) !(nlev) pressure
   real(r8), intent(in) :: t(lcnlev) !(nlev) temperature
   real(r8), intent(in) :: q(lcnlev) !(nlev) water vapor mixing ratio

   real(r8), intent(in) :: cld(lcnlev) !(nlev) cloud fraction

   real(r8), intent(out) :: scv(lcnlev)
   real(r8), intent(out) :: stab

!local variables
   real(r8)  :: qs(lcnlev) !(nlev)
   real(r8)  :: schs(lcnlev) !(nlev)
   real(r8)  :: sch(lcnlev) !(nlev)

   integer :: k
   integer :: ip500, ip950
   real(r8)  :: hs500, h950
   real(r8)  :: g=9.80616_R8, cpp=1.00464e3_R8, latvap=2.501e6_R8


   ip500=0
   ip950=0

   if (lcnlev .ne. size(z) ) then
      write(*,*) "subcol_calv inconsistent number of lev"
      write(*,*) lcnlev, size(z)
      call exit
   end if
   if ( hscale .lt. 1._r8) then
      scv = 6._r8
   else
      do k=1,lcnlev
!XX
         qs(k)  = 0.622*2.53e11*exp(-5420/t(k))/p(k)
         schs(k) = cpp*t(k) + g*z(k) + latvap*qs(k)
         sch(k)  = cpp*t(k) + g*z(k) + latvap*q(k)
         if ( (ip500 .eq. 0 ) .and. (p(k) > 50000.) ) then
            ip500 = k
         end if
         if ( (ip950 .eq. 0 ) .and. (p(k) > 95000.) ) then
            ip950 = k
         end if
      end do

!      write(*,*) "test detail:", ip500, ip950

      if ( (ip500 .gt. 0) .and. (ip950 .gt. 0) ) then
!        write(*,*) p(i,ip500), "   ", p(i,ip950), p(i,ip950)-p(i,ip500)
         hs500 = schs(ip500-1)+(50000._r8-p(ip500-1))*(schs(ip500)-schs(ip500-1))/(p(ip500)-p(ip500-1))
         h950  =  sch(ip950-1)+(95000._r8-p(ip950-1))*( sch(ip950)- sch(ip950-1))/(p(ip950)-p(ip950-1))
         stab  = (h950-hs500)/45000._r8


!         do k=1,lcnlev
!            scv(k) = 0.44_r8+8.3*(0.6-stab)*(0.05+(hscale )**(-2./3))
            !scv(k) = 0.67_r8-0.38*stab+4.96*(hscale)**(-2./3)-8.32*stab*(hscale)**(-2./3)
            !write(*,*) scv(k)
            !scv(k) = 0.6_r8-0.62*stab+6.6*(hscale)**(-2./3)-14.5*stab*(hscale)**(-2./3)
            !write(*,*) scv(k)
            !write(*,*)
!         end do

!            scv(k) = 0.44_r8+8.3*(0.6-stab)*(0.05+(hscale )**(-2./3))
!            write(*,*) 'stab:',stab,'hscale:',hscale
            scv = 0.67_r8-0.38*stab+4.96*(hscale)**(-2./3)-8.32*stab*(hscale)**(-2./3)
!            write(*,*) scv(1)
!            scv = 0.6_r8-0.62*stab+6.6*(hscale)**(-2./3)-14.5*stab*(hscale)**(-2./3)
!            write(*,*) scv(1)
!            write(*,*)

!         if (cape(i) .gt. 0.00001_r8) then
!            scv(i) = 0.20_r8+(5.1-4.7*stab(i))*(0.087+hscale**(-2./3))
!         else
!            scv(i) = 0.37_r8+(6.3-6.0*stab(i))*(0.048+hscale**(-2./3))
!         end if

      else if ( (ip950 .eq. 0) .and. (ip500 .ne. lcnlev)) then

         hs500 = schs(ip500-1)+(50000._r8-p(ip500-1))*(schs(ip500)-schs(ip500-1))/(p(ip500)-p(ip500-1))
         ip950 = lcnlev
         stab  = (sch(ip950)-hs500)/(p(ip950)-p(ip500))
!        stab(i)  = (sch(i,ip950)-schs(i,ip500-1))/(p(i,ip950)-p(i,ip500-1))

         !do k=1,lcnlev
!!            scv(k) = 0.44_r8+8.3*(0.6-stab)*(0.05+(hscale )**(-2./3))
            !scv(k) = 0.67_r8-0.38*stab+4.96*(hscale)**(-2./3)-8.32*stab*(hscale)**(-2./3)
            !write(*,*) scv(k)
            !scv(k) = 0.6_r8-0.62*stab+6.6*(hscale)**(-2./3)-14.5*stab*(hscale)**(-2./3)
            !write(*,*) scv(k)
            !write(*,*)
         !end do

!         write(*,*) 'stab:',stab,'hscale:',hscale
         scv = 0.67_r8-0.38*stab+4.96*(hscale)**(-2./3)-8.32*stab*(hscale)**(-2./3)
!         write(*,*) scv(1)
!         scv = 0.6_r8-0.62*stab+6.6*(hscale)**(-2./3)-14.5*stab*(hscale)**(-2./3)
!         write(*,*) scv(1)
!         write(*,*)

!         if (cape(i) .gt. 0.00001_r8) then
!            scv(i) = 0.20_r8+(5.1-4.7*stab(i))*(0.087+hscale**(-2./3))
!         else
!            scv(i) = 0.37_r8+(6.3-6.0*stab(i))*(0.048+hscale**(-2./3))
!         end if
      else
         scv = 1._r8
      end if

   end if

   do k=1,lcnlev
      scv(k) = max(0.4_r8, min(6._r8, scv(k)) )
   end do

end subroutine subcol_calv



subroutine subcol_gencld( overlap_opt, randseed, cldfrc)
   integer , intent(in) :: overlap_opt
   real(r8), intent(in) :: randseed(:, :)  ! (ncol, nlev)
   real(r8), intent(in) :: cldfrc(:,:)     ! (ncol, nlev)

!  real(r8), intent(out) :: output(:, :, :) ! (nsubcol, ncol, nlev)

   integer  :: i, j, k
   integer  :: seed1(ncol),seed2(ncol)
   integer  :: seed3(ncol),seed4(ncol)
   real(r8) :: rand_tmp(nsubcol, ncol, nlev)

   if ( (size(cldfrc, 1) /= ncol) .or. &
      (size(cldfrc, 2) /= nlev) ) then
      write(*,*) "subcolumn shape no match!"
      call exit(1)
   end if


   do i = 1, ncol
      seed1(i) = (randseed(i,nlev)   - int(randseed(i,nlev))  )  * 1000000000
      seed2(i) = (randseed(i,nlev-1) - int(randseed(i,nlev-1)))  * 1000000000
      seed3(i) = (randseed(i,nlev-2) - int(randseed(i,nlev-2)))  * 1000000000
      seed4(i) = (randseed(i,nlev-3) - int(randseed(i,nlev-3)))  * 1000000000
   end do

   call subcol_kissvec_2d(seed1, seed2, seed3, seed4, rand_tmp)

   cld = cldfrc

!overlapping algorithm from Raisanen et al. 2004
   select case (overlap_opt)
!random overlap
   case(1)
      do k = 1, nlev
         do i = 1, ncol
            do j = 1, nsubcol
               if ( rand_tmp(j,i,k) .gt. (1-cldfrc(i,k)) ) then
                  sc_cld(j,i,k) = 1.0_r8
               else
                  sc_cld(j,i,k) = 0._r8
               end if
            end do
         end do
      end do

!maximum-random overlap 
   case(2)
      do k = 2, nlev
         do i = 1, ncol
            do j = 1, nsubcol
               if ( rand_tmp(j,i,k-1) .gt. (1-cldfrc(i,k-1)) ) then
                  rand_tmp(j,i,k) = rand_tmp(j,i,k-1)
               else
                  rand_tmp(j,i,k) = rand_tmp(j,i,k)*(1-cldfrc(i,k-1))
               end if
            end do
         end do
      end do

   end select

   do k = 1, nlev
      do i = 1, ncol
         do j = 1, nsubcol
            if ( rand_tmp(j,i,k) .gt. (1-cldfrc(i,k)) ) then
               sc_cld(j,i,k) = 1.0_r8
            else
               sc_cld(j,i,k) = 0._r8
            end if
         end do
      end do
   end do

   cldcount = 0._r8
   do i = 1, nsubcol
      cldcount = cldcount + sc_cld(i,:,:)
   end do

   !do k = 1, nlev
      !do i = 1, ncol
          !write(*,*) "k=", k, cldcount(i,k)/nsubcol
       !end do
   !end do

end subroutine



subroutine subcol_invgamma(randseed, inshape, inscale, output)
   real(r8), intent(in)  :: randseed(:, :)  ! (ncol, nlev)
   real(r8), intent(in)  :: inshape(:, :)
   real(r8), intent(in)  :: inscale(:, :)   ! (ncol, nlev)

   real(r8), intent(out) :: output(:, :, :) ! (nsubcol, ncol, nlev)

   integer :: i, j, k
   integer :: seed1(ncol),seed2(ncol)
   integer :: seed3(ncol),seed4(ncol)
!   real(r8) :: rand_tmp(ncol)
   real(r8) :: rand_tmp(nsubcol, ncol, nlev)

   do i = 1, ncol
      seed1(i) = (randseed(i,nlev)   - int(randseed(i,nlev))  )  * 1000000000
      seed2(i) = (randseed(i,nlev-1) - int(randseed(i,nlev-1)))  * 1000000000
      seed3(i) = (randseed(i,nlev-2) - int(randseed(i,nlev-2)))  * 1000000000
      seed4(i) = (randseed(i,nlev-3) - int(randseed(i,nlev-3)))  * 1000000000
   end do

   call subcol_kissvec_2d(seed1, seed2, seed3, seed4, rand_tmp)
!   do k = 1, nlev
!     do i = 1, ncol
!       write(*,"(30f20.15)") rand_tmp(:,i,k)
!     end do
!   end do

   do k = 1, nlev
      do i = 1, ncol

         if (cld(i,k) > 0._r8) then
            do j = 1, nsubcol
  !       call subcol_kissvec(seed1, seed2, seed3, seed4, rand_tmp)
  !         write(*,*) i, j, k
  !         write(*,*) inshape, inscale(k,i)
               if (sc_cld(j,i,k) > 0._r8) then
                  output(j, i, k) = subcol_inverse_gamma_cdf(rand_tmp(j,i,k), &
                     inshape(i,k), inscale(i,k) )
               else
                  output(j, i, k) = 0._r8
               end if
  !         write(*,"(2f20.15)") rand_tmp(j,i,k), output(j,i,k)
            end do
         else
            output(1:nsubcol,i,k) = 0._r8
         end if

      end do
   end do
end subroutine subcol_invgamma


subroutine subcol_gengamma(watertype, randseed, inshape, qc )
   character(len=3), intent(in)  :: watertype
   !real(r8), intent(in)  :: randseed(:, :)  ! (ncol, nlev)
   !real(r8), intent(in)  :: inshape(:, :)
   !real(r8), intent(in)  :: inscale(:, :)   ! (ncol, nlev)
   real(r8), intent(in)  :: randseed(ncol, nlev)  ! (ncol, nlev)
   real(r8), intent(in)  :: inshape(ncol, nlev)
   real(r8), intent(in)  :: qc(:, :)   ! (ncol, nlev)


!local variables
   integer i, j, k
   real(r8) :: icqc(ncol, nlev) !in-cloud qc
   real(r8) :: adjfac(ncol, nlev)
!   real(r8) :: cldcount(ncol, nlev)
   real(r8), pointer :: ptr(:,:,:)

   if ( (size(qc, 1) /= ncol) .or. &
      (size(qc, 2) /= nlev) ) then
      write(*,*) "subcolumn shape no match!"
      call exit(1)
   end if

   !cldcount = 0._r8
   !do i = 1, nsubcol
      !cldcount = cldcount + sc_cld(i,:,:)
   !end do

   icqc = qc/(cldcount/nsubcol) !obtain in-cloud quantity

   if ( watertype .eq. 'lmr') then
      call subcol_invgamma(  randseed, inshape, icqc/inshape, sc_cldlmr)
      ptr => sc_cldlmr
   else if ( watertype .eq. 'imr') then
      call subcol_invgamma(  randseed, inshape, icqc/inshape, sc_cldimr)
      ptr => sc_cldimr
   else if ( watertype .eq. 'lwp') then
      call subcol_invgamma(  randseed, inshape, icqc/inshape, sc_cldlwp)
      ptr => sc_cldlwp
   else if ( watertype .eq. 'iwp') then
      call subcol_invgamma(  randseed, inshape, icqc/inshape, sc_cldiwp)
      ptr => sc_cldiwp
   end if

   adjfac = 0._r8
   do i = 1, nsubcol
      adjfac = adjfac + ptr(i,:,:)
   end do
   adjfac = adjfac/nsubcol

   do i = 1, ncol
      do j = 1, nlev
         if ( (adjfac(i,j) == 0._r8) .or. (cldcount(i,j) == 0._r8) ) then
            adjfac(i,j) = 0._r8
         else
!           write(*,"(5f15.9)") inshape(i,j)*inscale(i,j)
            adjfac(i,j) = qc(i,j)/adjfac(i,j)
         end if
      end do
   end do

   do i = 1, nsubcol
      ptr(i,:,:) = ptr(i,:,:)*adjfac
   end do

   !adjfac = 0._r8
   !do i = 1, nsubcol
      !adjfac = adjfac + ptr(i,:,:)
   !end do
   !adjfac = adjfac/nsubcol
   
   !do j = 1, nlev
      !write(*,*) adjfac(1,j), qc(1,j), cldcount(1,j)/nsubcol
   !end do

end subroutine subcol_gengamma



subroutine subcol_invgammafast(randseed, inshape, inscale, output)
   real(r8), intent(in)  :: randseed(:, :)  ! (ncol, nlev)
   real(r8), intent(in)  :: inshape(:, :)
   real(r8), intent(in)  :: inscale(:, :)   ! (ncol, nlev)

   real(r8), intent(out) :: output(:, :, :) ! (nsubcol, ncol, nlev)

   integer :: i, j, k, icld
   integer :: seed1(ncol),seed2(ncol)
   integer :: seed3(ncol),seed4(ncol)
!   real(r8) :: rand_tmp(ncol)
   real(r8) :: rand_tmp(nsubcol, ncol, nlev)
   real(r8) :: dx, x, p

   do i = 1, ncol
      seed1(i) = (randseed(i,nlev)   - int(randseed(i,nlev))  )  * 1000000000
      seed2(i) = (randseed(i,nlev-1) - int(randseed(i,nlev-1)))  * 1000000000
      seed3(i) = (randseed(i,nlev-2) - int(randseed(i,nlev-2)))  * 1000000000
      seed4(i) = (randseed(i,nlev-3) - int(randseed(i,nlev-3)))  * 1000000000
   end do

   do k = 1, nlev
      do i = 1, ncol
         if (cld(i,k) > 0._r8) then
!            write(*,*) cldcount(i,k)
!            p = 0
            dx = 1._r8/cldcount(i,k)
            x = 0.5*dx
            do j = 1, nsubcol
               if (sc_cld(j,i,k) > 0._r8) then
                  output(j, i, k) = subcol_inverse_gamma_cdf(x, &
                     inshape(i,k), inscale(i,k) )
!                  write(*,*) j, x, p, output(j,i,k)
!                  p = p + 1
                  x = x + dx
               else
                  output(j, i, k) = 0._r8
               end if
            end do
         else
            output(1:nsubcol,i,k) = 0._r8
         end if

      end do
   end do
end subroutine subcol_invgammafast



subroutine subcol_gengammafast(watertype, randseed, inshape, qc )
   character(len=3), intent(in)  :: watertype
   !real(r8), intent(in)  :: randseed(:, :)  ! (ncol, nlev)
   !real(r8), intent(in)  :: inshape(:, :)
   !real(r8), intent(in)  :: inscale(:, :)   ! (ncol, nlev)
   real(r8), intent(in)  :: randseed(ncol, nlev)  ! (ncol, nlev)
   real(r8), intent(in)  :: inshape(ncol, nlev)
   real(r8), intent(in)  :: qc(:, :)   ! (ncol, nlev)


!local variables
   integer i, j, k
   real(r8) :: icqc(ncol, nlev) !in-cloud qc
   real(r8) :: adjfac(ncol, nlev)
!   real(r8) :: cldcount(ncol, nlev)
   real(r8), pointer :: ptr(:,:,:)

   if ( (size(qc, 1) /= ncol) .or. &
      (size(qc, 2) /= nlev) ) then
      write(*,*) "subcolumn shape no match!"
      call exit(1)
   end if

   !cldcount = 0._r8
   !do i = 1, nsubcol
      !cldcount = cldcount + sc_cld(i,:,:)
   !end do

   icqc = qc/(cldcount/nsubcol) !obtain in-cloud quantity

   if ( watertype .eq. 'lmr') then
      call subcol_invgammafast(  randseed, inshape, icqc/inshape, sc_cldlmr)
      ptr => sc_cldlmr
   else if ( watertype .eq. 'imr') then
      call subcol_invgammafast(  randseed, inshape, icqc/inshape, sc_cldimr)
      ptr => sc_cldimr
   else if ( watertype .eq. 'lwp') then
      call subcol_invgammafast(  randseed, inshape, icqc/inshape, sc_cldlwp)
      ptr => sc_cldlwp
   else if ( watertype .eq. 'iwp') then
      call subcol_invgammafast(  randseed, inshape, icqc/inshape, sc_cldiwp)
      ptr => sc_cldiwp
   end if

   adjfac = 0._r8
   do i = 1, nsubcol
      adjfac = adjfac + ptr(i,:,:)
   end do
   adjfac = adjfac/nsubcol

   do i = 1, ncol
      do j = 1, nlev
         if ( (adjfac(i,j) == 0._r8) .or. (cldcount(i,j) == 0._r8) ) then
            adjfac(i,j) = 0._r8
         else
!           write(*,"(5f15.9)") inshape(i,j)*inscale(i,j)
            adjfac(i,j) = qc(i,j)/adjfac(i,j)
         end if
      end do
   end do

   do i = 1, nsubcol
      ptr(i,:,:) = ptr(i,:,:)*adjfac
   end do

   !adjfac = 0._r8
   !do i = 1, nsubcol
      !adjfac = adjfac + ptr(i,:,:)
   !end do
   !adjfac = adjfac/nsubcol
   
   !do j = 1, nlev
      !write(*,*) adjfac(1,j), qc(1,j), cldcount(1,j)/nsubcol
   !end do

end subroutine subcol_gengammafast



subroutine subcol_genuni(watertype, cldwater)
   character(len=3), intent(in)  :: watertype
   real(r8), intent(in)  :: cldwater(:, :)   ! (ncol, nlev)

!local variables
   integer i, j, k
   real(r8), pointer :: ptr(:,:,:)

   if ( (size(cldwater, 1) /= ncol) .or. &
      (size(cldwater, 2) /= nlev) ) then
      write(*,*) "subcolumn shape no match!"
      call exit(1)
   end if

   if ( watertype .eq. 'lmr') then
      ptr => sc_cldlmr
   else if ( watertype .eq. 'imr') then
      ptr => sc_cldimr
   else if ( watertype .eq. 'lwp') then
      ptr => sc_cldlwp
   else if ( watertype .eq. 'iwp') then
      ptr => sc_cldiwp
   end if
   
   ptr = 0._r8
   do i = 1, ncol
      do k = 1, nlev
         if (cld(i,k) > 0._r8) then
            ptr(1:nsubcol,i,k) = cldwater(i,k)
            do j = 1, nsubcol
               if ( sc_cld(j,i,k) == 0._r8 ) then
                  ptr(j,i,k) = 0._r8
               end if
            end do
         else
            ptr(1:nsubcol,i,k) = 0._r8
         end if
      end do
   end do

end subroutine subcol_genuni



subroutine subcol_get(watertype, numsubcol, output )
   character*(*), intent(in)  :: watertype
   integer, intent(in)    :: numsubcol
   real(r8), intent(out)  :: output(:, :, :)   ! (numsubcol, ncol, nlev)

   integer n
!local variables
   integer i, j, k
   real(r8) :: adjfac(ncol, nlev)
   integer innsubcol, inncol, innlev


   innsubcol = size( output, 1)
   inncol = size( output, 2)
   innlev = size( output, 3)
!   write(*,*) size(sc_cldlmr,1), size(sc_cldlmr,2), size(sc_cldlmr,3)

   if ( (numsubcol .ne. innsubcol) .or. (innsubcol .gt. nsubcol)) then
      write(*,*) 'ERROR: getting inconsistent number of subcolumn'
      call exit
   else
!      write(*,*) 'getting ', watertype
      if ( trim(watertype) .eq. 'lmr') then
         output = sc_cldlmr(1:nsubcol, :, :)
      else if ( trim(watertype) .eq. 'imr') then
         output = sc_cldimr(1:nsubcol, :, :)
      else if ( trim(watertype) .eq. 'lwp') then
         output = sc_cldlwp(1:nsubcol, :, :)
      else if ( trim(watertype) .eq. 'iwp') then
         output = sc_cldiwp(1:nsubcol, :, :)
      else if ( trim(watertype) .eq. 'cld') then
         output = sc_cld(1:nsubcol, :, :)
      else
         write(*,*) 'ERROR: getting unrecognized subcolumn'
         call exit
      end if
   end if

   !adjfac = 0._r8
   !cldcount = 0._r8
   !do i = 1, n
      !adjfac = adjfac + output(i,:,:)
      !cldcount = cldcount + sc_cld(i,:,:)
   !end do
   !!adjfac = adjfac/cldcount

   !do i = 1, ncol
      !do j = 1, nlev
         !if ( (adjfac(i,j) .eq. 0._r8) .or. (cldcount(i,j) .eq. 0._r8) ) then
            !adjfac(i,j) = 0._r8
         !else
            !adjfac(i,j) = cldw(i,j)/adjfac(i,j)*cldcount(i,j)
!!           write(*,"(5f15.9)") inshape(i,j)*inscale(i,j)

!!           adjfac(i,j) = cldw(i,j)/adjfac(i,j)
         !end if
      !end do
   !end do

   !do i = 1, n
      !output(i,:,:) = output(i,:,:)*adjfac
   !end do

end subroutine subcol_get



subroutine subcol_ptr(watertype, numsubcol, output )
   character*(*), intent(in)  :: watertype
   integer, intent(in)    :: numsubcol
   real(r8), intent(inout), pointer  :: output(:, :, :)   ! (numsubcol, ncol, nlev)

   if ( numsubcol .gt. nsubcol) then
      write(*,*) 'ERROR: pointing to inconsistent number of subcolumn'
      call exit
   else
!     write(*,*) 'pointing to ', watertype
      if ( trim(watertype) .eq. 'lmr') then
         output => sc_cldlmr(1:numsubcol, :, :)
      else if ( trim(watertype) .eq. 'imr') then
         output => sc_cldimr(1:numsubcol, :, :)
      else if ( trim(watertype) .eq. 'lwp') then
         output => sc_cldlwp(1:numsubcol, :, :)
      else if ( trim(watertype) .eq. 'iwp') then
         output => sc_cldiwp(1:numsubcol, :, :)
      else if ( trim(watertype) .eq. 'cld') then
         output => sc_cld(1:numsubcol, :, :)
      else
         write(*,*) 'ERROR: pointing to unregcognized subcolumn'
         call exit
      end if
   end if
end subroutine subcol_ptr



integer function subcol_getnumsubcol()
   subcol_getnumsubcol = nsubcol
end function subcol_getnumsubcol



subroutine subcol_diag(watertype)
   integer i, j, k
   character*(*) watertype
   real(r8) :: subcol_mean(nlev)
   real(r8) :: cldcount(nlev)

   real(r8), pointer :: ptr(:,:,:)

   write(*,"(a5, a15,i5,a15,i5,a15,i5)") trim(watertype), "nsubcol:", nsubcol,&
      "ncol:", ncol, "nlev:", nlev

   if ( watertype .eq. 'lmr') then
      ptr => sc_cldlmr
   else if ( watertype .eq. 'imr') then
      ptr => sc_cldimr
   else if ( watertype .eq. 'lwp') then
      ptr => sc_cldlwp
   else if ( watertype .eq. 'iwp') then
      ptr => sc_cldiwp
   else if ( watertype .eq. 'cld') then
      ptr => sc_cld
   else
      write(*,*) "ERROR: diagnostise an nonexisting subcolum variable"
      call exit
   end if

   do i = 1, ncol
      subcol_mean = 0._r8
      cldcount = 0._r8
      do j = 1, nsubcol

         if ( mod(j,10) == 1 ) then
            write(*,"(a5,a10,i5,a10,i5)") trim(watertype), "icol:", i, "isubcol:", j
            write(*,"(5f15.9)") ptr(j, i, :)
         end if

         subcol_mean = subcol_mean+ptr(j, i, :)
         cldcount = cldcount+sc_cld(j, i, :)
      end do
      if ( watertype == 'cld') then
         subcol_mean = subcol_mean/nsubcol
      else
         do k=1, nlev
            if ( cldcount(k) == 0._r8) then
               subcol_mean(k) = 0._r8
            else
               subcol_mean(k) = subcol_mean(k)/cldcount(k)
            end if
         end do
      end if

      write(*,"(a5,a15)") trim(watertype),"subcolmean:"
      write(*,"(5f15.9)") subcol_mean(:)
   end do

end subroutine subcol_diag



! use the bisection method to find the inverse of gamma function range from 0. to 500.
function subcol_inverse_gamma_cdf( x, shape_p, scale_p)
   real(r8), intent(in)  :: x, shape_p, scale_p
   real(r8) :: subcol_inverse_gamma_cdf

   integer  :: i, ifault, n
   real(r8) :: xmin, xmax, error, minerror=1.e-12_r8
   real(r8) :: xa, xb, xc, fa, fb, fc
   real(r8) :: p

!   write(*,*) "subcol_inverse_gamma_cdf:", x, shape_p, scale_p
   if ( (scale_p .le. 1.e-12_r8) .or. (shape_p .le. 1.e-12_r8) ) then
      subcol_inverse_gamma_cdf = scale_p*shape_p
!     write(*,*) "a=0!!!"
      return
   end if

   xa = 0._r8
   xb = 10._r8
   fa = gammad(xa/scale_p, shape_p, ifault)-x
   fb = gammad(xb/scale_p, shape_p, ifault)-x

   if ( fa*fb > 0 ) then
      write(*,*) "(subcol_inverse_gamma_cdf)There might be zero or more than one roots between initial a and b!"
      subcol_inverse_gamma_cdf = scale_p*shape_p
      return
   end if

   n = 1
   error = abs(xa-xb)
   do while (error > minerror)
      xc = (xa+xb)/2
      fc = gammad(xc/scale_p, shape_p, ifault)-x
      if (fc*fa > 0) then
         xa = xc
      else
         xb = xc
      end if
      error = abs(xa-xb)
      n = n + 1
!     write(*,"(3f15.8)") xc, fc, error
   end do

   subcol_inverse_gamma_cdf = xc
end function subcol_inverse_gamma_cdf


!-------------------------------------------------------------------------------------------------- 
subroutine subcol_kissvec_2d(seed1,seed2,seed3,seed4,ran_arr)
!-------------------------------------------------------------------------------------------------- 
! public domain code from NCAR CAM

   real(kind=r8), dimension(:,:,:), intent(inout)  :: ran_arr        ! nsubcol X ncol X nlev
   integer, dimension(:), intent(inout) :: seed1,seed2,seed3,seed4   ! ncol
   integer :: i,j,sz,kiss
   integer :: nc, ns, nl
   integer :: m, k, n

! inline function 
   m(k, n) = ieor (k, ishft (k, n) )

   ns = size(ran_arr, 1)
   nc = size(ran_arr, 2)
   nl = size(ran_arr, 3)

   do i = 1, nc

      do k = 1, nl
         do j = 1, ns
            seed1(i) = 69069 * seed1(i) + 1327217885
            seed2(i) = m (m (m (seed2(i), 13), - 17), 5)
            seed3(i) = 18000 * iand (seed3(i), 65535) + ishft (seed3(i), - 16)
            seed4(i) = 30903 * iand (seed4(i), 65535) + ishft (seed4(i), - 16)
            kiss = seed1(i) + seed2(i) + ishft (seed3(i), 16) + seed4(i)
            ran_arr(j,i,k) = kiss*2.328306e-10 + 0.5_r8
         end do
      end do

   end do
end subroutine subcol_kissvec_2d


!-------------------------------------------------------------------------------------------------- 
subroutine subcol_kissvec(seed1,seed2,seed3,seed4,ran_arr)
!-------------------------------------------------------------------------------------------------- 
! public domain code from NCAR CAM

   real(kind=r8), dimension(:), intent(inout)  :: ran_arr
   integer, dimension(:), intent(inout) :: seed1,seed2,seed3,seed4
   integer :: i,sz,kiss
   integer :: m, k, n

! inline function 
   m(k, n) = ieor (k, ishft (k, n) )

   sz = size(ran_arr)
   do i = 1, sz
      seed1(i) = 69069 * seed1(i) + 1327217885
      seed2(i) = m (m (m (seed2(i), 13), - 17), 5)
      seed3(i) = 18000 * iand (seed3(i), 65535) + ishft (seed3(i), - 16)
      seed4(i) = 30903 * iand (seed4(i), 65535) + ishft (seed4(i), - 16)
      kiss = seed1(i) + seed2(i) + ishft (seed3(i), 16) + seed4(i)
      ran_arr(i) = kiss*2.328306e-10 + 0.5_r8
   end do
end subroutine subcol_kissvec


FUNCTION subcol_gamma(X)

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!D    DOUBLE PRECISION FUNCTION DGAMMA(X)
!----------------------------------------------------------------------
!
! THIS ROUTINE CALCULATES THE GAMMA FUNCTION FOR A REAL ARGUMENT X.
!   COMPUTATION IS BASED ON AN ALGORITHM OUTLINED IN REFERENCE 1.
!   THE PROGRAM USES RATIONAL FUNCTIONS THAT APPROXIMATE THE GAMMA
!   FUNCTION TO AT LEAST 20 SIGNIFICANT DECIMAL DIGITS.  COEFFICIENTS
!   FOR THE APPROXIMATION OVER THE INTERVAL (1,2) ARE UNPUBLISHED.
!   THOSE FOR THE APPROXIMATION FOR X .GE. 12 ARE FROM REFERENCE 2.
!   THE ACCURACY ACHIEVED DEPENDS ON THE ARITHMETIC SYSTEM, THE
!   COMPILER, THE INTRINSIC FUNCTIONS, AND PROPER SELECTION OF THE
!   MACHINE-DEPENDENT CONSTANTS.
!
!
!*******************************************************************
!*******************************************************************
!
! EXPLANATION OF MACHINE-DEPENDENT CONSTANTS
!
! BETA   - RADIX FOR THE FLOATING-POINT REPRESENTATION
! MAXEXP - THE SMALLEST POSITIVE POWER OF BETA THAT OVERFLOWS
! XBIG   - THE LARGEST ARGUMENT FOR WHICH GAMMA(X) IS REPRESENTABLE
!          IN THE MACHINE, I.E., THE SOLUTION TO THE EQUATION
!                  GAMMA(XBIG) = BETA**MAXEXP
! XINF   - THE LARGEST MACHINE REPRESENTABLE FLOATING-POINT NUMBER;
!          APPROXIMATELY BETA**MAXEXP
! EPS    - THE SMALLEST POSITIVE FLOATING-POINT NUMBER SUCH THAT
!          1.0+EPS .GT. 1.0
! XMININ - THE SMALLEST POSITIVE FLOATING-POINT NUMBER SUCH THAT
!          1/XMININ IS MACHINE REPRESENTABLE
!
!     APPROXIMATE VALUES FOR SOME IMPORTANT MACHINES ARE:
!
!                            BETA       MAXEXP        XBIG
!
! CRAY-1         (S.P.)        2         8191        966.961
! CYBER 180/855
!   UNDER NOS    (S.P.)        2         1070        177.803
! IEEE (IBM/XT,
!   SUN, ETC.)   (S.P.)        2          128        35.040
! IEEE (IBM/XT,
!   SUN, ETC.)   (D.P.)        2         1024        171.624
! IBM 3033       (D.P.)       16           63        57.574
! VAX D-FORMAT   (D.P.)        2          127        34.844
! VAX G-FORMAT   (D.P.)        2         1023        171.489
!
!                            XINF         EPS        XMININ
!
! CRAY-1         (S.P.)   5.45E+2465   7.11E-15    1.84E-2466
! CYBER 180/855
!   UNDER NOS    (S.P.)   1.26E+322    3.55E-15    3.14E-294
! IEEE (IBM/XT,
!   SUN, ETC.)   (S.P.)   3.40E+38     1.19E-7     1.18E-38
! IEEE (IBM/XT,
!   SUN, ETC.)   (D.P.)   1.79D+308    2.22D-16    2.23D-308
! IBM 3033       (D.P.)   7.23D+75     2.22D-16    1.39D-76
! VAX D-FORMAT   (D.P.)   1.70D+38     1.39D-17    5.88D-39
! VAX G-FORMAT   (D.P.)   8.98D+307    1.11D-16    1.12D-308
!
!*******************************************************************
!*******************************************************************
!
! ERROR RETURNS
!
!  THE PROGRAM RETURNS THE VALUE XINF FOR SINGULARITIES OR
!     WHEN OVERFLOW WOULD OCCUR.  THE COMPUTATION IS BELIEVED
!     TO BE FREE OF UNDERFLOW AND OVERFLOW.
!
!
!  INTRINSIC FUNCTIONS REQUIRED ARE:
!
!     INT, DBLE, EXP, LOG, REAL, SIN
!
!
! REFERENCES:  AN OVERVIEW OF SOFTWARE DEVELOPMENT FOR SPECIAL
!              FUNCTIONS   W. J. CODY, LECTURE NOTES IN MATHEMATICS,
!              506, NUMERICAL ANALYSIS DUNDEE, 1975, G. A. WATSON
!              (ED.), SPRINGER VERLAG, BERLIN, 1976.
!
!              COMPUTER APPROXIMATIONS, HART, ET. AL., WILEY AND
!              SONS, NEW YORK, 1968.
!
!  LATEST MODIFICATION: OCTOBER 12, 1989
!
!  AUTHORS: W. J. CODY AND L. STOLTZ
!           APPLIED MATHEMATICS DIVISION
!           ARGONNE NATIONAL LABORATORY
!           ARGONNE, IL 60439
!
!----------------------------------------------------------------------
      INTEGER I,N
      LOGICAL PARITY

      real(r8) subcol_gamma
      REAL(r8) &
!D    DOUBLE PRECISION
         C,CONV,EPS,FACT,HALF,ONE,P,PI,Q,RES,SQRTPI,SUM,TWELVE, &
         TWO,X,XBIG,XDEN,XINF,XMININ,XNUM,Y,Y1,YSQ,Z,ZERO
      DIMENSION C(7),P(8),Q(8)
!----------------------------------------------------------------------
!  MATHEMATICAL CONSTANTS
!----------------------------------------------------------------------
      DATA ONE,HALF,TWELVE,TWO,ZERO/1.0E0_r8,0.5E0_r8,12.0E0_r8,2.0E0_r8,0.0E0_r8/, &
          SQRTPI/0.9189385332046727417803297E0_r8/, &
          PI/3.1415926535897932384626434E0_r8/
!D    DATA ONE,HALF,TWELVE,TWO,ZERO/1.0D0,0.5D0,12.0D0,2.0D0,0.0D0/,
!D   1     SQRTPI/0.9189385332046727417803297D0/,
!D   2     PI/3.1415926535897932384626434D0/
!----------------------------------------------------------------------
!  MACHINE DEPENDENT PARAMETERS
!----------------------------------------------------------------------
      DATA XBIG,XMININ,EPS/35.040E0_r8,1.18E-38_r8,1.19E-7_r8/, &
          XINF/3.4E38_r8/
!D    DATA XBIG,XMININ,EPS/171.624D0,2.23D-308,2.22D-16/,
!D   1     XINF/1.79D308/
!----------------------------------------------------------------------
!  NUMERATOR AND DENOMINATOR COEFFICIENTS FOR RATIONAL MINIMAX
!     APPROXIMATION OVER (1,2).
!----------------------------------------------------------------------
      DATA P/-1.71618513886549492533811E+0_r8,2.47656508055759199108314E+1_r8,&
            -3.79804256470945635097577E+2_r8,6.29331155312818442661052E+2_r8,&
            8.66966202790413211295064E+2_r8,-3.14512729688483675254357E+4_r8,&
            -3.61444134186911729807069E+4_r8,6.64561438202405440627855E+4_r8/
      DATA Q/-3.08402300119738975254353E+1_r8,3.15350626979604161529144E+2_r8,&
           -1.01515636749021914166146E+3_r8,-3.10777167157231109440444E+3_r8,&
             2.25381184209801510330112E+4_r8,4.75584627752788110767815E+3_r8,&
           -1.34659959864969306392456E+5_r8,-1.15132259675553483497211E+5_r8/
!D    DATA P/-1.71618513886549492533811D+0,2.47656508055759199108314D+1,
!D   1       -3.79804256470945635097577D+2,6.29331155312818442661052D+2,
!D   2       8.66966202790413211295064D+2,-3.14512729688483675254357D+4,
!D   3       -3.61444134186911729807069D+4,6.64561438202405440627855D+4/
!D    DATA Q/-3.08402300119738975254353D+1,3.15350626979604161529144D+2,
!D   1      -1.01515636749021914166146D+3,-3.10777167157231109440444D+3,
!D   2        2.25381184209801510330112D+4,4.75584627752788110767815D+3,
!D   3      -1.34659959864969306392456D+5,-1.15132259675553483497211D+5/
!----------------------------------------------------------------------
!  COEFFICIENTS FOR MINIMAX APPROXIMATION OVER (12, INF).
!----------------------------------------------------------------------
      DATA C/-1.910444077728E-03_r8,8.4171387781295E-04_r8, &
          -5.952379913043012E-04_r8,7.93650793500350248E-04_r8,&
          -2.777777777777681622553E-03_r8,8.333333333333333331554247E-02_r8,&
           5.7083835261E-03_r8/
!D    DATA C/-1.910444077728D-03,8.4171387781295D-04,
!D   1     -5.952379913043012D-04,7.93650793500350248D-04,
!D   2     -2.777777777777681622553D-03,8.333333333333333331554247D-02,
!D   3      5.7083835261D-03/
!----------------------------------------------------------------------
!  STATEMENT FUNCTIONS FOR CONVERSION BETWEEN INTEGER AND FLOAT
!----------------------------------------------------------------------
      CONV(I) = REAL(I,r8)
!D    CONV(I) = DBLE(I)
      PARITY=.FALSE.
      FACT=ONE
      N=0
      Y=X
      IF(Y.LE.ZERO)THEN
!----------------------------------------------------------------------
!  ARGUMENT IS NEGATIVE
!----------------------------------------------------------------------
        Y=-X
        Y1=AINT(Y)
        RES=Y-Y1
        IF(RES.NE.ZERO)THEN
          IF(Y1.NE.AINT(Y1*HALF)*TWO)PARITY=.TRUE.
          FACT=-PI/SIN(PI*RES)
          Y=Y+ONE
        ELSE
          RES=XINF
          GOTO 900
        ENDIF
      ENDIF
!----------------------------------------------------------------------
!  ARGUMENT IS POSITIVE
!----------------------------------------------------------------------
      IF(Y.LT.EPS)THEN
!----------------------------------------------------------------------
!  ARGUMENT .LT. EPS
!----------------------------------------------------------------------
        IF(Y.GE.XMININ)THEN
          RES=ONE/Y
        ELSE
          RES=XINF
          GOTO 900
        ENDIF
      ELSEIF(Y.LT.TWELVE)THEN
        Y1=Y
        IF(Y.LT.ONE)THEN
!----------------------------------------------------------------------
!  0.0 .LT. ARGUMENT .LT. 1.0
!----------------------------------------------------------------------
          Z=Y
          Y=Y+ONE
        ELSE
!----------------------------------------------------------------------
!  1.0 .LT. ARGUMENT .LT. 12.0, REDUCE ARGUMENT IF NECESSARY
!----------------------------------------------------------------------
          N=INT(Y)-1
          Y=Y-CONV(N)
          Z=Y-ONE
        ENDIF
!----------------------------------------------------------------------
!  EVALUATE APPROXIMATION FOR 1.0 .LT. ARGUMENT .LT. 2.0
!----------------------------------------------------------------------
        XNUM=ZERO
        XDEN=ONE
        DO 260 I=1,8
          XNUM=(XNUM+P(I))*Z
          XDEN=XDEN*Z+Q(I)
  260   CONTINUE
        RES=XNUM/XDEN+ONE
        IF(Y1.LT.Y)THEN
!----------------------------------------------------------------------
!  ADJUST RESULT FOR CASE  0.0 .LT. ARGUMENT .LT. 1.0
!----------------------------------------------------------------------
          RES=RES/Y1
        ELSEIF(Y1.GT.Y)THEN
!----------------------------------------------------------------------
!  ADJUST RESULT FOR CASE  2.0 .LT. ARGUMENT .LT. 12.0
!----------------------------------------------------------------------
          DO 290 I=1,N
            RES=RES*Y
            Y=Y+ONE
  290     CONTINUE
        ENDIF
      ELSE
!----------------------------------------------------------------------
!  EVALUATE FOR ARGUMENT .GE. 12.0,
!----------------------------------------------------------------------
        IF(Y.LE.XBIG)THEN
          YSQ=Y*Y
          SUM=C(7)
          DO 350 I=1,6
            SUM=SUM/YSQ+C(I)
  350     CONTINUE
          SUM=SUM/Y-Y+SQRTPI
          SUM=SUM+(Y-HALF)*LOG(Y)
          RES=EXP(SUM)
        ELSE
          RES=XINF
          GOTO 900
        ENDIF
      ENDIF
!----------------------------------------------------------------------
!  FINAL ADJUSTMENTS AND RETURN
!----------------------------------------------------------------------
      IF(PARITY)RES=-RES
      IF(FACT.NE.ONE)RES=FACT/RES
  900 SUBCOL_GAMMA=RES
!D900 DGAMMA = RES
      RETURN

! ---------- LAST LINE OF GAMMA ----------
END function subcol_gamma


!-------------------------------------------------------
!Some outdated code for CESM subcolumn
!--------------------------------------------------------

subroutine subcol_cesm_rand(randseed, randomnumber)
   real(r8), intent(in)   :: randseed(:, :)  ! (ncol, nlev)
   real(r8), intent(out)  :: randomnumber(:, :, :)   ! (nsubcol, ncol, nlev)

   integer :: i, j, k
   integer :: istat
   integer :: ndi1, ndi2
   integer :: ndo1, ndo2, ndo3

   integer, allocatable :: seed1(:),seed2(:)
   integer, allocatable :: seed3(:),seed4(:)

   ndi1 = size(randseed, 1)
   ndi2 = size(randseed, 2)
   ndo1 = size(randomnumber, 1)
   ndo2 = size(randomnumber, 2)
   ndo3 = size(randomnumber, 3)

   write(*,"(a10,i5,a1,i5)") "input", ndi1, "X", ndi2
   write(*,"(a10,i5,a1,i5,a1,i5)") "output", ndo1, "X", ndo2, "X", ndo3

   allocate( seed1(ndi1), stat=istat )
   allocate( seed2(ndi1), stat=istat )
   allocate( seed3(ndi1), stat=istat )
   allocate( seed4(ndi1), stat=istat )

   do i = 1, ndi1
      seed1(i) = (randseed(i,ndi2)   - int(randseed(i,ndi2))  )  * 1000000000
      seed2(i) = (randseed(i,ndi2-1) - int(randseed(i,ndi2-1)))  * 1000000000
      seed3(i) = (randseed(i,ndi2-2) - int(randseed(i,ndi2-2)))  * 1000000000
      seed4(i) = (randseed(i,ndi2-3) - int(randseed(i,ndi2-3)))  * 1000000000
   end do
   call subcol_kissvec_2d(seed1, seed2, seed3, seed4, randomnumber)
!  do k = 1, ndo3
!     do i = 1, ndo2
!        write(*,"(2i5)") k, i
!        do j = 1, ndo1
!           write(*,"(f20.15)")  randomnumber(j,i,k)
!        end do
!     end do
!  end do

   deallocate( seed1 )
   deallocate( seed2 )
   deallocate( seed3 )
   deallocate( seed4 )
end subroutine subcol_cesm_rand



subroutine subcol_cesm_invgamma(inshape, inscale, randomnumber, output)
   real(r8), intent(in)  :: inshape(:, :)
   real(r8), intent(in)  :: inscale(:, :)   ! (ncol, nlev)
   real(r8), intent(in)  :: randomnumber(:, :, :) ! (nsubcol, ncol, nlev)

   real(r8), intent(out) :: output(:, :, :) ! (nsubcol, ncol, nlev)

   integer :: i, j, k
   integer :: ndi1, ndi2, ndi3

   ndi1 = size( randomnumber, 1)
   ndi2 = size( randomnumber, 2)
   ndi3 = size( randomnumber, 3)

   do k = 1, ndi3
      do i = 1, ndi2

         if ( (inshape(i,k) .le. 1.e-12_r8) .or. (inscale(i,k) .le. 1.e-12_r8) ) then
            output(j, i, k) = 0._r8
         else
            do j = 1, ndi1
               output(j, i, k) = subcol_inverse_gamma_cdf(randomnumber(j,i,k), &
                  inshape(i,k), inscale(i,k) )
            end do
         end if

      end do
   end do
end subroutine subcol_cesm_invgamma


end module subcolphys
