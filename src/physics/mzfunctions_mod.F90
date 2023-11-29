module mzfunctions_mod
!------------------------------------------------------------------------------------------------
! Purpose:    Calculate ramp functions
! Author :    Minghua Zhang
! Completed : 2018-08-15
! Update    
!------------------------------------------------------------------------------------------------

   use shr_kind_mod, only: r8 => shr_kind_r8
   use units,           only: getunit, freeunit
   use time_manager,    only: get_nstep
   use ppgrid,        only: pcols, pver, pverp

   implicit none

   save
   public

CONTAINS
!================================================================================================
!================================================================================================

  subroutine mzfunc1(nx,x,y1, xmin, xmax,x0,xscale, iflag)

! =====================================================
!  x0 is the infection point
!  xscale is the contraction xscale in x
!  returned value from -1. to 1.
! -------------------------------------
! input
  integer :: nx
  integer :: iflag  ! -1 to return (-1. 1.), else return (0.0, 1.)
  real(8) :: x(nx)
  real(8) :: xmin, xmax, x0, xscale

! output
  real(8) :: y1(nx)

! local
  real(8) :: x1,x1min, x1max
  integer :: i

   if(xmin .eq. xmax) then 
    write(*,*)'xmin cannot be the same as xmax, stop in zmh_function'
    stop
   end if

   x1min = (xmin - x0)*xscale
   x1max = (xmax - x0)*xscale

   do i=1,nx
    x1     = (x(i)    - x0)*xscale 
    y1(i)  = (atan( -x1) - atan(-x1min) )/(atan(-x1max) - atan(-x1min)) 

    if(iflag .eq.  -1)then
       y1(i)  = 2.0*y1(i) - 1. 
    end if

   end do

  return

 end subroutine

!================================================================================================

  subroutine fout2d(d2,nnx,nny,nx0,ny0,var)

! work with IDL program iap_proc1.pro to view fields

  real(8)      :: d2(nnx,nny)
  integer      :: nnx,nny,nx0,ny0
  integer      :: nstep
  character(*) :: var

  character(len=160) :: filename
  character(len=6) :: strt,strx,stry
  logical      :: file_exists
  integer      :: unitn,J,K

  nstep = get_nstep()

  write(strt,'(I6.6)')nstep
  write(strx,'(I6.6)')nx0
  write(stry,'(I6.6)')ny0

  filename = 'fout/'//var//'_'//strt//'_'//strx//'_'//stry//'.txt'
  INQUIRE(FILE=trim(filename), EXIST=file_exists)

  if(file_exists)then 
  !    return   !!
      filename = trim(filename)//'1'
  endif

  write(*,*)filename
  unitn = getunit()
  open( unitn, file=trim(filename),status='unknown')
   write(unitn,'(8I8)') nstep,nnx,nny,nx0,ny0
   do J = 1, nny
     write(unitn,'(1000(5E15.7/))') d2(1:nnx,j)
   enddo
  write(unitn,*)'min=',minval(d2)
  write(unitn,*)'max=',maxval(d2)
  close(unitn)
  call freeunit(unitn)
               

  return

 end subroutine


!================================================================================================

  subroutine fout3d(d3,nnx,nnz,nny,nx0,nz0,ny0,var)

! work with IDL program iap_proc1.pro to view fields

  real(8)      :: d3(nnx,nnz,nny)
  integer      :: nnx,nnz,nny,nx0,nz0,ny0
  integer      :: nstep
  character(*) :: var

  character(len=160) :: filename
  character(len=6) :: strt,strx,stry,strz
  logical      :: file_exists
  integer      :: unitn,J,K

  nstep = get_nstep()

  write(strt,'(I6.6)')nstep
  write(strx,'(I6.6)')nx0
  write(strz,'(I6.6)')nz0
  write(stry,'(I6.6)')ny0

  filename = 'fout/'//var//'_'//strt//'_'//strx//'_'//strz//'_'//stry//'.txt'
  INQUIRE(FILE=trim(filename), EXIST=file_exists)

  if(file_exists)then 
      return !!
      filename = trim(filename)//'1'
  endif

  write(*,*)filename
  unitn = getunit()
  open( unitn, file=trim(filename),status='unknown')
   write(unitn,'(8I8)') nstep,nnx,nnz,nny,nx0,nz0,ny0
   do J = 1, nny
    do k = 1, nnz
     write(unitn,'(1000(5E15.7/))') d3(1:nnx,k,j)
    enddo
   enddo
  write(unitn,*)'min=',minval(d3)
  write(unitn,*)'max=',maxval(d3)
  close(unitn)
  call freeunit(unitn)


  return

 end subroutine


!================================================================================================

  subroutine fout_phy(d3,lchnk,nnx,nnz,lat,lon,var)

! work with IDL program iap_proc3.pro to view fields

  real(8)      :: d3(nnx,nnz),lat(nnx),lon(nnx)
  integer      :: nnx,nnz,lchnk
  integer      :: nstep
  character(*) :: var

  character(len=160) :: filename
  character(len=6) :: strt,strx
  logical      :: file_exists
  integer      :: unitn,K

  nstep = get_nstep()

  write(strt,'(I6.6)')nstep
  write(strx,'(I6.6)')lchnk

  filename = 'fout/'//var//'_'//strt//'_'//strx//'.txt'
  INQUIRE(FILE=trim(filename), EXIST=file_exists)

  if(file_exists)then 
      return !!
      filename = trim(filename)//'1'
  endif

  write(*,*)filename
  unitn = getunit()
  open( unitn, file=trim(filename),status='unknown')
   write(unitn,'(8I8)') nstep,lchnk,nnx,nnz

     write(unitn,*)'lat'
     write(unitn,'(1000(5f15.3/))') lat(1:nnx)*180./3.1416
     write(unitn,*)'lon'
     write(unitn,'(1000(5f15.3/))') lon(1:nnx)*180./3.1416

  if(nnz == 1)then
     write(unitn,'(1000(5E15.7/))') d3
  else
    do k = 1, nnz
     write(unitn,'(1000(5E15.7/))') d3(1:nnx,k)
    enddo
  endif

  write(unitn,*)'min=',minval(d3)
  write(unitn,*)'max=',maxval(d3)
  close(unitn)
  call freeunit(unitn)

  return

 end subroutine

! =====================================================

real(r8) function zmh_ramp(x,x0,xint)

! =====================================================
!  =0 when x<x0
!  =1 when x>x0+xint
!  = linear in between
!
! -------------------------------------
! input
  implicit none
  real(r8), intent(in) :: x,x0,xint
  real(r8)  x2

   if(x < x0) then
     x2 = 0.
   else if(x > x0+xint)then
     x2=1.
   else
     x2 = (x-x0)/max(xint,1.e-20)
   endif
     x2 = min(x2,1.)

  zmh_ramp = x2

 return

 end function zmh_ramp

 
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

!===============================================================================

subroutine zmh_rand_normal(seed1,seed2,seed3,seed4,ran_arr,ran_arr2)

!by Minghua Zhang using  Box-Muller transform

   integer, dimension(:), intent(inout) :: seed1,seed2,seed3,seed4
   real(kind=r8), dimension(:), intent(out)  :: ran_arr,ran_arr2

   real(kind=r8) pi2, u1,u2,u, v

   integer :: i,sz

   pi2=6.2832_r8

   sz = size(ran_arr)

   call subcol_kissvec(seed1,seed2,seed3,seed4,ran_arr)

   call subcol_kissvec(seed2,seed3,seed4,seed1,ran_arr2)
!-------------------------------------------------------------------------------------------------- 
   do i = 1, sz
     U1 = max(ran_arr(i),1.0e-5_r8)
     U2 = -2._r8* log(U1)
     U = sqrt( U2)
     V = pi2*ran_arr2(i)
     ran_arr(i)  = U*cos(V)
     ran_arr2(i) = U*sin(V)
   enddo

end subroutine zmh_rand_normal 

end module mzfunctions_mod
