!------------------------------------------------------------------------------------------------
! juanxiong he, 2017/05/10
!------------------------------------------------------------------------------------------------
module geatm_aerosol_cycle

!------------------------------------------------------------------------------------------------
! GEATM AEROSOL was used in radiation calculation.
!
! Purpose:
! Provides distributions of aerosols from GEATM
! Read aerosol flux from dataset 
!
! Author: Juanxiong He              
!                                              
!------------------------------------------------------------------------------------------------

use shr_kind_mod,   only: r8 => shr_kind_r8
use spmd_utils,     only: masterproc
use ppgrid,         only: pver
use abortutils,     only: endrun

implicit none
 
! Public interfaces
public geatm_aerosol_register                  ! register consituents

! Namelist variables
logical :: geatm_aerosol_flag            = .true.      ! true => turn on geatm
!-----------------------------------------------------------------------
! new constituents
integer, parameter :: gcnst=18                      ! number of constituents implemented
integer, dimension(gcnst) :: gptr ! index in pbuf
character(len=7), dimension(gcnst), parameter :: & ! constituent names
   g_names = (/'dust01 ','dust02 ','dust03 ','dust04 ','sea01  ','sea02  ','sea03  ','sea04  ',&
               'ppm    ','bc     ','oc     ','nh4    ','no3    ','so4    ','h2o2   ','so2    ',&
               'dms    ','nh3    '/)

!================================================================================================
contains
!================================================================================================

subroutine geatm_aerosol_register
use phys_buffer, only: pbuf_add
!----------------------------------------------------------------------- 
! 
! Purpose: register advected constituents 
! 
!-----------------------------------------------------------------------
   integer  :: i

   if (.not. geatm_aerosol_flag) return

   do i = 1, gcnst
    call pbuf_add(g_names(i), 'physpkg', 1, pver, 1, gptr(i))
   end do

end subroutine geatm_aerosol_register
 
end module geatm_aerosol_cycle
