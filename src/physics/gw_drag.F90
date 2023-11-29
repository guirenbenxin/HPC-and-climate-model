!
module gw_drag

!---------------------------------------------------------------------------------
! Purpose:
!
! Module to compute the forcing due to parameterized gravity waves. Both an 
! orographic and an internal source spectrum are considered.
!
! Author: Byron Boville
!
! Modified: Zhang He, 2012-01-14, important revision in subroutine gw_oro
!           Zhang He, 2013-02-25, ubi(i,pver) < 1E-12 --> abs(ubi(i,pver)) < 1E-12
!           Zhang He, 2013-03-12
!---------------------------------------------------------------------------------
  use shr_kind_mod,  only: r8 => shr_kind_r8
  use spmd_utils,    only: masterproc
!czy20181116  use ppgrid,        only: pcols, pver
!czy20181116  use physics_types, only: physics_state, physics_ptend
!czy20181116  use cam_history,   only: outfld
!czy20181116  use scamMod,       only: single_column
  use cam_logfile,   only: iulog
  use abortutils,    only: endrun

  implicit none
  save
  private                         ! Make default type private to the module
!
! PUBLIC: interfaces
!
  public gw_drag_readnl           ! Read namelist
!  public gw_inti                  ! Initialization
!  public gw_intr                  ! interface to actual parameterization

  !+czy20181116
  integer, public :: gw_drag_scheme = 0  ! 1 use cam5 gw_drag
                                  ! 2 use waccm gw_drag
                                  ! 0 not defined, reset to 1 to make model
                                  ! robust and give warning

  real(r8), parameter :: unset_r8 = huge(1._r8)

  ! fcrit2 has been made a namelist variable to facilitate backwards compatibility 
  ! with the CAM3 version of this parameterization.  In CAM3 fcrit2=0.5
  real(r8), public :: fcrit2 = unset_r8   ! critical froude number squared
  real(r8), public :: effgw_beres=0.1_r8  ! beres source tendency efficiency
  character(len=256), public :: gw_drag_file = 'Beres04_file'
  integer, public  :: idx_zmdt = -1

  !-czy20181116
!===============================================================================
contains
!===============================================================================

subroutine gw_drag_readnl(nlfile)

   use namelist_utils,  only: find_group_name
   use units,           only: getunit, freeunit
   use mpishorthand

   character(len=*), intent(in) :: nlfile  ! filepath for file containing namelist input

   ! Local variables
   integer :: unitn, ierr
   character(len=*), parameter :: subname = 'gw_drag_readnl'

   namelist /gw_drag_nl/ gw_drag_scheme, fcrit2, gw_drag_file, effgw_beres
   !-----------------------------------------------------------------------------

   if (masterproc) then
      unitn = getunit()
      open( unitn, file=trim(nlfile), status='old' )
      call find_group_name(unitn, 'gw_drag_nl', status=ierr)
      if (ierr == 0) then
         read(unitn, gw_drag_nl, iostat=ierr)
         if (ierr /= 0) then
            call endrun(subname // ':: ERROR reading namelist')
         end if
      end if
      close(unitn)
      call freeunit(unitn)
   end if
!+czy20181116
   if (masterproc) then
   ! Error checking:
   ! check if gw_drag_scheme was set.
           if (gw_drag_scheme == 0) then
                gw_drag_scheme = 1
                write(iulog,*)subname // ':: WARNING gw_drag_scheme = 0, reset to ',gw_drag_scheme
           endif
           if (gw_drag_scheme == 1) then
                write(iulog,*)subname//':: Use NCAR CAM5 gravity-wave drag scheme'
           elseif (gw_drag_scheme == 2) then
                write(iulog,*)subname//':: Use NCAR WACCM gravity-wave drag scheme'
           elseif (gw_drag_scheme == 3) then
                write(iulog,*)subname//':: Use Xiejinbo gravity-wave drag scheme'
           else
                write(iulog,*)subname//':: ERROR gw_drag_scheme = ',gw_drag_scheme
                call endrun(subname//':: ERROR gw_drag_scheme must be set 1,2,3')
           endif
   ! check if fcrit2 was set.
           if (fcrit2 == unset_r8) then
                 call endrun('gw_drag_readnl: fcrit2 must be set via the namelist')
           end if
           if (gw_drag_scheme == 2) then
                fcrit2 = 1.0    ! reset for waccm gw drag
                write(iulog,*)subname//':: WARNING gw_drag_scheme = ',gw_drag_scheme
                write(iulog,*)subname//':: WARNING Use NCAR WACCM gravity-wave drag scheme'
                write(iulog,*)subname//':: WARNING For WACCM gw drag, Reset fcrit2 to ',fcrit2 
           endif

   endif
!-czy20181116

#ifdef SPMD
   ! Broadcast namelist variables
   call mpibcast (fcrit2      , 1                , mpir8  , 0, mpicom)
   call mpibcast (gw_drag_scheme, 1                , mpiint , 0, mpicom) !czy20181116
   call mpibcast (gw_drag_file, len(gw_drag_file), mpichar, 0, mpicom) !czy20181116
   call mpibcast (effgw_beres , 1                , mpir8  , 0, mpicom) !czy20181116
#endif



end subroutine gw_drag_readnl

!================================================================================

end module gw_drag

