#include<xjb_drag3.inc>
!#define continuous 1
!-----------------------------------------------------------------------
!
! !MODULE: comsrf
!
! !DESCRIPTION:	Module to handle surface fluxes for the subcomponents of cam/csm
!                    Currently this is a hodge-podge of 2D arrays without a lot
!                    of thought or design. We are under the process of removing
!                    this completely and moving the relevent arrays to the modules
!                    that actually use the data.
!
!			See: http://swiki.ucar.edu/start/66
!
!-----------------------------------------------------------------------
module comsrf
!
! USES:
!
  use shr_kind_mod, only: r8 => shr_kind_r8, r4 => shr_kind_r4
!====Jinbo Xie====
  !use ppgrid, only: pcols, begchunk, endchunk
   use ppgrid, only: pcols, begchunk, endchunk,nvar_dirOA,nvar_dirOL,indexb
!====Jinbo Xie====
  use infnan, only: nan
  use abortutils, only: endrun

  implicit none

!----------------------------------------------------------------------- 
! PRIVATE: Make default data and interfaces private
!----------------------------------------------------------------------- 
  private     ! By default all data is private to this module
!
! ! PUBLIC MEMBER FUNCTIONS:
!
  public initialize_comsrf          ! Set the surface temperature and sea-ice fraction
!====Jinbo Xie====
public initialize_comsrf2
!====Jinbo Xie====
!
! Public data
!
  public landm, sgh, sgh30, fv, ram1, fsns, fsds
  public fsnt, flns, flnt, srfrpdel, psm1, prcsnw
  public trefmxav, trefmnav

  real(r8), allocatable:: landm(:,:)     ! land/ocean/sea ice flag
  real(r8), allocatable:: sgh(:,:)       ! land/ocean/sea ice flag
  real(r8), allocatable:: sgh30(:,:)     ! land/ocean/sea ice flag
  real(r8), allocatable:: fv(:,:)        ! needed for dry dep velocities (over land)
  real(r8), allocatable:: ram1(:,:)      ! needed for dry dep velocities (over land)
  real(r8), allocatable:: fsns(:,:)      ! surface absorbed solar flux
  real(r8), allocatable:: fsds(:,:)      ! downward solar flux
  real(r8), allocatable:: fsnt(:,:)      ! Net column abs solar flux at model top
  real(r8), allocatable:: flns(:,:)      ! Srf longwave cooling (up-down) flux
  real(r8), allocatable:: flnt(:,:)      ! Net outgoing lw flux at model top
  real(r8), allocatable:: srfrpdel(:,:)  ! 1./(pint(k+1)-pint(k))
  real(r8), allocatable:: psm1(:,:)      ! surface pressure
  real(r8), allocatable:: prcsnw(:,:)    ! cam tot snow precip
  real(r8), allocatable:: trefmxav(:,:)  ! diagnostic: tref max over the day
  real(r8), allocatable:: trefmnav(:,:)  ! diagnostic: tref min over the day

!================
! Jinbo Xie1
!================
        public var,oc,oadir
        real(r8), allocatable:: var(:,:) ! Convexity
        !real(r8), allocatable:: hgt_m(:,:) !
        real(r8), allocatable:: oc(:,:) ! Convexity
        real(r8), allocatable:: oadir(:,:,:) ! Asymmetry
#ifndef continuous
        public ol
        real(r8), allocatable:: ol(:,:,:) ! Effective length
        !public dxydir
        !real(r8), allocatable:: dxydir(:,:,:) ! Effective length
#else
        !for oldir
        public  terrout
        real(r8), allocatable,dimension(:,:,:,:) :: terrout
#endif
!================
! Jinbo Xie1
!================

! Private module data

!===============================================================================
CONTAINS
!===============================================================================

!======================================================================
! PUBLIC ROUTINES: Following routines are publically accessable
!======================================================================
!----------------------------------------------------------------------- 
! 
! BOP
!
! !IROUTINE: initialize_comsrf
!
! !DESCRIPTION:
!
! Initialize the procedure for specifying sea surface temperatures
! Do initial read of time-varying ice boundary dataset, reading two
! consecutive months on either side of the current model date.
!
! Method: 
! 
! Author: 
! 
!-----------------------------------------------------------------------
!
! !INTERFACE
!
  subroutine initialize_comsrf
  use cam_control_mod,  only: ideal_phys, adiabatic
!-----------------------------------------------------------------------
!
! Purpose:
! Initialize surface data
!
! Method:
!
! Author: Mariana Vertenstein
!
!-----------------------------------------------------------------------
    integer k,c      ! level, constituent indices

    if(.not. (adiabatic .or. ideal_phys)) then
       allocate (landm   (pcols,begchunk:endchunk))
       allocate (sgh     (pcols,begchunk:endchunk))
       allocate (sgh30   (pcols,begchunk:endchunk))

       allocate (fv      (pcols,begchunk:endchunk))
       allocate (ram1    (pcols,begchunk:endchunk))
       allocate (fsns    (pcols,begchunk:endchunk))         
       allocate (fsds    (pcols,begchunk:endchunk))         
       allocate (fsnt    (pcols,begchunk:endchunk))         
       allocate (flns    (pcols,begchunk:endchunk))         
       allocate (flnt    (pcols,begchunk:endchunk))         
       allocate (srfrpdel(pcols,begchunk:endchunk))
       allocate (psm1    (pcols,begchunk:endchunk))
       allocate (prcsnw  (pcols,begchunk:endchunk))
       allocate (trefmxav(pcols,begchunk:endchunk))
       allocate (trefmnav(pcols,begchunk:endchunk))


       !
       ! Initialize to NaN or Inf
       ! elements of the array outside valid surface points must be set to
       ! zero if these fields are to be written to netcdf history files.
       !
       landm    (:,:) = nan
       sgh      (:,:) = nan
       sgh30    (:,:) = nan
       fsns     (:,:) = nan
       fsds     (:,:) = nan
       fsnt     (:,:) = nan
       flns     (:,:) = nan
       flnt     (:,:) = nan
       srfrpdel (:,:) = nan
       psm1     (:,:) = nan
       prcsnw   (:,:) = nan
       trefmxav (:,:) = -1.0e36_r8
       trefmnav (:,:) =  1.0e36_r8
    end if
  end subroutine initialize_comsrf

        
  subroutine initialize_comsrf2
  use cam_control_mod,  only: ideal_phys, adiabatic
  integer k,c      ! level, constituent indices
  if(.not. (adiabatic .or. ideal_phys)) then
                allocate (var(pcols,begchunk:endchunk))
                allocate (oc(pcols,begchunk:endchunk))
                allocate (oadir(pcols,nvar_dirOA,begchunk:endchunk))
#ifndef continuous
                allocate (ol(pcols,nvar_dirOL,begchunk:endchunk))
                ol  (:,:,:) = nan
                !allocate (dxydir(pcols,nvar_dirOL,begchunk:endchunk))
                !dxydir  (:,:,:) = nan
#else
        !=====Jinbo Xie=====
        !for oldir
        allocate(terrout(4,pcols,indexb,begchunk:endchunk))
        terrout=nan
#endif
        var   (:,:) = nan
        oc    (:,:) = nan
        oadir (:,:,:) = nan
  endif
  end subroutine initialize_comsrf2
end module comsrf
