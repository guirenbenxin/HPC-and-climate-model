module startup_initialconds
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Module to set the initial conditions for model startup.
! 
!-----------------------------------------------------------------------
! $Id$
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r8 => shr_kind_r8
   use spmd_utils,   only: masterproc
   use pio,     only: file_desc_t
   implicit none
   private
!
! Public methods
!
   public setup_initial         ! Initial setup to prepare to read in initial conditions
   public initial_conds         ! Read in initial conditions
   public initial_file_get_id
   public topo_file_get_id
!=======Jinbo Xie=========
public topoGWD_file_get_id
public setup_initialGWD
public close_initial_fileGWD
!=======Jinbo Xie=========
   public close_initial_file

   type(file_desc_t), pointer :: ncid_ini, ncid_topo  

!============
! Jinbo Xie0
!============
type(file_desc_t), pointer :: ncid_topoGWD
!============
! Jinbo Xie0
!============

!======================================================================= 
contains
!======================================================================= 
function initial_file_get_id()
  type(file_desc_t), pointer :: initial_file_get_id
  initial_file_get_id => ncid_ini
end function initial_file_get_id

function topo_file_get_id()
  type(file_desc_t), pointer :: topo_file_get_id
  topo_file_get_id => ncid_topo
end function topo_file_get_id


!====================================
!       Jinbo Xie
function topoGWD_file_get_id()
  type(file_desc_t), pointer :: topoGWD_file_get_id
  topoGWD_file_get_id => ncid_topoGWD
end function topoGWD_file_get_id
!====================================



!
!======================================================================= 
!

subroutine setup_initial( )
!----------------------------------------------------------------------- 
! 
! Purpose:  Do initial setup needed for startup initial conditions. Open
! files and start time-manager.
! 
!----------------------------------------------------------------------- 
   use filenames,        only: ncdata, bnd_topo    ,&
                                       bnd_topo2 !Jinbo Xie
                                        !Jinbo Xie                  
   use ioFileMod,        only: getfil

   use cam_pio_utils,    only: cam_pio_openfile
   use pio,              only: pio_nowrite

   use readinitial,      only: read_initial
!
! Input arguments
!
!-----------------------------------------------------------------------
   include 'netcdf.inc'
!-----------------------------------------------------------------------

   character(len=256) :: ncdata_loc     ! filepath of initial file on local disk
   character(len=256) :: bnd_topo_loc   ! filepath of topo file on local disk
!=========Jinbo Xie==========
   !character(len=256) :: bnd_topo_loc2   ! filepath of topo file on local disk
!=========Jinbo Xie==========
   
   ! Open initial, topography, and landfrac datasets
   call getfil (ncdata, ncdata_loc)

   allocate(ncid_ini)
   call cam_pio_openfile(ncid_ini, ncdata_loc, PIO_NOWRITE, .TRUE.)
   ! Backward compatibility: look for topography data on initial file if topo file name not provided.
   if (trim(bnd_topo) /= 'bnd_topo' .and. len_trim(bnd_topo) > 0) then
      allocate(ncid_topo)
      call getfil(bnd_topo, bnd_topo_loc)
      call cam_pio_openfile(ncid_topo, bnd_topo_loc, PIO_NOWRITE)

!===================Jinbo Xie========================
!czy                allocate(ncid_topoGWD)
!czy                call getfil(bnd_topo2, bnd_topo_loc2)
!czy                call cam_pio_openfile(ncid_topoGWD, bnd_topo_loc2, PIO_NOWRITE)
!===================Jinbo Xie========================
   else
      ncid_topo => ncid_ini
   end if
!+czy20181120
!   if (trim(bnd_topo2) /= 'bnd_topo2' .and. len_trim(bnd_topo2) > 0) then
!      allocate(ncid_topoGWD)
!      call getfil(bnd_topo2, bnd_topo_loc2)
!      call cam_pio_openfile(ncid_topoGWD, bnd_topo_loc2, PIO_NOWRITE)
!   else
!      ncid_topoGWD => ncid_ini ! ????czy20181120
!   endif
!-czy20181120

   !
   ! Check for consistent settings on initial dataset
   !
   call read_initial (ncid_ini)
end subroutine setup_initial

!
!======================================================================= 
!
subroutine initial_conds( dyn_in )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Convert from inital to startup_initialconds module. Define initial 
! conditions for first run of case
! 
! Method: 
! 
! Author: 
! Original version:  CCM1
! Standardized:      L. Bath, June 1992
!                    T. Acker, March 1996
! Reviewed:          B. Boville, April 1996
!
!-----------------------------------------------------------------------
   use inidat,        only: read_inidat
   use buffer,        only: initialize_buffer
   use comsrf,        only: initialize_comsrf
   use radae,         only: initialize_radbuffer
   use phys_buffer,   only: pbuf_allocate
   use dyn_comp,      only: dyn_import_t
   use cam_pio_utils, only: clean_iodesc_list
#if (defined BFB_CAM_SCAM_IOP )
   use history_defaults, only: initialize_iop_history
#endif

!-----------------------------------------------------------------------
   implicit none
!------------------------------Parameters-------------------------------
!---------------------------Arguments-----------------------------------
   type(dyn_import_t),  intent(inout) :: dyn_in

!---------------------------Local variables-----------------------------

   !
   ! Initialize buffer, comsrf, and radbuffer variables 
   ! (which must occur after the call to phys_grid_init)
   !
   call pbuf_allocate('global')
   call initialize_buffer
   call initialize_comsrf
   call initialize_radbuffer

#if (defined BFB_CAM_SCAM_IOP )
   call initialize_iop_history
#endif
   !
   ! Read in initial data
   !
!---------------------------- test zhh ------------------------
   if (masterproc) print*, 'In sub. initial_conds: before calling read_inidat'
!---------------------------- test zhh ------------------------

   call read_inidat( ncid_ini, ncid_topo, dyn_in )
!---------------------------- test zhh ------------------------
   if (masterproc) print*, 'In sub. initial_conds: after calling read_inidat'
!---------------------------- test zhh ------------------------

   call clean_iodesc_list()

   call print_memusage ('post-inidat')

end subroutine initial_conds


subroutine close_initial_file
  use pio,          only: pio_closefile

  if(associated(ncid_ini)) then
     if(.not. associated(ncid_ini, target=ncid_topo)) then
        call pio_closefile(ncid_topo)
        deallocate(ncid_topo)
     end if
!+czy20181122
     !if(.not. associated(ncid_ini, target=ncid_topoGWD)) then
     !   call pio_closefile(ncid_topoGWD)
     !   deallocate(ncid_topoGWD)
     !end if
!-czy20181122
     
     call pio_closefile(ncid_ini)
     deallocate(ncid_ini)
     nullify(ncid_ini)
     nullify(ncid_topo)
     !nullify(ncid_topoGWD) !czy20181122
  end if
end subroutine close_initial_file



!======Jinbo Xie file==============
subroutine setup_initialGWD()
   use filenames,        only: bnd_topo2 !Jinbo Xie
   use ioFileMod,        only: getfil
   use cam_pio_utils,    only: cam_pio_openfile
   use pio,              only: pio_nowrite
!
! Input arguments
!
!-----------------------------------------------------------------------
   include 'netcdf.inc'
!-----------------------------------------------------------------------
!=========Jinbo Xie==========
   character(len=256) :: bnd_topo_loc2   ! filepath of topo file on local disk
!=========Jinbo Xie==========
!+czy20181120
      allocate(ncid_topoGWD)
      call getfil(bnd_topo2, bnd_topo_loc2)
      call cam_pio_openfile(ncid_topoGWD, bnd_topo_loc2, PIO_NOWRITE)
!-czy20181120
end subroutine setup_initialGWD


subroutine close_initial_fileGWD
  use pio,          only: pio_closefile
        call pio_closefile(ncid_topoGWD)
        deallocate(ncid_topoGWD)
        nullify(ncid_topoGWD) !czy20181122
end subroutine close_initial_fileGWD
!======Jinbo Xie file==============
!
!======================================================================= 
!

end module startup_initialconds
