
module ppgrid

!----------------------------------------------------------------------- 
! 
! Purpose: 
! Initialize physics grid resolution parameters
!  for a chunked data structure
! 
! Author: 
! 
!-----------------------------------------------------------------------

  implicit none
  private
  save
 
  public begchunk
  public endchunk
  public pcols
  public pver
  public pverp
!====Jinbo Xie====
public nvar_dirOA
public nvar_dirOL
public indexb
!====Jinbo Xie====

! Grid point resolution parameters

   integer pcols      ! number of columns (max)
   integer pver       ! number of vertical levels
   integer pverp      ! pver + 1
!====Jinbo Xie====
   integer nvar_dirOA
   integer nvar_dirOL
   integer indexb
!====Jinbo Xie====


!Jinbo Xie
!parameter (nvar =18 )
!Jinbo Xie
  ! parameter (pcols  = PCOLS)
   parameter (pcols  = 1024)  !by zhenzhen wang
   parameter (pver   = PLEV)!PLEV
   parameter (pverp  = pver + 1  )
   parameter (nvar_dirOA =2+1 )!Jinbo Xie avoid bug when nvar_dirOA is 2
   parameter (nvar_dirOL =720)!180)!4)!360)
   parameter (indexb = 3232)!63)

!
! start, end indices for chunks owned by a given MPI task
! (set in phys_grid_init).
!
   integer :: begchunk = 0            ! 
   integer :: endchunk = -1           ! 

end module ppgrid
