

module zm_conv_intr
!---------------------------------------------------------------------------------
! Purpose:
!
! CAM interface to the Zhang-McFarlane deep convection scheme
!
! Author: D.B. Coleman
! January 2010 modified by J. Kay to add COSP simulator fields to physics buffer
!---------------------------------------------------------------------------------
   use shr_kind_mod, only: r8=>shr_kind_r8
   use physconst,    only: cpair                              
   use ppgrid,       only: pver, pcols, pverp, begchunk, endchunk
   use zm_conv,      only: zm_conv_evap, zm_convr, convtran, momtran
   use cam_history,  only: outfld, addfld, add_default, phys_decomp
   use perf_mod
   use cam_logfile,  only: iulog
   use phys_buffer,  only: pbuf_add
   implicit none
   private
   save

   ! Public methods

   public ::&
      zm_conv_register,           &! register fields in physics buffer
      zm_conv_init,               &! initialize donner_deep module
      zm_conv_tend,               &! return tendencies
      zm_conv_tend_2               ! return tendencies

   ! Private module data

   real(r8), allocatable, dimension(:,:,:) :: mu  !(pcols,pver,begchunk:endchunk)
   real(r8), allocatable, dimension(:,:,:) :: eu  !(pcols,pver,begchunk:endchunk)
   real(r8), allocatable, dimension(:,:,:) :: du  !(pcols,pver,begchunk:endchunk)
   real(r8), allocatable, dimension(:,:,:) :: md  !(pcols,pver,begchunk:endchunk)
   real(r8), allocatable, dimension(:,:,:) :: ed  !(pcols,pver,begchunk:endchunk)
   real(r8), allocatable, dimension(:,:,:) :: dp  !(pcols,pver,begchunk:endchunk) 
	! wg layer thickness in mbs (between upper/lower interface).
   real(r8), allocatable, dimension(:,:)   :: dsubcld  !(pcols,begchunk:endchunk)
	! wg layer thickness in mbs between lcl and maxi.

   integer, allocatable, dimension(:,:) :: jt   !(pcols,begchunk:endchunk)
        ! wg top  level index of deep cumulus convection.
   integer, allocatable, dimension(:,:) :: maxg !(pcols,begchunk:endchunk)
        ! wg gathered values of maxi.
   integer, allocatable, dimension(:,:) :: ideep !(pcols,begchunk:endchunk)               
	! w holds position of gathered points vs longitude index

   integer, allocatable, dimension(:) :: lengath !(begchunk:endchunk)

   integer ::& ! indices for fields in the physics buffer
      dp_flxprc_idx, &
      dp_flxsnw_idx, &
      dp_cldliq_idx, &
      dp_cldice_idx

!  indices for fields in the physics buffer
   integer  ::    cld_idx          = 0    
   integer  ::    icwmrdp_idx      = 0     
   integer  ::    rprddp_idx       = 0    
   integer  ::    fracis_idx       = 0   
   integer  ::    nevapr_dpcu_idx  = 0    

!wxc zmh
   integer  ::    slflxdp_idx      = 0
   integer  ::    qtflxdp_idx      = 0


!=========================================================================================
contains
!=========================================================================================

subroutine zm_conv_register

!----------------------------------------
! Purpose: register fields with the physics buffer
!----------------------------------------


  implicit none

  integer idx

   call pbuf_add('DP_FLXPRC', 'global', 1, pverp, 1, dp_flxprc_idx) ! Flux of precipitation from deep convection (kg/m2/s)
   call pbuf_add('DP_FLXSNW', 'global', 1, pverp, 1, dp_flxsnw_idx) ! Flux of snow from deep convection (kg/m2/s) 
   call pbuf_add('DP_CLDLIQ', 'global', 1, pver,  1, dp_cldliq_idx) ! deep gbm cloud liquid water (kg/kg)
   call pbuf_add('DP_CLDICE', 'global', 1, pver,  1, dp_cldice_idx) ! deep gbm cloud liquid water (kg/kg)    

end subroutine zm_conv_register

!=========================================================================================

subroutine zm_conv_init(hypi)

!----------------------------------------
! Purpose:  declare output fields, initialize variables needed by convection
!----------------------------------------

  use cam_history,    only: outfld, addfld, add_default, phys_decomp
  use ppgrid,         only: pcols, pver
  use zm_conv,        only: zm_convi
  use pmgrid,         only: plev,plevp
  use spmd_utils,     only: masterproc
  use error_messages, only: alloc_err	
  use phys_control,   only: phys_deepconv_pbl, phys_getopts, cam_physpkg_is
  use phys_buffer,   only:  pbuf_get_fld_idx

  implicit none

  real(r8),intent(in) :: hypi(plevp)        ! reference pressures at interfaces


  logical :: no_deep_pbl    ! if true, no deep convection in PBL
  integer  limcnv           ! top interface level limit for convection
  integer k, istat
  logical :: history_budget ! output tendencies and state variables for CAM4
                            ! temperature, water vapor, cloud ice and cloud
                            ! liquid budgets.
  integer :: history_budget_histfile_num ! output history file number for budget fields

!
! Allocate space for arrays private to this module
!
     allocate( mu(pcols,pver,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'mu', &
                      pcols*pver*((endchunk-begchunk)+1) )
     allocate( eu(pcols,pver,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'eu', &
                      pcols*pver*((endchunk-begchunk)+1) )
     allocate( du(pcols,pver,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'du', &
                      pcols*pver*((endchunk-begchunk)+1) )
     allocate( md(pcols,pver,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'md', &
                      pcols*pver*((endchunk-begchunk)+1) )
     allocate( ed(pcols,pver,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'ed', &
                      pcols*pver*((endchunk-begchunk)+1) )
     allocate( dp(pcols,pver,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'dp', &
                      pcols*pver*((endchunk-begchunk)+1) )
     allocate( dsubcld(pcols,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'dsubcld', &
                      pcols*((endchunk-begchunk)+1) )
     allocate( jt(pcols,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'jt', &
                      pcols*((endchunk-begchunk)+1) )
     allocate( maxg(pcols,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'maxg', &
                      pcols*((endchunk-begchunk)+1) )
     allocate( ideep(pcols,begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'ideep', &
                      pcols*((endchunk-begchunk)+1) )
     allocate( lengath(begchunk:endchunk), stat=istat )
      call alloc_err( istat, 'zm_conv_tend', 'lengath', &
                      ((endchunk-begchunk)+1) )


! 
! Register fields with the output buffer
!


    call addfld ('PRECZ   ','m/s     ',1,    'A','total precipitation from ZM convection',        phys_decomp)
    call addfld ('ZMDT    ','K/s     ',pver, 'A','T tendency - Zhang-McFarlane moist convection', phys_decomp)
    call addfld ('ZMDQ    ','kg/kg/s ',pver, 'A','Q tendency - Zhang-McFarlane moist convection', phys_decomp)
    call addfld ('ZMDICE ','kg/kg/s ',pver, 'A','Cloud ice tendency - Zhang-McFarlane convection',phys_decomp)
    call addfld ('ZMDLIQ ','kg/kg/s ',pver, 'A','Cloud liq tendency - Zhang-McFarlane convection',phys_decomp)
    call addfld ('EVAPTZM ','K/s     ',pver, 'A','T tendency - Evaporation/snow prod from Zhang convection',phys_decomp)
    call addfld ('FZSNTZM ','K/s     ',pver, 'A','T tendency - Rain to snow conversion from Zhang convection',phys_decomp)
    call addfld ('EVSNTZM ','K/s     ',pver, 'A','T tendency - Snow to rain prod from Zhang convection',phys_decomp)
    call addfld ('EVAPQZM ','kg/kg/s ',pver, 'A','Q tendency - Evaporation from Zhang-McFarlane moist convection',phys_decomp)
    
    call addfld ('ZMFLXPRC','kg/m2/s ',pverp, 'A','Flux of precipitation from ZM convection'       ,phys_decomp)
    call addfld ('ZMFLXSNW','kg/m2/s ',pverp, 'A','Flux of snow from ZM convection'                ,phys_decomp)
    call addfld ('ZMNTPRPD','kg/kg/s ',pver , 'A','Net precipitation production from ZM convection',phys_decomp)
    call addfld ('ZMNTSNPD','kg/kg/s ',pver , 'A','Net snow production from ZM convection'         ,phys_decomp)
    call addfld ('ZMEIHEAT','W/kg'    ,pver , 'A','Heating by ice and evaporation in ZM convection',phys_decomp)
    
    call addfld ('CMFMCDZM','kg/m2/s ',pverp,'A','Convection mass flux from ZM deep ',phys_decomp)
    call addfld ('PRECCDZM','m/s     ',1,    'A','Convective precipitation rate from ZM deep',phys_decomp)
    call add_default ('CMFMCDZM', 1, ' ')
    call add_default ('PRECCDZM', 1, ' ')

    call addfld ('PCONVB','Pa'    ,1 , 'A','convection base pressure',phys_decomp)
    call addfld ('PCONVT','Pa'    ,1 , 'A','convection top  pressure',phys_decomp)
    call add_default ('PCONVB', 1, ' ')
    call add_default ('PCONVT', 1, ' ')
    

    call addfld ('CAPE',   'J/kg',       1, 'A', 'Convectively available potential energy', phys_decomp)
    call addfld ('FREQZM ','fraction  ',1  ,'A', 'Fractional occurance of ZM convection',phys_decomp) 
    call add_default ('FREQZM', 1, ' ')

    call addfld ('ZMMTT ', 'K/s',     pver, 'A', 'T tendency - ZM convective momentum transport',phys_decomp)
    call addfld ('ZMMTU',  'm/s2',    pver, 'A', 'U tendency - ZM convective momentum transport',  phys_decomp)
    call addfld ('ZMMTV',  'm/s2',    pver, 'A', 'V tendency - ZM convective momentum transport',  phys_decomp)

    call addfld ('ZMMU',   'kg/m2/s', pver, 'A', 'ZM convection updraft mass flux',   phys_decomp)
    call addfld ('ZMMD',   'kg/m2/s', pver, 'A', 'ZM convection downdraft mass flux', phys_decomp)

    call addfld ('ZMUPGU', 'm/s2',    pver, 'A', 'zonal force from ZM updraft pressure gradient term',       phys_decomp)
    call addfld ('ZMUPGD', 'm/s2',    pver, 'A', 'zonal force from ZM downdraft pressure gradient term',     phys_decomp)
    call addfld ('ZMVPGU', 'm/s2',    pver, 'A', 'meridional force from ZM updraft pressure gradient term',  phys_decomp)
    call addfld ('ZMVPGD', 'm/s2',    pver, 'A', 'merdional force from ZM downdraft pressure gradient term', phys_decomp)

    call addfld ('ZMICUU', 'm/s',     pver, 'A', 'ZM in-cloud U updrafts',      phys_decomp)
    call addfld ('ZMICUD', 'm/s',     pver, 'A', 'ZM in-cloud U downdrafts',    phys_decomp)
    call addfld ('ZMICVU', 'm/s',     pver, 'A', 'ZM in-cloud V updrafts',      phys_decomp)
    call addfld ('ZMICVD', 'm/s',     pver, 'A', 'ZM in-cloud V downdrafts',    phys_decomp)
    
    call phys_getopts(history_budget_out = history_budget, history_budget_histfile_num_out = history_budget_histfile_num)
    if ( history_budget ) then
       call add_default('EVAPTZM  ', history_budget_histfile_num, ' ')
       call add_default('EVAPQZM  ', history_budget_histfile_num, ' ')
       call add_default('ZMDT     ', history_budget_histfile_num, ' ')
       call add_default('ZMDQ     ', history_budget_histfile_num, ' ')
       call add_default('ZMDLIQ   ', history_budget_histfile_num, ' ')
       call add_default('ZMDICE   ', history_budget_histfile_num, ' ')

       if( cam_physpkg_is('cam4') .or. cam_physpkg_is('cam5') ) then
          call add_default('ZMMTT    ', history_budget_histfile_num, ' ')
       end if

    end if
!
! Limit deep convection to regions below 40 mb
! Note this calculation is repeated in the shallow convection interface
!
    limcnv = 0   ! null value to check against below
    if (hypi(1) >= 4.e3_r8) then
       limcnv = 1
    else
       do k=1,plev
          if (hypi(k) < 4.e3_r8 .and. hypi(k+1) >= 4.e3_r8) then
             limcnv = k
             exit
          end if
       end do
       if ( limcnv == 0 ) limcnv = plevp
    end if
    
    if (masterproc) then
       write(iulog,*)'ZM_CONV_INIT: Deep convection will be capped at intfc ',limcnv, &
            ' which is ',hypi(limcnv),' pascals'
    end if
        
    no_deep_pbl = phys_deepconv_pbl()
    call zm_convi(limcnv,no_deep_pbl_in = no_deep_pbl)

    cld_idx          = pbuf_get_fld_idx('CLD')
    icwmrdp_idx      = pbuf_get_fld_idx('ICWMRDP')
    rprddp_idx       = pbuf_get_fld_idx('RPRDDP')
    fracis_idx       = pbuf_get_fld_idx('FRACIS')
    nevapr_dpcu_idx  = pbuf_get_fld_idx('NEVAPR_DPCU')
!wxc zmh
    slflxdp_idx      = pbuf_get_fld_idx('slflxdp')
    qtflxdp_idx      = pbuf_get_fld_idx('qtflxdp')


end subroutine zm_conv_init
!=========================================================================================
!subroutine zm_conv_tend(state, ptend, tdt, pbuf)

subroutine zm_conv_tend(prec    , &
     pblh    ,mcon    ,cme     ,          &
     tpert   ,dlf     ,pflx    ,zdu      , &
     rliq    , &
     ztodt   ,snow    ,&
     jctop   ,jcbot , &
     state   ,ptend_all   ,landfrac   ,pbuf  )
  

   use cam_history,   only: outfld
   use physics_types, only: physics_state, physics_ptend, physics_tend
   use physics_types, only: physics_ptend_init,  physics_tend_init,physics_update
   use physics_types, only: physics_state_copy
   use physics_types, only: physics_ptend_sum

   use phys_grid,     only: get_lat_p, get_lon_p
   use time_manager,  only: get_nstep, is_first_step
   use phys_buffer,   only: pbuf_size_max, pbuf_fld, pbuf_old_tim_idx
   use constituents,  only: pcnst, cnst_get_ind
   use check_energy,  only: check_energy_chng
   use physconst,     only: gravit
   use phys_control,  only: cam_physpkg_is

   ! Arguments

   type(physics_state), intent(in ) :: state          ! Physics state variables
   type(physics_ptend), intent(out) :: ptend_all          ! indivdual parameterization tendencies
   type(pbuf_fld), intent(inout), dimension(pbuf_size_max) :: pbuf  ! physics buffer

   real(r8), intent(in) :: ztodt                       ! 2 delta t (model time increment)
   real(r8), intent(in) :: pblh(pcols)                 ! Planetary boundary layer height
   real(r8), intent(in) :: tpert(pcols)                ! Thermal temperature excess
   real(r8), intent(in) :: landfrac(pcols)             ! RBN - Landfrac 

   real(r8), intent(out) :: mcon(pcols,pverp)  ! Convective mass flux--m sub c
   real(r8), intent(out) :: dlf(pcols,pver)    ! scattrd version of the detraining cld h2o tend
   real(r8), intent(out) :: pflx(pcols,pverp)  ! scattered precip flux at each level
   real(r8), intent(out) :: cme(pcols,pver)    ! cmf condensation - evaporation
   real(r8), intent(out) :: zdu(pcols,pver)    ! detraining mass flux

   real(r8), intent(out) :: prec(pcols)   ! total precipitation
   real(r8), intent(out) :: snow(pcols)   ! snow from ZM convection 
   real(r8), intent(out) :: rliq(pcols) ! reserved liquid (not yet in cldliq) for energy integrals


   ! Local variables

   integer :: i,k,m
   integer :: ilon                      ! global longitude index of a column
   integer :: ilat                      ! global latitude index of a column
   integer :: nstep
   integer :: ixcldice, ixcldliq      ! constituent indices for cloud liquid and ice water.
   integer :: lchnk                   ! chunk identifier
   integer :: ncol                    ! number of atmospheric columns
   integer :: itim                    ! for physics buffer fields

   real(r8) :: ftem(pcols,pver)              ! Temporary workspace for outfld variables
   real(r8) :: ntprprd(pcols,pver)    ! evap outfld: net precip production in layer
   real(r8) :: ntsnprd(pcols,pver)    ! evap outfld: net snow production in layer
   real(r8) :: tend_s_snwprd  (pcols,pver) ! Heating rate of snow production
   real(r8) :: tend_s_snwevmlt(pcols,pver) ! Heating rate of evap/melting of snow
   real(r8) :: fake_dpdry(pcols,pver) ! used in convtran call

   ! physics types
   type(physics_state) :: state1        ! locally modify for evaporation to use, not returned
   type(physics_tend ) :: tend          ! Physics tendencies (empty, needed for physics_update call)
   type(physics_ptend) :: ptend_loc     ! package tendencies

   ! physics buffer fields
   real(r8), pointer, dimension(:,:) :: cld
   real(r8), pointer, dimension(:,:) :: ql           ! wg grid slice of cloud liquid water.
   real(r8), pointer, dimension(:,:) :: rprd         ! rain production rate
   real(r8), pointer, dimension(:,:,:) :: fracis  ! fraction of transported species that are insoluble
   real(r8), pointer, dimension(:,:) :: evapcdp      ! Evaporation of deep convective precipitation
   real(r8), pointer, dimension(:,:) :: flxprec      ! Convective-scale flux of precip at interfaces (kg/m2/s)
   real(r8), pointer, dimension(:,:) :: flxsnow      ! Convective-scale flux of snow   at interfaces (kg/m2/s)
   real(r8), pointer, dimension(:,:) :: dp_cldliq
   real(r8), pointer, dimension(:,:) :: dp_cldice

!wxc zmh
   real(r8), pointer, dimension(:,:) :: slflx           !
   real(r8), pointer, dimension(:,:) :: qtflx           !

   real(r8) :: jctop(pcols)  ! o row of top-of-deep-convection indices passed out.
   real(r8) :: jcbot(pcols)  ! o row of base of cloud indices passed out.

   real(r8) :: pcont(pcols), pconb(pcols), freqzm(pcols)

   ! history output fields
   real(r8) :: cape(pcols)        ! w  convective available potential energy.
   real(r8) :: mu_out(pcols,pver)
   real(r8) :: md_out(pcols,pver)

   ! used in momentum transport calculation
   real(r8) :: winds(pcols, pver, 2)
   real(r8) :: wind_tends(pcols, pver, 2)
   real(r8) :: pguall(pcols, pver, 2)
   real(r8) :: pgdall(pcols, pver, 2)
   real(r8) :: icwu(pcols,pver, 2)
   real(r8) :: icwd(pcols,pver, 2)
   real(r8) :: seten(pcols, pver)
   logical  :: l_windt(2)
   real(r8) :: tfinal1, tfinal2
   integer  :: ii
   !----------------------------------------------------------------------

   ! initialize
   lchnk = state%lchnk
   ncol  = state%ncol
   nstep = get_nstep()

   ftem = 0._r8   
   mu_out(:,:) = 0._r8
   md_out(:,:) = 0._r8
   wind_tends(:ncol,:pver,:) = 0.0_r8

   call physics_state_copy(state,state1)   ! copy state to local state1.
   call physics_ptend_init(ptend_loc)  ! initialize local ptend type
   call physics_ptend_init(ptend_all)  ! initialize output ptend type
   call physics_tend_init(tend)        ! tend type here is a null place holder


!
! Associate pointers with physics buffer fields
!
   itim = pbuf_old_tim_idx()
   cld => pbuf(cld_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)

   ql => pbuf(icwmrdp_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
   rprd => pbuf(rprddp_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
   fracis  => pbuf(fracis_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,1:pcnst)
   evapcdp => pbuf(nevapr_dpcu_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,1)

!wxc zmh
   slflx => pbuf(slflxdp_idx)%fld_ptr(1,1:pcols,1:pverp,lchnk,1)
   qtflx => pbuf(qtflxdp_idx)%fld_ptr(1,1:pcols,1:pverp,lchnk,1)

!
! Begin with Zhang-McFarlane (1996) convection parameterization
!
   call t_startf ('zm_convr')

   call zm_convr(   lchnk   ,ncol    , &
                    state%t       ,state%q     ,prec    ,jctop   ,jcbot   , &
                    pblh    ,state%zm      ,state%phis    ,state%zi      ,ptend_loc%q(:,:,1)    , &
                    ptend_loc%s    ,state%pmid     ,state%pint    ,state%pdel     , &
                    .5_r8*ztodt    ,mcon    ,cme     , cape,      &
                    tpert   ,dlf     ,pflx    ,zdu     ,rprd    , &
                    mu(:,:,lchnk),md(:,:,lchnk),du(:,:,lchnk),eu(:,:,lchnk),ed(:,:,lchnk)      , &
                    dp(:,:,lchnk) ,dsubcld(:,lchnk) ,jt(:,lchnk),maxg(:,lchnk),ideep(:,lchnk)   , &
!wxc                    lengath(lchnk) ,ql      ,rliq  ,landfrac   )
                    lengath(lchnk) ,ql      ,rliq  ,landfrac   , slflx, qtflx   )
i=1
if(i<0)then
write(*,*)'nstep=',nstep
write(*,*)'t',state%t 
write(*,*)'q',state%q(:,:,1)
write(*,*)'pbl',pblh
write(*,*)'tpert',tpert
write(*,*)'state%zm',state%zm  
write(*,*)'state%zi',state%zi  
write(*,*)'landfrac',landfrac
write(*,*)'state%pmid',state%pmid
write(*,*)'state%pdel',state%pdel
write(*,*)'cape',cape
write(*,*)'prec',prec
write(*,*)''
endif
   call outfld('CAPE', cape, pcols, lchnk)        ! RBN - CAPE output
!
! Output fractional occurance of ZM convection
!
   freqzm(:) = 0._r8
   do i = 1,lengath(lchnk)
      freqzm(ideep(i,lchnk)) = 1.0
   end do
   call outfld('FREQZM  ',freqzm          ,pcols   ,lchnk   )
!
! Convert mass flux from reported mb/s to kg/m^2/s
!
   mcon(:ncol,:pver) = mcon(:ncol,:pver) * 100._r8/gravit

   ! Store upward and downward mass fluxes in un-gathered arrays
   ! + convert from mb/s to kg/m^2/s
   do i=1,lengath(lchnk) 
      do k=1,pver
         ii = ideep(i,lchnk)
         mu_out(ii,k) = mu(i,k,lchnk) * 100._r8/gravit
         md_out(ii,k) = md(i,k,lchnk) * 100._r8/gravit
      end do
   end do

   call outfld('ZMMU', mu_out(1,1), pcols, lchnk)
   call outfld('ZMMD', md_out(1,1), pcols, lchnk)

   ptend_loc%name  = 'zm_convr'
   ptend_loc%ls    = .TRUE.
   ptend_loc%lq(1) = .TRUE.

   ftem(:ncol,:pver) = ptend_loc%s(:ncol,:pver)/cpair
   call outfld('ZMDT    ',ftem           ,pcols   ,lchnk   )
   call outfld('ZMDQ    ',ptend_loc%q(1,1,1) ,pcols   ,lchnk   )
   call t_stopf ('zm_convr')

!    do i = 1,pcols
!    do i = 1,nco
   pcont(:ncol) = state%ps(:ncol)
   pconb(:ncol) = state%ps(:ncol)
   do i = 1,lengath(lchnk)
       if (maxg(i,lchnk).gt.jt(i,lchnk)) then
          pcont(ideep(i,lchnk)) = state%pmid(ideep(i,lchnk),jt(i,lchnk))  ! gathered array (or jctop ungathered)
          pconb(ideep(i,lchnk)) = state%pmid(ideep(i,lchnk),maxg(i,lchnk))! gathered array
       endif
       !     write(iulog,*) ' pcont, pconb ', pcont(i), pconb(i), cnt(i), cnb(i)
    end do
    call outfld('PCONVT  ',pcont          ,pcols   ,lchnk   )
    call outfld('PCONVB  ',pconb          ,pcols   ,lchnk   )

  ! add tendency from this process to tendencies from other processes
  call physics_ptend_sum(ptend_loc,ptend_all, state)

  ! update physics state type state1 with ptend_loc 
  call physics_update(state1, tend, ptend_loc, ztodt)

  ! initialize ptend for next process
  call physics_ptend_init(ptend_loc)

   call t_startf ('zm_conv_evap')
!
! Determine the phase of the precipitation produced and add latent heat of fusion
! Evaporate some of the precip directly into the environment (Sundqvist)
! Allow this to use the updated state1 and the fresh ptend_loc type
! heating and specific humidity tendencies produced
!
    ptend_loc%name  = 'zm_conv_evap'
    ptend_loc%ls    = .TRUE.
    ptend_loc%lq(1) = .TRUE.

    flxprec    => pbuf(dp_flxprc_idx)%fld_ptr(1,1:pcols,1:pverp,lchnk,1)
    flxsnow    => pbuf(dp_flxsnw_idx)%fld_ptr(1,1:pcols,1:pverp,lchnk,1)
    dp_cldliq  => pbuf(dp_cldliq_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
    dp_cldice  => pbuf(dp_cldice_idx)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
    dp_cldliq(:ncol,:) = 0._r8
    dp_cldice(:ncol,:) = 0._r8

    call zm_conv_evap(state1%ncol,state1%lchnk, &
         state1%t,state1%pmid,state1%pdel,state1%q(:pcols,:pver,1), &
         ptend_loc%s, tend_s_snwprd, tend_s_snwevmlt, ptend_loc%q(:pcols,:pver,1), &
         rprd, cld, ztodt, &
         prec, snow, ntprprd, ntsnprd , flxprec, flxsnow)

    evapcdp(:ncol,:pver) = ptend_loc%q(:ncol,:pver,1)
!
! Write out variables from zm_conv_evap
!
   ftem(:ncol,:pver) = ptend_loc%s(:ncol,:pver)/cpair
   call outfld('EVAPTZM ',ftem           ,pcols   ,lchnk   )
   ftem(:ncol,:pver) = tend_s_snwprd  (:ncol,:pver)/cpair
   call outfld('FZSNTZM ',ftem           ,pcols   ,lchnk   )
   ftem(:ncol,:pver) = tend_s_snwevmlt(:ncol,:pver)/cpair
   call outfld('EVSNTZM ',ftem           ,pcols   ,lchnk   )
   call outfld('EVAPQZM ',ptend_loc%q(1,1,1) ,pcols   ,lchnk   )
   call outfld('ZMFLXPRC', flxprec, pcols, lchnk)
   call outfld('ZMFLXSNW', flxsnow, pcols, lchnk)
   call outfld('ZMNTPRPD', ntprprd, pcols, lchnk)
   call outfld('ZMNTSNPD', ntsnprd, pcols, lchnk)
   call outfld('ZMEIHEAT', ptend_loc%s, pcols, lchnk)
   call outfld('CMFMCDZM   ',mcon ,  pcols   ,lchnk   )
   call outfld('PRECCDZM   ',prec,  pcols   ,lchnk   )


   call t_stopf ('zm_conv_evap')

   call outfld('PRECZ   ', prec   , pcols, lchnk)

  ! add tendency from this process to tend from other processes here
  call physics_ptend_sum(ptend_loc,ptend_all, state)

  ! update physics state type state1 with ptend_loc 
  call physics_update(state1, tend, ptend_loc, ztodt)

  ! initialize ptend for next process
  call physics_ptend_init(ptend_loc)

  ! Momentum Transport (non-cam3 physics)

  if ( .not. cam_physpkg_is('cam3')) then

     winds(:ncol,:pver,1) = state1%u(:ncol,:pver)
     winds(:ncol,:pver,2) = state1%v(:ncol,:pver)
   
     l_windt(1) = .true.
     l_windt(2) = .true.

     call t_startf ('momtran')
     call momtran (lchnk, ncol,                                        &
                   l_windt,winds, 2,  mu(1,1,lchnk), md(1,1,lchnk),   &
                   du(1,1,lchnk), eu(1,1,lchnk), ed(1,1,lchnk), dp(1,1,lchnk), dsubcld(1,lchnk),  &
                   jt(1,lchnk),maxg(1,lchnk), ideep(1,lchnk), 1, lengath(lchnk),  &
                   nstep,  wind_tends, pguall, pgdall, icwu, icwd, ztodt, seten )  
     call t_stopf ('momtran')

     ptend_loc%lu = .TRUE.
     ptend_loc%lv = .TRUE.
     ptend_loc%ls = .TRUE.
   
     ptend_loc%u(:ncol,:pver) = wind_tends(:ncol,:pver,1)
     ptend_loc%v(:ncol,:pver) = wind_tends(:ncol,:pver,2)
     ptend_loc%s(:ncol,:pver) = seten(:ncol,:pver)  

     call physics_ptend_sum(ptend_loc,ptend_all, state)

     ! update physics state type state1 with ptend_loc 
     call physics_update(state1, tend, ptend_loc, ztodt)


     call physics_ptend_init(ptend_loc)

     ftem(:ncol,:pver) = seten(:ncol,:pver)/cpair
     call outfld('ZMMTT', ftem             , pcols, lchnk)
     call outfld('ZMMTU', wind_tends(1,1,1), pcols, lchnk)
     call outfld('ZMMTV', wind_tends(1,1,2), pcols, lchnk)
   
     ! Output apparent force from  pressure gradient
     call outfld('ZMUPGU', pguall(1,1,1), pcols, lchnk)
     call outfld('ZMUPGD', pgdall(1,1,1), pcols, lchnk)
     call outfld('ZMVPGU', pguall(1,1,2), pcols, lchnk)
     call outfld('ZMVPGD', pgdall(1,1,2), pcols, lchnk)

     ! Output in-cloud winds
     call outfld('ZMICUU', icwu(1,1,1), pcols, lchnk)
     call outfld('ZMICUD', icwd(1,1,1), pcols, lchnk)
     call outfld('ZMICVU', icwu(1,1,2), pcols, lchnk)
     call outfld('ZMICVD', icwd(1,1,2), pcols, lchnk)

   end if

   ! Transport cloud water and ice only
   call cnst_get_ind('CLDLIQ', ixcldliq)
   call cnst_get_ind('CLDICE', ixcldice)
   ptend_loc%name = 'convtran1'
   ptend_loc%lq(ixcldice) = .true.
   ptend_loc%lq(ixcldliq) = .true.

   ! dpdry is not used in this call to convtran since the cloud liquid and ice mixing
   ! ratios are moist
   fake_dpdry(:,:) = 0._r8

   call t_startf ('convtran1')
   call convtran (lchnk,                                        &
                  ptend_loc%lq,state1%q, pcnst,  mu(:,:,lchnk), md(:,:,lchnk),   &
                  du(:,:,lchnk), eu(:,:,lchnk), ed(:,:,lchnk), dp(:,:,lchnk), dsubcld(:,lchnk),  &
                  jt(:,lchnk),maxg(:,lchnk), ideep(:,lchnk), 1, lengath(lchnk),  &
                  nstep,   fracis,  ptend_loc%q, fake_dpdry)
   call t_stopf ('convtran1')

   call outfld('ZMDICE ',ptend_loc%q(1,1,ixcldice) ,pcols   ,lchnk   )
   call outfld('ZMDLIQ ',ptend_loc%q(1,1,ixcldliq) ,pcols   ,lchnk   )

  ! add tendency from this process to tend from other processes here
  call physics_ptend_sum(ptend_loc,ptend_all, state)

  ! ptend_all will be applied to original state on return to tphysbc
  ! This name triggers a special case in physics_types.F90:physics_update()
  ptend_all%name = 'convect_deep'

end subroutine zm_conv_tend
!=========================================================================================


subroutine zm_conv_tend_2( state,  ptend,  ztodt, pbuf  )

   use physics_types, only: physics_state, physics_ptend, physics_ptend_init
   use time_manager,  only: get_nstep
   use phys_buffer,   only: pbuf_size_max, pbuf_fld, pbuf_old_tim_idx, pbuf_get_fld_idx
   use constituents,  only: pcnst, cnst_get_ind
   use error_messages, only: alloc_err	
 
! Arguments
   type(physics_state), intent(in ) :: state          ! Physics state variables
   type(physics_ptend), intent(out) :: ptend          ! indivdual parameterization tendencies
   type(pbuf_fld), intent(inout), dimension(pbuf_size_max) :: pbuf  ! physics buffer
   real(r8), intent(in) :: ztodt                          ! 2 delta t (model time increment)

! Local variables
   integer :: i, lchnk, istat
   integer :: nstep
   integer :: ixcldice, ixcldliq              ! constituent indices for cloud liquid and ice water.
   real(r8), dimension(pcols,pver) :: dpdry

! physics buffer fields 
   integer itim, ifld
   real(r8), pointer, dimension(:,:,:) :: fracis  ! fraction of transported species that are insoluble

!
! Initialize
!
  call physics_ptend_init(ptend)

!
! Associate pointers with physics buffer fields
!
   ifld = pbuf_get_fld_idx('FRACIS')
   fracis  => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,state%lchnk,1:pcnst)

!
! Transport all constituents except cloud water and ice
!

  lchnk = state%lchnk

   nstep = get_nstep()

!
!     Convective transport of all trace species except cloud liquid 
!     and cloud ice done here because we need to do the scavenging first
!     to determine the interstitial fraction.
!
   call cnst_get_ind('CLDLIQ', ixcldliq)
   call cnst_get_ind('CLDICE', ixcldice)


   ptend%name  = 'convtran2'
   ptend%lq(:) = .true.
   ptend%lq(1)        = .false.
   ptend%lq(ixcldice) = .false.
   ptend%lq(ixcldliq) = .false.

   ! initialize dpdry for call to convtran
   ! it is used for tracers of dry mixing ratio type
   dpdry = 0._r8
   do i = 1,lengath(lchnk)
      dpdry(i,:) = state%pdeldry(ideep(i,lchnk),:)/100._r8
   end do

   call t_startf ('convtran2')
   call convtran (lchnk,                                        &
                  ptend%lq,state%q, pcnst,  mu(:,:,lchnk), md(:,:,lchnk),   &
                  du(:,:,lchnk), eu(:,:,lchnk), ed(:,:,lchnk), dp(:,:,lchnk), dsubcld(:,lchnk),  &
                  jt(:,lchnk),maxg(:,lchnk),ideep(:,lchnk), 1, lengath(lchnk),  &
                  nstep,   fracis,  ptend%q, dpdry)
   call t_stopf ('convtran2')

end subroutine zm_conv_tend_2

!=========================================================================================



end module zm_conv_intr
