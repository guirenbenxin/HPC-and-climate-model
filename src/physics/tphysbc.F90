#if ((defined geatm))
subroutine tphysbc (ztodt,   pblht,   tpert,   qpert,   tpert2,   qpert2,            &
                    fsns,    fsnt,    flns,    flnt,    state,   &
                    tend,    pbuf,    fsds,    landm,            &
                    cam_out, cam_in, cam_state ) ! juanxiong he
#elif ((defined wrf))
subroutine tphysbc (ztodt,   pblht,   tpert,   qpert,   tpert2,   qpert2,            &
                    fsns,    fsnt,    flns,    flnt,    state,   &
                    tend,    pbuf,    fsds,    landm,            &
                    cam_out, cam_in, cam_state, cam_tend ) ! juanxiong he
#else
subroutine tphysbc (ztodt,   pblht,   tpert,   qpert,   tpert2,   qpert2,            &
                    fsns,    fsnt,    flns,    flnt,    state,   &
                    tend,    pbuf,    fsds,    landm,            &
                    cam_out, cam_in )
#endif
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Evaluate and apply physical processes that are calculated BEFORE 
! coupling to land, sea, and ice models.  
!
! Processes currently included are: 
! dry adjustment, moist convection, stratiform, wet deposition, radiation
!
! Pass surface fields for separate surface flux calculations
! Dump appropriate fields to history file.
! 
! Method: 
!
! Each parameterization should be implemented with this sequence of calls:
!  1)  Call physics interface
!  2)  Check energy
!  3)  Call physics_update
! See Interface to Column Physics and Chemistry Packages 
!   http://www.ccsm.ucar.edu/models/atm-cam/docs/phys-interface/index.html
! 
! Author: CCM1, CMS Contact: J. Truesdale
!         modified by A. Gettelman and C. Craig Nov 2010 to separate micro/macro physics
! 
!-----------------------------------------------------------------------

   use shr_kind_mod,    only: r8 => shr_kind_r8
   use ppgrid
   use phys_buffer,     only: pbuf_size_max, pbuf_fld, pbuf_old_tim_idx, pbuf_get_fld_idx, pbuf_times
   use stratiform,      only: stratiform_tend
   use phys_control,    only: phys_getopts
   use microp_driver,   only: microp_driver_tend
   use macrop_driver,   only: macrop_driver_tend
   use physics_types,   only: physics_state, physics_tend, physics_ptend, physics_update, physics_ptend_init
   use cam_diagnostics, only: diag_conv_tend_ini, diag_phys_writeout, diag_conv, diag_export, diag_state_b4_phys_write
   use cam_history,     only: outfld
   use physconst,       only: cpair
   use constituents,    only: pcnst, qmin, cnst_get_ind
   use convect_deep,    only: convect_deep_tend, convect_deep_tend_2, deep_scheme_does_scav_trans
   use time_manager,    only: is_first_step, get_nstep
   use convect_shallow, only: convect_shallow_tend
   use check_energy,    only: check_energy_chng, check_energy_fix
   use check_energy,    only: check_tracers_data, check_tracers_init, check_tracers_chng
   use dycore,          only: dycore_is
   use aerosol_intr,    only: aerosol_wet_intr
   use camsrfexch_types,only: cam_out_t, cam_in_t
   use radiation,       only: radiation_tend
   use cloud_diagnostics, only: put_cloud_diagnostics
   use perf_mod
#ifdef MODAL_AERO
   use modal_aero_data, only: qneg3_worst_thresh_amode
#endif
   use mo_gas_phase_chemdr,only: map2chm
   use clybry_fam,         only: clybry_fam_adj
   use sslt_rebin,      only: sslt_rebin_adv
   use tropopause,      only: tropopause_output
!zmh
   use comsrf,          only: sgh30
   use mzfunctions_mod, only: fout_phy,zmh_ramp

   implicit none

!
! ===================================================

!
   real(r8), intent(in) :: ztodt                          ! 2 delta t (model time increment)
   real(r8), intent(in) :: pblht(pcols)                ! Planetary boundary layer height
!zmh
   real(r8), intent(in) :: tpert(pcols)                ! changed to inout
   real(r8), intent(in) :: qpert(pcols,pcnst)         ! Thermal humidity & constituent excess1
   real(r8), intent(inout) :: tpert2(pcols)                ! changed to inout
   real(r8), intent(inout) :: qpert2(pcols)         ! Thermal humidity & constituent excess1
   real(r8), intent(inout) :: fsns(pcols)                   ! Surface solar absorbed flux
   real(r8), intent(inout) :: fsnt(pcols)                   ! Net column abs solar flux at model top
   real(r8), intent(inout) :: flns(pcols)                   ! Srf longwave cooling (up-down) flux
   real(r8), intent(inout) :: flnt(pcols)                   ! Net outgoing lw flux at model top
   real(r8), intent(inout) :: fsds(pcols)                   ! Surface solar down flux
   real(r8), intent(in) :: landm(pcols)                   ! land fraction ramp

   type(physics_state), intent(inout) :: state
   type(physics_tend ), intent(inout) :: tend
   type(pbuf_fld),      intent(inout) :: pbuf(pbuf_size_max)
   type(cam_out_t),     intent(inout) :: cam_out
   type(cam_in_t),      intent(in)    :: cam_in
#if ((defined wrf) || (defined geatm))
   type(physics_state), intent(inout) :: cam_state ! juanxiong he
#endif
#if ((defined wrf))
   type(physics_tend ), intent(inout) :: cam_tend ! juanxiong he
#endif
!
!---------------------------Local workspace-----------------------------
!

   type(physics_state)   :: state_eq         ! equilibrium state variables
   type(physics_ptend)   :: ptend            ! indivdual parameterization tendencies

   integer :: nstep                          ! current timestep number
   integer      lat(pcols)                   ! current latitudes(indices)
   integer      lon(pcols)                   ! current longtitudes(indices)

   real(r8) :: net_flx(pcols)

   real(r8) :: zdu(pcols,pver)               ! detraining mass flux from deep convection
   real(r8) :: cmfmc(pcols,pverp)            ! Convective mass flux--m sub c

   real(r8) cmfcme(pcols,pver)                ! cmf condensation - evaporation
   real(r8) cmfmc2(pcols,pverp)               ! Moist convection cloud mass flux
   real(r8) coszrs(pcols)                     ! Cosine solar zenith angle
   real(r8) dlf(pcols,pver)                   ! Detraining cld H20 from shallow + deep convections
   real(r8) dlf2(pcols,pver)                  ! Detraining cld H20 from shallow convections
   real(r8) pflx(pcols,pverp)                 ! Conv rain flux thru out btm of lev
   real(r8) rtdt                              ! 1./ztodt

   integer lchnk                              ! chunk identifier
   integer ncol                               ! number of atmospheric columns

   integer  i,k,m                             ! Longitude, level, constituent indices
   integer :: ixcldice, ixcldliq              ! constituent indices for cloud liquid and ice water.
                                           

   real(r8) dellow(pcols)                     ! delta p for bottom three levels of model
   real(r8) tavg(pcols)                       ! mass weighted average temperature for 

! physics buffer fields to compute tendencies for stratiform package
   integer itim, ifld
   real(r8), pointer, dimension(:,:) :: cld        ! cloud fraction


! physics buffer fields for total energy and mass adjustment
   real(r8), pointer, dimension(:  ) :: teout
   real(r8), pointer, dimension(:,:) :: tini
   real(r8), pointer, dimension(:,:) :: qini
   real(r8), pointer, dimension(:,:) :: cldliqini
   real(r8), pointer, dimension(:,:) :: cldiceini
   real(r8), pointer, dimension(:,:) :: dtcore

   real(r8), pointer, dimension(:,:,:) :: fracis  ! fraction of transported species that are insoluble
!---------------- zhh debug ---------------
   real(r8), pointer, dimension(:,:) :: tmp1    !
   real(r8) :: tmp
!---------------- zhh debug ---------------

! convective precipitation variables
   real(r8) :: prec_zmc(pcols)                ! total precipitation from ZM convection
   real(r8) :: snow_zmc(pcols)                ! snow from ZM convection
   real(r8) :: prec_cmf(pcols)                ! total precipitation from Hack convection
   real(r8) :: snow_cmf(pcols)                ! snow from Hack convection

! stratiform precipitation variables
   real(r8) :: prec_str(pcols)    ! sfc flux of precip from stratiform (m/s)
   real(r8) :: snow_str(pcols)     ! sfc flux of snow from stratiform   (m/s)
   real(r8) :: prec_pcw(pcols)     ! total precip from prognostic cloud scheme
   real(r8) :: snow_pcw(pcols)     ! snow from prognostic cloud scheme
   real(r8) :: prec_sed(pcols)     ! total precip from cloud sedimentation
   real(r8) :: snow_sed(pcols)     ! snow from cloud ice sedimentation
   real(r8), pointer, dimension(:,:) :: cldo 

! energy checking variables
   real(r8) :: zero(pcols)                    ! array of zeros
   real(r8) :: rliq(pcols)                    ! vertical integral of liquid not yet in q(ixcldliq)
   real(r8) :: rliq2(pcols)                   ! vertical integral of liquid from shallow scheme
   real(r8) :: flx_cnd(pcols)
   real(r8) :: flx_heat(pcols)
   type(check_tracers_data):: tracerint             ! energy integrals and cummulative boundary fluxes
   real(r8) :: zero_tracers(pcols,pcnst)

!++ debug code to be removed after PBL scheme validation
   integer :: kmx
!-- debug code to be removed after PBL scheme validation

  real(r8)  :: cmeliq(pcols,pver)                      ! Rate of cond-evap of liq within the cloud
!  pass macro to micro

! zmh 2018-12-30
!   real(r8) :: tpert3(pcols)
!   real(r8) :: qpert3(pcols,pcnst)
   real(r8) :: gradt(pcols)
   real(r8) :: gradq(pcols)
   real(r8) :: vort3(pcols)
!   real(r8) :: dzm(pcols)
   real(r8) :: calday
   real(r8) :: drhlims(pcols)
!zmh

   character(len=16) :: microp_scheme 
   call phys_getopts( microp_scheme_out = microp_scheme )

!-----------------------------------------------------------------------
   call t_startf('bc_init')

   zero = 0._r8
   zero_tracers(:,:) = 0._r8

   lchnk = state%lchnk
   ncol  = state%ncol

   rtdt = 1._r8/ztodt

   nstep = get_nstep()


! Associate pointers with physics buffer fields
   itim = pbuf_old_tim_idx()
   ifld = pbuf_get_fld_idx('CLD')
   cld  => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)

   ifld   = pbuf_get_fld_idx('TEOUT')
   teout  => pbuf(ifld)%fld_ptr(1,1:pcols,1,lchnk,itim)
   ifld   =  pbuf_get_fld_idx('QINI')
   qini   => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1)
   ifld   =  pbuf_get_fld_idx('CLDLIQINI')
   cldliqini => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1)
   ifld   =  pbuf_get_fld_idx('CLDICEINI')
   cldiceini => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1)
   ifld   =  pbuf_get_fld_idx('TINI')
   tini   => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1)
   ifld   =  pbuf_get_fld_idx('DTCORE')
   dtcore => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, itim)

   ifld    = pbuf_get_fld_idx('FRACIS')
   fracis  => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk, 1:pcnst)

   if(is_first_step())then
      ifld = pbuf_get_fld_idx('CLDO')
      do m=1,pbuf_times
         cldo => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,m)
         cldo(:ncol,:) = 0
      enddo
   endif
!
! Set physics tendencies to 0
   tend %dTdt(:ncol,:pver)  = 0._r8
   tend %dudt(:ncol,:pver)  = 0._r8
   tend %dvdt(:ncol,:pver)  = 0._r8

   call physics_ptend_init (ptend) ! Initialize parameterization tendency structure
!
! Make sure that input tracers are all positive (probably unnecessary)
!
    
#ifdef MODAL_AERO
   call qneg3_modalx1( &
              'TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q, qneg3_worst_thresh_amode )
#else
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q )
#endif

   call clybry_fam_adj( ncol, lchnk, map2chm, state%q )

   fracis (:ncol,:,1:pcnst) = 1._r8
!
! Dump out "before physics" state
!
   call diag_state_b4_phys_write (state)

! compute mass integrals of input tracers state
   call check_tracers_init(state, tracerint)

   call t_stopf('bc_init')

!===================================================
! Global mean total energy fixer
!===================================================
   call t_startf('energy_fixer')

   !*** BAB's FV heating kludge *** save the initial temperature
   tini(:ncol,:pver) = state%t(:ncol,:pver)
   if (dycore_is('LR')) then
      call check_energy_fix(state, ptend, nstep, flx_heat)
      call physics_update(state, tend, ptend, ztodt)
      call check_energy_chng(state, tend, "chkengyfix", nstep, ztodt, zero, zero, zero, flx_heat)
   end if
! Save state for convective tendency calculations.
   call diag_conv_tend_ini(state,pbuf)

   call cnst_get_ind('CLDLIQ', ixcldliq)
   call cnst_get_ind('CLDICE', ixcldice)
   qini     (:ncol,:pver) = state%q(:ncol,:pver,       1)
   cldliqini(:ncol,:pver) = state%q(:ncol,:pver,ixcldliq)
   cldiceini(:ncol,:pver) = state%q(:ncol,:pver,ixcldice)

   call outfld('TEOUT', teout       , pcols, lchnk   )
   call outfld('TEINP', state%te_ini, pcols, lchnk   )
   call outfld('TEFIX', state%te_cur, pcols, lchnk   )

! set and output the dse change due to dynpkg
   if( nstep > pbuf_times-1 ) then
      do k = 1,pver
         dtcore(:ncol,k) = (state%s(:ncol,k) - dtcore(:ncol,k))/(cpair*ztodt)
      end do
      call outfld( 'DTCORE', dtcore, pcols, lchnk )
   end if

   call t_stopf('energy_fixer')
!
!===================================================
! Dry adjustment
! This code block is not a good example of interfacing a parameterization
!===================================================
   call t_startf('dry_adjustment')

! Copy state info for input to dadadj
! This is a kludge, so that dadadj does not have to be correctly reformulated in dry static energy

   ptend%s(:ncol,:pver)   = state%t(:ncol,:pver)
   ptend%q(:ncol,:pver,1) = state%q(:ncol,:pver,1)

   call dadadj (lchnk, ncol, state%pmid,  state%pint,  state%pdel,  &
                ptend%s, ptend%q(1,1,1))
   ptend%name  = 'dadadj'
   ptend%ls    = .TRUE.
   ptend%lq(1) = .TRUE.
   ptend%s(:ncol,:)   = (ptend%s(:ncol,:)   - state%t(:ncol,:)  )/ztodt * cpair
   ptend%q(:ncol,:,1) = (ptend%q(:ncol,:,1) - state%q(:ncol,:,1))/ztodt
   call physics_update (state, tend, ptend, ztodt)

! ===============================
!zmh 2018-12-30
    tpert2(:) = tpert(:)
    qpert2(:) = qpert(:,1)


#if (defined BFB_CAM_SCAM_IOP )
    gradt(:) = 0.0 !0.5
    gradq(:) = 0.0 !state%q(:,pver,1)*0.02
    vort3(:) = 0.0  ! level 3 from the bottom, reversed, see gravity_waves.F90
#else
    gradt(:) = state%frontgf(:,1)
    gradq(:) = state%frontgf(:,2)
    vort3(:) = state%frontga(:,3)  ! level 3 from the bottom, reversed, see gravity_waves.F90

#endif

  drhlims(:) = 0.0_r8
  do i=1,ncol
    drhlims(i) = 0.1*zmh_ramp(tpert(i) , 0.5_r8   ,   1.0_r8) &
                +0.1*zmh_ramp(qpert(i,1) ,state%q(i,pver,1)*0.02, state%q(i,pver,1)*0.05)
    drhlims(i) = drhlims(i)*0.5
  enddo

!
!if(nstep=0)then
!  call fout_phy(state%ps(1:ncol),lchnk,ncol,1,state%lat,state%lon,'ps')
!  call fout_phy(state%q(1:ncol,:,1),lchnk,ncol,pver,state%lat,state%lon,'q')
!  call fout_phy(tpert(1:ncol),lchnk,ncol,1,state%lat,state%lon,'tpert')
!  call fout_phy(qpert(1:ncol,1),lchnk,ncol,1,state%lat,state%lon,'qpert')
!  call fout_phy(qpert2(1:ncol),lchnk,ncol,1,state%lat,state%lon,'qpert2')
!  call fout_phy(tpert2(1:ncol),lchnk,ncol,1,state%lat,state%lon,'tpert2')
!  call fout_phy(state%frontga(1:ncol,3),lchnk,ncol,1,state%lat,state%lon,'vort3')
!  call fout_phy(state%frontga(1:ncol,3)*sin(state%lat),lchnk,ncol,1,state%lat,state%lon,'vort3b')
!  call fout_phy(state%frontgf(1:ncol,1),lchnk,ncol,1,state%lat,state%lon,'tdiff3')
!  call fout_phy(state%frontgf(1:ncol,2),lchnk,ncol,1,state%lat,state%lon,'qdiff3')
!  call fout_phy(sgh30(1:ncol,lchnk),lchnk,ncol,1,state%lat,state%lon,'sgh30')
!  call fout_phy(state%frontgf(1:ncol,3),lchnk,ncol,1,state%lat,state%lon,'qdiffx')
!  call fout_phy(state%frontgf(1:ncol,4),lchnk,ncol,1,state%lat,state%lon,'qdiffy')
!  call fout_phy(state%frontgf(1:ncol,5),lchnk,ncol,1,state%lat,state%lon,'qdiffg')
!endif

! ===============================

   call t_stopf('dry_adjustment')
!
!===================================================
! Moist convection
!===================================================
   call t_startf('moist_convection')

!print*,'state%t1',state%t(2,1)
!
! Since the PBL doesn't pass constituent perturbations, they
! are zeroed here for input to the moist convection routine
!print*,'ptend%s1',ptend%s(2,1)
!
   call t_startf ('convect_deep_tend')
   call convect_deep_tend(  prec_zmc,   &
        pblht,    cmfmc,      cmfcme,             &
        tpert,    qpert(:,1), gradt, gradq, vort3, dlf, pflx,    zdu,       & 
        rliq,    &
        ztodt,    snow_zmc,  &
        !state,   ptend, cam_in%landfrac,  pbuf ) 
        state,   ptend, cam_in%landfrac, cam_in%lhf, cam_in%shf, pbuf )  !zmh
   call t_stopf('convect_deep_tend')

!zmh
#ifdef MODAL_AERO
   call qneg3_modalx1( &
              'TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q, qneg3_worst_thresh_amode )
#else
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q )
#endif


   call physics_update(state, tend, ptend, ztodt)

! Check energy integrals, including "reserved liquid"
   flx_cnd(:ncol) = prec_zmc(:ncol) + rliq(:ncol)
   call check_energy_chng(state, tend, "convect_deep", nstep, ztodt, zero, flx_cnd, snow_zmc, zero)

!
! Call Hack (1994) convection scheme to deal with shallow/mid-level convection
!
   call t_startf ('convect_shallow_tend')

   call convect_shallow_tend (ztodt   ,&
        tpert      ,qpert     , gradt, gradq,  &                 !zmh
        pblht      ,     &
        cmfmc      ,cmfmc2  ,  prec_cmf,   &
        dlf        , dlf2,   rliq      , rliq2, & 
        !snow_cmf   , state, ptend,  ,pbuf       )   !zmh
        snow_cmf   , state, ptend,  cam_in%landfrac,pbuf       )   !zmh
   call t_stopf ('convect_shallow_tend')

!zmh
#ifdef MODAL_AERO
   call qneg3_modalx1( &
              'TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q, qneg3_worst_thresh_amode )
#else
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q )
#endif
   call physics_update (state, tend, ptend, ztodt)

   flx_cnd(:ncol) = prec_cmf(:ncol) + rliq2(:ncol)
   call check_energy_chng(state, tend, "convect_shallow", nstep, ztodt, zero, flx_cnd, snow_cmf, zero)

   call check_tracers_chng(state, tracerint, "convect_shallow", nstep, ztodt, zero_tracers)

   call t_stopf('moist_convection')

   ! Rebin the 4-bin version of sea salt into bins for coarse and accumulation
   ! modes that correspond to the available optics data.  This is only necessary
   ! for CAM-RT.  But it's done here so that the microphysics code which is called
   ! from the stratiform interface has access to the same aerosols as the radiation
   ! code.
   call sslt_rebin_adv(pbuf, state)

!if CAM4/RK microphysics
!print*,'state%t4',state%t(2,1)

   if( microp_scheme .eq. 'RK' ) then

!===================================================
! Calculate stratiform tendencey (sedimentation, detrain, cloud fraction and microphysics )
!===================================================
      call t_startf('stratiform_tend')

      call stratiform_tend(state, ptend, ztodt, &
           cam_in%icefrac, cam_in%landfrac, cam_in%ocnfrac, &
           landm, cam_in%snowhland, & ! sediment
           dlf, dlf2, & ! detrain
           rliq  , & ! check energy after detrain
           cmfmc,   cmfmc2, &
           cam_in%ts,      cam_in%sst,        zdu,  tpert, qpert, &  !zmh tpert2 qpert2
           prec_str, snow_str, prec_sed, snow_sed, prec_pcw, snow_pcw, & 
           pbuf, state_eq)

      call physics_update (state, tend, ptend, ztodt)
      call check_energy_chng(state, tend, "cldwat_tend", nstep, ztodt, zero, prec_str, snow_str, zero)
   
      call t_stopf('stratiform_tend')

    elseif( microp_scheme .eq. 'MG' .or. microp_scheme .eq. 'M2M' ) then

!===================================================
! Calculate macrophysical tendencey (sedimentation, detrain, cloud fraction)
!===================================================

      call t_startf('macrop_tend')
    
      call macrop_driver_tend(state, ptend, ztodt, pblht, &  !zmh added pblht
           cam_in%landfrac, cam_in%ocnfrac, &
           cam_in%snowhland, & ! sediment
           dlf, dlf2, & ! detrain
           cmfmc,   cmfmc2, &
           cam_in%ts,      cam_in%sst,   cam_in%lhf, cam_in%shf,     zdu, &
           tpert,qpert,gradt, gradq, drhlims, &  !zmh
           pbuf, state_eq, cmeliq)

!      call physics_update (state, tend, ptend, ztodt)
!      call check_energy_chng(state, tend, "macrop_tend", nstep, ztodt, zero, prec_str, snow_str, zero)

!zmh
#ifdef MODAL_AERO
   call qneg3_modalx1( &
              'TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q, qneg3_worst_thresh_amode )
#else
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q )
#endif
      call t_stopf('macrop_tend')
!===================================================
! Calculate cloud microphysics 
!===================================================

!print*,'state%t5',state%t(2,1)
      call t_startf('microp_tend')

      call microp_driver_tend(state, ptend, ztodt, &
#ifdef MODAL_AERO
           cam_in%cflx, & ! constituent sources
#endif
           rliq  , & ! check energy after detrain
           prec_str, snow_str, prec_sed, snow_sed, prec_pcw, snow_pcw, & 
           pbuf, state_eq, cmeliq)
!zmh
#ifdef MODAL_AERO
   call qneg3_modalx1( &
              'TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q, qneg3_worst_thresh_amode )
#else
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q )
#endif
      call physics_update (state, tend, ptend, ztodt)
      call check_energy_chng(state, tend, "microp_tend", nstep, ztodt, zero, prec_str, snow_str, zero)

      call t_stopf('microp_tend')

   endif

!end microphysics conditional.

   if ( .not. deep_scheme_does_scav_trans() ) then

      !===================================================
      !  Aerosol wet chemistry determines scavenging fractions, and transformations
      !
      !
      !  Then do convective transport of all trace species except water vapor and
      !     cloud liquid and ice (we needed to do the scavenging first
      !     to determine the interstitial fraction) 
      !===================================================

      call t_startf('bc_aerosols')
      call aerosol_wet_intr (state, ptend, ztodt, pbuf, cam_out, dlf)
      call physics_update (state, tend, ptend, ztodt)

      call t_startf ('convect_deep_tend2')
      call convect_deep_tend_2( state,   ptend,  ztodt,  pbuf ) 
      call t_stopf ('convect_deep_tend2')

      call physics_update (state, tend, ptend, ztodt)

      ! check tracer integrals
      call check_tracers_chng(state, tracerint, "cmfmca", nstep, ztodt, ptend%cflx_srf)

   endif

   call t_stopf('bc_aerosols')

   !===================================================
   ! Moist physical parameteriztions complete: 
   ! send dynamical variables, and derived variables to history file
   !===================================================

   call t_startf('bc_history_write')
   call diag_phys_writeout(state, cam_out%psl)
#if ((defined wrf) || (defined geatm))
!--------------------------------------------------------------------------------------
! get value from phys_state, added juanxiong he
!--------------------------------------------------------------------------------------
    cam_state=state
!--------------------------------------------------------------------------------------
! get value from phys_state, added juanxiong he
!--------------------------------------------------------------------------------------
#endif
   call diag_conv(state, ztodt,    &
        prec_zmc, snow_zmc, prec_cmf, snow_cmf, prec_sed, snow_sed, prec_pcw, snow_pcw)

   call t_stopf('bc_history_write')

   !===================================================
   ! Write cloud diagnostics on history file
   !===================================================

   call t_startf('bc_cld_diag_history_write')

   call put_cloud_diagnostics(state, pbuf)

   call t_stopf('bc_cld_diag_history_write')

   !===================================================
   ! Radiation computations
   !===================================================
   call t_startf('radiation')
!zmh
#ifdef MODAL_AERO
   call qneg3_modalx1( &
              'TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q, qneg3_worst_thresh_amode )
#else
   call qneg3('TPHYSBCb',lchnk  ,ncol    ,pcols   ,pver    , &
              1, pcnst, qmin  ,state%q )
#endif

!   call fout_phy(cam_in%ts(1:ncol),lchnk,ncol,1,state%lat,state%lon,'ts')

   call radiation_tend(state,ptend,pbuf, &
        cam_out, cam_in, &
        cam_in%landfrac,landm,cam_in%icefrac, cam_in%snowhland, &
        fsns,    fsnt, flns,    flnt,  &
        fsds, net_flx)
   ! Set net flux used by spectral dycores
   do i=1,ncol
      tend%flx_net(i) = net_flx(i)
   end do
   call physics_update(state, tend, ptend, ztodt)
   call check_energy_chng(state, tend, "radheat", nstep, ztodt, zero, zero, zero, net_flx)

   call t_stopf('radiation')

   ! Diagnose the location of the tropopause and its location to the history file(s).
   call t_startf('tropopause')
   call tropopause_output(state)
   call t_stopf('tropopause')

   ! Save atmospheric fields to force surface models
   call t_startf('srfxfer')
   call srfxfer (state,cam_out,prec_zmc,snow_zmc, &
        prec_cmf,snow_cmf,prec_sed,snow_sed, &
        prec_pcw,snow_pcw)
   call t_stopf('srfxfer')

   ! Write export state to history file
   call t_startf('diag_export')
   call diag_export(cam_out)
   call t_stopf('diag_export')

end subroutine tphysbc
