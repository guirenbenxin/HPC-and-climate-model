
subroutine tphysac (ztodt,   pblh,    qpert,   tpert,  tpert2, qpert2, cam_in,  &  !zmh added tpert2
                    sgh,     sgh30,                                     &
                    cam_out,  state,   tend,    pbuf,           &
                    fsds    )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Tendency physics after coupling to land, sea, and ice models.
! Computes the following:
!   o Radon surface flux and decay (optional)
!   o Vertical diffusion and planetary boundary layer
!   o Multiple gravity wave drag
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: CCM1, CMS Contact: J. Truesdale
! 
!-----------------------------------------------------------------------
   use shr_kind_mod,       only: r8 => shr_kind_r8
   use ppgrid,             only: pcols, pver, pverp
   use chemistry,          only: chem_is_active, chem_timestep_tend
   use cam_diagnostics,    only: diag_phys_tend_writeout
!czy20181116   use gw_drag,            only: gw_intr
!+czy20181116
   use gw_drag,            only: gw_drag_scheme
   use gw_drag_cam,        only: gw_intr_cam => gw_intr
   use gw_drag_waccm,      only: gw_intr_waccm => gw_intr
   use gw_drag_xjb,        only: gw_intr_xjb   => gw_intr !czy20181120
   use spmd_utils,         only: masterproc
   use cam_logfile,        only: iulog
!-czy20181116
   use vertical_diffusion, only: vertical_diffusion_tend
   use rayleigh_friction,  only: rayleigh_friction_tend
   use constituents,       only: cnst_get_ind
   use physics_types,      only: physics_state, physics_tend, physics_ptend, physics_update,    &
                                 physics_ptend_init, physics_dme_adjust, set_dry_to_wet
   use phys_buffer,        only: pbuf_size_max, pbuf_fld, pbuf_old_tim_idx, pbuf_get_fld_idx
   use tracers,            only: tracers_timestep_tend
   use aoa_tracers,        only: aoa_tracers_timestep_tend
   use physconst,          only: zvir, gravit, rhoh2o, latvap,latice, cpair, rair
   use aerosol_intr,       only: aerosol_emis_intr, aerosol_drydep_intr
   use camsrfexch_types,   only: cam_out_t, cam_in_t     
   use check_energy,       only: check_energy_chng
   use check_energy,       only: check_tracers_data, check_tracers_init, check_tracers_chng
   use time_manager,       only: get_nstep
   use abortutils,         only: endrun
   use dycore,             only: dycore_is
   use cam_control_mod,    only: aqua_planet 
   use mo_gas_phase_chemdr,only: map2chm
   use clybry_fam,         only: clybry_fam_set
#if ( defined WACCM_PHYS )
   use charge_neutrality,  only: charge_fix
   use iondrag,            only: iondrag_calc, do_waccm_ions
   use qbo,                only: qbo_relax
#endif
   use perf_mod
   use phys_control,       only: phys_do_flux_avg
   use flux_avg,           only: flux_avg_run
   use ppgrid,             only: begchunk, endchunk   ! zhh for debug

   implicit none

!
! Arguments
!
   real(r8), intent(in) :: ztodt                  ! Two times model timestep (2 delta-t)
#if ( defined MODAL_AERO )
   real(r8), intent(inout) :: pblh(pcols)           ! Planetary boundary layer height
#else
   real(r8), intent(out) :: pblh(pcols)           ! Planetary boundary layer height
#endif   
   real(r8), intent(in) :: fsds(pcols)            ! down solar flux
   real(r8), intent(inout) :: qpert(pcols)          ! zmh changed to inout
   real(r8), intent(inout) :: tpert(pcols)          ! Temperature perturbation (PBL)

   real(r8), intent(inout) :: qpert2(pcols)          ! 
   real(r8), intent(inout) :: tpert2(pcols)          !
   real(r8), intent(in) :: sgh(pcols)             ! Std. deviation of orography for gwd
   real(r8), intent(in) :: sgh30(pcols)           ! Std. deviation of 30s orography for tms

   type(cam_in_t),      intent(inout) :: cam_in
   type(cam_out_t),     intent(inout) :: cam_out
   type(physics_state), intent(inout) :: state
   type(physics_tend ), intent(inout) :: tend
   type(pbuf_fld),      intent(inout) :: pbuf(pbuf_size_max)

   type(check_tracers_data):: tracerint             ! tracer mass integrals and cummulative boundary fluxes

!
!---------------------------Local workspace-----------------------------
!
   character(len=*), parameter :: subname = 'tphysac' !czy20181116
   type(physics_ptend)     :: ptend               ! indivdual parameterization tendencies

   integer  :: nstep                              ! current timestep number
   real(r8) :: zero(pcols)                        ! array of zeros

   integer :: lchnk                                ! chunk identifier
   integer :: ncol                                 ! number of atmospheric columns
   integer i,k,m                 ! Longitude, level indices
   integer :: yr, mon, day, tod       ! components of a date
   integer :: ixcldice, ixcldliq      ! constituent indices for cloud liquid and ice water.

   logical :: labort                            ! abort flag

   real(r8) tvm(pcols,pver)           ! virtual temperature
   real(r8) prect(pcols)              ! total precipitation
   real(r8) surfric(pcols)            ! surface friction velocity
   real(r8) obklen(pcols)             ! Obukhov length
   real(r8) :: fh2o(pcols)            ! h2o flux to balance source from methane chemistry
   real(r8) :: tmp_q     (pcols,pver) ! tmp space
   real(r8) :: tmp_cldliq(pcols,pver) ! tmp space
   real(r8) :: tmp_cldice(pcols,pver) ! tmp space
   real(r8) :: tmp_t     (pcols,pver) ! tmp space

! physics buffer fields for total energy and mass adjustment
   integer itim, ifld
   real(r8), pointer, dimension(:  ) :: teout
   real(r8), pointer, dimension(:,:) :: tini
   real(r8), pointer, dimension(:,:) :: cld
   real(r8), pointer, dimension(:,:) :: kvh
   real(r8), pointer, dimension(:,:) :: qini
   real(r8), pointer, dimension(:,:) :: cldliqini
   real(r8), pointer, dimension(:,:) :: cldiceini
   real(r8), pointer, dimension(:,:) :: dtcore
   real(r8), pointer, dimension(:,:) :: ast     ! relative humidity cloud fraction 
!
!-----------------------------------------------------------------------
!
   lchnk = state%lchnk
   ncol  = state%ncol

   nstep = get_nstep()

   ! Adjust the surface fluxes to reduce instabilities in near sfc layer
   if (phys_do_flux_avg()) then 
      call flux_avg_run(state, cam_in, pbuf, nstep, ztodt)
   endif

   call t_startf('tphysac_init')
! Associate pointers with physics buffer fields
! juanxiong he
   nullify(teout)
   nullify(tini) 
   nullify(cld)
   nullify(kvh) 
   nullify(qini) 
   nullify(cldliqini) 
   nullify(cldiceini) 
   nullify(dtcore) 
   nullify(ast) 
   itim = pbuf_old_tim_idx()
   ifld = pbuf_get_fld_idx('TEOUT')
   teout  => pbuf(ifld)%fld_ptr(1,1:pcols,1,lchnk,itim)
   ifld = pbuf_get_fld_idx('DTCORE')
   dtcore => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
   ifld = pbuf_get_fld_idx('QINI')
   qini => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
   ifld = pbuf_get_fld_idx('CLDLIQINI')
   cldliqini => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
   ifld = pbuf_get_fld_idx('CLDICEINI')
   cldiceini => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
   ifld = pbuf_get_fld_idx('TINI')
   tini  => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,1)
   ifld = pbuf_get_fld_idx('CLD')
   cld => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)
   ifld = pbuf_get_fld_idx('kvh')
   kvh => pbuf(ifld)%fld_ptr(1,1:pcols,1:pverp,lchnk,itim)
   ifld = pbuf_get_fld_idx('AST')
   ast => pbuf(ifld)%fld_ptr(1,1:pcols,1:pver,lchnk,itim)

!
! accumulate fluxes into net flux array for spectral dycores
! jrm Include latent heat of fusion for snow
!
   do i=1,ncol
      tend%flx_net(i) = tend%flx_net(i) + cam_in%shf(i) + (cam_out%precc(i) &
                        + cam_out%precl(i))*latvap*rhoh2o &
                        + (cam_out%precsc(i) + cam_out%precsl(i))*latice*rhoh2o
   end do
 
! Initialize parameterization tendency structure

   call physics_ptend_init(ptend)

! emission of aerosols at surface
#if ( defined MODAL_AERO )
   call aerosol_emis_intr (state, ptend, cam_in%cflx, ztodt, cam_in%ocnfrac, cam_in%sst)
#else
   call aerosol_emis_intr (state, ptend, cam_in%cflx, ztodt, cam_in%ocnfrac)
#endif
   call physics_update (state, tend, ptend, ztodt)

! get nstep and zero array for energy checker
   zero = 0._r8
   nstep = get_nstep()
   call check_tracers_init(state, tracerint)

! Check if latent heat flux exceeds the total moisture content of the
! lowest model layer, thereby creating negative moisture.

   call qneg4('TPHYSAC '       ,lchnk               ,ncol  ,ztodt ,               &
              state%q(1,pver,1),state%rpdel(1,pver) ,cam_in%shf ,         &
              cam_in%lhf , cam_in%cflx )

   call t_stopf('tphysac_init')
!===================================================
! Source/sink terms for advected tracers.
!===================================================
! zmh   
#if (! defined BFB_CAM_SCAM_IOP )

   call t_startf('adv_tracer_src_snk')
! Test tracers

   call tracers_timestep_tend(state, ptend, cam_in%cflx, cam_in%landfrac, ztodt)      
   call physics_update (state, tend, ptend, ztodt)
   call check_tracers_chng(state, tracerint, "tracers_timestep_tend", nstep, ztodt,   &
                           cam_in%cflx)

   call aoa_tracers_timestep_tend(state, ptend, cam_in%cflx, cam_in%landfrac, ztodt)      
   call physics_update (state, tend, ptend, ztodt)
   call check_tracers_chng(state, tracerint, "aoa_tracers_timestep_tend", nstep, ztodt,   &
                           cam_in%cflx)

   ! Chemistry calculation
   if (chem_is_active()) then
      call chem_timestep_tend(state, ptend, cam_in, cam_out, ztodt, pbuf, fh2o, fsds &
#if ( defined MODAL_AERO )
                              , pblh                                                                 &
#endif
                                                                                                     )
      call physics_update (state, tend, ptend, ztodt)
      call check_energy_chng(state, tend, "chem", nstep, ztodt, fh2o, zero, zero, zero)
      call check_tracers_chng(state, tracerint, "chem_timestep_tend", nstep, ztodt, &
                              cam_in%cflx)
   end if
   call t_stopf('adv_tracer_src_snk')
#endif 

!write(*,*)'in physac1, tpert,qpert', tpert, qpert
!===================================================
! Vertical diffusion/pbl calculation
! Call vertical diffusion code (pbl, free atmosphere and molecular)
!===================================================
   call t_startf('vertical_diffusion_tend')
   call vertical_diffusion_tend (ztodt ,state ,cam_in%wsx, cam_in%wsy,   &
                                 cam_in%shf     ,cam_in%cflx     ,pblh     ,&
                                 tpert    ,qpert    , tpert2, qpert2, & !zmh
                                 surfric  ,obklen   ,ptend    ,ast    ,&
                                 cam_in%ocnfrac  , cam_in%landfrac ,        &
                                 sgh30    ,pbuf     )
   call physics_update (state, tend, ptend, ztodt)
   call t_stopf ('vertical_diffusion_tend')
!write(*,*)'in physac2, tpert,qpert', tpert, qpert
!write(*,*)'in physac2, tpert,qpert', tpert, qpert

!===================================================
! Rayleigh friction calculation
!===================================================
   call t_startf('rayleigh_friction')
   call rayleigh_friction_tend( ztodt, state, ptend)
   call physics_update (state, tend, ptend, ztodt)
   call t_stopf('rayleigh_friction')

   call check_energy_chng(state, tend, "vdiff", nstep, ztodt, cam_in%cflx(:,1), zero, &
                          zero, cam_in%shf)
   call check_tracers_chng(state, tracerint, "vdiff", nstep, ztodt, cam_in%cflx)

   !  aerosol dry deposition processes
   call t_startf('aero_drydep')
   call aerosol_drydep_intr (state, ptend, cam_in, cam_out, ztodt,  &
                             fsds, obklen, surfric, prect, pblh, pbuf )
   call physics_update (state, tend, ptend, ztodt)
   call t_stopf('aero_drydep')

#if ( defined WACCM_PHYS )
!---------------------------------------------------------------------------------
!	... enforce charge neutrality
!---------------------------------------------------------------------------------
      call charge_fix( ncol, state%q(:,:,:) )
#endif
!===================================================
! Gravity wave drag
!===================================================
   call t_startf('gw_intr')
!czy20181116   call gw_intr (state   ,sgh     ,pblh    ,ztodt   , ptend , cam_in%landfrac, &
!czy20181116                 kvh)
!+czy20181116
   if (gw_drag_scheme == 1) then
           call gw_intr_cam(state   ,sgh     ,pblh    ,ztodt   , ptend , cam_in%landfrac, kvh) 
   elseif (gw_drag_scheme == 2) then
           call gw_intr_waccm(state   ,sgh     ,pblh    ,ztodt   , ptend , cam_in%landfrac, kvh) 
   elseif (gw_drag_scheme == 3) then
           call gw_intr_xjb(state   ,sgh     ,pblh    ,ztodt   , ptend , cam_in%landfrac, kvh) 
   else
            if (masterproc) write(iulog,*)subname//':: ERROR gw_drag_scheme = ',gw_drag_scheme
            call endrun(subname//':: ERROR gw_drag_scheme is not set right')
   endif
!-czy20181116

   call physics_update (state, tend, ptend, ztodt)
! Check energy integrals
   call check_energy_chng(state, tend, "gwdrag", nstep, ztodt, zero, zero, zero, zero)
   call t_stopf('gw_intr')

#if ( defined WACCM_PHYS )

! QBO relaxation
   call qbo_relax(state,ptend,state%uzm)
   call physics_update (state, tend, ptend, ztodt)
   ! Check energy integrals
   call check_energy_chng(state, tend, "qborelax", nstep, ztodt, zero, zero, zero, zero)
! Ion drag calculation
   call t_startf ( 'iondrag' )
   if ( do_waccm_ions ) then
      call iondrag_calc( lchnk, ncol, state, ptend, pbuf, ztodt )
   else
      call iondrag_calc( lchnk, ncol, state, ptend, pbuf )
   endif
   call physics_update (state, tend, ptend, ztodt)
! Check energy integrals
   call check_energy_chng(state, tend, "iondrag", nstep, ztodt, zero, zero, zero, zero)
   call t_stopf  ( 'iondrag' )

#endif

   call physics_update (state, tend, ptend, ztodt)

!-------------- Energy budget checks vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
   teout(:ncol) = state%te_cur(:ncol)

! store dse after tphysac in buffer
   do k = 1,pver
      dtcore(:ncol,k) = state%s(:ncol,k)
   end do

!*** BAB's FV heating kludge *** apply the heating as temperature tendency.
!*** BAB's FV heating kludge *** modify the temperature in the state structure
   tmp_t(:ncol,:pver) = state%t(:ncol,:pver)
   state%t(:ncol,:pver) = tini(:ncol,:pver) + ztodt*tend%dtdt(:ncol,:pver)

!
! FV: convert dry-type mixing ratios to moist here because physics_dme_adjust
!     assumes moist. This is done in p_d_coupling for other dynamics. Bundy, Feb 2004.


   if ( dycore_is('LR') .or. dycore_is('HOMME')) call set_dry_to_wet(state)    ! Physics had dry, dynamics wants moist


! Scale dry mass and energy (does nothing if dycore is EUL or SLD)
   call cnst_get_ind('CLDLIQ', ixcldliq)
   call cnst_get_ind('CLDICE', ixcldice)
   tmp_q     (:ncol,:pver) = state%q(:ncol,:pver,1)
   tmp_cldliq(:ncol,:pver) = state%q(:ncol,:pver,ixcldliq)
   tmp_cldice(:ncol,:pver) = state%q(:ncol,:pver,ixcldice)
   call physics_dme_adjust(state, tend, qini, ztodt)
!!!   REMOVE THIS CALL, SINCE ONLY Q IS BEING ADJUSTED. WON'T BALANCE ENERGY. TE IS SAVED BEFORE THIS
!!!   call check_energy_chng(state, tend, "drymass", nstep, ztodt, zero, zero, zero, zero)

!-------------- Energy budget checks ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   if (aqua_planet) then
      labort = .false.
      do i=1,ncol
         if (cam_in%ocnfrac(i) /= 1._r8) labort = .true.
      end do
      if (labort) then
         call endrun ('TPHYSAC error:  grid contains non-ocean point')
      endif
   endif

   call diag_phys_tend_writeout (state, pbuf, tend, ztodt, tmp_q, tmp_cldliq, tmp_cldice, &
                                 tmp_t, qini, cldliqini, cldiceini)

   call clybry_fam_set( ncol, lchnk, map2chm, state%q )

end subroutine tphysac
