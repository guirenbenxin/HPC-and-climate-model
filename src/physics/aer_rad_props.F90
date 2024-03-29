module aer_rad_props

!------------------------------------------------------------------------------------------------
! Converts aerosol masses to bulk optical properties for sw and lw radiation
! computations.  
!------------------------------------------------------------------------------------------------

use shr_kind_mod,     only: r8 => shr_kind_r8
use cam_logfile,      only: iulog  !added by fkc
use ppgrid,           only: pcols, pver, pverp
use physconst,        only: rga
use physics_types,    only: physics_state
use phys_buffer,      only: pbuf_size_max, pbuf_fld, pbuf_get_fld_idx
use radconstants,     only: nrh, nswbands, nlwbands, idx_sw_diag, ot_length
use rad_constituents, only: rad_cnst_get_info, rad_cnst_get_aer_mmr, &
                            rad_cnst_get_aer_props, new_rad_cnst_get_aer_mmr !! added by fkc
use time_manager,     only: get_nstep  ! added by fkc
use wv_saturation,    only: aqsat
use cam_history,      only: fieldname_len, addfld, phys_decomp, outfld, add_default
use cam_history_support, only : fillvalue

use abortutils,       only: endrun
use constituents,     only: pcnst
use phys_control,     only: phys_getopts

implicit none
private
save

public :: &
   aer_rad_props_init,        &
   aer_rad_props_sw,          & ! return SW optical props of aerosols
   aer_rad_props_lw             ! return LW optical props of aerosols

! Private data
character(len=fieldname_len), pointer :: odv_names(:)  ! outfld names for visible OD


!==============================================================================
contains
!==============================================================================

subroutine aer_rad_props_init()


   integer                    :: i
   integer                    :: numaerosols  ! number of aerosols
   character(len=64), pointer :: aernames(:)  ! aerosol names
   logical                    :: history_aerosol      ! Output the MAM aerosol tendencies

   !----------------------------------------------------------------------------

   call phys_getopts( history_aerosol_out        = history_aerosol   )

   call addfld ('AEROD_v ', '1', 1, 'A', &
      'Total Aerosol Optical Depth in visible band', phys_decomp, flag_xyfill=.true.)
   !added by fkc
#ifdef GEATM
   write(iulog,*)'adding GEATM aerosols'
   call addfld( 'GEATM_dust1', 'kg/m3   ', pver, 'A', 'GEATM_dust1' , phys_decomp )
   call addfld( 'GEATM_dust2', 'kg/m3   ', pver, 'A', 'GEATM_dust2' , phys_decomp )
   call addfld( 'GEATM_dust3', 'kg/m3   ', pver, 'A', 'GEATM_dust3' , phys_decomp )
   call addfld( 'GEATM_dust4', 'kg/m3   ', pver, 'A', 'GEATM_dust4' , phys_decomp )
   call addfld( 'GEATM_sslt1', 'kg/m3   ', pver, 'A', 'GEATM_sslt1' , phys_decomp )
   call addfld( 'GEATM_sslt2', 'kg/m3   ', pver, 'A', 'GEATM_sslt2' , phys_decomp )
   call addfld( 'GEATM_sslt3', 'kg/m3   ', pver, 'A', 'GEATM_sslt3' , phys_decomp )
   call addfld( 'GEATM_sslt4', 'kg/m3   ', pver, 'A', 'GEATM_sslt4' , phys_decomp )
   call addfld( 'GEATM_bcar1', 'kg/m3   ', pver, 'A', 'GEATM_bcar1' , phys_decomp )
   call addfld( 'GEATM_bcar2', 'kg/m3   ', pver, 'A', 'GEATM_bcar2' , phys_decomp )
   call addfld( 'GEATM_sulf',  'kg/m3   ', pver, 'A', 'GEATM_sulf'  , phys_decomp )
   call addfld( 'GEATM_ocar1', 'kg/m3   ', pver, 'A', 'GEATM_ocar1' , phys_decomp )
   call addfld( 'GEATM_ocar2', 'kg/m3   ', pver, 'A', 'GEATM_ocar2' , phys_decomp )
#endif
   call add_default ('AEROD_v', 1, ' ')

   ! Contributions to AEROD_v from individual aerosols (climate species).

   ! number of aerosols in climate list
   call rad_cnst_get_info(0,naero=numaerosols)

   ! get names of aerosols
   allocate(aernames(numaerosols))
   call rad_cnst_get_info(0,aernames=aernames)

   ! create outfld names for visible OD and call addfld
   allocate(odv_names(numaerosols))
   do i = 1, numaerosols
      odv_names(i) = 'ODV_'//trim(aernames(i))
      call addfld (odv_names(i), '1', 1, 'A', &
         trim(aernames(i))//' optical depth in visible band', phys_decomp, flag_xyfill=.true.)
      if ( history_aerosol ) then 
         call add_default (odv_names(i), 1, ' ')
      endif
   end do
   deallocate(aernames)

end subroutine aer_rad_props_init

!==============================================================================

subroutine aer_rad_props_sw(list_idx, state, pbuf, nnite, idxnite, &
                            tau, tau_w, tau_w_g, tau_w_f)

! Return bulk layer tau, omega, g, f for all spectral intervals.

#ifdef MODAL_AERO
   use modal_aer_opt,     only: modal_aero_sw
#endif

   ! Arguments
   integer,             intent(in) :: list_idx      ! index of the climate or a diagnostic list
   type(physics_state), intent(in) :: state
   type(pbuf_fld),      intent(in) :: pbuf(:)
   integer,             intent(in) :: nnite                ! number of night columns
   integer,             intent(in) :: idxnite(:)           ! local column indices of night columns

   real(r8), intent(out) :: tau    (pcols,0:pver,nswbands) ! aerosol extinction optical depth
   real(r8), intent(out) :: tau_w  (pcols,0:pver,nswbands) ! aerosol single scattering albedo * tau
   real(r8), intent(out) :: tau_w_g(pcols,0:pver,nswbands) ! aerosol assymetry parameter * tau * w
   real(r8), intent(out) :: tau_w_f(pcols,0:pver,nswbands) ! aerosol forward scattered fraction * tau * w

   ! Local variables

   integer :: ncol
   integer :: lchnk
   integer :: nstep   ! added by fkc, CAM nstep
   integer :: k, i    ! lev and daycolumn indices
   integer :: iswband ! sw band indices

   ! optical props for each aerosol
   ! hygroscopic
   real(r8), pointer :: h_ext(:,:)
   real(r8), pointer :: h_ssa(:,:)
   real(r8), pointer :: h_asm(:,:)
   character(len=64), pointer :: aernames(:)  ! added by fkc, aerosol names

   ! non-hygroscopic
   real(r8), pointer :: n_ext(:)
   real(r8), pointer :: n_ssa(:)
   real(r8), pointer :: n_asm(:)
   real(r8), pointer :: n_scat(:)
   real(r8), pointer :: n_ascat(:)
   ! radius-dependent
   real(r8), pointer :: r_ext(:,:)    ! radius-dependent mass-specific extinction
   real(r8), pointer :: r_scat(:,:)
   real(r8), pointer :: r_ascat(:,:)
   real(r8), pointer :: r_mu(:)       ! log(radius) domain variable for r_ext, r_scat, r_ascat

   ! radiative properties for each aerosol
   real(r8) :: ta (pcols,pver,nswbands)
   real(r8) :: tw (pcols,pver,nswbands)
   real(r8) :: twf(pcols,pver,nswbands)
   real(r8) :: twg(pcols,pver,nswbands)

   ! aerosol masses
   real(r8), pointer :: aermmr(:,:),aermmr1(:,:)    ! added by fkc, mass mixing ratio of aerosols
   real(r8) :: mmr_to_mass(pcols,pver) ! conversion factor for mmr to mass
   real(r8) :: aermass(pcols,pver), aermass1(pcols,pver)     ! added by fkc, mass of aerosols
   real(r8) :: maxaermass, maxaermass1 ! added by fkc
   ! for table lookup into rh grid
   real(r8) :: esat(pcols,pver)     ! saturation vapor pressure
   real(r8) :: qsat(pcols,pver)     ! saturation specific humidity
   real(r8) :: rh(pcols,pver)
   real(r8) :: rhtrunc(pcols,pver)
   real(r8) :: wrh(pcols,pver)
   integer  :: krh(pcols,pver)
 
   integer  :: iaerosol        ! aerosol index
   integer  :: numaerosols     ! number of aerosols in climate list

   character(len=ot_length) :: opticstype       ! hygro or nonhygro
   !-----------------------------------------------------------------------------

   ncol  = state%ncol
   lchnk = state%lchnk

   ! compute mixing ratio to mass conversion
   do k = 1, pver
      mmr_to_mass(:ncol,k) = rga * state%pdeldry(:ncol,k)
   enddo

   ! initialize to conditions that would cause failure
   tau     (:,:,:) = -100._r8
   tau_w   (:,:,:) = -100._r8
   tau_w_g (:,:,:) = -100._r8
   tau_w_f (:,:,:) = -100._r8

   ! top layer (ilev = 0) has no aerosol (ie tau = 0)
   ! also initialize rest of layers to accumulate od's
   tau    (1:ncol,:,:) = 0._r8
   tau_w  (1:ncol,:,:) = 0._r8
   tau_w_g(1:ncol,:,:) = 0._r8
   tau_w_f(1:ncol,:,:) = 0._r8

   ! calculate relative humidity for table lookup into rh grid
   call aqsat(state%t, state%pmid, esat, qsat, pcols, &
               ncol, pver, 1, pver)
   rh(1:ncol,1:pver) = state%q(1:ncol,1:pver,1) / qsat(1:ncol,1:pver)

   rhtrunc(1:ncol,1:pver) = min(rh(1:ncol,1:pver),1._r8)
   krh(1:ncol,1:pver) = min(floor( rhtrunc(1:ncol,1:pver) * nrh ) + 1, nrh - 1) ! index into rh mesh
   wrh(1:ncol,1:pver) = rhtrunc(1:ncol,1:pver) * nrh - krh(1:ncol,1:pver)       ! (-) weighting on left side values

   ! number of aerosols
   call rad_cnst_get_info(list_idx, naero=numaerosols)
   ! added by fkc
   allocate(aernames(numaerosols))
   call rad_cnst_get_info(list_idx, aernames=aernames)
   ! end

#ifdef MODAL_AERO
   write(iulog,*)'MODAL_AERO defined'
   if(list_idx==0) then
      call modal_aero_sw(state, pbuf, nnite, idxnite, &
                         tau, tau_w, tau_w_g, tau_w_f)
   else
      tau    (1:ncol,:,:) = 0._r8
      tau_w  (1:ncol,:,:) = 0._r8
      tau_w_g(1:ncol,:,:) = 0._r8
      tau_w_f(1:ncol,:,:) = 0._r8
   end if

   ! Loop over aerosols and look for volcanic aerosol.  This is the only aerosol
   ! that will be treated in this loop and externally mixed with the modal aerosols.
   do iaerosol = 1, numaerosols

      ! get aerosol type
      call rad_cnst_get_aer_props(list_idx,iaerosol, opticstype=opticstype)

      if (trim(opticstype) == 'volcanic_radius') then

         ! get aerosol mass mixing ratio
         call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr)
         aermass(1:ncol,1:pver) = aermmr(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)

     ! get optical properties for volcanic aerosols

         call rad_cnst_get_aer_props(list_idx,iaerosol, r_sw_ext=r_ext, r_sw_scat=r_scat, r_sw_ascat=r_ascat, mu=r_mu)
         call get_volcanic_radius_rad_props(lchnk, ncol, aermass, pbuf, r_ext, r_scat, r_ascat, r_mu, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)
         call aer_vis_diag_out(lchnk, ncol, nnite, idxnite, iaerosol, ta(:,:,idx_sw_diag), list_idx)
      end if
   end do
#else
   write(iulog,*)'MODAL_AERO NOT defined'
   ! Loop over aerosols in list.  All aerosols are externally mixed when
   ! not using modal aerosols.
   do iaerosol = 1, numaerosols
! modified by fkc
#ifdef GEATM
      nstep = get_nstep()
      if (nstep.eq.0) then
          call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr1)
          aermass(1:ncol,1:pver) = aermmr1(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
      else
      ! get aerosol mass mixing ratio
          call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr1)
          aermass1(1:ncol,1:pver) = aermmr1(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
          write(iulog,*)'calling sw new_rad_cnst'
          call new_rad_cnst_get_aer_mmr(list_idx, iaerosol,aernames(iaerosol), state, pbuf, aermass)
          call gea_aer_out(lchnk, aermass, aernames(iaerosol))
          aermass(1:ncol,1:pver) = aermass(1:ncol,1:pver)*mmr_to_mass(1:ncol,1:pver)
          aermass(1:ncol,1:pver) = aermass(1:ncol,1:pver)*1.0E-9_r8/(1.293_r8*273.15_r8/state%t(1:ncol,1:pver))
          maxaermass=maxval(aermass)
          maxaermass1=maxval(aermass1)
      end if
! end       
#else
      ! get aerosol mass mixing ratio
      call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr)
      aermass(1:ncol,1:pver) = aermmr(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
#endif

      call rad_cnst_get_aer_props(list_idx, iaerosol, opticstype=opticstype)

      select case (trim(opticstype))
      case('hygro')
         ! get optical properties for hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, sw_hygro_ext=h_ext, sw_hygro_ssa=h_ssa, sw_hygro_asm=h_asm)
         call get_hygro_rad_props(ncol, krh, wrh, aermass, h_ext, h_ssa, h_asm, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)

      case('hygroscopic','hygroscopi')
         ! get optical properties for hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, sw_hygro_ext=h_ext, sw_hygro_ssa=h_ssa, sw_hygro_asm=h_asm)
         call get_hygro_rad_props(ncol, krh, wrh, aermass, h_ext, h_ssa, h_asm, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)

      case('nonhygro')
         ! get optical properties for non-hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, sw_nonhygro_ext=n_ext, sw_nonhygro_ssa=n_ssa, sw_nonhygro_asm=n_asm)

         call get_nonhygro_rad_props(ncol, aermass, n_ext, n_ssa, n_asm, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)

      case('insoluble ')
         ! get optical properties for non-hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, sw_nonhygro_ext=n_ext, sw_nonhygro_ssa=n_ssa, sw_nonhygro_asm=n_asm)

         call get_nonhygro_rad_props(ncol, aermass, n_ext, n_ssa, n_asm, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)

      case('volcanic')
         ! get optical properties for volcanic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, sw_nonhygro_ext=n_ext, sw_nonhygro_scat=n_scat, sw_nonhygro_ascat=n_ascat)

         call get_volcanic_rad_props(ncol, aermass, n_ext, n_scat, n_ascat, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)

      case('volcanic_radius')
         ! get optical properties for volcanic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, r_sw_ext=r_ext, r_sw_scat=r_scat, r_sw_ascat=r_ascat, mu=r_mu)

         call get_volcanic_radius_rad_props(lchnk, ncol, aermass, pbuf, r_ext, r_scat, r_ascat, r_mu, ta, tw, twg, twf)
         tau    (1:ncol,1:pver,:) = tau    (1:ncol,1:pver,:) + ta (1:ncol,:,:)
         tau_w  (1:ncol,1:pver,:) = tau_w  (1:ncol,1:pver,:) + tw (1:ncol,:,:)
         tau_w_g(1:ncol,1:pver,:) = tau_w_g(1:ncol,1:pver,:) + twg(1:ncol,:,:)
         tau_w_f(1:ncol,1:pver,:) = tau_w_f(1:ncol,1:pver,:) + twf(1:ncol,:,:)

      case('zero')
         ! no effect of "zero" aerosols, so update nothing
      case default
         call endrun('aer_rad_props_sw: unsupported opticstype :'//trim(opticstype)//':')
      end select

      ! diagnostic output of individual aerosol optical properties
      ! currently implemented for climate list only
      call aer_vis_diag_out(lchnk, ncol, nnite, idxnite, iaerosol, ta(:,:,idx_sw_diag), list_idx)

   enddo
#endif

   ! diagnostic output of total aerosol optical properties
   ! currently implemented for climate list only
   call aer_vis_diag_out(lchnk, ncol, nnite, idxnite, 0, tau(:,:,idx_sw_diag), list_idx)
   ! added by fkc
   deallocate(aernames)
   ! end
end subroutine aer_rad_props_sw

!==============================================================================

subroutine aer_rad_props_lw(list_idx, state, pbuf, odap_aer)

   use radconstants,  only: ot_length
   use phys_buffer,   only: pbuf_get_fld_idx
#ifdef MODAL_AERO
   use modal_aer_opt, only: modal_aero_lw
#endif

   ! Purpose: Compute aerosol transmissions needed in absorptivity/
   !    emissivity calculations

   ! lw extinction is the same representation for all 
   ! species.  If this changes, this routine will need to do something
   ! similar to the sw with routines like get_hygro_lw_abs

   ! Arguments
   integer,             intent(in)  :: list_idx                      ! index of the climate or a diagnostic list
   type(physics_state), intent(in)  :: state
   type(pbuf_fld),      intent(in)  :: pbuf(:)
   real(r8),            intent(out) :: odap_aer(pcols,pver,nlwbands) ! [fraction] absorption optical depth, per layer

   ! Local variables

   integer :: bnd_idx     ! LW band index
   integer :: i           ! column index
   integer :: k           ! lev index
   integer :: ncol        ! number of columns
   integer :: nstep       ! CAM nstep, added by fkc
   integer :: lchnk       ! chunk index
   integer :: iaerosol    ! index into aerosol list
   integer :: numaerosols ! index into aerosol list
   character(len=ot_length) :: opticstype       ! hygro or nonhygro
   character(len=64), pointer :: aernames(:)  ! aerosol names, added by fkc

   ! optical props for each aerosol
   real(r8), pointer :: lw_abs(:)
   real(r8), pointer :: lw_hygro_abs(:,:)
   real(r8), pointer :: geometric_radius(:,:)
 
   ! volcanic lookup table
   real(r8), pointer :: r_lw_abs(:,:)  ! radius dependent mass-specific absorption coefficient
   real(r8), pointer :: r_mu(:)        ! log(geometric_mean_radius) domain samples of r_lw_abs(:,:)
   integer  :: idx                     ! index to pbuf for geometric radius
   real(r8) :: mu(pcols,pver)          ! log(geometric_radius)
   real(r8) :: r_mu_min, r_mu_max, wmu, mutrunc
   integer  :: nmu, kmu

   ! for table lookup into rh grid
   real(r8) :: esat(pcols,pver)     ! saturation vapor pressure
   real(r8) :: qsat(pcols,pver)     ! saturation specific humidity
   real(r8) :: rh(pcols,pver)
   real(r8) :: rhtrunc(pcols,pver)
   real(r8) :: wrh(pcols,pver)
   integer  :: krh(pcols,pver)

   ! aerosol (vertical) mass path and extinction
   ! aerosol masses
   real(r8), pointer :: aermmr(:,:),aermmr1(:,:)    ! mass mixing ratio of aerosols
   real(r8) :: mmr_to_mass(pcols,pver) ! conversion factor for mmr to mass
   real(r8) :: aermass(pcols,pver),aermass1(pcols,pver)     ! mass of aerosols
   !-----------------------------------------------------------------------------

   ncol = state%ncol
   lchnk = state%lchnk
   ! added by fkc
   call rad_cnst_get_info(list_idx, naero=numaerosols)
   allocate(aernames(numaerosols))
   call rad_cnst_get_info(list_idx, aernames=aernames)
   ! end
#ifdef MODAL_AERO
   if(list_idx==0) then
      call modal_aero_lw(state, pbuf, odap_aer)
   else
      odap_aer = 0._r8
   end if

   ! number of aerosols
   call rad_cnst_get_info(list_idx, naero=numaerosols)

   ! Loop over aerosols and look for volcanic aerosol.  This is the only aerosol
   ! that will be treated in this loop and externally mixed with the modal aerosols.
   do iaerosol = 1, numaerosols

      call rad_cnst_get_aer_props(list_idx, iaerosol, opticstype=opticstype)

      if (trim(opticstype) == 'volcanic_radius') then

         ! compute mixing ratio to mass conversion
         do k = 1, pver
            mmr_to_mass(:ncol,k) = rga * state%pdeldry(:ncol,k)
         end do
            
         ! compute optical depths odap_aer (summed over all aerosols)
         ! total optical depths from top (odap_aer_ttl)

         ! calculate relative humidity for table lookup into rh grid
         call aqsat(state%t, state%pmid, esat, qsat, pcols, &
              ncol, pver, 1, pver)
         rh(1:ncol,1:pver) = state%q(1:ncol,1:pver,1) / qsat(1:ncol,1:pver)
            
         rhtrunc(1:ncol,1:pver) = min(rh(1:ncol,1:pver),1._r8)
         krh(1:ncol,1:pver) = min(floor( rhtrunc(1:ncol,1:pver) * nrh ) + 1, nrh - 1) ! index into rh mesh
         wrh(1:ncol,1:pver) = rhtrunc(1:ncol,1:pver) * nrh - krh(1:ncol,1:pver)       ! (-) weighting on left side values

      ! get aerosol mass mixing ratio
      ! modified by fkc
#ifdef GEATM
      nstep = get_nstep()
      if (nstep.eq.0) then
          call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr1)
          aermass(1:ncol,1:pver) = aermmr1(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
      else
      ! get aerosol mass mixing ratio
          write(iulog,*)'calling lw new_rad_cnst' 
          call new_rad_cnst_get_aer_mmr(list_idx, iaerosol,aernames(iaerosol), state, pbuf, aermass1)
          aermass(1:ncol,1:pver) = aermass1(1:ncol,1:pver)*mmr_to_mass(1:ncol,1:pver)
          aermass(1:ncol,1:pver) = aermass(1:ncol,1:pver)*1.0E-9_r8/(1.293_r8*273.15_r8/state%t(1:ncol,1:pver))
      end if
      ! end 
#else
         call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr)
         aermass(1:ncol,1:pver) = aermmr(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
#endif

         ! get optical properties for hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, r_lw_abs=r_lw_abs, mu=r_mu)

         ! get microphysical properties for volcanic aerosols
         idx = pbuf_get_fld_idx('VOLC_RAD_GEOM')
         geometric_radius => pbuf(idx)%fld_ptr( 1, 1:pcols, 1:pver, lchnk, 1)

         ! interpolate in radius
         ! caution: clip the table with no warning when outside bounds
         nmu = size(r_mu)
         r_mu_max = r_mu(nmu)
         r_mu_min = r_mu(1)
         do i = 1, ncol
            do k = 1, pver
               if(geometric_radius(i,k) > 0.) then
                  mu(i,k) = log(geometric_radius(i,k))
               else
                  mu(i,k) = 0.
               endif
               mutrunc = max(min(mu(i,k),r_mu_max),r_mu_min)
               kmu = max(min(1 + (mutrunc-r_mu_min)/(r_mu_max-r_mu_min)*(nmu-1),nmu-1._r8),1._r8)
               wmu = max(min( (mutrunc -r_mu(kmu)) / (r_mu(kmu+1) - r_mu(kmu)) ,1._r8),0._r8)
               do bnd_idx = 1, nlwbands
                  odap_aer(i,k,bnd_idx) = odap_aer(i,k,bnd_idx) + &
                       aermass(i,k) * &
                       ((1._r8 - wmu) * r_lw_abs(bnd_idx, kmu  ) + &
                       (wmu) * r_lw_abs(bnd_idx, kmu+1))
               end do
            end do
         end do

      end if ! volcanic aerosol
   end do

#else
   ! compute mixing ratio to mass conversion
   do k = 1, pver
      mmr_to_mass(:ncol,k) = rga * state%pdeldry(:ncol,k)
   end do

   ! compute optical depths odap_aer (summed over all aerosols)
   ! total optical depths from top (odap_aer_ttl)
   odap_aer = 0._r8

   ! calculate relative humidity for table lookup into rh grid
   call aqsat(state%t, state%pmid, esat, qsat, pcols, &
               ncol, pver, 1, pver)
   rh(1:ncol,1:pver) = state%q(1:ncol,1:pver,1) / qsat(1:ncol,1:pver)

   rhtrunc(1:ncol,1:pver) = min(rh(1:ncol,1:pver),1._r8)
   krh(1:ncol,1:pver) = min(floor( rhtrunc(1:ncol,1:pver) * nrh ) + 1, nrh - 1) ! index into rh mesh
   wrh(1:ncol,1:pver) = rhtrunc(1:ncol,1:pver) * nrh - krh(1:ncol,1:pver)       ! (-) weighting on left side values


   ! number of aerosols
   call rad_cnst_get_info(list_idx, naero=numaerosols)

   ! Loop over aerosols in list.  All aerosols are externally mixed when
   ! not using modal aerosols.
   do iaerosol = 1, numaerosols

      ! modified by fkc
#ifdef GEATM
      nstep = get_nstep()
      if (nstep.eq.0) then
          call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr1)
          aermass(1:ncol,1:pver) = aermmr1(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
      else
      ! get aerosol mass mixing ratio
          call new_rad_cnst_get_aer_mmr(list_idx, iaerosol,aernames(iaerosol), state, pbuf, aermass1)
          aermass(1:ncol,1:pver) = aermass1(1:ncol,1:pver)*mmr_to_mass(1:ncol,1:pver)
          aermass(1:ncol,1:pver) = aermass(1:ncol,1:pver)*1.0E-9_r8/(1.293_r8*273.15_r8/state%t(1:ncol,1:pver))
      end if
      ! end 
#else
         call rad_cnst_get_aer_mmr(list_idx, iaerosol, state, pbuf, aermmr)
         aermass(1:ncol,1:pver) = aermmr(1:ncol,1:pver) * mmr_to_mass(1:ncol,1:pver)
#endif



      call rad_cnst_get_aer_props(list_idx, iaerosol, opticstype=opticstype)
      select case (trim(opticstype))
      case('hygroscopic')
          ! get optical properties for hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, lw_hygro_ext=lw_hygro_abs)
         do bnd_idx = 1, nlwbands
            do k = 1, pver
               do i = 1, ncol
                  odap_aer(i, k, bnd_idx) = odap_aer(i, k, bnd_idx) + &
                       aermass(i, k) * &
                       ((1 + wrh(i,k)) * lw_hygro_abs(krh(i,k)+1,bnd_idx) &
                       - wrh(i,k)  * lw_hygro_abs(krh(i,k),  bnd_idx))
               end do
            end do
         end do
      case('insoluble','nonhygro','hygro','volcanic')
          ! get optical properties for hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, lw_ext=lw_abs)
         do bnd_idx = 1, nlwbands
            do k = 1, pver          
               do i = 1, ncol
                  odap_aer(i,k,bnd_idx) = odap_aer(i,k,bnd_idx) + lw_abs(bnd_idx)*aermass(i,k)
               end do
            end do
         end do
         
      case('volcanic_radius')
          ! get optical properties for hygroscopic aerosols
         call rad_cnst_get_aer_props(list_idx, iaerosol, r_lw_abs=r_lw_abs, mu=r_mu)
         ! get microphysical properties for volcanic aerosols
         idx = pbuf_get_fld_idx('VOLC_RAD_GEOM')
         geometric_radius => pbuf(idx)%fld_ptr( 1, 1:pcols, 1:pver, lchnk, 1)
         
         ! interpolate in radius
         ! caution: clip the table with no warning when outside bounds
         nmu = size(r_mu)
         r_mu_max = r_mu(nmu)
         r_mu_min = r_mu(1)
         do i = 1, ncol
            do k = 1, pver
               if(geometric_radius(i,k) > 0.) then
                  mu(i,k) = log(geometric_radius(i,k))
               else
                  mu(i,k) = 0.
                endif
                mutrunc = max(min(mu(i,k),r_mu_max),r_mu_min)
                kmu = max(min(1 + (mutrunc-r_mu_min)/(r_mu_max-r_mu_min)*(nmu-1),nmu-1._r8),1._r8)
                wmu = max(min( (mutrunc -r_mu(kmu)) / (r_mu(kmu+1) - r_mu(kmu)) ,1._r8),0._r8)
                do bnd_idx = 1, nlwbands
                   odap_aer(i,k,bnd_idx) = odap_aer(i,k,bnd_idx) + &
                        aermass(i,k) * &
                        ((1._r8 - wmu) * r_lw_abs(bnd_idx, kmu  ) + &
                        (wmu) * r_lw_abs(bnd_idx, kmu+1))
                end do
             end do
          end do

       case('zero')
          ! zero aerosols types have no optical effect, so do nothing.
       case default
          call endrun('aer_rad_props_lw: unsupported opticstype: '//trim(opticstype))
       end select
    end do

#endif
    ! added by fkc
    deallocate(aernames)
    ! end
end subroutine aer_rad_props_lw

!==============================================================================
! Private methods
!==============================================================================

subroutine get_hygro_rad_props(ncol, krh, wrh, mass, ext, ssa, asm, &
                               tau, tau_w, tau_w_g, tau_w_f)

   ! Arguments
   integer,  intent(in) :: ncol
   integer,  intent(in) :: krh(pcols,pver)  ! index for linear interpolation of optics on rh
   real(r8), intent(in) :: wrh(pcols,pver)  ! weight for linear interpolation of optics on rh
   real(r8), intent(in) :: mass(pcols,pver)
   real(r8), intent(in) :: ext(:,:)
   real(r8), intent(in) :: ssa(:,:)
   real(r8), intent(in) :: asm(:,:)

   real(r8), intent(out) :: tau    (pcols,pver,nswbands)
   real(r8), intent(out) :: tau_w  (pcols,pver,nswbands)
   real(r8), intent(out) :: tau_w_g(pcols,pver,nswbands)
   real(r8), intent(out) :: tau_w_f(pcols,pver,nswbands)

   ! Local variables
   real(r8) :: ext1, ssa1, asm1
   integer :: icol, ilev, iswband
   !-----------------------------------------------------------------------------

   do iswband = 1, nswbands
      do icol = 1, ncol
         do ilev = 1, pver
            ext1 = (1 + wrh(icol,ilev)) * ext(krh(icol,ilev)+1,iswband) &
                      - wrh(icol,ilev)  * ext(krh(icol,ilev),  iswband)
            ssa1 = (1 + wrh(icol,ilev)) * ssa(krh(icol,ilev)+1,iswband) &
                      - wrh(icol,ilev)  * ssa(krh(icol,ilev),  iswband)
            asm1 = (1 + wrh(icol,ilev)) * asm(krh(icol,ilev)+1,iswband) &
                      - wrh(icol,ilev)  * asm(krh(icol,ilev),  iswband)
  
            tau    (icol, ilev, iswband) = mass(icol, ilev) * ext1
            tau_w  (icol, ilev, iswband) = mass(icol, ilev) * ext1 * ssa1
            tau_w_g(icol, ilev, iswband) = mass(icol, ilev) * ext1 * ssa1 * asm1
            tau_w_f(icol, ilev, iswband) = mass(icol, ilev) * ext1 * ssa1 * asm1 * asm1
         enddo
      enddo
   enddo

end subroutine get_hygro_rad_props 

!==============================================================================
    
subroutine get_nonhygro_rad_props(ncol, mass, ext, ssa, asm, &
                                  tau, tau_w, tau_w_g, tau_w_f)

   ! Arguments
   integer,  intent(in) :: ncol
   real(r8), intent(in) :: mass(pcols, pver)
   real(r8), intent(in) :: ext(:)
   real(r8), intent(in) :: ssa(:)
   real(r8), intent(in) :: asm(:)

   real(r8), intent(out) :: tau    (pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w  (pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w_g(pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w_f(pcols, pver, nswbands) 

   ! Local variables
   integer  :: iswband
   real(r8) :: ext1, ssa1, asm1
   !-----------------------------------------------------------------------------
   
   do iswband = 1, nswbands
      ext1 = ext(iswband)
      ssa1 = ssa(iswband)
      asm1 = asm(iswband)
      tau    (1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * ext1
      tau_w  (1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * ext1 * ssa1
      tau_w_g(1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * ext1 * ssa1 * asm1
      tau_w_f(1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * ext1 * ssa1 * asm1 * asm1
   enddo

end subroutine get_nonhygro_rad_props

!==============================================================================
    
subroutine get_volcanic_radius_rad_props(lchnk, ncol, mass, pbuf, r_ext, r_scat, r_ascat, r_mu, &
                                  tau, tau_w, tau_w_g, tau_w_f)

   use phys_buffer,       only: pbuf_get_fld_idx, pbuf_size_max

   ! Arguments
   integer,  intent(in) :: lchnk
   integer,  intent(in) :: ncol
   real(r8), intent(in) :: mass(pcols, pver)
   type(pbuf_fld),      intent(in) :: pbuf(pbuf_size_max)
   real(r8), intent(in) :: r_ext(:,:)
   real(r8), intent(in) :: r_scat(:,:)
   real(r8), intent(in) :: r_ascat(:,:)
   real(r8), intent(in) :: r_mu(:) ! log(radius) domain of mass-specific optics

   real(r8), intent(out) :: tau    (pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w  (pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w_g(pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w_f(pcols, pver, nswbands) 

   ! Local variables
   integer  :: iswband
   real(r8) :: g

   integer  :: idx                             ! index to radius in physics buffer
   real(r8), pointer :: geometric_radius(:,:)  ! geometric mean radius of volcanic aerosol
   real(r8) :: mu(pcols,pver)                  ! log(geometric mean radius of volcanic aerosol)
   integer  :: imu                             ! index into table values of log radius
   integer  :: kmu, nmu
   real(r8) :: wmu, mutrunc, r_mu_max, r_mu_min
 
   ! interpolated values from table
   real(r8) :: ext(nswbands)
   real(r8) :: scat(nswbands)
   real(r8) :: ascat(nswbands)

   integer :: i, k ! column level iterator
   !-----------------------------------------------------------------------------

   tau    =0._r8                 
   tau_w  =0._r8                 
   tau_w_g=0._r8                 
   tau_w_f=0._r8                  

   ! get microphysical properties for volcanic aerosols
   idx = pbuf_get_fld_idx('VOLC_RAD_GEOM')
   geometric_radius => pbuf(idx)%fld_ptr( 1, 1:pcols, 1:pver, lchnk, 1 )

   ! interpolate in radius
   ! caution: clip the table with no warning when outside bounds
   nmu = size(r_mu)
   r_mu_max = r_mu(nmu)
   r_mu_min = r_mu(1)
   do i = 1, ncol
      do k = 1, pver
         if(geometric_radius(i,k) > 0.) then
            mu(i,k) = log(geometric_radius(i,k))
         else
            mu(i,k) = 0.
         endif
         mutrunc = max(min(mu(i,k),r_mu_max),r_mu_min)
         kmu = max(min(1 + (mutrunc-r_mu_min)/(r_mu_max-r_mu_min)*(nmu-1),nmu-1._r8),1._r8)
         wmu = max(min( (mutrunc -r_mu(kmu)) / (r_mu(kmu+1) - r_mu(kmu)) ,1._r8),0._r8)
         do iswband = 1, nswbands
            ext(iswband) =  &
               ((1._r8 - wmu) * r_ext(iswband, kmu  ) + &
               (wmu) * r_ext(iswband, kmu+1))
            scat(iswband) =  &
               ((1._r8 - wmu) * r_scat(iswband, kmu  ) + &
               (wmu) * r_scat(iswband, kmu+1))
            ascat(iswband) =  &
               ((1._r8 - wmu) * r_ascat(iswband, kmu  ) + &
               (wmu) * r_ascat(iswband, kmu+1))
            if (scat(iswband).gt.0.) then
               g = ascat(iswband)/scat(iswband)
            else
               g=0._r8
            endif
            tau    (i,k,iswband) = mass(i,k) * ext(iswband)  
            tau_w  (i,k,iswband) = mass(i,k) * scat(iswband)  
            tau_w_g(i,k,iswband) = mass(i,k) * ascat(iswband)  
            tau_w_f(i,k,iswband) = mass(i,k) * g * ascat(iswband)  
         end do
      enddo
   enddo

end subroutine get_volcanic_radius_rad_props

!==============================================================================
    
subroutine get_volcanic_rad_props(ncol, mass, ext, scat, ascat, &
                                  tau, tau_w, tau_w_g, tau_w_f)

   ! Arguments
   integer,  intent(in) :: ncol
   real(r8), intent(in) :: mass(pcols, pver)
   real(r8), intent(in) :: ext(:)
   real(r8), intent(in) :: scat(:)
   real(r8), intent(in) :: ascat(:)

   real(r8), intent(out) :: tau    (pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w  (pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w_g(pcols, pver, nswbands)
   real(r8), intent(out) :: tau_w_f(pcols, pver, nswbands) 

   ! Local variables
   integer  :: iswband
   real(r8) :: g
   !-----------------------------------------------------------------------------
   
   do iswband = 1, nswbands
      if (scat(iswband).gt.0.) then
         g = ascat(iswband)/scat(iswband)
      else
         g=0._r8
      endif
      tau    (1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * ext(iswband) 
      tau_w  (1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * scat(iswband) 
      tau_w_g(1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * ascat(iswband) 
      tau_w_f(1:ncol,1:pver,iswband) = mass(1:ncol,1:pver) * g * ascat(iswband) 
   enddo

end subroutine get_volcanic_rad_props

!==============================================================================

subroutine aer_vis_diag_out(lchnk, ncol, nnite, idxnite, iaer, tau, diag_idx)

   ! output aerosol optical depth for the visible band

   integer,          intent(in) :: lchnk
   integer,          intent(in) :: ncol           ! number of columns
   integer,          intent(in) :: nnite          ! number of night columns
   integer,          intent(in) :: idxnite(:)     ! local column indices of night columns
   integer,          intent(in) :: iaer           ! aerosol index -- if 0 then tau is a total for all aerosols
   real(r8),         intent(in) :: tau(:,:)       ! aerosol optical depth for the visible band
   integer,          intent(in) :: diag_idx       ! identifies whether the aerosol optics
                                                  ! is for the climate calc or a diagnostic calc
 
   ! Local variables
   integer  :: i
   real(r8) :: tmp(pcols)
   !-----------------------------------------------------------------------------

   ! currently only implemented for climate calc
   if (diag_idx > 0) return

   ! compute total column aerosol optical depth
   tmp(1:ncol) = sum(tau(1:ncol,:), 2)
   ! use fillvalue to indicate night columns
   do i = 1, nnite
      tmp(idxnite(i)) = fillvalue
   end do

   if (iaer > 0) then
      call outfld(odv_names(iaer), tmp, pcols, lchnk)
   else
      call outfld('AEROD_v', tmp, pcols, lchnk)
   end if

end subroutine aer_vis_diag_out

! added by fkc
#ifdef GEATM
subroutine gea_aer_out(lchnk, aermass, aername)
      integer,                     intent(in) :: lchnk
      character(len=64),           intent(in) :: aername
      real(r8)                   , intent(in):: aermass(pcols,pver)     ! mass of aerosols

      if (trim(aername)=='dust1') then
          call outfld('GEATM_dust1', aermass, pcols, lchnk)
      else if (trim(aername)=='dust2') then
          call outfld('GEATM_dust2', aermass, pcols, lchnk)
      else if (trim(aername)=='dust3') then
          call outfld('GEATM_dust3', aermass, pcols, lchnk)
      else if (trim(aername)=='dust4') then
          call outfld('GEATM_dust4', aermass, pcols, lchnk)
      else if (trim(aername)=='sslt1') then
          call outfld('GEATM_sslt1', aermass, pcols, lchnk)
      else if (trim(aername)=='sslt2') then
          call outfld('GEATM_sslt2', aermass, pcols, lchnk)
      else if (trim(aername)=='sslt3') then
          call outfld('GEATM_sslt3', aermass, pcols, lchnk)
      else if (trim(aername)=='sslt4') then
          call outfld('GEATM_sslt4', aermass, pcols, lchnk)
      else if (trim(aername)=='bcar1') then
          call outfld('GEATM_bcar1', aermass, pcols, lchnk)
      else if (trim(aername)=='bcar2') then
          call outfld('GEATM_bcar2', aermass, pcols, lchnk)
      else if (trim(aername)=='sulf') then
          call outfld('GEATM_sulf', aermass, pcols, lchnk)
      else if (trim(aername)=='ocar1') then
          call outfld('GEATM_ocar1', aermass, pcols, lchnk)
      else if (trim(aername)=='ocar2') then
          call outfld('GEATM_ocar2', aermass, pcols, lchnk)
      end if
end subroutine gea_aer_out
#endif


!==============================================================================

end module aer_rad_props
