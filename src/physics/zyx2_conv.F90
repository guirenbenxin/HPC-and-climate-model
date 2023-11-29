module zyx2_conv   
!------------------------------------------------------
! Stochastic Convective Parameterization Scheme
! contributors: Minghua Zhang, Xin Xie, Haiyang Yu
!------------------------------------------------------
! Convention:
! SI unit
! ***************************************************
!                ATMOSPHERE TOP k=1
! ***************************************************
!                         .
!                         .
!                         .
!                         .  CLOUD TOP (NEUTRAL BOUYANCY)[kuptop]
!                         .
!                         .
!                         .
! ********cloud vars are at the interface************       K+1/2 K_cloud=1
!
! **variables : all input vars are in the mid level**       K=1
!                         .
! ********cloud vars are at the interface************       K+1/2 K_cloud=2
!                     ...........
!                     ...........
!                     ...........
!                     ...........
! ********cloud vars are at the interface************       K+1/2 K_cloud=nlev
!
! **variables : all input vars are in the mid level**       K=nlev
!
! ********cloud vars are at the interface************       K+1/2 K_cloud=nlev+1
!                         .
!                         .  CLOUD BASE (POSITIVE BOUYANCY) [kupbase]
!                         .
!                         .
!                         .  LCL: [kuplcl]
!                         .
!                         .
!                         .  LAUNCH LEVEL: [kuplaunch] max MSE
!                         .
! ***************************************************
!                ATMOSPHERE BOTTOM k=nlev
! ***************************************************
!
!------------------------------------------------------

#ifdef SCMDIAG
    use scmdiag, only: subcol_netcdf_nextstep
    use scmdiag, only: subcol_netcdf_putclm, subcol_netcdf_putfld
#endif

!============================
!MZ ===== to add CAM ZM option
  use spmd_utils,      only: masterproc
  use cloud_fraction,  only: cldfrc_fice
  use abortutils,      only: endrun
  use cam_logfile,     only: iulog
  use perf_mod
  use shr_kind_mod,    only: r8 => shr_kind_r8
  use phys_control, only: phys_getopts, phys_deepconv_pbl, &
                          plume_model, trigger_scheme
  use units,           only: getunit, freeunit
!MZ ===== done 

    implicit none
    private
    save

    !integer,parameter :: r8 = selected_real_kind(12)

    public :: zyx2_conv_init, zyx2_conv_tend
    public :: ecp_readnl
    public :: nplume_sh_zyx2, nplume_dp_zyx2
#ifndef OFFLINECP
    integer :: ncol=0, nlev=0, nlevp=0
#endif
    real(r8), parameter :: unset_r8 = huge(1.0_r8)  ! set namelist variables
    integer,  parameter :: unset_int = -1       ! set namelist variables
!--------------------------------------------------------------
! tunable variables from namelist
!--------------------------------------------------------------
    integer :: ecp_ctopflag = unset_int
    integer :: ecp_nplume_sh = unset_int
    integer :: ecp_nplume_dp = unset_int
!MZ<
    integer :: nplume_sh_zyx2 
    integer :: nplume_dp_zyx2 
!>
    real(r8) :: ecp_turb_enhance = unset_r8
    real(r8) :: ecp_basemass_enhance = unset_r8
    real(r8) :: ecp_org_enhance = unset_r8
    real(r8) :: ecp_org_shape = unset_r8
    integer  :: ecp_flagorgent = unset_int
    real(r8) :: ecp_evap_enhance = unset_r8
    real(r8) :: ecp_evap_shape = unset_r8
    real(r8) :: ecp_qnegtop = unset_r8
    real(r8) :: ecp_trig_eps0 = unset_r8
    real(r8) :: ecp_trig_c2 = unset_r8
    real(r8) :: ecp_fixcldsr = unset_r8
    real(r8) :: ecp_ratio_ent_rad = unset_r8
    real(r8) :: ecp_orgent_a = unset_r8
    real(r8) :: ecp_orgent_beta0 = unset_r8
    real(r8) :: ecp_w_up_init_sh_beg = unset_r8
    real(r8) :: ecp_w_up_init_sh_end = unset_r8
    real(r8) :: ecp_w_up_init_dp_beg = unset_r8
    real(r8) :: ecp_w_up_init_dp_end = unset_r8
    real(r8) :: ecp_w_init_shape_sh = unset_r8
    real(r8) :: ecp_w_init_shape_dp = unset_r8
    real(r8) :: ecp_rain_z0 = unset_r8
    real(r8) :: ecp_rain_zp = unset_r8
    real(r8) :: ecp_dn_be = unset_r8
    real(r8) :: ecp_dn_ae = unset_r8
    real(r8) :: ecp_dn_vt = unset_r8
    real(r8) :: ecp_dn_frac_sh = unset_r8
    real(r8) :: ecp_dn_frac_dp = unset_r8
    real(r8) :: ecp_pmf_alpha_sh = unset_r8
    real(r8) :: ecp_pmf_tau_sh = unset_r8
    real(r8) :: ecp_pmf_alpha_dp = unset_r8
    real(r8) :: ecp_pmf_tau_dp = unset_r8
    real(r8) :: ecp_capelmt_sh   = unset_r8
    real(r8) :: ecp_capelmt_dp   = unset_r8
    real(r8) :: ecp_tpertglob   = unset_r8
    real(r8) :: ecp_qpertglob   = unset_r8
    integer  :: ecp_meanorsum   = unset_int
    real(r8) :: ecp_facdlf = unset_r8

!--------------------------------------------------------------
! physical parameter
!--------------------------------------------------------------
    real(r8), parameter :: gravit = 9.80616         ! gravitational acceleration (m/s**2)
    real(r8), parameter :: pi     = 3.141592653     ! Pi
    real(r8), parameter :: cpair  = 1004.64         ! specific heat of dry air (J/K/kg)
    real(r8), parameter :: cpliq  = 4188.           ! specific heat of fresh h2o (J/K/kg)
    real(r8), parameter :: cpwv   = 1810.           ! specific heat of water vapor (J/K/kg)
    real(r8), parameter :: latvap = 2.501e6         ! latent heat of vaporization (J/kg)
    real(r8), parameter :: latice = 3.34e5          ! latent heat of freezing (J/kg)
    real(r8), parameter :: epsilo = 0.6219705862    ! ratio of h2o to dry air molecular weights
    real(r8), parameter :: tmelt  = 273.15          ! Freezing point of water (K)
    real(r8), parameter :: rair   = 287.042311365   ! Dry air gas constant (J/K/kg)
    real(r8), parameter :: rh2o   = 461.5046398202  ! Water vapor gas constant (J/K/kg)
    real(r8), parameter :: rhofw  = 1000.           ! liquid water density (kg/m3)
    real(r8), parameter :: tveps  = 1.0/epsilo - 1  ! coefficient of virsual temperature
   
!--------------------------------------------------------------
! parameters of the new scheme: MZhang scheme
!--------------------------------------------------------------
    integer,  parameter :: ischeme = 2          ! 1: CS2010;  2: MZhang Group
    integer,  parameter :: flagbspdf = 2        ! 1: uniform distribution;  2: new pdf
    integer,  parameter :: flagqcheck = 2       ! 1: Xin's method;  2: new method
    integer,  parameter :: cldhiteration = 2    ! iteration for cloud height
    integer :: flagorgent = unset_int       ! 1: using beta0 and minMSE as the division between entr and detr
                                                ! 2: new organized entr and detr, and use half of H as division
                                                ! 3,4: only organized entr, no orgnized detr
                                                ! 5: when B<=0, use detrain
                                                ! 6: same as 5, but add a profile shape
                                                ! 7: for shallow plume, when dB/dz<0, use detrain
                                                ! 8: when dB/dz<0, use detrain
    integer,  parameter :: flagtotent = 3       ! 1: organized only; 2: turbulence only; 3: sum of org and turb
    integer,  parameter :: flagturbent = 1      ! 1: using low interface only; 
                                                ! 2: use low interface at the first iteration, then averaged
    integer,  parameter :: flagbuoysort = 2     ! 1: old codes from CAM UW
                                                ! 2: new codes 
    real(r8) :: qnegtop = unset_r8
    real(r8) :: trig_eps0  = unset_r8   ! trigger parameters: w -> R (default: 0.003)
    real(r8) :: trig_c2    = unset_r8   ! trigger parameters: w -> R: 23.5 ~ 240m; default: 117.5 ~ 1km
    real(r8) :: turb_enhance = unset_r8   ! enhance turbulence entrainment and detrainment (1.0)
    real(r8) :: basemass_enhance = unset_r8   ! enhance cloud base mass flux for shallow plumes (>1.0)
    real(r8) :: org_enhance = unset_r8   ! enhance organized entrainment and detrainment (2.0)
    real(r8) :: org_shape = unset_r8     ! shape parameter of organized entrainment and detrainment (2.0)
    real(r8) :: evap_enhance = unset_r8   ! enhance evaporation (5.0)
    real(r8) :: evap_shape = unset_r8     ! shape parameter of evaporation profile (2.0)
    real(r8), parameter :: fixcldrad0 = -1.0      ! if +: fixed cloud base radius;  if -: diagnostic from winit
    real(r8) :: fixcldsr   = unset_r8   ! default: 1.0; if +: fixed cloud size ratio;  if -: iteration
    real(r8) :: ratio_ent_rad = unset_r8  ! relation between turbulent entr/detr and cloud radius (0.2)
    real(r8) :: orgent_a     = unset_r8  ! organized entrainment parameters: default is 1.23
    real(r8) :: orgent_beta0 = unset_r8   ! organized entrainment parameters: default is 2.0
    real(r8), parameter :: wupmin = 0.01        ! threshold of w for stopping convection
    real(r8), parameter :: wupmax = 100.0       ! upbound of w
!#ifdef SCMDIAG
!    real(r8), parameter :: fixbasemf = 0.01    ! if +: fixed cloud base mass flux; if -: prognostic
!#endif
!#if (! defined SCMDIAG)
    real(r8), parameter :: fixbasemf = -0.01    ! if +: fixed cloud base mass flux; if -: prognostic
!#endif
    
!MZ
    real(r8), parameter :: zpbltop = -1000.0    ! if +: downdraft mass flux decreases gradually in PBL
                                                ! if -: no decreasing 
    integer :: meanorsum = unset_int            ! 1: mean of plumes
                                                ! 2: sum of plumes
    real(r8) :: facdlf = unset_r8
!--------------------------------------------------------------
! GRE and NSJ parameters
!--------------------------------------------------------------
    integer, parameter :: flagmeaniter = 0      ! 0: no mean;  
                                                ! 1: mean of entr/detr of the last iteration
    integer, parameter :: maxiteration = 2      ! maximum iteration number for mseQi
    integer :: ctopflag = unset_int          ! 1: B<0; 2: w<0
    integer, parameter :: buoyflag = 2          ! 1: B=Tv'/Tv; 2: B=Tv'/Tv - qliq - qice
    integer, parameter :: mse2tsatflag = 1      ! 1: Taylor; 2: bi-section
    integer, parameter :: mseqiflag = 1         ! 1: use Qi; 0: Qi=0
    integer, parameter :: entratemidflag = 1    ! 1: averaged; 2: recalculated with B and w
    integer, parameter :: bsflag = 1            ! 0: no buoy sort 1: buoy sort
    integer, parameter :: mflxmeanflag = 1      ! 0: averaged; 1: with log weighted
    integer, parameter :: negcondflag  = 1      ! 0: exit when c<0; 1: recalculate qv,T when c<0

    real(r8), parameter :: max_ent_rate = 4.e-3_r8      ! maximum entrainment rate (1/m)
    real(r8), parameter :: max_det_rate = 4.e-3_r8      ! maximum detrainment rate (1/m)

!MZ 2018-08-04
    !real(r8), parameter :: zuplaunchtop = 6000.0      ! default: 3000; max cloud parcel launch height [m]
    !real(r8), parameter :: zuplaunchlow = 0.0         ! default: 0; min cloud parcel launch height [m]
    real(r8), parameter :: zuplaunchtop = 3000.0      ! default: 3000; max cloud parcel launch height [m]
    real(r8), parameter :: zuplaunchlow = 100.0         ! default: 0; min cloud parcel launch height [m]
    
    integer :: nplume_sh = unset_int               ! shallow plumes
    integer :: nplume_dp = unset_int                ! deep plumes

!--------------------------------------------------------------
! shallow plumes parameters
!--------------------------------------------------------------
    real(r8), parameter :: greg_z0_sh = 1.e4_r8
    real(r8), parameter :: greg_ent_a_sh_beg = 0.15_r8
    real(r8), parameter :: greg_ent_a_sh_end = 0.15_r8
    real(r8), parameter :: greg_ce_sh_beg    = 0.8_r8
    real(r8), parameter :: greg_ce_sh_end    = 0.8_r8
    real(r8) :: w_up_init_sh_beg  = unset_r8 ! 0.1
    real(r8) :: w_up_init_sh_end  = unset_r8 ! 1.2
    real(r8) :: w_init_shape_sh  = unset_r8
    real(r8) :: w_init_shape_dp  = unset_r8
!--------------------------------------------------------------
! deep plumes parameters
!--------------------------------------------------------------
    real(r8), parameter :: greg_z0_dp = 1.e4_r8 
    real(r8), parameter :: greg_ent_a_dp_beg = 0.15_r8 
    real(r8), parameter :: greg_ent_a_dp_end = 0.15_r8 
    real(r8), parameter :: greg_ce_dp_beg    = 0.5_r8  
    real(r8), parameter :: greg_ce_dp_end    = 0.5_r8  
    real(r8) :: w_up_init_dp_beg  = unset_r8 ! 0.2    
    real(r8) :: w_up_init_dp_end  = unset_r8 ! 3.0

!--------------------------------------------------------------
! parameters for NSJ scheme
!--------------------------------------------------------------
    real(r8), parameter :: nsj_ent_a = 0.9_r8     
    real(r8), parameter :: nsj_coef  = 1.8e-3_r8  

!--------------------------------------------------------------
! rain fraction parameters
!--------------------------------------------------------------
    real(r8) :: rain_z0 = unset_r8      ! default: 0; CS2010: 1500
    real(r8) :: rain_zp = unset_r8   ! default: 1500; CS2010: 4000

!--------------------------------------------------------------
! cloud ice fraction parameters
!--------------------------------------------------------------
!MZ to be consistent with macrop_driver.F90
    !real(r8), parameter :: cloud_t1 = 258.15    ! mixed / ice cloud
    !real(r8), parameter :: cloud_t2 = 273.15    ! liq / mixed cloud
    real(r8), parameter :: cloud_t1 = 238.15    ! mixed / ice cloud
    real(r8), parameter :: cloud_t2 = 268.15    ! liq / mixed cloud
    
!--------------------------------------------------------------
! downdraft parameters
!--------------------------------------------------------------
    real(r8) :: dn_be = unset_r8  ! default: 5.0e-4
    real(r8) :: dn_ae = unset_r8  ! default: 0.3
    real(r8) :: dn_vt = unset_r8  ! default: 10
    real(r8) :: dn_frac_sh = unset_r8     ! default: 0.3; ratio of downdraft base massflux to updraft
    real(r8) :: dn_frac_dp = unset_r8     ! default: 0.3; ratio of downdraft base massflux to updraft

!--------------------------------------------------------------
! parameter for prognostics mass flux calculation
!--------------------------------------------------------------
! default setting: 
!    real(r8), parameter :: pmf_alpha_dp = 4000.e7_r8 , pmf_tau_dp = 800.e3_r8 ! cntr deep
!    real(r8), parameter :: pmf_alpha_sh = 50000.e7_r8, pmf_tau_sh = 20000.e3_r8
    real(r8) :: pmf_alpha_dp = unset_r8
    real(r8) :: pmf_tau_dp = unset_r8
    real(r8) :: pmf_alpha_sh = unset_r8
    real(r8) :: pmf_tau_sh = unset_r8

    real(r8) :: pmf_alpha, pmf_tau

!--------------------------------------------------------------
! parameter for diagnostic mass flux calculation
!--------------------------------------------------------------
    real(r8), parameter :: cape_timescale = 10.e7_r8  ! default: not used
    real(r8) :: capelmt_sh = unset_r8  ! default: 80; threshold of CAPE for triggering convection
    real(r8) :: capelmt_dp = unset_r8  ! default: 80; threshold of CAPE for triggering convection

!--------------------------------------------------------------
! parameter for diagnostic mass flux calculation
!--------------------------------------------------------------
    real(r8) :: tpertglob = unset_r8   ! default: 0.0
    real(r8) :: qpertglob = unset_r8   ! default: 0.0

! -------------------------------------------------------------------
! local variables
! -------------------------------------------------------------------
    integer :: ent_opt ! 0=EC, 2=GREG, 3=NSJ
    integer :: nplume
    integer :: nplume_tot, ind_offset
    real(r8) :: w_up_init_beg 
    real(r8) :: w_up_init_end, w_init_shape
    real(r8) :: greg_ent_a_beg, greg_ent_a_end, greg_ent_a_delta
    real(r8) :: greg_ce_beg, greg_ce_end, greg_ce_delta
    real(r8) :: greg_z0, greg_ent_a, greg_ce
    real(r8) :: c0              ! parameter for converting cloud water to rainfall

    real(r8) :: f_dcape
    real(r8) :: f_cape
    real(r8) :: f_w

!MZ ====== to include optional CAM fields in zm_conv.F90 =========

   !character(len=16)  :: plume_model ! from namelist= 'zyx2'   ! 'cam' or 'zyx2' set by namelist
   character(len=16)  :: deep_scheme! 

   !public zyx2_convi                !  zyx2 scheme not needed
   public zyx2_conv_evap            ! evaporation of precip 
   public convtran                 ! convective transport
   public momtran                  ! convective momentum transport

! Private data
!
   !real(r8), parameter :: unset_r8 = huge(1.0_r8)
   real(r8) :: zmconv_c0_lnd = unset_r8
   real(r8) :: zmconv_c0_ocn = unset_r8
   real(r8) :: zmconv_ke     = unset_r8

   real(r8) rl         ! wg latent heat of vaporization.
   real(r8) cpres      ! specific heat at constant pressure in j/kg-degk.
   real(r8), parameter :: capelmt = 70._r8  ! threshold value for cape for deep convection.
   real(r8) :: ke           ! Tunable evaporation efficiency set from namelist input zmconv_ke
   real(r8) :: c0_lnd       ! set from namelist input zmconv_c0_lnd
   real(r8) :: c0_ocn       ! set from namelist input zmconv_c0_ocn
   real(r8) tau   ! convective time scale
   real(r8),parameter :: c1 = 6.112_r8
   real(r8),parameter :: c2 = 17.67_r8
   real(r8),parameter :: c3 = 243.5_r8
   real(r8) :: tfreez
   real(r8) :: eps1
   logical :: no_deep_pbl = .false. ! default = .false.  !! assigned
                          ! no_deep_pbl = .true. eliminates deep convection entirely within PBL

   real(r8) :: rgrav       ! reciprocal of grav
   real(r8) :: rgas        ! gas constant for dry air
   real(r8) :: grav        ! = gravit
   real(r8) :: cp          ! = cpres = cpair

   integer ::  limcnv      ! MZ note should be limited with a pressure such as 100 mb

   real(r8),parameter ::  tiedke_add = 0.5_r8

!MZ CAM done
!=============

    contains

! ==============================================================================
! initialize the convection scheme
! ==============================================================================
subroutine ecp_readnl(nlfile)
!MZ
    use namelist_utils,  only: find_group_name

#if ((defined SCMDIAG) | (defined OFFLINECP))
    use shr_nl_mod,  only: shr_nl_find_group_name
#endif

#if ((! defined SCMDIAG) & (! defined OFFLINECP))  
!MZ
    !use namelist_utils,  only: find_group_name
    use spmd_utils,      only: masterproc
    use abortutils,      only: endrun
    use units,           only: getunit, freeunit
    use mpishorthand
#endif

   character(len=*), intent(in) :: nlfile  ! filepath for file containing namelist input

   ! Local variables
   integer :: unitn, ierr
   character(len=*), parameter :: subname = 'ecp_readnl'

   namelist /ecp_nl/ ecp_ctopflag, ecp_nplume_sh, ecp_nplume_dp, ecp_qnegtop, ecp_trig_eps0, ecp_trig_c2, &
       ecp_turb_enhance, ecp_basemass_enhance, ecp_fixcldsr, &
       ecp_ratio_ent_rad, ecp_orgent_a, ecp_orgent_beta0, ecp_org_enhance, ecp_org_shape, ecp_flagorgent, &
       ecp_w_up_init_sh_beg, ecp_w_up_init_sh_end, ecp_w_up_init_dp_beg, ecp_w_up_init_dp_end, &
       ecp_w_init_shape_sh, ecp_w_init_shape_dp, &
       ecp_rain_z0, ecp_rain_zp, ecp_dn_be, ecp_dn_ae, ecp_dn_vt, ecp_dn_frac_sh, ecp_dn_frac_dp, &
       ecp_pmf_alpha_sh, ecp_pmf_tau_sh, ecp_pmf_alpha_dp, ecp_pmf_tau_dp, ecp_capelmt_sh, ecp_capelmt_dp, &
       ecp_tpertglob, ecp_qpertglob, ecp_meanorsum, ecp_facdlf, ecp_evap_enhance, ecp_evap_shape
   !-----------------------------------------------------------------------------
!MZ ==== add CAM option from zm_conv_init and zm_convi
!MZ< 
   !namelist /zyx2conv_nl/ plume_model

   namelist /zmconv_nl/ zmconv_c0_lnd, zmconv_c0_ocn, zmconv_ke
   !-----------------------------------------------------------------------------
!MZ>
 
#if ((! defined SCMDIAG) & (! defined OFFLINECP)) 
   if (masterproc) then
       unitn = getunit()
      open( unitn, file=trim(nlfile), status='old' )
      call find_group_name(unitn, 'ecp_nl', status=ierr)
      if (ierr == 0) then
         read(unitn, ecp_nl, iostat=ierr)
         if (ierr /= 0) then
            call endrun(subname // ':: ERROR reading namelist')
         end if
      end if
      close(unitn)
      call freeunit(unitn)
#endif

#if ((defined SCMDIAG) | (defined OFFLINECP))
      open( 10, file=trim(nlfile), status='old' )
      call shr_nl_find_group_name(10, 'ecp_nl', status=ierr)
      if (ierr == 0) then
         read(10, ecp_nl, iostat=ierr)
         if (ierr /= 0) then
            write(*,*) 'Haiyang: ERROR reading namelist'
         end if
      end if
      close(10)
#endif

!!MZ #if ((! defined SCMDIAG) & (! defined OFFLINECP)) 
   
!MZ ===== READ CAM VALUES
      unitn = getunit()
      open( unitn, file=trim(nlfile), status='old' )
      call find_group_name(unitn, 'zmconv_nl', status=ierr)
      if (ierr == 0) then
         read(unitn, zmconv_nl, iostat=ierr)
         if (ierr /= 0) then
            call endrun(subname // ':: ERROR reading namelist')
         end if
      end if

!MZ      
      !call phys_getopts(plume_model_out = plume_model)
      call phys_getopts(deep_scheme_out = deep_scheme)
      write(*,*) '==============================='
      write(*,*) ' deep_scheme is ', deep_scheme 
      write(*,*) ' plume_model is ', plume_model
      write(*,*) ' trigger_scheme is ', trigger_scheme
      write(*,*) '==============================='

!      call find_group_name(unitn, 'zyx2conv_nl', status=ierr)
!      if (ierr == 0) then
!         read(unitn, zyx2conv_nl, iostat=ierr)
!         if (ierr /= 0) then
!            call endrun(subname // ':: ERROR reading namelist zyx2conv_nl')
!         end if
!      end if
!      write(*,*) 'plume model= ',plume_model
!MZ
!      close(unitn)
!      call freeunit(unitn)
!   end if

!!MZ #endif
! ==== MZ done


    ! set local variables
        ctopflag  = ecp_ctopflag
        nplume_sh = ecp_nplume_sh
        nplume_dp = ecp_nplume_dp
        qnegtop   = ecp_qnegtop
        trig_eps0 = ecp_trig_eps0
        trig_c2   = ecp_trig_c2
        turb_enhance = ecp_turb_enhance
        basemass_enhance = ecp_basemass_enhance
        org_enhance = ecp_org_enhance
        org_shape = ecp_org_shape
        flagorgent = ecp_flagorgent
        fixcldsr  = ecp_fixcldsr
        ratio_ent_rad = ecp_ratio_ent_rad
        orgent_a      = ecp_orgent_a
        orgent_beta0  = ecp_orgent_beta0
        w_up_init_sh_beg = ecp_w_up_init_sh_beg
        w_up_init_sh_end = ecp_w_up_init_sh_end
        w_up_init_dp_beg = ecp_w_up_init_dp_beg
        w_up_init_dp_end = ecp_w_up_init_dp_end
        w_init_shape_sh = ecp_w_init_shape_sh
        w_init_shape_dp = ecp_w_init_shape_dp

        rain_z0 = ecp_rain_z0
        rain_zp = ecp_rain_zp
        dn_be   = ecp_dn_be
        dn_ae   = ecp_dn_ae
        dn_vt   = ecp_dn_vt
        dn_frac_sh = ecp_dn_frac_sh
        dn_frac_dp = ecp_dn_frac_dp
        pmf_alpha_sh = ecp_pmf_alpha_sh
        pmf_tau_sh   = ecp_pmf_tau_sh
        pmf_alpha_dp = ecp_pmf_alpha_dp
        pmf_tau_dp   = ecp_pmf_tau_dp
        capelmt_sh = ecp_capelmt_sh
        capelmt_dp = ecp_capelmt_dp
        tpertglob = ecp_tpertglob
        qpertglob = ecp_qpertglob
        meanorsum = ecp_meanorsum 
        facdlf = ecp_facdlf
        evap_enhance = ecp_evap_enhance
        evap_shape = ecp_evap_shape
!MZ<
        nplume_sh_zyx2 = ecp_nplume_sh
        nplume_dp_zyx2 = ecp_nplume_dp

        c0_lnd = zmconv_c0_lnd
        c0_ocn = zmconv_c0_ocn
        ke = zmconv_ke
!MZ>

#if ((! defined SCMDIAG) & (! defined OFFLINECP)) 
    end if
#endif

#ifdef SPMD
   ! Broadcast namelist variables
   call mpibcast(ctopflag,  1, mpiint,  0, mpicom)
   call mpibcast(nplume_sh,  1, mpiint,  0, mpicom)
   call mpibcast(nplume_dp,  1, mpiint,  0, mpicom)
   call mpibcast(meanorsum,  1, mpiint,  0, mpicom)
   call mpibcast(qnegtop,  1, mpir8,  0, mpicom)
   call mpibcast(trig_eps0,  1, mpir8,  0, mpicom)
   call mpibcast(trig_c2,    1, mpir8,  0, mpicom)
   call mpibcast(turb_enhance,    1, mpir8,  0, mpicom)
   call mpibcast(basemass_enhance,    1, mpir8,  0, mpicom)
   call mpibcast(org_enhance,    1, mpir8,  0, mpicom)
   call mpibcast(org_shape,    1, mpir8,  0, mpicom)
   call mpibcast(flagorgent,    1, mpiint,  0, mpicom)
   call mpibcast(fixcldsr,   1, mpir8,  0, mpicom)
   call mpibcast(ratio_ent_rad,  1, mpir8,  0, mpicom)
   call mpibcast(orgent_a,  1, mpir8,  0, mpicom)
   call mpibcast(orgent_beta0,  1, mpir8,  0, mpicom)
   call mpibcast(w_up_init_sh_beg,  1, mpir8,  0, mpicom)
   call mpibcast(w_up_init_sh_end,  1, mpir8,  0, mpicom)
   call mpibcast(w_up_init_dp_beg,  1, mpir8,  0, mpicom)
   call mpibcast(w_up_init_dp_end,  1, mpir8,  0, mpicom)
   call mpibcast(w_init_shape_sh,  1, mpir8,  0, mpicom)
   call mpibcast(w_init_shape_dp,  1, mpir8,  0, mpicom)
   call mpibcast(rain_z0,  1, mpir8,  0, mpicom)
   call mpibcast(rain_zp,  1, mpir8,  0, mpicom)
   call mpibcast(dn_be,  1, mpir8,  0, mpicom)
   call mpibcast(dn_ae,  1, mpir8,  0, mpicom)
   call mpibcast(dn_vt,  1, mpir8,  0, mpicom)
   call mpibcast(dn_frac_sh,  1, mpir8,  0, mpicom)
   call mpibcast(dn_frac_dp,  1, mpir8,  0, mpicom)
   call mpibcast(pmf_alpha_sh,  1, mpir8,  0, mpicom)
   call mpibcast(pmf_tau_sh,  1, mpir8,  0, mpicom)
   call mpibcast(pmf_alpha_dp,  1, mpir8,  0, mpicom)
   call mpibcast(pmf_tau_dp,  1, mpir8,  0, mpicom)
   call mpibcast(capelmt_sh,  1, mpir8,  0, mpicom)
   call mpibcast(capelmt_dp,  1, mpir8,  0, mpicom)
   call mpibcast(tpertglob,  1, mpir8,  0, mpicom)
   call mpibcast(qpertglob,  1, mpir8,  0, mpicom)
   call mpibcast(facdlf,  1, mpir8,  0, mpicom)
   call mpibcast(evap_enhance,  1, mpir8,  0, mpicom)
   call mpibcast(evap_shape,  1, mpir8,  0, mpicom)
!MZ<
   call mpibcast(c0_lnd,            1, mpir8,  0, mpicom)
   call mpibcast(c0_ocn,            1, mpir8,  0, mpicom)
   call mpibcast(ke,                1, mpir8,  0, mpicom)
!MZ>
#endif
!MZ<
   if ( masterproc ) then
!MZ>

    write(*, *) "ctopflag: ", ctopflag
    write(*, *) "nplume_sh: ", nplume_sh
    write(*, *) "nplume_dp: ", nplume_dp
    write(*, *) "meanorsum: ", meanorsum
    write(*, *) "qnegtop: ", qnegtop
    write(*, *) "trig_eps0: ", trig_eps0
    write(*, *) "trig_c2: ", trig_c2
    write(*, *) "turb_enhance: ", turb_enhance
    write(*, *) "basemass_enhance: ", basemass_enhance
    write(*, *) "org_enhance: ", org_enhance
    write(*, *) "org_shape: ", org_shape
    write(*, *) "flagorgent: ", flagorgent
    write(*, *) "fixcldsr: ", fixcldsr
    write(*, *) "ratio_ent_rad: ", ratio_ent_rad
    write(*, *) "orgent_a: ", orgent_a
    write(*, *) "orgent_beta0: ", orgent_beta0
    write(*, *) "w_up_init_sh_beg: ", w_up_init_sh_beg
    write(*, *) "w_up_init_sh_end: ", w_up_init_sh_end
    write(*, *) "w_up_init_dp_beg: ", w_up_init_dp_beg
    write(*, *) "w_up_init_dp_end: ", w_up_init_dp_end
    write(*, *) "w_init_shape_sh: ", w_init_shape_sh
    write(*, *) "w_init_shape_dp: ", w_init_shape_dp
    write(*, *) "rain_z0: ", rain_z0
    write(*, *) "rain_zp: ", rain_zp
    write(*, *) "dn_be: ", dn_be
    write(*, *) "dn_ae: ", dn_ae 
    write(*, *) "dn_vt: ", dn_vt
    write(*, *) "dn_frac_sh: ", dn_frac_sh
    write(*, *) "dn_frac_dp: ", dn_frac_dp
    write(*, *) "pmf_alpha_sh: ", pmf_alpha_sh
    write(*, *) "pmf_tau_sh: ", pmf_tau_sh
    write(*, *) "pmf_alpha_dp: ", pmf_alpha_dp
    write(*, *) "capelmt_sh: ", capelmt_sh
    write(*, *) "capelmt_dp: ", capelmt_dp
    write(*, *) "tpertglob: ", tpertglob
    write(*, *) "qpertglob: ", qpertglob
    write(*, *) "facdlf: ", facdlf
    write(*, *) "evap_enhance: ", evap_enhance
    write(*, *) "evap_shape: ", evap_shape

!MZ
  endif !masterproc


!MZ ========= ZM scheme 

   ! Initialization of ZM constants


   tfreez = tmelt
   eps1   = epsilo
   rl     = latvap
   cpres  = cpair
   rgrav  = 1.0_r8/gravit
   rgas   = rair
   grav   = gravit
   cp     = cpres

   tau = 3600._r8

!   no_deep_pbl = phys_deepconv_pbl()
!   if(present(no_deep_pbl ) )then
!     no_deep_pbl = phys_deepconv_pbl()
!   else
!     no_deep_pbl = .false.
!   endif

   if ( masterproc ) then
      write(iulog,*) 'tuning parameters zm_convi: tau',tau
      write(iulog,*) 'tuning parameters zm_convi: c0_lnd',c0_lnd, ', c0_ocn', c0_ocn
      write(iulog,*) 'tuning parameters zm_convi: ke',ke
      write(iulog,*) 'tuning parameters zm_convi: no_deep_pbl',no_deep_pbl
   endif
#ifdef SCMDIAG
    write(*,*) "[zyx2_conv_init]"
    write(*,*) "Parameter"
    write(*,"(a20f20.10)") "gravit", gravit
    write(*,"(a20f20.10)") "cpair", cpair
    write(*,"(a20f20.10)") "cpliq", cpliq
    write(*,"(a20f20.10)") "cpwv", cpwv
    write(*,"(a20f20.10)") "latvap", latvap
    write(*,"(a20f20.10)") "epsilo", epsilo
    write(*,"(a20f20.10)") "tmelt", tmelt
    write(*,"(a20f20.10)") "rair", rair
    write(*,"(a20f20.10)") "rh2o", rh2o
    write(*,"(a20f20.10)") "rhofw", rhofw
#endif


end subroutine ecp_readnl

! ==============================================================================
! initialize the convection scheme
! ==============================================================================
subroutine zyx2_conv_init(innlev, limcnv_in)

!MZ<
integer limcnv_in
!>    

!input
    integer, intent(in) :: innlev
#ifndef OFFLINECP
    nlev  = innlev
    nlevp = innlev+1
#endif

!MZ<
    limcnv = limcnv_in
!>

#ifdef SCMDIAG
    write(*,*) "[zyx2_conv_init]"
    write(*,*) "Parameter"
    write(*,"(a20f20.10)") "gravit", gravit
    write(*,"(a20f20.10)") "cpair", cpair
    write(*,"(a20f20.10)") "cpliq", cpliq
    write(*,"(a20f20.10)") "cpwv", cpwv
    write(*,"(a20f20.10)") "latvap", latvap
    write(*,"(a20f20.10)") "epsilo", epsilo
    write(*,"(a20f20.10)") "tmelt", tmelt
    write(*,"(a20f20.10)") "rair", rair
    write(*,"(a20f20.10)") "rh2o", rh2o
    write(*,"(a20f20.10)") "rhofw", rhofw
#endif
    
    call ecp_readnl('atm_in')

end subroutine zyx2_conv_init


! ==============================================================================
! master route of convection scheme
! ==============================================================================
!subroutine zyx2_conv_tend( &
subroutine zyx2_conv_tend(lchnk, nstep, &
!input
        inncol, &
#ifdef OFFLINECP
!!MZ        nlev, nlevp, &
#endif
        in_ent_opt, dtime, qmin, &
        lat, landfrac, lhflx,shflx, &
        psrf, p, dp, zsrf, z, dz, &
        t_in, q_in, bfls_t, bfls_q, &
        omega, pblh, tpert, nn_prec, nn_stend, nn_qtend, &
!in/output
        massflxbase_p, &
!output
        jctop, jcbot, &
        stend, qtend, &
        qliqtend,  prec, qliq,  &
        precrate_out, &
!        qliqtend, qicetend, prec, qliq, qice, &
!        rainrate_out, snowrate_out, precrate_out, &
        mcon, &
        stendcomp, qtendcomp, &
!diagnostics
        dilucape, bfls_dilucape, &
        outtmp2d, outtmp3d, &
        outmb, outmse, outmsesat, outmseup, &
        outstend, outqtend, &
        outstendcond, outqtendcond, &
        outstendtranup, outqtendtranup, &
        outstendtrandn, outqtendtrandn, &
        outstendevap, outqtendevap &
!!#ifdef OFFLINECP
!!        , all_ent_org, all_det_org, all_ent_turb, all_det_turb, all_massflx, &
!!        all_mse_up, all_t_up, all_q_up, all_qliq_up, all_qice_up, all_w_up, all_buoy, all_mseqi, &
!!        all_condrate, all_rainrate, all_snowrate, all_precrate, all_accuprec, all_evaprate, &
!!        all_stend, all_qtend, all_qliqtend,  &
!!        all_stendcond, all_stendevap, all_stendtran_up, all_stendtran_dn, &
!!        all_qtendcond, all_qtendevap, all_qtendtran_up, all_qtendtran_dn, &
!!        all_cape, all_massflxbase, all_prec, all_surfprec &
!!#endif
!MZ additional output for optional CAM plume 
         ,geos, cme,cape,ideep,&
         dlf     ,pflx    ,zdu     ,rprd    , &
         mu      ,md      ,du      ,eu      ,ed      , &
         dsubcld ,jt,maxg, lengath, &
!MZ 2018-08-06 additionl input for trigger and closure)
         tdiff3,qdiff3,vort3,sgh30,tke)

!------------------------------------------------------
!Calculate convective tendency
!------------------------------------------------------
! Haiyang Yu
#if (! defined OFFLINECP)
    use nnparameter, only: cal_weight, cal_weight_eigen, nn_flag
#endif
! Main Interface
    integer , intent(in) :: inncol ! size of column dimension
#ifdef OFFLINECP
    integer, intent(in) :: nlev, nlevp
#endif
    integer , intent(in) :: in_ent_opt ! 0=ec, 1=greg
    real(r8), intent(in) :: dtime  ! [s] time step
    real(r8), intent(in) :: qmin   ! [kg/kg] minimum Q
    real(r8), dimension(inncol), intent(in) :: lat, landfrac, lhflx, shflx
    real(r8), dimension(inncol), intent(in) :: psrf, zsrf
    real(r8), dimension(inncol, nlev), intent(in) :: p, dp, z, dz ! [Pa] ; [m]
    real(r8), dimension(inncol, nlev), intent(in) :: t_in, q_in ! [K] ; [kg/kg]
       ! T and Q state after the large-scale forcing is applied, current state
    real(r8), dimension(inncol, nlev), intent(inout) :: bfls_t, bfls_q ! [K] ; [kg/kg]
       ! T and Q state before the large-scale forcing is applied
    real(r8), dimension(inncol, nlev), intent(in) :: omega ! [Pa/s]
    real(r8), dimension(inncol), intent(in) :: pblh  ! [m/s]
    real(r8), dimension(inncol), intent(in) :: tpert  ! [K]
    real(r8), dimension(inncol), intent(in) :: nn_prec  ! [m/s] NNprec
    real(r8), dimension(inncol, nlev), intent(in) :: nn_stend  ! [J/kg/s] NN
    real(r8), dimension(inncol, nlev), intent(in) :: nn_qtend  ! [kg/kg/s] NN

!MZ minghua added additional input for trigger and closure
    real(r8), dimension(inncol, nlev), intent(in) :: tdiff3,qdiff3,vort3
    real(r8), dimension(inncol, nlev), intent(in) ::  tke
    real(r8), dimension(inncol), intent(in) :: sgh30


!in/output
    real(r8), dimension(inncol, nlev), intent(inout) :: massflxbase_p !output convective precip[m/s]
!output
    real(r8), dimension(inncol), intent(out) :: jctop
    real(r8), dimension(inncol), intent(out) :: jcbot

    real(r8), dimension(inncol), intent(out) :: prec !output convective precip[m/s]
    real(r8), dimension(inncol, nlev), intent(out) :: stend, qtend, qliqtend
    ! [K/s] ; [kg/kg/s] output tendencies calculated by adding (condensate rate+transport)
    real(r8), dimension(inncol, nlev), intent(out) :: qliq! kg/kg
    real(r8), dimension(inncol, nlev), intent(out) :: precrate_out ! 1/s

    real(r8), dimension(inncol, nlevp), intent(out) :: mcon

!    real(r8), dimension(inncol, nlev), intent(out) :: stend, qtend, qliqtend, qicetend
!    ! [K/s] ; [kg/kg/s] output tendencies calculated by adding (condensate rate+transport)
!    real(r8), dimension(inncol, nlev), intent(out) :: qliq, qice ! kg/kg
!    real(r8), dimension(inncol, nlev), intent(out) :: rainrate_out, snowrate_out, precrate_out ! 1/s

    real(r8), dimension(inncol, nlev), intent(out) :: stendcomp, qtendcomp
    ! [K/s] ; [kg/kg/s] tendencies but calculated using the compensating way
    ! should be same as stend and qtend

!diagnostic output
    real(r8), dimension(inncol), intent(out) :: outmb, outtmp2d
    real(r8), dimension(inncol, nlev), intent(out) :: outtmp3d
    real(r8), dimension(inncol, nlev), intent(out) :: outmse, outmsesat, outmseup

    real(r8), dimension(inncol, nlev), intent(out) :: outstend, outqtend
    real(r8), dimension(inncol, nlev), intent(out) :: outstendcond, outqtendcond
    real(r8), dimension(inncol, nlev), intent(out) :: outstendtranup, outqtendtranup
    real(r8), dimension(inncol, nlev), intent(out) :: outstendtrandn, outqtendtrandn
    real(r8), dimension(inncol, nlev), intent(out) :: outstendevap, outqtendevap


!local
    integer i, j, k, begk, endk, iconv
    real(r8), dimension(inncol, nlev) :: t, q
    real(r8), dimension(inncol, nlev) :: dse !environment [J/kg]
    real(r8), dimension(inncol, nlev) :: mse, msesat ! [J/kg] ; [J/kg]
    real(r8), dimension(inncol, nlev) :: twet !environment wet bulb temperature [K]
    real(r8), dimension(inncol, nlevp) :: twetint !environment wet bulb temperature [K]
 
    real(r8), dimension(inncol, nlev) :: esat, qsat ! [Pa] ; [kg/kg]
    real(r8), dimension(inncol, nlev) :: rh  ! [1] relative humidity
    real(r8), dimension(inncol, nlev) :: rho ! [kg/m3]

    integer, dimension(inncol) :: trigdp ! [1] 1 trigger ; 0 no trigger
    integer, dimension(inncol) :: trigsh ! [1] 1 trigger ; 0 no trigger

    integer, dimension(inncol) :: kuplaunch ! [1] cloud launching level
    integer, dimension(inncol) :: kuplcl    ! [1] LCL
    integer, dimension(inncol) :: kupbase   ! [1] cloud base

    integer, dimension(inncol) :: kuptop ! [1] cloud top where buoyancy is zero
    real(r8),dimension(inncol) :: zuptop ! [m] exact z at kuptop


    real(r8), dimension(inncol, nlev)  :: lvmid !
    real(r8), dimension(inncol, nlevp) :: lvint !

!for bulk TOTAL fractional en/detrainment rate depending on vertical velocity
    real(r8), dimension(inncol, nlevp) :: zint ! [1] height at the interface
    real(r8), dimension(inncol, nlevp) :: pint ! [1] height at the interface
    real(r8), dimension(inncol, nlevp) :: tint ! [1] height at the interface
    real(r8), dimension(inncol, nlevp) :: qint ! [1] height at the interface
    real(r8), dimension(inncol, nlevp) :: rhoint ! [1] density at the interface

    real(r8), dimension(inncol, nlevp) :: mseint ! [1] height at the interface
    real(r8), dimension(inncol, nlevp) :: dseint ! [1] height at the interface
    real(r8), dimension(inncol, nlevp) :: esatint, qsatint ! [Pa] ; [kg/kg]
    real(r8), dimension(inncol, nlevp) :: msesatint ! [1] height at the interface

    real(r8), dimension(inncol, nlevp) :: normassflx_up ! [1]  bulk normalized updraft mass flux
    real(r8), dimension(inncol, nlevp) :: normassflx_up_tmp ! [1]  bulk normalized updraft mass flux
    real(r8), dimension(inncol, nlev)  :: ent_rate_dp_up ! [1] solved PARCEL fractional entrainment rates
    real(r8), dimension(inncol, nlev)  :: det_rate_dp_up ! [1] solved PARCEL fractional entrainment rates
    real(r8), dimension(inncol, nlev)  :: ent_rate_sh_up ! [1] solved PARCEL fractional entrainment rates
    real(r8), dimension(inncol, nlev)  :: det_rate_sh_up ! [1] solved PARCEL fractional entrainment rates
    ! Haiyang: diagnostic purpose
    real(r8), dimension(inncol, nlev)  :: ent_org 
    real(r8), dimension(inncol, nlev)  :: det_org 
    real(r8), dimension(inncol, nlev)  :: ent_turb 
    real(r8), dimension(inncol, nlev)  :: det_turb
    real(r8), dimension(inncol, nlev)  :: cldrad
    real(r8), dimension(inncol, nlev)  :: bs_xc

    real(r8), dimension(inncol, nlev) :: ent_fc ! [1] an entrainment rate modifier

    real(r8), dimension(inncol, nlevp) :: mse_up ! [J/kg]  bulk in-cloud MSE given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: dse_up ! [J/kg]  bulk in-cloud DSE given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: buoy ! [J/kg]  bulk in-cloud MSE given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: t_up   ! [K]  bulk in-cloud temperatur given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: tv_up  ! [K]  bulk in-cloud temperatur given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: q_up   ! [kg/kg]  bulk in-cloud sat water vapor given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: qliq_up   ! [kg/kg]  bulk in-cloud liquid water given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: qice_up   ! [kg/kg]  bulk in-cloud liquid water given en/detrainment rate.

    real(r8), dimension(inncol, nlev) :: mse_up_mid ! [J/kg]  bulk in-cloud MSE given en/detrainment rate.
    real(r8), dimension(inncol, nlev) :: t_up_mid ! [J/kg]  bulk in-cloud MSE given en/detrainment rate.
    real(r8), dimension(inncol, nlev) :: q_up_mid ! [J/kg]  bulk in-cloud MSE given en/detrainment rate.
    real(r8), dimension(inncol, nlev) :: buoy_mid ! [m/s-2] bulk in-cloud buoyancy
    real(r8), dimension(inncol, nlev) :: normassflx_up_mid ! [1]  bulk normalized updraft mass flux

    real(r8), dimension(inncol, nlev) :: mseqi      ! freezing term in mse equation
    real(r8), dimension(inncol, nlev) :: condrate   ! [m2/kg]  condensation rate.
    real(r8), dimension(inncol, nlev) :: rainrate   ! [1]  bulk precipitation production
    real(r8), dimension(inncol, nlev) :: snowrate   ! [1]  bulk precipitation production
    real(r8), dimension(inncol, nlev) :: precrate   ! [1]  bulk precipitation production

    real(r8), dimension(inncol, nlev)  :: normassflx_dn ! [1]  bulk normalized updraft mass flux
    real(r8), dimension(inncol, nlevp) :: normassflx_dn_tmp ! [1]  bulk normalized updraft mass flux

    real(r8), dimension(inncol, nlevp) :: mse_dn ! [J/kg]  bulk in-cloud MSE given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: t_dn   ! [K]  bulk in-cloud temperatur given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: q_dn   ! [kg/kg]  bulk in-cloud sat water vapor given en/detrainment rate.
    real(r8), dimension(inncol, nlevp) :: dse_dn ! [J/kg]  bulk in-cloud DSE given en/detrainment rate.
    
    real(r8) :: dn_frac   ! downdraft fraction

!for evaporation
    real(r8), dimension(inncol, nlev) :: accuprec  ! [1]  bulk precipitation production
    real(r8), dimension(inncol, nlev) :: evaprate  ! [1]  bulk evaporation production

    real(r8), dimension(inncol) :: surfprec      ! [1]  bulk evaporation production
    real(r8), dimension(inncol) :: surfprecsum      ! [1]  bulk evaporation production
    real(r8), dimension(inncol) :: netprec       ! [1]  bulk evaporation production

!calculated w from Grell's en/detrainment rate scheme
    real(r8), dimension(inncol) :: w_up_init  ! [J/kg]  bulk in-cloud vertical velocity
    real(r8), dimension(inncol, nlev)  :: w_up_mid ! [J/kg]  bulk in-cloud vertical velocity
    real(r8), dimension(inncol, nlevp) :: w_up ! [J/kg]  bulk in-cloud vertical velocity


! These varaiables are used to calcualte the closure parameter F
! (cloud work function change after applying tendencies with unit base mass flux)
    integer, dimension(inncol) :: trig_closure
    real(r8),dimension(inncol, nlev) :: t_closure  ! [K] adjusted temperature for closure use
                                                   ! by unit base mass flux
    real(r8),dimension(inncol, nlev) :: q_closure  ! [kg/kg] adjusted moisture for closure use
                                                   ! by unit base mass flux
    real(r8),dimension(inncol, nlev) :: rh_closure   ! [K] adjusted temperature for closure use
    real(r8),dimension(inncol, nlev) :: qsat_closure ! [K] adjusted temperature for closure use
    real(r8),dimension(inncol, nlev) :: mse_closure
    real(r8),dimension(inncol, nlev) :: msesat_closure

    integer, dimension(inncol) :: kuplaunch_closure ! [1] cloud launching level
    integer, dimension(inncol) :: kuplcl_closure    ! [1] cloud base
    integer, dimension(inncol) :: kupbase_closure   ! [1] cloud base
    integer, dimension(inncol) :: kuptop_closure ! [1] cloud top where buoyancy is zero
    real(r8),dimension(inncol) :: zuptop_closure ! [m] exact kuptop

    real(r8),dimension(inncol, nlev) :: mse_up_closure
    real(r8),dimension(inncol, nlev) :: t_up_closure
    real(r8),dimension(inncol, nlev) :: q_up_closure
    real(r8),dimension(inncol, nlev) :: tv_up_closure
    real(r8),dimension(inncol, nlev) :: normassflx_up_closure

    real(r8),dimension(inncol, nlev) :: ent_rate_dp_up_closure
    real(r8),dimension(inncol, nlev) :: det_rate_dp_up_closure

    real(r8),dimension(inncol, nlev) :: tv_closure ! [K]
    real(r8),dimension(inncol, nlev) :: buoy_closure  ! [kg/kg] adjusted buoyancy for closure use
    real(r8),dimension(inncol, nlev) :: w_up_closure ! [J/kg]  bulk in-cloud vertical velocity


! These variables are for closure calculation, OR base mass flux
    real(r8),dimension(inncol, nlev) :: w         ! [m/s] environment vertical velocity
    real(r8),dimension(inncol) :: mconv           ! [1] moisture convergence
    real(r8),dimension(inncol) :: conv            ! [1] wind convergence
    real(r8),dimension(inncol, nlev), intent(out) :: dilucape  ! [1] CAPE cloud work function
    real(r8),dimension(inncol, nlev) :: cwf       ! [1] CAPE cloud work function
    real(r8),dimension(inncol), intent(out) :: bfls_dilucape   ! [1] CAPE cloud work function before LS forcing
    real(r8),dimension(inncol) :: dilucape_closure

    real(r8),dimension(inncol) :: capeclm
    integer,dimension(inncol)  :: kclm

    real(r8),dimension(inncol) :: massflxbase       ! [1]
    real(r8),dimension(inncol) :: massflxbase_cape  ! [1]
    real(r8),dimension(inncol) :: massflxbase_clm   ! [1]
    real(r8),dimension(inncol) :: massflxbase_clmcape  ! [1]
    real(r8),dimension(inncol) :: massflxbase_w     ! [1]
    real(r8),dimension(inncol) :: massflxbase_mconv ! [1]
    real(r8),dimension(inncol) :: massflxbase_conv  ! [1]
    real(r8),dimension(inncol) :: massflxbase_dcape ! [1]
    real(r8),dimension(inncol) :: capefc
    real(r8),dimension(inncol) :: convfrc_up ! [1] conv fraction, for scale-aware, not used yet
    real(r8),dimension(inncol) :: lat_coef

! These variables are used to prevent negative water vapor
    real(r8), dimension(inncol, nlev) :: qcheck ! [1]  used to check negative q
    real(r8), dimension(inncol) :: qcheckout    ! [1]  used to check negative q
    real(r8) :: qguess, qcheckf, minqcheckf             ! [1]

    real(r8), dimension(inncol) :: tmp2d    ! [1]  used to check negative q


!for diag
    real(r8), dimension(inncol, nlevp) :: diffdse_up ! [1]  Delta DSE
    real(r8), dimension(inncol, nlevp) :: diffq_up   ! [1]  Delta Q
    real(r8), dimension(inncol, nlev) :: stendcond  ! [K/s] DSE tendency
    real(r8), dimension(inncol, nlev) :: qtendcond  ! [K/s] Q tendency
    real(r8), dimension(inncol, nlev) :: stendtran_up  ! [K/s] DSE tendency
    real(r8), dimension(inncol, nlev) :: qtendtran_up  ! [K/s] Q tendency
    real(r8), dimension(inncol, nlev) :: stendtran_dn  ! [K/s] DSE tendency
    real(r8), dimension(inncol, nlev) :: qtendtran_dn  ! [K/s] Q tendency
    real(r8), dimension(inncol, nlev) :: qliqtend_det  ! [K/s] liq tendency due to detrainment
    real(r8), dimension(inncol, nlev) :: stendevap  ! [K/s] DSE tendency
    real(r8), dimension(inncol, nlev) :: qtendevap  ! [K/s] Q tendency

    real(r8), dimension(inncol, nlev) :: stendsum
    real(r8), dimension(inncol, nlev) :: qtendsum
    real(r8), dimension(inncol, nlev) :: qliqtendsum
    real(r8), dimension(inncol, nlev) :: precratesum
    real(r8), dimension(inncol) :: precsum
    real(r8), dimension(inncol) :: massflxbasesum
    real(r8), dimension(inncol, nlevp) :: massflxsum
    real(r8), dimension(inncol, nlevp) :: massflx

    real(r8), dimension(inncol, nlev) :: tmp1stend, tmp1qtend
    real(r8), dimension(inncol, nlev) :: tmp2stend, tmp2qtend

    real(r8), dimension(inncol) :: bg_q
    real(r8), dimension(inncol) :: bg_qtend
    real(r8), dimension(inncol) :: bg_qtendcond
    real(r8), dimension(inncol) :: bg_precrate
    real(r8), dimension(inncol) :: bg_netprecrate
    real(r8), dimension(inncol) :: bg_net
    real(r8), dimension(inncol) :: bg_qtendup
    real(r8), dimension(inncol) :: bg_qtenddn
    real(r8), dimension(inncol) :: bg_qtendevap
    real(r8), dimension(inncol) :: bg_stendcond
    real(r8), dimension(inncol) :: bg_stendup
    real(r8), dimension(inncol) :: bg_stenddn
    real(r8), dimension(inncol) :: bg_stendevap
    real(r8), dimension(inncol) :: bg_qliqtenddet
    real(r8), dimension(inncol) :: bg_qtendsum
    real(r8), dimension(inncol) :: bg_factor

! Haiyang Yu: all plumes
#ifdef OFFLINECP
    real(r8), dimension(inncol, nlevp, 30), intent(out) :: &
        all_mse_up, all_t_up, all_q_up, all_qliq_up, all_qice_up, all_w_up, all_buoy
    real(r8), dimension(inncol, nlev, 30), intent(out) :: &
        all_ent_org, all_det_org, all_ent_turb, all_det_turb, all_massflx, all_mseqi, &
        all_condrate, all_rainrate, all_snowrate, all_precrate, all_accuprec, all_evaprate, &
        all_stend, all_qtend, all_qliqtend,  &
        all_stendcond, all_stendevap, all_stendtran_up, all_stendtran_dn, &
        all_qtendcond, all_qtendevap, all_qtendtran_up, all_qtendtran_dn
    real(r8), dimension(inncol, 30), intent(out) :: &
        all_cape, all_massflxbase, all_prec, all_surfprec
#else
    real(r8), dimension(inncol, nlev, 30) :: &
        all_massflx, all_precrate, all_stend, all_qtend, all_qliqtend
    real(r8), dimension(inncol, 30) :: &
        all_massflxbase, all_prec, all_surfprec
#endif
    ! internal variables
    real(r8), dimension(inncol) :: tpert_plume, qpert_plume
    real(r8), dimension(inncol, 30) :: weights
    real(r8), dimension(inncol) :: totalweight
    integer, dimension(inncol) :: validplume
    integer, dimension(inncol, 30) :: valid

#ifdef OFFLINECP
    integer :: ncol
#endif

!for test
    real(r8), dimension(inncol) :: tmp ! [1] number of convective lev
    real(r8) :: diffz, dw_up_init, basemass_scale
    logical :: flag

!
!MZ to include optional CAM plume calculations, from zm_conv_tend
!==========================
   integer :: lchnk                   ! chunk identifier
   integer :: nstep                   ! chunk identifier
   integer, intent(out) :: ideep(inncol)
   integer :: lengath

   !real(r8) :: t(inncol,nlev)          ! grid slice of temperature at mid-layer.
   real(r8) :: qh(inncol,nlev)   ! grid slice of specific humidity.
   real(r8) :: pap(inncol,nlev)
   real(r8) :: paph(inncol,nlev+1)
   real(r8) :: dpp(inncol,nlev)        ! local sigma half-level thickness (i.e. dshj).
   real(r8) :: zm(inncol,nlev)
   real(r8), intent(in) :: geos(inncol)
   real(r8) :: zi(inncol,nlev+1)
   !real(r8), intent(in) :: pblh(inncol)
   !real(r8), intent(in) :: tpert(inncol)
   !real(r8), intent(in) :: landfrac(inncol) ! RBN Landfrac
!
! output arguments
!
   real(r8) :: qtnd(inncol,nlev)           ! specific humidity tendency (kg/kg/s)
   real(r8) :: heat(inncol,nlev)           ! heating rate (dry static energy tendency, W/kg)
   !real(r8), intent(out) :: mcon(inncol,nlevp)
   real(r8), intent(out) :: dlf(inncol,nlev)    ! scattrd version of the detraining cld h2o tend
   real(r8), intent(out) :: pflx(inncol,nlevp)  ! scattered precip flux at each level
   real(r8), intent(out) :: cme(inncol,nlev)
   real(r8), intent(out) :: cape(inncol)        ! w  convective available potential energy.
   real(r8), intent(out) :: zdu(inncol,nlev)
   real(r8), intent(out) :: rprd(inncol,nlev)     ! rain production rate
! move these vars from local storage to output so that convective
! transports can be done in outside of conv_cam.
   real(r8), intent(out) :: mu(inncol,nlev)
   real(r8), intent(out) :: eu(inncol,nlev)
   real(r8), intent(out) :: du(inncol,nlev)
   real(r8), intent(out) :: md(inncol,nlev)
   real(r8), intent(out) :: ed(inncol,nlev)
   real(r8) :: dd(inncol,nlev)  !! calculated, but not sent out
   !real(r8), intent(out) :: dp(inncol,nlev)       ! wg layer thickness in mbs (between upper/lower interface).
   real(r8), intent(out) :: dsubcld(inncol)       ! wg layer thickness in mbs between lcl and maxi.
   !real(r8), intent(out) :: jctop(inncol)  ! o row of top-of-deep-convection indices passed out.
   !real(r8), intent(out) :: jcbot(inncol)  ! o row of base of cloud indices passed out.
   !real(r8), intent(out) :: prec(inncol)
   real(r8) :: rliq(inncol) ! reserved liquid (not yet in cldliq) for energy integrals
   
   !work arrayes
    real(r8), dimension(inncol, nlevp) :: massflxsum_dn
    real(r8), dimension(inncol, nlevp) :: massflx_dn

    real(r8), dimension(inncol, nlev)  :: ent_up !work array for entrainment E = episilon*Mass
    real(r8), dimension(inncol, nlev)  :: det_up !for single plme
    real(r8), dimension(inncol, nlev)  :: ent_dn
    real(r8), dimension(inncol, nlev)  :: det_dn

    real(r8), dimension(inncol, nlev)  :: ent_up_sum  !this is the total plume rate
    real(r8), dimension(inncol, nlev)  :: det_up_sum
    real(r8), dimension(inncol, nlev)  :: ent_dn_sum
    real(r8), dimension(inncol, nlev)  :: det_dn_sum

    real(r8), dimension(inncol, nlev)  :: accuprecsum
    real(r8), dimension(inncol, nlev)  :: cmesum
    real(r8), dimension(inncol, nlev)  :: qlsum
    real(r8), dimension(inncol) :: capemax
    real(r8), dimension(inncol) :: dsubcldmax
    integer, dimension(inncol) :: maxgmax
    real, dimension(inncol) :: jctopmin, jcbotmax

!xin added to CAM
! general work fields (local variables):
   real(r8) :: condtends(inncol,nlev)
   real(r8) :: condtendq(inncol,nlev)
   real(r8) :: tranuptends(inncol,nlev)
   real(r8) :: tranuptendq(inncol,nlev)
   real(r8) :: trandntends(inncol,nlev)
   real(r8) :: trandntendq(inncol,nlev)
   real(r8) :: outtends(inncol,nlev)
   real(r8) :: outtendq(inncol,nlev)
   real(r8) :: outcondtends(inncol,nlev)
   real(r8) :: outcondtendq(inncol,nlev)
   real(r8) :: outtranuptends(inncol,nlev)
   real(r8) :: outtranuptendq(inncol,nlev)
   real(r8) :: outtrandntends(inncol,nlev)
   real(r8) :: outtrandntendq(inncol,nlev)
!   real(r8) :: outmb(inncol)
   real(r8) :: outmu(inncol,nlev)
   real(r8) :: outsu(inncol,nlev)
   real(r8) :: outqu(inncol,nlev)

   real(r8) zs(inncol)
   real(r8) dlg(inncol,nlev)    ! gathrd version of the detraining cld h2o tend
   real(r8) pflxg(inncol,nlevp) ! gather precip flux at each level
   real(r8) cug(inncol,nlev)    ! gathered condensation rate
   real(r8) evpg(inncol,nlev)   ! gathered evap rate of rain in downdraft
   real(r8) mumax(inncol)
   integer jt(inncol)                          ! wg top  level index of deep cumulus convection.
   integer maxg(inncol)                        ! wg gathered values of maxi.
!     diagnostic field used by chem/wetdep codes
   real(r8) ql(inncol,nlev)                    ! wg grid slice of cloud liquid water.
!
   real(r8) pblt(inncol)           ! i row of pbl top indices.

   real(r8) wq(inncol,nlev)              ! MZ name change w  grid slice of mixing ratio.
   real(r8) wdp(inncol,nlev)             ! MZ name change  
   real(r8) wp(inncol,nlev)              ! w  grid slice of ambient mid-layer pressure in mbs.
   real(r8) wz(inncol,nlev)              ! w  grid slice of ambient mid-layer height in metres.
   real(r8) s(inncol,nlev)              ! w  grid slice of scaled dry static energy (t+gz/cp).
   real(r8) tp(inncol,nlev)             ! w  grid slice of parcel temperatures.
   real(r8) zf(inncol,nlev+1)           ! w  grid slice of ambient interface height in metres.
   real(r8) pf(inncol,nlev+1)           ! w  grid slice of ambient interface pressure in mbs.
   real(r8) qstp(inncol,nlev)           ! w  grid slice of parcel temp. saturation mixing ratio.

   real(r8) tl(inncol)                  ! w  row of parcel temperature at lcl.

   integer lcl(inncol)                  ! w  base level index of deep cumulus convection.
   integer lel(inncol)                  ! w  index of highest theoretical convective plume.
   integer lon(inncol)                  ! w  index of onset level for deep convection.
   integer maxi(inncol)                 ! w  index of level with largest moist static energy.
   integer index(inncol)
   real(r8) precip
!
! gathered work fields:
   real(r8) qg(inncol,nlev)             ! wg grid slice of gathered values of q.
   real(r8) tg(inncol,nlev)             ! w  grid slice of temperature at interface.
   real(r8) pg(inncol,nlev)             ! wg grid slice of gathered values of p.
   real(r8) zg(inncol,nlev)             ! wg grid slice of gathered values of z.
   real(r8) sg(inncol,nlev)             ! wg grid slice of gathered values of s.
   real(r8) tpg(inncol,nlev)            ! wg grid slice of gathered values of tp.
   real(r8) zfg(inncol,nlev+1)          ! wg grid slice of gathered values of zf.
   real(r8) qstpg(inncol,nlev)          ! wg grid slice of gathered values of qstp.
   real(r8) ug(inncol,nlev)             ! wg grid slice of gathered values of u.
   real(r8) vg(inncol,nlev)             ! wg grid slice of gathered values of v.
   real(r8) cmeg(inncol,nlev)

   real(r8) rprdg(inncol,nlev)           ! wg gathered rain production rate
   real(r8) capeg(inncol)               ! wg gathered convective available potential energy.
   real(r8) tlg(inncol)                 ! wg grid slice of gathered values of tl.
   real(r8) landfracg(inncol)            ! wg grid slice of landfrac

   integer lclg(inncol)       ! wg gathered values of lcl.
   integer lelg(inncol)
!
! work fields arising from gathered calculations.
!
   real(r8) dqdt(inncol,nlev)           ! wg mixing ratio tendency at gathered points.
   real(r8) dsdt(inncol,nlev)           ! wg dry static energy ("temp") tendency at gathered points.
!      real(r8) alpha(inncol,nlev)      ! array of vertical differencing used (=1. for upstream).
   real(r8) sd(inncol,nlev)             ! wg grid slice of dry static energy in downdraft.
   real(r8) qd(inncol,nlev)             ! wg grid slice of mixing ratio in downdraft.
   real(r8) mc(inncol,nlev)             ! wg net upward (scaled by mb) cloud mass flux.
   real(r8) qhat(inncol,nlev)           ! wg grid slice of upper interface mixing ratio.
   real(r8) qu(inncol,nlev)             ! wg grid slice of mixing ratio in updraft.
   real(r8) su(inncol,nlev)             ! wg grid slice of dry static energy in updraft.
   real(r8) qs(inncol,nlev)             ! wg grid slice of saturation mixing ratio.
   real(r8) shat(inncol,nlev)           ! wg grid slice of upper interface dry static energy.
   real(r8) hmn(inncol,nlev)            ! wg moist static energy.
   real(r8) hsat(inncol,nlev)           ! wg saturated moist static energy.
   real(r8) qlg(inncol,nlev)
   real(r8) dudt(inncol,nlev)           ! wg u-wind tendency at gathered points.
   real(r8) dvdt(inncol,nlev)           ! wg v-wind tendency at gathered points.
!      real(r8) ud(inncol,nlev)
!      real(r8) vd(inncol,nlev)

   real(r8) mb(inncol)                  ! wg cloud base mass flux.

   integer jlcl(inncol)
   integer j0(inncol)                 ! wg detrainment initiation level index.
   integer jd(inncol)                 ! wg downdraft initiation level index.

   real(r8) delt                     ! length of model time-step in seconds.

   !integer i
   integer ii
   !integer k
   integer msg                      !  ic number of missing moisture levels at the top of model.
   real(r8) qdifr
   real(r8) sdifr

!MZ CAM declaration done

!MZ 2018-08-04 to add new variables for trigger and closure design   
!              bfls_ stands for "because of large-scale forcing"
!   -------------------------------------------------------------

    real(r8), dimension(inncol, nlev) :: bfls_buoy_mid
    real(r8), dimension(inncol, nlev) :: bfls_qsat, bfls_mse, bfls_msesat
    real(r8), dimension(inncol, nlevp) :: bfls_tint,   bfls_qint,   bfls_qsatint
    real(r8), dimension(inncol, nlevp) :: bfls_mseint, bfls_msesatint 
    real(r8), dimension(inncol) ::        cin
    real(r8), dimension(inncol) ::        bfls_cwf, bfls_cin

!   -------------------------------------------------------------
 

!MZ
      !call phys_getopts(plume_model_out = plume_model)
      call phys_getopts(deep_scheme_out = deep_scheme)

!!!!!!!!!!!!!!!!!MZ
!!    deep_scheme = 'zyx2'
!!    plume_model = 'cam'


! weights of plumes
    weights = 0.0
    totalweight = 0.0
    validplume(:) = 0
    valid = 0

!setting the internal dimension size same as the input
    ncol = inncol

    ent_opt = in_ent_opt

!intialize output
    prec = 0._r8
    stend = 0._r8
    qtend = 0._r8
    qliqtend = 0._r8
    qliq  = 0._r8
    precrate_out = 0._r8

    stendcomp  = 0._r8
    qtendcomp  = 0._r8

    outmb = 0._r8
    outtmp2d = 0._r8
    outtmp3d = 0._r8
    outmse = 0._r8
    outmsesat = 0._r8
    outmseup = 0._r8
    outstend = 0._r8
    outqtend = 0._r8
    outstendcond = 0._r8
    outqtendcond = 0._r8
    outstendtranup = 0._r8
    outqtendtranup = 0._r8
    outstendtrandn = 0._r8
    outqtendtrandn = 0._r8
    outstendevap = 0._r8
    outqtendevap = 0._r8


!zero local variables

    mse = 0._r8
    dse = 0._r8

    qliq_up = 0._r8
    qice_up = 0._r8
    condrate = 0._r8
    rainrate = 0._r8
    snowrate = 0._r8
    precrate = 0._r8
    accuprec = 0._r8
    evaprate = 0._r8

    surfprec = 0._r8
    surfprecsum = 0._r8

    w = 0._r8
    tv_closure = 0._r8
    q_closure = 0._r8
    buoy_closure = 0._r8

    dilucape = 0._r8
    cwf = 0._r8
    bfls_dilucape = 0._r8
    dilucape_closure = 0._r8

    stendcond = 0._r8
    qtendcond = 0._r8
    stendtran_up = 0._r8
    qtendtran_up = 0._r8
    stendtran_dn = 0._r8
    qtendtran_dn = 0._r8
    stendevap = 0._r8
    qtendevap = 0._r8
    qliqtend_det = 0._r8

    stendsum = 0._r8
    qtendsum = 0._r8
    qliqtendsum = 0._r8
    precratesum = 0._r8
    precsum = 0._r8
    massflxbasesum = 0._r8
    massflxsum = 0._r8

!MZ
    massflxsum_dn = 0._r8
    ent_up_sum = 0._r8
    det_up_sum = 0._r8
    ent_dn_sum = 0._r8
    accuprecsum = 0._r8
    cmesum = 0._r8
    qlsum = 0._r8

    massflxbase = 0._r8
    massflxbase_w = 0._r8
    massflxbase_conv  = 0._r8
    massflxbase_mconv = 0._r8
    massflxbase_cape = 0._r8
    massflxbase_clm  = 0._r8
    massflxbase_dcape = 0._r8

!Calculation begins
#ifdef SCMDIAG
    write(*,*) "[zyx2_conv_tend]"
#endif


!------------------------------------------------------
!Calculate basic properties
!dse:dry static energy, mse: moist static energy
!esat:saturated water vapor pressure
!qsat:saturated water vapor mixing ratio
!msesat:saturated water vapor moist static energy
!------------------------------------------------------

    do i=1, inncol
        if (p(i,nlev) > psrf(i)) then
            zint(i,nlevp) = 0.0
            pint(i,nlevp) = p(i,nlev) + dp(i,nlev)/2.0
        else
            zint(i,nlevp) = max(0., zsrf(i) )
            pint(i,nlevp) = psrf(i)
        end if
    end do

!estimate z at the interface from z and dz
    do k=nlev,1,-1
        zint(:,k) = zint(:,k+1)+dz(:,k)
        pint(:,k) = pint(:,k+1)-dp(:,k)
    end do

    t = t_in
    q = q_in

!MZ 2018-08-04 to get values for bfls_t and bfls_q 
!   --------------------------------------------
    lvmid = latvap - (cpliq-cpwv) * (bfls_t-273.15)
    call cal_qsat2d(t(:,:), p(:,:), bfls_qsat(:,:))

    dse = cpair*bfls_t + gravit*z
    bfls_mse = dse + lvmid*bfls_q
    bfls_msesat = dse + lvmid*bfls_qsat

    bfls_tint(:,nlevp) = bfls_t(:,nlev)
    bfls_tint(:,1) = bfls_t(:,1)
    do k=2,nlev
        bfls_tint(:,k) = 0.5*( bfls_t(:,k)+bfls_t(:,k-1) )
    end do

    bfls_qint(:,nlevp) = bfls_q(:,nlev)
    bfls_qint(:,1) = bfls_q(:,1)
    do k=2,nlev
        bfls_qint(:,k) = 0.5*( bfls_q(:,k)+bfls_q(:,k-1) )
    end do

    lvint = latvap - (cpliq-cpwv) * (bfls_tint-273.15)
    call cal_qsat2d(bfls_tint, pint, bfls_qsatint)

    dseint = cpair*bfls_tint + gravit*zint
    bfls_mseint = dseint + lvint*bfls_qint
    bfls_msesatint = dseint + lvint*bfls_qsatint
!   --------------------------------------------
!MZ done with bfls

    lvmid = latvap - (cpliq-cpwv) * (t-273.15)
    call cal_qsat2d(t(:,:), p(:,:), qsat(:,:))

    dse = cpair*t + gravit*z
    mse = dse + lvmid*q
    msesat = dse + lvmid*qsat
    rh  = q/qsat
    rho = p/t/rair
    w = -omega/rho/gravit

    call cal_twet2d(t, rh, twet)

    do i=1, inncol
        do k=1, nlev
            twet(i,k) = min( t(i,k), twet(i,k) )
        end do
    end do

!estimate t at the interface from z and dz
    tint(:,nlevp) = t(:,nlev)
    tint(:,1) = t(:,1)
    do k=2,nlev
        tint(:,k) = 0.5*( t(:,k)+t(:,k-1) )
    end do

    qint(:,nlevp) = q(:,nlev)
    qint(:,1) = q(:,1)
    do k=2,nlev
        qint(:,k) = 0.5*( q(:,k)+q(:,k-1) )
    end do

    rhoint = pint/tint/rair

    lvint = latvap - (cpliq-cpwv) * (tint-273.15)
    call cal_qsat2d(tint, pint, qsatint)

    call cal_twet2d(tint, qint/qsatint, twetint)

    dseint = cpair*tint + gravit*zint
    mseint = dseint + lvint*qint
    msesatint = dseint + lvint*qsatint

!MZ start optional CAM initialization
!==============================

   !c0_lnd = zmconv_c0_lnd  !! from namelist using init
   !c0_ocn = zmconv_c0_ocn  !! assign values
   !ke = zmconv_ke

  ! Assign input values
  qh(:,:) = q_in(:,:)
  zm(:,:) = z(:,:)
  zi(:,:) = zint(:,:)
!  write(*,*) '---MZdp ',dp
  dpp(:,:) = dp(:,:)   !!*0.01_r8   !! checked unit!! 0.01_r8 ?
  pap(:,:) = p(:,:)    !!*0.01_r8
  paph(:,:) = pint(:,:)!!*0.01_r8
  delt = .5_r8*dtime  !!MZ

   msg = limcnv - 1
   ideep(:) = 1
!
! initialize necessary arrays.
! zero out variables not used in cam
!
   qtnd(:,:) = 0._r8
   heat(:,:) = 0._r8
   mcon(:,:) = 0._r8
   rliq(:)   = 0._r8
   dlf(:,:) = 0._r8
   pflx(:,:) = 0._r8
   zdu(:,:) = 0._r8
   rprd(:,:) = 0._r8
   mu(:,:) = 0._r8
   md(:,:) = 0._r8
   du(:,:) = 0._r8
   eu(:,:) = 0._r8
   ed(:,:) = 0._r8
   dsubcld(:) = 0._r8
   cape(:) = 0._r8

!xiex added to ZM
   condtends = 0._r8
   condtendq = 0._r8
   tranuptends = 0._r8
   tranuptendq = 0._r8
   trandntends = 0._r8
   trandntendq = 0._r8
   outtends = 0._r8
   outtendq = 0._r8
   outcondtends = 0._r8
   outcondtendq = 0._r8
   outtranuptends = 0._r8
   outtrandntendq = 0._r8
   outmb = 0._r8
   outmu = 0._r8
   outsu = 0._r8
   outqu = 0._r8
!
! initialize convective tendencies
!
   prec(:inncol) = 0._r8
   do k = 1,nlev
      do i = 1,inncol
         dqdt(i,k)  = 0._r8
         dsdt(i,k)  = 0._r8
         dudt(i,k)  = 0._r8
         dvdt(i,k)  = 0._r8
         pflx(i,k)  = 0._r8
         pflxg(i,k) = 0._r8
         cme(i,k)   = 0._r8
         rprd(i,k)  = 0._r8
         zdu(i,k)   = 0._r8
         ql(i,k)    = 0._r8
         qlg(i,k)   = 0._r8
         dlf(i,k)   = 0._r8
         dlg(i,k)   = 0._r8
      end do
   end do
   do i = 1,inncol
      pflx(i,nlevp) = 0
      pflxg(i,nlevp) = 0
   end do
!
!
   do i = 1,inncol
      pblt(i) = nlev
      dsubcld(i) = 0._r8

      jctop(i) = nlev
      jcbot(i) = 1._r8
!MZ
      jctopmin(i) = nlev
      jcbotmax(i) = 1._r8
      capemax(i)  = 1._r8
      dsubcldmax(i)  =  0._r8
      maxgmax(i) = 1._r8
      cmesum(i,:) = 0._r8
      qlsum(i,:) = 0._r8

   end do

! calculate local pressure (mbs) and height (m) for both interface
! and mid-layer locations.
!
   do i = 1,inncol
      zs(i) = geos(i)*rgrav
      pf(i,nlev+1) = paph(i,nlev+1)*0.01_r8
      zf(i,nlev+1) = zi(i,nlev+1) + zs(i)
   end do
   do k = 1,nlev
   do i = 1,inncol
   wp(i,k) = pap(i,k)*0.01_r8  !MZ  p,z,q  in ZM pressure is in mb
   pf(i,k) = paph(i,k)*0.01_r8
   wz(i,k) = zm(i,k) + zs(i)   !!MZ
   zf(i,k) = zi(i,k) + zs(i)
   end do
   end do

if(i<0)then
write(*,*)'qh',qh
write(*,*)'zm',zm
write(*,*)'zi',zi
write(*,*)'wp',wp
write(*,*)'pf',pf
write(*,*)'wz',wz
write(*,*)'zf',zf
endif      
   
   
!
   do k = nlev - 1,msg + 1,-1
      do i = 1,inncol
!MZ         if (abs(z(i,k)-zs(i)-pblh(i)) < (zf(i,k)-zf(i,k+1))*0.5_r8) pblt(i) = k
         if (abs(wz(i,k)-zs(i)-pblh(i)) < (zf(i,k)-zf(i,k+1))*0.5_r8) pblt(i) = k
      end do
   end do
   do k = 1,nlev
      do i = 1,inncol
         wq(i,k) = qh(i,k)   !MZ
         s(i,k) = t(i,k) + (grav/cpres)*wz(i,k)
         tp(i,k)=0.0_r8
         shat(i,k) = s(i,k)
         qhat(i,k) = wq(i,k) !MZ
      end do
      capeg(i) = 0._r8
      lclg(i) = 1
      lelg(i) = nlev
      maxg(i) = 1
      tlg(i) = 400._r8
      dsubcld(i) = 0._r8
   end do

!MZ optional CAM initialization done!
! ============================

!------------------------------------------------------
!Different methods to calculate entrainment rate
!------------------------------------------------------

    nplume_tot = nplume_sh + nplume_dp


    ! --- the big loop for dp and sh convection

    do iconv = 1, 2
        if ( iconv == 1 ) then
            ! shallow plumes
            greg_z0 = greg_z0_sh

            w_up_init_beg = w_up_init_sh_beg
            w_up_init_end = w_up_init_sh_end
            w_init_shape = w_init_shape_sh

            greg_ent_a_beg = greg_ent_a_sh_beg
            greg_ent_a_end = greg_ent_a_sh_end
            greg_ce_beg = greg_ce_sh_beg
            greg_ce_end = greg_ce_sh_end

            pmf_alpha = pmf_alpha_sh
            pmf_tau   = pmf_tau_sh
            nplume = nplume_sh
            ind_offset = 0

            dn_frac = dn_frac_sh

        else if ( iconv == 2 ) then
            ! deep plumes
            greg_z0 = greg_z0_dp

            w_up_init_beg = w_up_init_dp_beg
            w_up_init_end = w_up_init_dp_end
            w_init_shape = w_init_shape_dp

            greg_ent_a_beg = greg_ent_a_dp_beg
            greg_ent_a_end = greg_ent_a_dp_end
            greg_ce_beg = greg_ce_dp_beg
            greg_ce_end = greg_ce_dp_end

            pmf_alpha = pmf_alpha_dp
            pmf_tau   = pmf_tau_dp
            nplume = nplume_dp
            ind_offset = nplume_sh

            dn_frac = dn_frac_dp
        end if

        if (nplume > 1) then
            dw_up_init = (w_up_init_end - w_up_init_beg) / (nplume-1)
            greg_ent_a_delta = ( greg_ent_a_end - greg_ent_a_beg ) / (nplume-1)
            greg_ce_delta    = ( greg_ce_end - greg_ce_beg ) / (nplume-1)
        else
            dw_up_init = 0.0
            greg_ent_a_delta = 0.
            greg_ce_delta = 0.
        end if

        if (nplume <= 0) then
            cycle
        end if

        ! small loop for each regime: shallow or deep
        do j = ind_offset+1, ind_offset+nplume
    
            trigdp(:) = 1
            trigsh(:) = 1 

            greg_ent_a = greg_ent_a_beg + (j-ind_offset-1)*greg_ent_a_delta
            greg_ce    = greg_ce_beg + (j-ind_offset-1)*greg_ce_delta

            w_up_init = w_up_init_beg + (j-ind_offset-1) * dw_up_init
            
            !w_up_init = (((w_up_init - w_up_init_beg)/(w_up_init_end - w_up_init_beg))**w_init_shape) * &
            !    (w_up_init_end - w_up_init_beg) + w_up_init_beg

            w_up_init = (exp((w_up_init - w_up_init_beg)/ &
                (w_up_init_end - w_up_init_beg)*w_init_shape)-1) / (exp(w_init_shape) - 1) * &
                (w_up_init_end - w_up_init_beg) + w_up_init_beg

            !tpert_plume(:) = w_up_init/w_up_init_end * tpertglob
            !qpert_plume(:) = w_up_init/w_up_init_end * qpertglob
            !tpert_plume(:) = (1.0-w_up_init/w_up_init_end) * tpertglob
            !qpert_plume(:) = (1.0-w_up_init/w_up_init_end) * qpertglob
            tpert_plume(:) = tpertglob
            qpert_plume(:) = qpertglob

#ifdef SCMDIAG
            write(*, *) "---------------------------------------------------"
            write(*,'(a10,i5,a10,f10.5)') "plume:", j, ", w = ",w_up_init
#endif

            ! do some variable cleaning here
            ! variables w_up, buoy
            mse_up = 0._r8
            dse_up = 0._r8
            t_up = 0._r8
            q_up = 0._r8
            ent_rate_dp_up = 0._r8
            det_rate_dp_up = 0._r8
            ent_rate_sh_up = 0._r8
            det_rate_sh_up = 0._r8
            normassflx_up = 0._r8
!MZ       
            ent_dn = 0._r8
            det_dn = 0._r8
!        
            ent_org = 0.0_r8
            det_org = 0.0_r8
            ent_turb = 0.0_r8
            det_turb = 0.0_r8
            cldrad = 0.0_r8

            mse_dn = 0._r8
            dse_dn = 0._r8
            t_dn = 0._r8
            q_dn = 0._r8
            normassflx_dn = 0._r8

!MZ --------------------
i=1
if(i<0)then
              k=25
              write(*,*) 'plume model', plume_model, deep_scheme
              write(*,"(A50/,A30/,5I10/,6(A10/,3(5E15.7/)) )") &
                  ' ZZ1ZZZz in zyx2_conv at beinging  ....', &
              'state%lchnk, pcols,pver,ncol',lchnk, ncol,nlev,ncol,lengath, &
              'q(1)',q_in(1:15,k),        &
              'mcon',mcon(1:15,k),             &
              'rprd',rprd(1:15,k), &
              'qtendq(1)',qtend(1:15,k),        &
              'qliqtendq(1)',qliqtend(1:15,k),&
              'cape',cape(1:15)
endif

  if(plume_model .ne. 'cam') then

  !---------------------

            if ( iconv == 1 ) then   ! shallow plumes
                call cal_launchtocldbase( &
#ifdef OFFLINECP
                    ncol, nlev, nlevp, &
#endif
                    2, zsrf, z, zint, p, pint, t, tint, q, qint, qsat, qsatint, &
                    mse, mseint, msesat, msesatint, landfrac, lhflx, tpert_plume, qpert_plume, &
                    kuplaunch, kuplcl, mse_up, t_up, q_up, normassflx_up, trigdp)

                kupbase = kuplaunch


            else if ( iconv == 2 ) then    ! deep plumes
                call cal_launchtocldbase( &
#ifdef OFFLINECP
                    ncol, nlev, nlevp, &
#endif
                    1, zsrf, z, zint, p, pint, t, tint, q, qint, qsat, qsatint, &
                    mse, mseint, msesat, msesatint, landfrac, lhflx, tpert_plume, qpert_plume, &
                    kuplaunch, kuplcl, mse_up, t_up, q_up, normassflx_up, trigdp)

                kupbase = kuplcl

            end if

#ifdef OFFLINECP
    write(*, *) 'plume = ', j, ', lanuch process trig = ', trigdp, ', w = ', w_up_init
#endif
            dse_up = cpair*t_up+gravit*zint

            jcbot = kupbase
            do i = 1, inncol
                if (jcbot(i) > nlev) then
                    jcbot(i) = nlev
                end if
            end do
!MZ??
            !jcbot = kupbase
            jctop = kupbase


            do i=1, inncol
                if ( trigdp(i)<1 ) cycle
                !mse_up(i, 1:kupbase(i)-1) = mseint(i, 1:kupbase(i)-1)
                !t_up(i, 1:kupbase(i)-1) = tint(i, 1:kupbase(i)-1)
                !q_up(i, 1:kupbase(i)-1) = qint(i, 1:kupbase(i)-1)
            end do


   else ! 'cam'


!MZ cam Trig
 !first call to dilute
! -------------------------------------
  !limit DCAPE to be from above PBL
  kupbase = 1
  do i=1,inncol
    do k= nlev,msg,-1
      bfls_t(i,k) = t(i,k)
      bfls_q(i,k) = wq(i,k)
      if((z(i,k)-zsrf(i)) > min(pblh(i), 3000._r8) ) then
        kupbase(i) = k 
        exit
       endif
    enddo
   enddo

      !MZ call buoyan_dilute(lchnk, inncol    , &
      call buoyan_dilute(lchnk, inncol    , nlev, nlevp, &
                  !q       ,t       ,p       ,w       ,pf       , &
                  bfls_q       ,bfls_t       ,wp       ,wz       ,pf       , &
                  tp      ,qstp    ,tl      ,rl      ,bfls_dilucape, &
                  pblt    ,lcl     ,lel     ,lon     ,maxi     , &
                  rgas    ,grav    ,cpres   ,msg     , &
                  tpert,   &
!MZ
                  bfls_buoy_mid,cin)

    !write(*,*)'cape and cin 1-->', bfls_dilucape,cin

      call buoyan_dilute(lchnk, inncol    , nlev, nlevp, &
                  !q       ,t       ,p       ,w       ,pf       , &
                  wq       ,t       ,wp       ,wz       ,pf       , &
                  tp      ,qstp    ,tl      ,rl      ,cape     , &
                  pblt    ,lcl     ,lel     ,lon     ,maxi     , &
                  rgas    ,grav    ,cpres   ,msg     , &
                  tpert   , &
                  buoy_mid,cin )

    !write(*,*)'cape and cin 2-->', cape,cin

if(i<0)then
write(*,*)'!MZ in zyx2_conv     call buoyan_dilute(lchnk, inncol    , &'
write(*,*)'q',q
write(*,*)'wq',wq
write(*,*)'t',t
write(*,*)'p',p
write(*,*)'wp',wp
write(*,*)'z',z
write(*,*)'wz',wz
write(*,*)'pf',pf
write(*,*)'tp',tp
write(*,*)'qstp',qstp
write(*,*)'tl',tl
write(*,*)'rl',rl
write(*,*)'pblt',pblt
write(*,*)'lcl',lcl
write(*,*)'lel',lel
write(*,*)'lon',lon
write(*,*)'maxi',maxi
write(*,*)'rgas',rgas
write(*,*)'grav',grav
write(*,*)'cpres',cpres
write(*,*)'cape',cape
write(*,*)'msg',msg
write(*,*)'tpert',tpert
end if

i=1
if(i<0)then
   write(*,*)'in zyx2_conv 1.0 cape',lchnk,plume_model,deep_scheme,cape
endif

!!              write(*,*)' -- in zyx2_conv after buoyan_dilute cape'
!!              write(*,*)'cape',cape
!!              write(*,*)'cape q',q

    end if ! (plume model done for launching level)

  !---------------------


            normassflx_up_tmp = normassflx_up
            normassflx_dn_tmp = 0._r8

!MZ --------------------

  if(plume_model .ne. 'cam') then

  !---------------------
!      write(*,*)' here? ',lengath

!MZ 7/16/18 re-initialize!!
!!?            t_up = tint
!!?            q_up = qint
!!?            qliq_up = 0._r8
!!?            qice_up = 0._r8
!!?            q_dn = qint
!!?            dse_dn = dseint
!!MZ


!updraft properties
            if (ischeme == 2) then


                ! new scheme: MZhang


!MZ 2018-08-04 call cal_mse_up twice for DCAPE and trigger using by first using bfls_t,bfls_q

!  1st call for bfls
                call cal_mse_up( &
#ifdef OFFLINECP
                    ncol, nlev, nlevp, &
#endif
                    iconv, rho, rhoint, z, zint, dz, p, pint, &
                    bfls_t, bfls_tint, bfls_q, bfls_qint, bfls_qsat, bfls_qsatint, &
                    bfls_mse, bfls_mseint, bfls_msesat, bfls_msesatint, &
                    kuplaunch, kupbase, &
                    ent_rate_dp_up, det_rate_dp_up, ent_rate_sh_up, det_rate_sh_up, &
                    ent_org, det_org, ent_turb, det_turb, cldrad, &
                    bs_xc, w_up_init, &
                    mse_up, t_up, q_up, qliq_up, qice_up, mseqi, condrate, rainrate, snowrate, precrate, &
                    normassflx_up_tmp, w_up, w_up_mid, buoy, bfls_buoy_mid, kuptop, zuptop, &
                    trigdp)

                !i=1
                !k=25
                !write(*,*)'nstep-k1',trigdp(i), kupbase(i),kuptop(i),bfls_buoy_mid(i,k),bfls_t(i,k)
!  2st call for current state

                call cal_mse_up( &
#ifdef OFFLINECP
                    ncol, nlev, nlevp, &
#endif
                    iconv, rho, rhoint, z, zint, dz, p, pint, t, tint, q, qint, qsat, qsatint, &
                    mse, mseint, msesat, msesatint, &
                    kuplaunch, kupbase, &
                    ent_rate_dp_up, det_rate_dp_up, ent_rate_sh_up, det_rate_sh_up, &
                    ent_org, det_org, ent_turb, det_turb, cldrad, &
                    bs_xc, w_up_init, &
                    mse_up, t_up, q_up, qliq_up, qice_up, mseqi, condrate, rainrate, snowrate, precrate, &
                    normassflx_up_tmp, w_up, w_up_mid, buoy, buoy_mid, kuptop, zuptop, &
                    trigdp)

                !i=1
                !k=25
                !write(*,*)'nstep-k2',trigdp(i), kupbase(i),kuptop(i),buoy_mid(i,k),t(i,k)

             call cal_cape( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                dz, buoy_mid, normassflx_up_tmp, kupbase, kuptop, &
                dilucape(:,j), cwf(:,j), cin(:), &
                trigdp)

! here bfls_buoy_mid is manipulated to be above PBL
                         
          !  i=1
            lel = 1
          !        write(*,*)'kubase top',kuptop(i),kupbase(i)
           do i=1,inncol
             do k = kupbase(i),kuptop(i),-1
                  !write(*,*)'nstep,j, i,k,z(i,k),pblh(i)',nstep,j, i,k
                  !write(*,*)'nstep,j, i,k,z(i,k),pblh(i)',nstep,j, i,k,z(i,k),pblh(i)
                 if(z(i,k) < pblh(i)) then
                     bfls_buoy_mid(i,k) = buoy_mid(i,k)
                 else
                     lel(i) = k ! PBL index
                     exit
                 endif
              enddo
           enddo

            call cal_cape( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                dz, bfls_buoy_mid, normassflx_up_tmp, kupbase, kuptop, &
                bfls_dilucape(:), bfls_cwf(:), bfls_cin(:), &
                trigdp)


! Trigger below

            !use vort3, tdiff3 and qdiff3 later

               qdifr = (2-iconv)*capelmt_sh+(iconv-1)*capelmt_dp 
               do i = 1,inncol 
                if(dilucape(i,j) < qdifr)then
                    trigdp(i) = 0
                 endif

                 if(dilucape(i,j) < bfls_dilucape(i))then
                    trigdp(i) = 0
                 endif

                 if(w(i,lel(i)) < 0._r8)then   !downward motion at PBL top  a layer?
                    trigdp(i) = 0
                 endif

                 !if(pblh(i) < 200._r8)then   
                 !   trigdp(i) = 0
                 !endif

                 !trigdp(i) = 1  !for sensitivity tests
               enddo

               !i=1
               !k=25
               !write(*,*)'nstep,j, i,k,ktop,kbase,kpbl,',nstep,j, i,k,kupbase(i),kuptop(i)
               !write(*,*)'trigdp(i)',trigdp(i)
               !write(*,*)'dilucape(i,j),bfls_dilucape(i) ',dilucape(i,j),bfls_dilucape(i) 
               !write(*,*)'pblh(i),w(i,lel(i))',pblh(i),w(i,lel(i))



! --end of trig design-----------------------------------------------




            end if   ! for ischeme


            if (ischeme == 1) then
                ! old scheme: GRE and NSJ
                call cal_mse_up_old( &
#ifdef OFFLINECP
                    ncol, nlev, nlevp, &
#endif
                    ent_opt, rho, z, zint, dz, p, pint, t, tint, q, qint, qsat, qsatint, &
                    mse, mseint, msesat, msesatint, &
                    kuplaunch, kupbase, &
                    ent_rate_dp_up, det_rate_dp_up, ent_rate_sh_up, det_rate_sh_up, bs_xc, w_up_init, &
                    mse_up, t_up, q_up, qliq_up, qice_up, mseqi, condrate, rainrate, snowrate, precrate, &
                    normassflx_up_tmp, w_up, w_up_mid, buoy, buoy_mid, kuptop, zuptop, &
                    trigdp)
             end if

#ifdef OFFLINECP
    write(*, *) 'plume = ', j, ', updraft trig = ', trigdp
#endif
            do i=1, inncol
                if ( trigdp(i)<1 ) cycle
                if ( kuptop(i) < jctop(i) ) then
                    jctop(i) = kuptop(i)
                end if
            end do
        

            dse_up = cpair*t_up+gravit*zint
            stendcond =  latvap*condrate
            qtendcond = -condrate

!MZ
!! if(plume_model == 'scp')then
 !-----------------------------
            call cal_evap( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                ent_opt, kuptop, trigdp, zsrf, z, dz, p, rho, t, twet, q, &
                precrate, accuprec, surfprec, evaprate )

!downdraft properties
            call cal_mse_dn( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                ent_opt, kuptop, trigdp, dz, zint, p, pint, rho, t, twet, twetint, lvmid, &
                qint, dseint, accuprec, evaprate, buoy_mid, dn_frac, &
                dse_dn, q_dn, normassflx_dn_tmp)

!!    endif  ! scp

!!    if(plume_model == 'zyx2')then

!MZ Downdraft and evaporation are to be combined  --- later !!
!evaporation tendency
!!            call cal_mse_dn_evap( &
#ifdef OFFLINECP
!!                ncol, nlev, nlevp, &
#endif
!!                ent_opt, kuptop, trigdp, dz, zint, p, pint, rho, t, twet, twetint, lvmid, &
!!                qint, dseint, accuprec, evaprate, buoy_mid, dn_frac, &
!!                dse_dn, q_dn, normassflx_dn_tmp, &
!MZ combined with cal_evap
!!                q,kuplaunch,precrate,surfprec, ent_dn, det_dn)
!!
!!    endif  ! 'zyx2'

#ifdef OFFLINECP
    write(*, *) 'plume = ', j, ', evap trig = ', trigdp
#endif
            stendevap = -latvap*evaprate
            qtendevap =  evaprate

            mse_dn = dse_dn + lvint*q_dn


if(i<0)then
write(*,*)'--------    condrate, rainrate, snowrate, precrate limcnv'
write(*,*)'kuplaunch, kupbase,kuptop, zuptop',kuplaunch, kupbase,kuptop, zuptop,limcnv
write(*,*)'ent_rate_dp_up',ent_rate_dp_up
write(*,*)'det_rate_dp_up',det_rate_dp_up
write(*,*)'ent_rate_sh_up',ent_rate_sh_up
write(*,*)'det_rate_sh_up',det_rate_sh_up
write(*,*)'normassflx_up_tmp',normassflx_up_tmp
write(*,*)'ent_dn',ent_dn
write(*,*)'det_dn',det_dn
write(*,*)'normassflx_dn_tmp',normassflx_dn_tmp

write(*,*)'t_up-tint ',t_up-tint
write(*,*)'q_up-qint',(q_up-qint)*1.0e3
write(*,*)'qliq_up',qliq_up*1.0e3
write(*,*)'qice_up',qice_up*1.0e3
write(*,*)'t_dn-tint ',(dse_dn-dseint)/cpair
write(*,*)'q_dn-qint ',(q_dn-qint)*1.0e3
write(*,*)'condrate',condrate
write(*,*)'rainrate',rainrate
write(*,*)'snowrate',snowrate
write(*,*)'evaprate',evaprate
write(*,*)'precrate',precrate
write(*,*)'cme',condrate - evaprate
write(*,*)'cldcme',condrate-precrate
endif

! dilute CAPE
            call cal_cape( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                dz, buoy_mid, normassflx_up_tmp, kupbase, kuptop, &
                dilucape(:,j), cwf(:,j),cin(:), &
                trigdp)

#ifdef OFFLINECP
    write(*, *) 'plume = ', j, ', calcuate cape trig = ', trigdp
#endif

!MZ to obtain CAM type output
!----------------------------
   lengath = inncol
   maxg(:) = kuplaunch(:)
   cape(:) = dilucape(:,j)
!   cape(:)   = cwf(:,j)
   lcl(:)  = kuplcl(:)
   lel(:)  = jctop(:)

   do i = 1,inncol
    ideep(i) = i
    tl(i)   = t_up(i,kuplaunch(i))  !not used
   end do

   dsubcld(:) = 0.0_r8
   do k = msg + 1,nlev
      do i = 1,lengath
         if (k >= maxg(i)) then
            dsubcld(i) = dsubcld(i) + dp(i,k)*0.01_r8   ! in mb to be the same as cam
         end if
      end do
   end do


  else !    'cam'
! ---------------------

! determine whether grid points will undergo some deep convection
! (ideep=1) or not (ideep=0), based on values of cape,lcl,lel
! (require cape.gt. 0 and lel<lcl as minimum conditions).
!

!MZ cam Trig
! -------------------------------------
   lengath = 0

 select case (trigger_scheme)

  case('cape') !ZM
   do i=1,inncol
      if ( cape(i) > capelmt) then 
         lengath = lengath + 1
         index(lengath) = i
       endif
    enddo

  case('restricted') !additional limit

    do i=1,inncol
      flag =  (cape(i) > capelmt)                      &
          .and.  (cape(i) > bfls_dilucape(i)) &
          .and.  (w(i,kupbase(i)) > 0._r8)           &
          .and.  (pblh(i) > 200._r8)                &
          .and.  (rh(i,kupbase(i)-2) > 0.7_r8)        

       if(flag) then
         lengath = lengath + 1
         index(lengath) = i
        end if
    end do

  case('prognostic')
    do i=1,inncol
      flag =  (cape(i) > capelmt)                      &
          .and.  (cape(i) > bfls_dilucape(i)) &
          .and.  (w(i,kupbase(i)) > 0._r8)           &
          .and.  (pblh(i) > 200._r8)                &
          .and.  (rh(i,kupbase(i)-2) > 0.7_r8)        

       if(.not.flag) then
           cape(i) = 0._r8
        endif

         lengath = lengath + 1
         index(lengath) = i
    end do

   case default
      write(*,*)'--------- NO CLOSURE SCHEME IS SELECTED, STOPPED in zyx2CONV'
      stop
   end select


       !write(*,*)
       !write(*,*)' Trig in cam ---------i=',i
       !write(*,*)'cape(i),capelmt,bfls_dilucape(i)',cape(i),capelmt,bfls_dilucape(i)
       !write(*,*)'w(i,kupbase(i)),pblh(i) ,rh(i,kupbase(i)-2)',w(i,kupbase(i)),pblh(i) ,rh(i,kupbase(i)-2)
       !write(*,*)'lengath',lengath
       

i=1
if(i<0)then
   write(*,*)'in zyx2_conv 2.0 cape',lchnk,lengath,cape
endif

   if (lengath.eq.0) return
   do ii=1,lengath
      i=index(ii)
      ideep(ii)=i
   end do
!
! obtain gathered arrays necessary for ensuing calculations.
!
   do k = 1,nlev
      do i = 1,lengath
         wdp(i,k) = dpp(ideep(i),k) *0.01_r8 !MZ
         qg(i,k) = wq(ideep(i),k)  !MZ
         tg(i,k) = t(ideep(i),k)
         pg(i,k) = wp(ideep(i),k)  !MZ
         zg(i,k) = wz(ideep(i),k)  !MZ
         sg(i,k) = s(ideep(i),k)
         tpg(i,k) = tp(ideep(i),k)
         zfg(i,k) = zf(ideep(i),k)
         qstpg(i,k) = qstp(ideep(i),k)
         ug(i,k) = 0._r8
         vg(i,k) = 0._r8
      end do
   end do
!
   do i = 1,lengath
      zfg(i,nlev+1) = zf(ideep(i),nlev+1)
   end do
   do i = 1,lengath
      capeg(i) = cape(ideep(i))
      lclg(i) = lcl(ideep(i))
      lelg(i) = lel(ideep(i))
      maxg(i) = maxi(ideep(i))
      tlg(i) = tl(ideep(i))
      landfracg(i) = landfrac(ideep(i))
   end do
!
! calculate sub-cloud layer pressure "thickness" for use in
! closure and tendency routines.
!! write(*,*)'in zyx2_conv 2.3'
!
   do k = msg + 1,nlev
      do i = 1,lengath
         if (k >= maxg(i)) then
            dsubcld(i) = dsubcld(i) + wdp(i,k)
         end if
      end do
   end do
! define array of factors (alpha) which defines interfacial
! values, as well as interfacial values for (q,s) used in
! subsequent routines.
!! write(*,*)'in zyx2_conv 2.4'
!
   do k = msg + 2,nlev
      do i = 1,lengath
!            alpha(i,k) = 0.5
         sdifr = 0._r8
         qdifr = 0._r8
         if (sg(i,k) > 0._r8 .or. sg(i,k-1) > 0._r8) &
            sdifr = abs((sg(i,k)-sg(i,k-1))/max(sg(i,k-1),sg(i,k)))
         if (qg(i,k) > 0._r8 .or. qg(i,k-1) > 0._r8) &
            qdifr = abs((qg(i,k)-qg(i,k-1))/max(qg(i,k-1),qg(i,k)))
         if (sdifr > 1.E-6_r8) then
            shat(i,k) = log(sg(i,k-1)/sg(i,k))*sg(i,k-1)*sg(i,k)/(sg(i,k-1)-sg(i,k))
         else
            shat(i,k) = 0.5_r8* (sg(i,k)+sg(i,k-1))
         end if
         if (qdifr > 1.E-6_r8) then

            qhat(i,k) = log(qg(i,k-1)/qg(i,k))*qg(i,k-1)*qg(i,k)/(qg(i,k-1)-qg(i,k))
         else
            qhat(i,k) = 0.5_r8* (qg(i,k)+qg(i,k-1))
         end if
      end do
   end do
!
! obtain cloud properties.
!

if(i<0)then
 write(*,*)'in zyx2_conv 3'
write(*,*)'qg',qg
write(*,*)'tg',tg 
write(*,*)'ug',ug
write(*,*)'pg',pg 
write(*,*)'zg',zg 
write(*,*)'sg',sg 
endif
i=1
if(i<0)then
              k=25
              write(*,"(A50/,A30/,5I10/,3(A10,3(5I15/)),6(A10/,3(5E15.7/)) )") &
                  ' ZZ0ZZZz in zyx2_conv at end ....', &
              'state%lchnk, pcols,pver,ncol',lchnk, ncol,nlev,ncol,lengath, &
              'maxg',maxg(1:15),             &
              'jt',jt(1:15),             &
              'jlcl',jlcl(1:15),             &
              'qg',qg(1:15,k),        &
              'tg',tg(1:15,k),             &
              'vg',vg(1:15,k),             &
              'pg',pg(1:15,k),             &
              'zg',zg(1:15,k),             &
              'sg',sg(1:15,k)
endif


!write(*,*)'CC1',lchnk,c0_lnd,c0_ocn
!MZ   call cldprp(lchnk   , &
   call cldprp(lchnk   , inncol, nlev, nlevp,  &
               qg      ,tg      ,ug      ,vg      ,pg      , &
               zg      ,sg      ,mu      ,eu      ,du      , &
               md      ,ed      ,sd      ,qd      ,mc      , &
               qu      ,su      ,zfg     ,qs      ,hmn     , &
               hsat    ,shat    ,qlg     , &
               cmeg    ,maxg    ,lelg    ,jt      ,jlcl    , &
               maxg    ,j0      ,jd      ,rl      ,lengath , &
               rgas    ,grav    ,cpres   ,msg     , &
               pflxg   ,evpg    ,cug     ,rprdg   ,limcnv  ,landfracg)
!
! convert detrainment from units of "1/m" to "1/mb".
i=1
if(i<0)then
              k=25
              write(*,"(A50/,A30/,5I10/,3(A10/,3(5E15.7/)) )") &
                  ' ZZ2ZZZz in zyx2_conv at end ....', &
              'state%lchnk, pcols,pver,ncol',lchnk, ncol,nlev,ncol,lengath, &
              'q(1)',q_in(1:15,k),        &
              'mu',mu(1:15,k),             &
              'rprdg',rprdg(1:15,k)
endif
!
   do k = msg + 1,nlev
      do i = 1,lengath
         du   (i,k) = du   (i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
         eu   (i,k) = eu   (i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
         ed   (i,k) = ed   (i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
         cug  (i,k) = cug  (i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
         cmeg (i,k) = cmeg (i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
         rprdg(i,k) = rprdg(i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
         evpg (i,k) = evpg (i,k)* (zfg(i,k)-zfg(i,k+1))/wdp(i,k)
      end do
   end do


if(i<0)then
write(*,*)' in zyx2_conv.F90 for the zyx2cam option'
write(*,*)'mu',mu 
write(*,*)'eu',eu 
write(*,*)'qu',qu 
write(*,*)'su',su 
write(*,*)'qd',qd 
write(*,*)'capeg',capeg
write(*,*)'lelg',lelg
write(*,*)'pflxg',pflxg 
write(*,*)'rprdg',rprdg 
write(*,*)' evpg',evpg
write(*,*)'ideep',ideep 
write(*,*)'qhat',qhat
write(*,*)'shat',shat
write(*,*)'dp',dp
write(*,*)'wdp',wdp
write(*,*)'zfg',zfg
write(*,*)'qlg',qlg
write(*,*)'dsubcld',dsubcld
write(*,*)'mb',mb
write(*,*)'capeg',capeg
write(*,*)'tlg',tlg
write(*,*)'lclg',lclg
write(*,*)'lchnk   , inncol,nlev, nlevp msg,lengath'
write(*,*)lchnk   , inncol,nlev, nlevp,msg,lengath
write(*,*)'lclg,capelmt',lclg,capelmt
end if
!   write(*,*)'--MZwdp',wdp
  end if  ! plume model end cld properties
! ----------------------------------------



!Compute Closure

 if(plume_model == 'cam') then
 !------------------------
!! write(*,*)'in zyx2_conv before closure'
!MZ     call closure(lchnk   ,   &
     call closure(lchnk   , inncol,nlev, nlevp,  &
                qg      ,tg      ,pg      ,zg      ,sg      , &
                tpg     ,qs      ,qu      ,su      ,mc      , &
                du      ,mu      ,md      ,qd      ,sd      , &
!MZ                qhat    ,shat    ,dp      ,qstpg   ,zfg     , &
                qhat    ,shat    ,wdp      ,qstpg   ,zfg     , &
                qlg     ,dsubcld ,mb      ,capeg   ,tlg     , &
                lclg    ,lelg    ,jt      ,maxg    ,1       , &
                lengath ,rgas    ,grav    ,cpres   ,rl      , &
                msg     ,capelmt    )
!
! limit cloud base mass flux to theoretical upper bound.
!
   do i=1,lengath
      mumax(i) = 0
   end do
   do k=msg + 2,nlev
      do i=1,lengath
        mumax(i) = max(mumax(i), mu(i,k)/wdp(i,k))
      end do
   end do

   do i=1,lengath
      if (mumax(i) > 0._r8) then
         mb(i) = min(mb(i),0.5_r8/(delt*mumax(i)))
      else
         mb(i) = 0._r8
      endif
   end do
   ! If no_deep_pbl = .true., don't allow convection entirely
   ! within PBL (suggestion of Bjorn Stevens, 8-2000)

   if (no_deep_pbl) then
      do i=1,lengath
         if (zm(ideep(i),jt(i)) < pblh(ideep(i))) mb(i) = 0
      end do
   end if

   do k=msg+1,nlev
      do i=1,lengath
         mu   (i,k)  = mu   (i,k)*mb(i)
         md   (i,k)  = md   (i,k)*mb(i)
         mc   (i,k)  = mc   (i,k)*mb(i)
         du   (i,k)  = du   (i,k)*mb(i)
         eu   (i,k)  = eu   (i,k)*mb(i)
         ed   (i,k)  = ed   (i,k)*mb(i)
         cmeg (i,k)  = cmeg (i,k)*mb(i)
         rprdg(i,k)  = rprdg(i,k)*mb(i)
         cug  (i,k)  = cug  (i,k)*mb(i)
         evpg (i,k)  = evpg (i,k)*mb(i)
         pflxg(i,k+1)= pflxg(i,k+1)*mb(i)*100._r8/grav
      end do
   end do
 
if(i<0)then
write(*,*)'mb',mb 
write(*,*) 'after ... closure'
write(*,*)'mu',mu 
write(*,*)'eu',eu 
write(*,*)'qu',qu 
write(*,*)'su',su 
write(*,*)'qd',qd 
write(*,*)'capeg',capeg
write(*,*)'pflxg',pflxg 
write(*,*)'rprdg',rprdg 
write(*,*)' evag',evpg
write(*,*)'ideep',ideep 
write(*,*)'jt',jt
end if

! compute temperature and moisture changes due to convection.
!
!!write(*,*)'!MZ   call q1q2_pjr(lchnk   , &'
!MZ   call q1q2_pjr(lchnk   , &
   call q1q2_pjr(lchnk   ,inncol, nlev, nlevp,  &
                 dqdt    ,dsdt    ,qg      ,qs      ,qu      , &
!MZ                 su      ,du      ,qhat    ,shat    ,dp      , &
                 su      ,du      ,qhat    ,shat    ,wdp      , &
                 mu      ,md      ,sd      ,qd      ,qlg     , &
                 dsubcld ,jt      ,maxg    ,1       ,lengath , &
                 cpres   ,rl      ,msg     ,          &
!xiex
                 condtends, condtendq, tranuptends, tranuptendq, &
                 trandntends, trandntendq, &
                 dlg     ,evpg    ,cug     )

!
! gather back temperature and mixing ratio.
!
   do k = msg + 1,nlev
!DIR$ CONCURRENT
      do i = 1,lengath
!
! q is updated to compute net precip.
!
         q(ideep(i),k) = qh(ideep(i),k) + 2._r8*delt*dqdt(i,k)  !MZ
         !wq(ideep(i),k) = qh(ideep(i),k) + 2._r8*delt*dqdt(i,k)  !MZ
         qtnd(ideep(i),k) = dqdt (i,k)
         cme (ideep(i),k) = cmeg (i,k)
         rprd(ideep(i),k) = rprdg(i,k)
         zdu (ideep(i),k) = du   (i,k)
         mcon(ideep(i),k) = mc   (i,k)
         heat(ideep(i),k) = dsdt (i,k)*cpres
         dlf (ideep(i),k) = dlg  (i,k)
         pflx(ideep(i),k) = pflxg(i,k)
         ql  (ideep(i),k) = qlg  (i,k)
!xiex
         outmb(ideep(i)) = mb(i)
         outmu(ideep(i),k) = mu(i,k)
         outsu(ideep(i),k) = su(i,k)
         outqu(ideep(i),k) = qu(i,k)
         outtends(ideep(i),k) = dsdt(i,k)*cpres
         outtendq(ideep(i),k) = dqdt(i,k)
         outcondtends(ideep(i),k) = condtends(i,k)*cpres
         outcondtendq(ideep(i),k) = condtendq(i,k)
         outtranuptends(ideep(i),k) = tranuptends(i,k)*cpres
         outtranuptendq(ideep(i),k) = tranuptendq(i,k)
         outtrandntends(ideep(i),k) = trandntends(i,k)*cpres
         outtrandntendq(ideep(i),k) = trandntendq(i,k)
      end do
   end do
!
!DIR$ CONCURRENT
   do i = 1,lengath
      jctop(ideep(i)) = jt(i)
!++bee
      jcbot(ideep(i)) = maxg(i)
!--bee
      pflx(ideep(i),nlevp) = pflxg(i,nlevp)
   end do

! Compute precip by integrating change in water vapor minus detrained cloud water
if(i<0)then
write(*,*)'nlev,msg',nlev,msg
write(*,*)'prec',prec
write(*,*)'dpp',dpp
write(*,*)'wq',wq
write(*,*)'qh',qh
write(*,*)'dlf',dlf
write(*,*)'delt',delt
endif 
   do k = nlev,msg + 1,-1
      do i = 1,inncol
         prec(i) = prec(i) - dpp(i,k)* (q(i,k)-qh(i,k)) - dpp(i,k)*dlf(i,k)*2*delt !MZ  q here!
      end do
   end do
! obtain final precipitation rate in m/s.
   do i = 1,inncol
      prec(i) = rgrav*max(prec(i),0._r8)/ (2._r8*delt)/1000._r8
   end do

! Compute reserved liquid (not yet in cldliq) for energy integrals.
! Treat rliq as flux out bottom, to be added back later.
   do k = 1, nlev
      do i = 1, inncol
         rliq(i) = rliq(i) + dlf(i,k)*dpp(i,k)/gravit
      end do
   end do
   rliq(:inncol) = rliq(:inncol) /1000._r8


! Return ZM fields to zyx2_conv interface output

  massflxbase_p(:,nlev) = mb(:) ! for prognostic purpose 
  stend    = heat
  qtend    = qtnd
  qliqtend = dlf  
  precrate_out = rprd

  stendcomp = stend
  qtendcomp = qtend

if(i<0)then
  write(*,*)'------- in zyx2_conv_tend heat'
  write(*,*)'heat',heat
  write(*,*)'qtnd',qtnd
  write(*,*)'pflx',pflx
  write(*,*)'prec',prec
endif
   !return  ! done if it is CAM !
   goto 1001

!!   write(*,*) 'cam plume_model option done in zyx2_conv!'
   !============================
 endif  ! 'cam'

 !else  ! 'zyx2

!updraft transport tendency
            call cal_tendtransport( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                dz, kupbase, kuptop, &
                rho, dseint, qint, dse_up, q_up, &
                normassflx_up_tmp,  &
                stendtran_up, qtendtran_up, &
                trigdp)

#ifdef OFFLINECP
    write(*, *) 'plume = ', j, ', trans_up trig = ', trigdp
#endif
!downdraft transport tendency
            call cal_tendtransport( &
#ifdef OFFLINECP
                ncol, nlev, nlevp, &
#endif
                dz, kupbase, kuptop, &
                rho, dseint, qint, dse_dn, q_dn, &
                normassflx_dn_tmp,  &
                stendtran_dn, qtendtran_dn, &
                trigdp)
#ifdef OFFLINECP
    write(*, *) 'plume = ', j, ', trans_down trig = ', trigdp
#endif

!liquid detrainment tendency

            do i=1, inncol
                qliqtend_det(i,1:nlev) = 0.0
                if ( trigdp(i)<1 ) cycle

                ! New version: detrain occurs through cloud layers
                do k = kuptop(i)-1, kupbase(i)-1, 1
                    qliqtend_det(i,k) = max(0.0, &
                        normassflx_up_tmp(i,k+1) * min(det_rate_dp_up(i,k) + det_rate_sh_up(i,k), max_det_rate) &
                            * (qliq_up(i,k+1) + qice_up(i,k+1)) / rho(i,k) )

                end do

!!MZMZ top layer needed

                ! Old version: detrain occurs only at the cloud top layer
                !k = kuptop(i)-1
                !qliqtend_det(i,k) = max( 0.0, &
                !    normassflx_up_tmp(i,k+1)*( qliq_up(i,k+1)+qice_up(i,k+1) ) &
                !    /dz(i,k)/rho(i,k) )
            end do

            dlf = qliqtend_det


        ql = qliq_up + qice_up



if(i<0)then
write(*,*)'normassflx_up(i,k)',normassflx_up(i,:)
write(*,*)'qliq_up(i,k)',qliq_up(i,:)
write(*,*)'qliqtend_det(i,k)',qliqtend_det(i,:) 
endif                    


            do i=1, inncol
                if ( trigdp(i)<1 ) cycle
                if (iconv == 1) then  ! shallow plumes 
                    if (basemass_enhance > 1.0) then
                        basemass_scale = max(0.0, -(basemass_enhance-1.0)/w_up_init_end * w_up_init(i) &
                            + basemass_enhance )
                    else
                        basemass_scale = w_up_init_end/w_up_init(i)
                    end if
                    massflxbase_p(i,j) = min( 0.1, max( 0., &
                        massflxbase_p(i,j) + dtime*( max( (dilucape(i,j) - capelmt_sh), 0._r8 )/(2*pmf_alpha) &
                        * basemass_scale &
                        - massflxbase_p(i,j)/(2*pmf_tau) ) ) )
                else    ! deep plumes
                    massflxbase_p(i,j) = min( 0.1, max( 0., &
                        massflxbase_p(i,j) + dtime*( max( (dilucape(i,j) - capelmt_dp), 0._r8 )/(2*pmf_alpha) &
                        - massflxbase_p(i,j)/(2*pmf_tau) ) ) )
                end if
            end do

!SENS
!                    massflxbase_p(i,j) + dtime*( cwf(i,j)/(2*pmf_alpha) &
!                    massflxbase_p(i,j) + dtime*( dilucape(i,j)/(2*pmf_alpha) &
!                massflxbase_p(i,j) = min( 0.1, max( 0., (dilucape(i,j) - capelmt)/cape_timescale ) )



! ----------------------------------   
!MZ 2018-08-02 put limit to each plume for CFL condition

 do i=1,inncol
     if( trigdp(i) < 1 )then
      massflxbase_p(:,j) = 0._r8
     endif
 enddo
  
 massflxbase(:) = massflxbase_p(:,j)

   mumax(:) = 0._r8
   mb(:)    = 0._r8

   do k=msg + 1,nlev
      do i=1,inncol
        mumax(i) = max(mumax(i), normassflx_up_tmp(i,k)/(rho(i,k)*dz(i,k)) )
        mumax(i) = max(mumax(i), -normassflx_dn_tmp(i,k)/(rho(i,k)*dz(i,k)))
      end do
   end do

   mumax(:) = mumax(:)*massflxbase(:)

   do i=1,inncol
      if (mumax(i) > 1.e-20_r8) then
        mb(i) = min(1._r8, dtime*mumax(i)  )    ! 1.0 can be changed!
        !mb(i) = min(0.5_r8, dtime*mumax(i)  )    ! 1.0 can be changed!
        massflxbase(i) = massflxbase(i)*mb(i)/(dtime*mumax(i)) 
      else
        massflxbase(i) = 0._r8
      endif
   end do


            if (fixbasemf > 0) then
                massflxbase = fixbasemf
            end if
!MZ - done ----------------------

            
#ifdef SCMDIAG 
            write(*,'(a25,f10.5,a25,i10)') "massflxbase = ", massflxbase(1), "trigdp = ", trigdp(1)
!            write(*,'(a25,i10,a25,f10.5)') "kuptop = ", kuptop(1), "dilucape=", dilucape(1,j)
!        write(*,'(10f20.10)') dtime, dilucape(1,j), pmf_alpha, dtime*dilucape(1,j)/(2*pmf_alpha)
#endif

            netprec = 0._r8
            do i=1, inncol
                if ( trigdp(i)<1 ) massflxbase(i) = 0._r8
                condrate(i,:) = condrate(i,:) * massflxbase(i)  ! 1/s
                rainrate(i,:) = rainrate(i,:) * massflxbase(i)  ! 1/s
                snowrate(i,:) = snowrate(i,:) * massflxbase(i)  ! 1/s
                precrate(i,:) = precrate(i,:) * massflxbase(i)  ! 1/s
                accuprec(i,:) = accuprec(i,:) * massflxbase(i)  ! kg/m2/s
                evaprate(i,:) = evaprate(i,:) * massflxbase(i)  ! 1/s
                surfprec(i) = surfprec(i) * massflxbase(i) / rhofw  ! m/s

                stendtran_up(i,:) = stendtran_up(i,:)*massflxbase(i)
                qtendtran_up(i,:) = qtendtran_up(i,:)*massflxbase(i)

                stendtran_dn(i,:) = stendtran_dn(i,:) * massflxbase(i)
                qtendtran_dn(i,:) = qtendtran_dn(i,:) * massflxbase(i)

                stendcond(i,:) = stendcond(i,:)*massflxbase(i)
                qtendcond(i,:) = qtendcond(i,:)*massflxbase(i)

                stendevap(i,:) = stendevap(i,:)*massflxbase(i)
                qtendevap(i,:) = qtendevap(i,:)*massflxbase(i)

                qliqtend_det(i,:) = qliqtend_det(i,:)*massflxbase(i)

                massflx(i,:) = normassflx_up_tmp(i,:)*massflxbase(i)

!MZ ------------------
                massflx_dn(i,:) = normassflx_dn_tmp(i,:)*massflxbase(i)  !*dn_frac already considered

                ent_up(i,:) = (ent_rate_dp_up(i,:) + ent_rate_sh_up(i,:))*massflx(i,:) !MZ
                det_up(i,:) = (det_rate_dp_up(i,:) + det_rate_sh_up(i,:))*massflx(i,:)

                ent_dn(i,:) = ent_dn(i,:)*massflxbase(i)  ! this is already the actual entrainment
                det_dn(i,:) = det_dn(i,:)*massflxbase(i)  ! this is already the actual detrainment

                cme(i,:) = (condrate(i,:) - evaprate(i,:)) *massflxbase(i)  
!MZ ------------------

                stend(i,:) = stendcond(i,:) + stendevap(i,:) &
                    + stendtran_up(i,:) + stendtran_dn(i,:) 

                qtend(i,:) = qtendcond(i,:) + qtendevap(i,:) &
                    + qtendtran_up(i,:) + qtendtran_dn(i,:) 

                if(plume_model =='scp')then
                    stend(i,:) = stend(i,:)- lvmid(i,:) * qliqtend_det(i,:)
                    qtend(i,:) = qtend(i,:)+ qliqtend_det(i,:)
                endif 

                do k=1, nlev
                    netprec(i) = netprec(i) + max(0.0, - ( qtend(i,k) + qliqtend_det(i,k) )*rho(i,k)*dz(i,k))
                end do
                netprec(i) = netprec(i)/rhofw

            end do

            do i = 1, inncol, 1
#ifdef OFFLINECP
                all_ent_org(i,:,j) = ent_org(i,:)
                all_det_org(i,:,j) = det_org(i,:) 
                all_ent_turb(i,:,j) = ent_turb(i,:)
                all_det_turb(i,:,j) = det_turb(i,:)
                all_massflx(i,:,j) = normassflx_up_tmp(i,:)
                all_mse_up(i,:,j) = mse_up(i,:)
                all_t_up(i,:,j) = t_up(i,:)
                all_q_up(i,:,j) = q_up(i,:)
                all_qliq_up(i,:,j) = qliq_up(i,:)
                all_qice_up(i,:,j) = qice_up(i,:)
                all_w_up(i,:,j) = w_up(i,:)
                all_buoy(i,:,j) = buoy(i,:)
                all_mseqi(i,:,j) = mseqi(i,:)
                all_condrate(i,:,j) = condrate(i,:)
                all_rainrate(i,:,j) = rainrate(i,:)
                all_snowrate(i,:,j) = snowrate(i,:)
                all_precrate(i,:,j) = precrate(i,:)
                all_accuprec(i,:,j) = accuprec(i,:)
                all_evaprate(i,:,j) = evaprate(i,:)
                all_stend(i,:,j) = stend(i,:)
                all_qtend(i,:,j) = qtend(i,:)
                all_qliqtend(i,:,j) = qliqtend_det(i,:)
                all_stendcond(i,:,j) = stendcond(i,:)
                all_stendevap(i,:,j) = stendevap(i,:)
                all_stendtran_up(i,:,j) = stendtran_up(i,:)
                all_stendtran_dn(i,:,j) = stendtran_dn(i,:)
                all_qtendcond(i,:,j) = qtendcond(i,:)
                all_qtendevap(i,:,j) = qtendevap(i,:)
                all_qtendtran_up(i,:,j) = qtendtran_up(i,:)
                all_qtendtran_dn(i,:,j) = qtendtran_dn(i,:)
                all_cape(i,j) = dilucape(i,j)
                all_massflxbase(i,j) = massflxbase(i)
                all_prec(i,j) = netprec(i)
                all_surfprec(i,j) = surfprec(i)
#else
                all_stend(i,:,j) = stend(i,:)
                all_qtend(i,:,j) = qtend(i,:)
                all_qliqtend(i,:,j) = qliqtend_det(i,:)
                all_precrate(i,:,j) = precrate(i,:) - evaprate(i,:)
                all_massflx(i,:,j)  = massflx(i,:)
                all_massflxbase(i,j) = massflxbase(i)
                all_prec(i,j) = netprec(i)
                all_surfprec(i,j) = surfprec(i)
#endif
            end do

            do i=1, inncol
                stendsum(i,:) = stendsum(i,:)+stend(i,:)
                qtendsum(i,:) = qtendsum(i,:)+qtend(i,:)
                qliqtendsum(i,:) = qliqtendsum(i,:)+qliqtend_det(i,:)
                precratesum(i,:) = precratesum(i,:)+( precrate(i,:)-evaprate(i,:) )
                precsum(i) = precsum(i)+netprec(i)
                surfprecsum(i) = surfprecsum(i) + surfprec(i)
                massflxbasesum(i) = massflxbasesum(i) + massflxbase(i)
                massflxsum(i,:) = massflxsum(i,:) + massflx(i,:)

!MZ

                massflxsum_dn(i,:) = massflxsum_dn(i,:) + massflx_dn(i,:)
                ent_up_sum (i,:)   = ent_up_sum(i,:)    + ent_up(i,:)
                det_up_sum (i,:)   = det_up_sum(i,:)    + det_up(i,:)
                ent_dn_sum (i,:)   = ent_dn_sum(i,:)    + ent_dn(i,:)
                accuprecsum(i,:)     = accuprecsum(i,:) + accuprec(i,:)
                cmesum(i,:)     = cmesum(i,:) + cme(i,:)

                qlsum(i,:)     = qlsum(i,:) + ql(i,:)*massflx_dn(i,:)

                capemax(i)     = max(capemax(i) , cape(i))
                dsubcldmax(i) = max(dsubcldmax(i),dsubcld(i))
                maxgmax(i) = max(maxgmax(i),kuplaunch(i))

                jctopmin(i)     = min(jctopmin(i) , jctop(i))
                jcbotmax(i)     = max(jcbotmax(i) , 1._r8*kuplaunch(i))

!>
            end do
if(i<0)then 
    write(*,*)'-----j plume',j
    write(*,*)' qliqtendsum qliqtend_det ',qliqtend_det
    write(*,*)' qliqtendsum qliqtend_det ',qliqtendsum
    write(*,*)'-----massflxbase', massflxbase
    write(*,*)'-----massflx', massflx
    write(*,*)'-----massflxsum',massflxsum
    write(*,*)'stendtran_up',stendtran_up
    write(*,*)'qtendtran_up',qtendtran_up
    write(*,*)'stendtran_dn',stendtran_dn
    write(*,*)'qtendtran_dn',qtendtran_dn

endif


            diffdse_up = 0._r8
            diffq_up = 0._r8
!MZ            
          if(plume_model =='scp')then
            do i=1, inncol
                if ( trigdp(i)<1 ) cycle

                do k=nlevp, kuptop(i), -1
                    diffdse_up(i,k) = normassflx_up_tmp(i,k)*( dse_up(i,k)-dseint(i,k) )
                    diffq_up(i,k)   = normassflx_up_tmp(i,k)*( q_up(i,k)-qint(i,k) )
                end do
            end do
           endif

          if(plume_model =='zyx2')then
            do i=1, inncol
                if ( trigdp(i)<1 ) cycle

                do k=nlevp, kuptop(i), -1
                    diffdse_up(i,k) = ( dse_up(i,k)-dseint(i,k) )
                    diffq_up(i,k)   = ( q_up(i,k)-qint(i,k) )
                end do
            end do
         endif

#ifdef SCMDIAG
            call subcol_netcdf_putclm( "ent_rate", nlev, &
                min( max_ent_rate, ent_rate_dp_up(1,:)+ent_rate_sh_up(1,:)), j )
            call subcol_netcdf_putclm( "det_rate", nlev, &
                min( max_det_rate, det_rate_dp_up(1,:)+det_rate_sh_up(1,:)), j )
            call subcol_netcdf_putclm( "ent_rate_dp", nlev, ent_rate_dp_up(1,:), j )
            call subcol_netcdf_putclm( "det_rate_dp", nlev, det_rate_dp_up(1,:), j )
            call subcol_netcdf_putclm( "ent_rate_sh", nlev, ent_rate_sh_up(1,:), j )
            call subcol_netcdf_putclm( "det_rate_sh", nlev, det_rate_sh_up(1,:), j )
            call subcol_netcdf_putclm( "bs_xc", nlev, bs_xc(1,:), j )
            
            call subcol_netcdf_putclm( "radius_up", nlev, cldrad(1,:), j )
            call subcol_netcdf_putclm( "ent_rate_org", nlev, ent_org(1,:), j )
            call subcol_netcdf_putclm( "det_rate_org", nlev, det_org(1,:), j )
            call subcol_netcdf_putclm( "ent_rate_turb", nlev, ent_turb(1,:), j )
            call subcol_netcdf_putclm( "det_rate_turb", nlev, det_turb(1,:), j )

            call subcol_netcdf_putclm( "w_up_mid", nlev, w_up_mid(1,:), j )
            call subcol_netcdf_putclm( "buoy_mid", nlev, buoy_mid(1,:), j )

            call subcol_netcdf_putclm( "w_up_init", 1, w_up_init(1), j )
            call subcol_netcdf_putclm( "w_up", nlevp, w_up(1,:), j )
            call subcol_netcdf_putclm( "buoy", nlevp, buoy(1,:), j )
            call subcol_netcdf_putclm( "mse_up", nlevp, mse_up(1,:), j )
            call subcol_netcdf_putclm( "t_up", nlevp, t_up(1,:), j )
            call subcol_netcdf_putclm( "q_up", nlevp, q_up(1,:), j )
            call subcol_netcdf_putclm( "qliq_up", nlevp, qliq_up(1,:), j )
            call subcol_netcdf_putclm( "qice_up", nlevp, qice_up(1,:), j )
            call subcol_netcdf_putclm( "dse_up", nlevp, dse_up(1,:), j )
            call subcol_netcdf_putclm( "normassflx_up", nlevp, normassflx_up_tmp(1,:), j )

            call subcol_netcdf_putclm( "mse_dn", nlevp, mse_dn(1,:), j )
            call subcol_netcdf_putclm( "normassflx_dn", nlevp, normassflx_dn_tmp(1,:), j )

            call subcol_netcdf_putclm( "dilucape", 1, dilucape(1,j), j )
            ! call subcol_netcdf_putclm( "weight", 1, weights(1,j), j )
            call subcol_netcdf_putclm( "mseqi", nlev, mseqi(1,:), j )
            call subcol_netcdf_putclm( "condrate", nlev, condrate(1,:), j )
            call subcol_netcdf_putclm( "rainrate", nlev, rainrate(1,:), j )
            call subcol_netcdf_putclm( "snowrate", nlev, snowrate(1,:), j )
            call subcol_netcdf_putclm( "precrate", nlev, precrate(1,:), j )
            call subcol_netcdf_putclm( "accuprec", nlev, accuprec(1,:), j )
            call subcol_netcdf_putclm( "evaprate", nlev, evaprate(1,:), j )

            call subcol_netcdf_putclm( "stend", nlev, stend(1,:), j )
            call subcol_netcdf_putclm( "qtend", nlev, qtend(1,:), j )
            call subcol_netcdf_putclm( "stendcond", nlev, stendcond(1,:), j )
            call subcol_netcdf_putclm( "qtendcond", nlev, qtendcond(1,:), j )
            call subcol_netcdf_putclm( "stendevap", nlev, stendevap(1,:), j )
            call subcol_netcdf_putclm( "qtendevap", nlev, qtendevap(1,:), j )
            call subcol_netcdf_putclm( "stendtranup", nlev, stendtran_up(1,:), j )
            call subcol_netcdf_putclm( "qtendtranup", nlev, qtendtran_up(1,:), j )
            call subcol_netcdf_putclm( "stendtrandn", nlev, stendtran_dn(1,:), j )
            call subcol_netcdf_putclm( "qtendtrandn", nlev, qtendtran_dn(1,:), j )
            call subcol_netcdf_putclm( "qliqtenddet", nlev, qliqtend_det(1,:), j )

            call subcol_netcdf_putclm( "diffdse_up", nlevp, diffdse_up(1,:), j )
            call subcol_netcdf_putclm( "diffq_up", nlevp, diffq_up(1,:), j )

            call subcol_netcdf_putclm( "massflxbase", 1, massflxbase(1), j )
            call subcol_netcdf_putclm( "massflx", nlevp, massflx(1,:), j )
            !call subcol_netcdf_putclm( "prec", 1, surfprec(1), j )
            call subcol_netcdf_putclm( "prec", 1, netprec(1), j )

            tmp2d = trigdp
            call subcol_netcdf_putclm( "trigdp", 1, tmp2d(1), j )

#endif
        end do     ! loop of plume

    end do     ! loop of convection(dp/sh)

if(i<0)then
    write(*,*)'---massflxsum_1',massflxsum
endif

    !----------------------------------------------------------------------
    ! Haiyang Yu: normalized with weights
    do i = 1, inncol
#if (! defined OFFLINECP)        
        call cal_weight_eigen(nlev, nplume_tot, p(i,:), nn_stend(i,:), all_stend(i,:,1:nplume_tot), &
            nn_qtend(i,:), all_qtend(i,:,1:nplume_tot), nn_prec(i), all_prec(i,1:nplume_tot), &
            weights(i,1:nplume_tot), valid(i,1:nplume_tot))
       
        !if (sum(valid(i,1:nplume_tot)) >= 1) then
        if ( abs(sum(weights(i,1:nplume_tot))) > nplume_tot*1.0 ) then
            weights(i,1:nplume_tot) = weights(i,1:nplume_tot)/sum(weights(i,1:nplume_tot)) * nplume_tot
        end if
#ifdef SCMDIAG
        write(*, *) 'weights = ', weights(i,1:nplume_tot)
        write(*, *) 'valid = ', valid(i,1:nplume_tot)
#endif
        if (nn_flag > 0) then

            do k = 1, nlev, 1
                stendsum(i,k) = sum(all_stend(i,k,1:nplume_tot) * weights(i,1:nplume_tot))
                qtendsum(i,k) = sum(all_qtend(i,k,1:nplume_tot) * weights(i,1:nplume_tot))
                qliqtendsum(i,k) = sum(all_qliqtend(i,k,1:nplume_tot) * weights(i,1:nplume_tot))
                precratesum(i,k) = sum(all_precrate(i,k,1:nplume_tot) * weights(i,1:nplume_tot))
                massflxsum(i,k) = sum(all_massflx(i,k,1:nplume_tot) * weights(i,1:nplume_tot))
                massflxbasesum(i) = sum(all_massflxbase(i,1:nplume_tot) * weights(i,1:nplume_tot))
                precsum(i) = sum(all_prec(i,1:nplume_tot) * weights(i,1:nplume_tot))
                surfprecsum(i) = sum(all_surfprec(i,1:nplume_tot) * weights(i,1:nplume_tot))
            end do

            ! qliqtendsum(i,:) = qliqtendsum(i,:) / nplume_tot

            !if ( validplume(i) > 0 .and. abs(totalweight(i)) > 1e-6) then
            !    totalweight(i) = 1.0 / totalweight(i)
            !else
            !    totalweight(i) = 0.0
            !end if

            !stendsum(i,:) = stendsum(i,:) * totalweight(i)
            !qtendsum(i,:) = qtendsum(i,:) * totalweight(i)
            !qliqtendsum(i,:) = qliqtendsum(i,:) * totalweight(i)
            !precratesum(i,:) = precratesum(i,:) * totalweight(i)
            !precsum(i) = precsum(i) * totalweight(i)
            !surfprecsum(i) = surfprecsum(i) * totalweight(i)
            !massflxbasesum(i) = massflxbasesum(i) * totalweight(i)
            !massflxsum(i,:) = massflxsum(i,:) * totalweight(i) 
            
        else
#endif            

            ! without NN: mean
            if (meanorsum == 1) then
!!                write(*,*)'-------- This statement is done!!'
                stendsum(i,:) = stendsum(i,:) / nplume_tot  
                qtendsum(i,:) = qtendsum(i,:) / nplume_tot
                qliqtendsum(i,:) = qliqtendsum(i,:) / nplume_tot
                precratesum(i,:) = precratesum(i,:) / nplume_tot
                precsum(i) = precsum(i) / nplume_tot 
                surfprecsum(i) = surfprecsum(i) / nplume_tot 
                massflxbasesum(i) = massflxbasesum(i) / nplume_tot 
                massflxsum(i,:) = massflxsum(i,:) / nplume_tot 
!MZ
                massflxsum_dn(i,:) = massflxsum_dn(i,:) / nplume_tot
                ent_up_sum(i,:)    = ent_up_sum(i,:) / nplume_tot
                det_up_sum(i,:)    = det_up_sum(i,:) / nplume_tot
                ent_dn_sum(i,:)    = ent_dn_sum(i,:) / nplume_tot
                accuprecsum(i,:)   = accuprecsum(i,:) / nplume_tot
                cmesum(i,:)        = cmesum(i,:) / nplume_tot

            end if
#if (! defined OFFLINECP)        
        end if
#endif
    end do

if(i<0)then
    write(*,*)'---massflxsum_2',massflxsum
    write(*,*)'---qliqtendsum',qliqtendsum
endif
    !----------------------------------------------------------------------

    !for diag only
    mse_up_mid = 0._r8
    t_up_mid = 0._r8
    q_up_mid = 0._r8
    normassflx_up_mid = 0._r8

    do i=1, inncol
        do k=nlev, 1, -1
            if ( (mse_up(i,k)>0._r8) .and. (mse_up(i,k+1)>0._r8) ) then
                mse_up_mid(i,k) = 0.5*( mse_up(i,k) + mse_up(i,k+1) )
                t_up_mid(i,k) = 0.5*( t_up(i,k) + t_up(i,k+1) )
                q_up_mid(i,k) = 0.5*( q_up(i,k) + q_up(i,k+1) )
            end if
            normassflx_up_mid(i,k) = 0.5*( normassflx_up(i,k) + normassflx_up(i,k+1) )
        end do
    end do

!!    write(*,*)'---massflxsum_3',massflxsum

    !use sum as ouput
    prec = precsum
    stend = stendsum
    qtend = qtendsum
    qliqtend = qliqtendsum
    precrate_out = precratesum
!!
    mcon = massflxsum + massflxsum_dn

!    mcon(:,:) = massflxsum(:,:)
!    write(*,*)'mcon---',mcon
!    write(*,*)'massflxsum--',massflxsum

#ifdef SCMDIAG
do j = 1, nplume_tot, 1
    call subcol_netcdf_putclm( "weight", 1, weights(1,j), j )
end do
#endif

#ifdef SCMDIAG
!    call subcol_netcdf_putclm( "stendsum", nlev, stendsum(1,:), 1 )
!    call subcol_netcdf_putclm( "qtendsum", nlev, qtendsum(1,:), 1 )
!    call subcol_netcdf_putclm( "qliqtendsum", nlev, qliqtendsum(1,:), 1 )
!    call subcol_netcdf_putclm( "precratesum", nlev, precratesum(1,:), 1 )
!    call subcol_netcdf_putclm( "precsum", 1, precsum(1), 1 )
!    call subcol_netcdf_putclm( "massflxsum", nlevp, massflxsum(1,:), 1 )
!    call subcol_netcdf_putclm( "massflxbasesum", 1, massflxbasesum(1), 1 )
#endif

!------------------------------------------------------
!make sure no negative q
!------------------------------------------------------

    qcheckout = 1._r8
    minqcheckf = 1._r8
    
    
    if (flagqcheck == 1) then
        do i=1, inncol
    !whole column adjustment
            minqcheckf = 1._r8
            do k=nlev, 1, -1

                if ( (q(i,k)<=qmin*1.001) .and. (qtend(i,k)<0.) ) then
#ifdef SCMDIAG 
                    write(*,*) 'too small Q: ', p(i,k), q(i,k), qtend(i,k)*dtime
#endif
                    ! Haiyang Yu
                    qtend(i,k) = 0.0
                    stend(i,k) = 0.0
                    qliqtend(i,k) = 0.0
                    mcon(i,k) = 0.0
                    precrate_out(i,k) = 0.0

    !                minqcheckf = 0._r8
    !                trigdp(i) = -91
    !                exit
                end if

                qcheckf = q(i,k)+qtend(i,k)*dtime

                if( qcheckf<qmin ) then
                    if (abs(qtend(i,k)) > 1e-15) then
                        qcheckf = (qmin*1.001-q(i,k))/dtime/qtend(i,k)
                    else
                        qcheckf = 1.0
                    end if
                    if( qcheckf<minqcheckf ) then
                        minqcheckf = qcheckf
                    end if
                end if

            end do

            if( minqcheckf<1.0_r8 ) then
                massflxbase(i) = minqcheckf*massflxbase(i)

                stendcond(i,:) = minqcheckf*stendcond(i,:)
                qtendcond(i,:) = minqcheckf*qtendcond(i,:)
                stendtran_up(i,:) = minqcheckf*stendtran_up(i,:)
                qtendtran_up(i,:) = minqcheckf*qtendtran_up(i,:)
                stendevap(i,:) = minqcheckf*stendevap(i,:)
                qtendevap(i,:) = minqcheckf*qtendevap(i,:)
                
                qliqtend(i,:) = minqcheckf*qliqtend(i,:)
                mcon(i,:) = minqcheckf*mcon(i,:)

                stend(i,:) = minqcheckf*stend(i,:)
                qtend(i,:) = minqcheckf*qtend(i,:)
                prec(i) = minqcheckf*prec(i)
                precrate_out(i,:) = minqcheckf*precrate_out(i,:)

                stendcomp(i,:) = minqcheckf*stendcomp(i,:)
                qtendcomp(i,:) = minqcheckf*qtendcomp(i,:)

                tmp1stend(i,:) = minqcheckf*tmp1stend(i,:)
                tmp1qtend(i,:) = minqcheckf*tmp1qtend(i,:)
                tmp2stend(i,:) = minqcheckf*tmp2stend(i,:)
                tmp2qtend(i,:) = minqcheckf*tmp2qtend(i,:)

!MZ
                massflxsum(i,:) = minqcheckf*massflxsum(i,:)
                massflxsum_dn(i,:) = minqcheckf*massflxsum_dn(i,:)
                ent_up_sum(i,:) = minqcheckf*ent_up_sum(i,:)
                det_up_sum(i,:) = minqcheckf*det_up_sum(i,:)
                ent_dn_sum(i,:) = minqcheckf*ent_dn_sum(i,:)
                det_dn_sum(i,:) = minqcheckf*det_dn_sum(i,:)
                accuprec(i,:) = minqcheckf*accuprec(i,:)
                cmesum(i,:) = minqcheckf*cmesum(i,:)
            end if

            qcheckout(i) = minqcheckf
        end do
    end if

    if (flagqcheck == 2) then
        do i = 1, inncol
            
            minqcheckf = 1.0
            do k = nlev, 1, -1
                qcheckf = 1.0
                if (p(i,k) > qnegtop) then
                    qguess  = q(i,k)+qtend(i,k)*dtime
                    if( qguess < qmin .and. abs(qtend(i,k)) > 1e-15) then
                        qcheckf = (qmin*1.001-q(i,k))/dtime/qtend(i,k)
                    end if
                end if
                if (qcheckf < minqcheckf) then
                    minqcheckf = qcheckf
                end if
            end do

#ifdef SCMDIAG
    write(*,*) "yhyminqcheck:", minqcheckf
#endif

            do k = nlev, 1, -1
                if (p(i,k) <= qnegtop) then
                    qcheckf = 0.0
                else
                    qcheckf = minqcheckf
                end if
                qtend(i,k) = qcheckf * qtend(i,k)
                stend(i,k) = qcheckf * stend(i,k)
                qliqtend(i,k) = qcheckf * qliqtend(i,k)
                mcon(i,k) = qcheckf * mcon(i,k)
                precrate_out(i,k) = qcheckf * precrate_out(i,k)
            end do
        end do

    end if


!clean output.
    !do i=1, inncol
    !    if ( trigdp(i)<1 ) then
    !        mse_up(i,:) = 0._r8
    !    end if
    !end do

    outmb = massflxbasesum
    outtmp2d = qcheckout
    outmse = mse
    outmsesat= msesat
    outmseup = mse_up_mid
    outstend = stend
    outqtend = qtend
    outstendcond = stendcond
    outqtendcond = qtendcond
    outstendtranup = stendtran_up
    outqtendtranup = qtendtran_up
    outstendtrandn = stendtran_dn
    outqtendtrandn = qtendtran_dn
    outstendtrandn = 0._r8
    outqtendtrandn = 0._r8
    outstendevap = stendevap
    outqtendevap = qtendevap

!MZ Postprocessing below ==============    

!CAM MZ type output 1003
! convert detrainment from units of "1/m" to "1/mb".
! mass flux from kg/m2/s to mb/s
 do k= 1,nlev
   eu(:,k)   = ent_up_sum(:,k)*dz(:,k)/dp(:,k)*100.  !unit (kg/m2/s)/m * m/mb 
   du(:,k)   = det_up_sum(:,k)*dz(:,k)/dp(:,k)*100. 
   ed(:,k)   = ent_dn_sum(:,k)*dz(:,k)/dp(:,k)*100.
   dd(:,k)   = det_dn_sum(:,k)*dz(:,k)/dp(:,k)*100.
   massflxsum(:,k) = massflxsum(:,k)            
   massflxsum_dn(:,k) = massflxsum_dn(:,k)     
 end do

!MZ 180801
   massflxsum =massflxsum /(100._r8/gravit) !kg/m2/s to mb/s 
   massflxsum_dn =massflxsum_dn /(100._r8/gravit)
   eu = eu/(100._r8/gravit) !unit mb/s /mb
   du = du/(100._r8/gravit)
   ed = ed/(100._r8/gravit)
   dd = dd/(100._r8/gravit)

!MZ to put a limit on mass flux
! ----------------------------------   

   mu = massflxsum
   do i=1,lengath
      mumax(i) = 0
   end do
   do k=msg + 2,nlev
      do i=1,lengath
        mumax(i) = max(mumax(i), mu(i,k)/dp(i,k))
      end do
   end do

   do i=1,lengath
      !if (mumax(i) > 0._r8) then
        !mb(i) = min(mb(i),0.5_r8/(delt*mumax(i)))
      if (mumax(i) > 1.e-20_r8) then
        mb(i) = min(1._r8, dtime*mumax(i)  )   !! 0.5 limit for half level
        !mb(i) = min(0.5_r8, dtime*mumax(i)  )
        mb(i) = mb(i)/(dtime*mumax(i)) 
      else
        mb(i) = 0._r8
      endif
   end do
!!!!MZ
!!!   mb = 1.0

   do i=1,lengath
     massflxsum(i,:) =massflxsum(i,:) * mb(i)
     massflxsum_dn(i,:) =massflxsum_dn(i,:) * mb(i) 
     eu(i,:) = eu(i,:) * mb(i)
     du(i,:) = du(i,:) * mb(i)
     ed(i,:) = ed(i,:) * mb(i)
     dd(i,:) = dd(i,:) * mb(i)
     qtnd(i,:) = qtend(i,:) * mb(i)
     heat(i,:) = heat(i,:)  * mb(i)
     qliqtend(i,:) = qliqtend(i,:) * mb(i)
     mcon(i,:) = (massflxsum(i,:) + massflxsum_dn(i,:))* mb(i)
     dlf(i,:)  = qliqtend(i,:) * mb(i)  !kg/kg/s
     zdu(i,:)  = du(i,:)* mb(i)
     rprd(i,:) = precratesum(i,:) * mb(i)
     accuprec(i,:) = accuprec(i,:)* mb(i)
     cmesum(i,:) = cmesum(i,:)* mb(i)
   enddo
! ----------------------------------   

   do k=1,nlev 
    pflx(:,k+1) = accuprec(:,k)
   end do
   cme(:,:)  = cmesum(:,:)

   do i= 1,inncol
    do k=1,nlev
        if( massflxsum(i,k) < 1.e-10_r8)then
         ql(i,k) = 0._r8
        else
         ql(i,k) = qlsum(i,k)/massflxsum(i,k)
        endif
    enddo
   enddo
   do k=1,nlev
    mu(:,k)   = 0.5_r8*(massflxsum(:,k)+massflxsum(:,k+1))  
    md(:,k)   = 0.5_r8*(massflxsum_dn(:,k)+massflxsum_dn(:,k+1))  
   enddo

   dsubcld(:)= dsubcldmax(:) 
   cape(:)   = capemax(:)
   maxg(:)   = maxgmax(:)
   jctop(:)  = jctopmin(:)
   jcbot(:)  = jcbotmax(:)
   jt(:)     = jctop(:)

  if(plume_model == 'zyx2')then

   prec(:) = 0.0_r8
   do k = nlev,msg + 1,-1
      do i = 1,inncol
         !prec(i) = prec(i) - dp(i,k)* (qtend(i,k)+qliqtend(i,k))
         prec(i) = prec(i) - dp(i,k)* (qtnd(i,k)+qliqtend(i,k))
      end do
   end do
! obtain final precipitation rate in m/s.
   do i = 1,inncol
      prec(i) = rgrav*max(prec(i),0._r8)/ 1000._r8
   end do

! Compute reserved liquid (not yet in cldliq) for energy integrals.
! Treat rliq as flux out bottom, to be added back later.
   do k = 1, nlev
      do i = 1, inncol
         rliq(i) = rliq(i) + dlf(i,k)*dp(i,k)/gravit
      end do
   end do
   rliq(:inncol) = rliq(:inncol) /1000._r8

  endif ! 'zyx2' recalculation
!----------------

#ifdef SCMDIAG 
    write(*,"(a20,f20.10)") "dtime:", dtime
    write(*,"(a20,f20.10,a20,f20.10,a20,f20.10)") "lat:", lat, "psrf:", psrf
    write(*,"(a20,i4,a20,i4)") "uplaunch:", kuplaunch, " upbase:  ", kupbase, " uplcl:", kuplcl
    write(*,"(a20,i4,a20,i4)") "uptop:", kuptop
    write(*,"(a20,i4,a20,i4)") "trigdp:", trigdp, "trigsh:", trigsh
    write(*,"(a20,f20.10)") "zsrf:",zsrf 
    !write(*,"(a20,f20.10)") "bflsdilucape:", bfls_dilucape
    write(*,"(a20,50f20.10)") "dilucape:", dilucape(1,1:nplume_tot)
    write(*,"(a20,50f20.10)") "cwf:", cwf(1,1:nplume_tot)
    !write(*,"(a20,f20.10)") "dilucape_closure:", dilucape_closure
    !!write(*,"(a20,f20.10)") "capefc:", capefc
    !write(*,"(a20,f20.10)") "capeclm:", capeclm
    !write(*,"(a20,f20.10)") "mconv:", mconv
    write(*,"(a20,50f20.10)") "massflxbase_p:", massflxbase_p(1,1:nplume_tot)
    !write(*,"(a20,f20.10)") "massflxbase_cape:", massflxbase_cape
    !write(*,"(a20,f20.10)") "massflxbase_dcape:", massflxbase_dcape
    !write(*,"(a20,f20.10)") "massflxbase_clm:", massflxbase_clm
    !write(*,"(a20,f20.10)") "massflxbase_w:", massflxbase_w
    !write(*,"(a20,f20.10)") "massflxbase_mconv:", massflxbase_mconv
    !write(*,"(a20,f20.10)") "massflxbase:", massflxbase
    write(*,"(a20,f20.10)") "prec:", prec*3600*24*1000
    !write(*,"(a20,f20.10)") "surfprec:", surfprecsum*3600*24*1000
    write(*,"(a20,f20.10)") "minqcheckf:", minqcheckf

    !netcdf output
    call subcol_netcdf_putclm( "mse", nlev, mse(1,:), 1 )
    call subcol_netcdf_putclm( "dse", nlev, dse(1,:), 1 )
    call subcol_netcdf_putclm( "msesat", nlev, msesat(1,:), 1 )
    call subcol_netcdf_putclm( "z", nlev, z(1,:), 1 )
    call subcol_netcdf_putclm( "p", nlev, p(1,:), 1 )
    call subcol_netcdf_putclm( "rho", nlev, rho(1,:), 1 )

    call subcol_netcdf_putclm( "mseint", nlevp, mseint(1,:), 1 )
    call subcol_netcdf_putclm( "msesatint", nlevp, msesatint(1,:), 1 )

    call subcol_netcdf_putclm( "zint", nlevp, zint(1,:), 1 )
    call subcol_netcdf_putclm( "pint", nlevp, pint(1,:), 1 )
    call subcol_netcdf_putclm( "tint", nlevp, tint(1,:), 1 )
    call subcol_netcdf_putclm( "qint", nlevp, qint(1,:), 1 )
    call subcol_netcdf_putclm( "qsatint", nlevp, qsatint(1,:), 1 )

    call subcol_netcdf_putclm( "t", nlev, t(1,:), 1 )
    call subcol_netcdf_putclm( "q", nlev, q(1,:), 1 )
    call subcol_netcdf_putclm( "qsat", nlev, qsat(1,:), 1 )

    !call subcol_netcdf_putclm( "prec", prec(1), 1 )
    !call subcol_netcdf_putclm( "pmassflxbase", massflxbase_p(1), 1 )
    !call subcol_netcdf_putclm( "massflxbase_cape", massflxbase_cape(1), 1 )
    !call subcol_netcdf_putclm( "massflxbase_w", massflxbase_w(1), 1 )
    !call subcol_netcdf_putclm( "massflxbase_mconv", massflxbase_mconv(1), 1 )
    !call subcol_netcdf_putclm( "qcheck", 1, qcheckout(1), 1 )

    !tmp = kupbase-kuptop+1
    !call subcol_netcdf_putclm( "nconvlev", 1, tmp(1), 1 )
    !tmp = kuplaunch
    !call subcol_netcdf_putclm( "kuplaunch", 1, tmp(1), 1 )
    !tmp = kupbase
    !call subcol_netcdf_putclm( "kupbase", 1, tmp(1), 1 )
    !tmp = kuplcl
    !call subcol_netcdf_putclm( "kuplcl", 1, tmp(1), 1 )

#endif

    !-------------------------------------------------
    ! Haiyang test
     qliqtend = qliqtend * facdlf
    !-------------------------------------------------
1001 Continue     

i=1
if(i<0)then
              k=25
              write(*,"(A50/,A30/,5I10/,6(A10/,3(5E15.7/)) )") &
                  ' ZZZZZz in zyx2_conv at end ....', &
              'state%lchnk, pcols,pver,ncol',lchnk, ncol,nlev,ncol,lengath, &
              'q(1)',q_in(1:15,k),        &
              'mcon',mcon(1:15,k),             &
              'rprd',rprd(1:15,k), &
              'qtendq(1)',qtend(1:15,k),        &
              'qliqtendq(1)',qliqtend(1:15,k),&
              'cape',cape(1:15)
endif


if(i<0)then
   write(*,*)'-------------1001-----------'
   write(*,*)'-- after call zym_conv_tend line 2931'
   write(*,*)'dlf',dlf
   write(*,*)'qliqtend',qliqtend
   !write(*,*)'heat',heat
   !write(*,*)'qtnd',qtnd
   write(*,*)'stend(:ncol,:)',stend
   write(*,*)'qtend(:ncol,:)',qtend
   write(*,*)'mcon',mcon
   write(*,*)'pflx',pflx
   write(*,*)'cme',cme
   write(*,*)'zdu',zdu
   write(*,*)'rprd',rprd
   write(*,*)'mu',mu
   write(*,*)'md',md
   write(*,*)'eu',eu
   write(*,*)'du',du
   write(*,*)'ed',ed
   write(*,*)'dsubcld',dsubcld
   write(*,*)'cape',cape
   write(*,*)'capemax',capemax
   write(*,*)'jctop',jctop
   write(*,*)'jcbot',jcbot
   write(*,*)'jt',jt
   write(*,*)'prec',prec
   write(*,*)'rliq',rliq
   write(*,*)'-------------'
endif


end subroutine zyx2_conv_tend

! ==============================================================================
! calculate the launch processes
! ==============================================================================
subroutine cal_launchtocldbase( &
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
!input
        opt, zsrf, z, zint, p, pint, t, tint, q, qint, qsat, qsatint, &
        mse, mseint, msesat, msesatint, landfrac, lhflx, tpert_plume, qpert_plume, &
!output
        kuplaunch, kuplcl, mse_up, t_up, q_up, normassflx_up,  &
!in
        trig)
!------------------------------------------------------
!launch to LCL, no entrainment up, in-cloud properties
!------------------------------------------------------
!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    integer, intent(in) :: opt ! 1:LCL  2:launch point
    real(r8), dimension(ncol),  intent(in) :: zsrf     ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: z     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: zint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: p     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: pint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: t     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: tint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: q     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: qint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: qsat  ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: qsatint! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: mse   ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: mseint ! [J/kg]
    real(r8), dimension(ncol, nlev),  intent(in) :: msesat ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: msesatint ! [J/kg]
    real(r8), dimension(ncol), intent(in) :: landfrac    ! 
    real(r8), dimension(ncol), intent(in) :: lhflx       ! 
    real(r8), dimension(ncol), intent(in) :: tpert_plume    !
    real(r8), dimension(ncol), intent(in) :: qpert_plume    !

!output
    integer, dimension(ncol), intent(out) :: kuplaunch ! [1]
    integer, dimension(ncol), intent(out) :: kuplcl    ! [1]

    real(r8), dimension(ncol, nlevp), intent(out) :: mse_up ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(out) :: t_up ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(out) :: q_up ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(out) :: normassflx_up ! [kg/kg]

!input/output
    integer, dimension(ncol), intent(in) :: trig     ! [1]

!local
    integer :: i, k, stat
    real(r8) :: msemax, q_up_test, t_up_test
    real(r8) :: diffmse, dt, dq, buoy
    integer, dimension(ncol) :: kuplaunchmin  ! [1]
    integer, dimension(ncol) :: kuplaunchmax  ! [1]
    integer, dimension(ncol) :: kcbase  ! [1]

    !intialize output
    kuplaunch = 1
    kuplcl = 1
    kcbase = nlevp

    do i=1, ncol
        kuplaunchmin(i) = nlevp
        kuplaunchmax(i) = 1

        if ( trig(i) < 1 ) cycle
        do k=nlevp, 1, -1
!MZ 2018-08-04
            !if ( zint(i,k) >= max(zuplaunchlow, zsrf(i)) ) then
            if ( (zint(i,k)-zsrf(i)) >= zuplaunchlow ) then
                kuplaunchmin(i) = k
                exit
            end if
        end do
        

        do k=nlevp, 1, -1
!MZ 2018-08-04
           !if ( zint(i,k) >= max(zuplaunchtop, zsrf(i)) ) then
            if ( (zint(i,k)-zsrf(i)) >= zuplaunchtop ) then
                kuplaunchmax(i) = k
                exit
            end if
        end do
    end do

    kuplaunch = nlev
    do i=1, ncol
        if ( trig(i) < 1 ) cycle
        
        !find the maximun MSE level as cloud parcel launching point
        msemax = 0._r8
        do k=kuplaunchmin(i), kuplaunchmax(i), -1
            if ( mseint(i,k) >= msemax ) then
                msemax = mseint(i,k)
                kuplaunch(i) = k
            end if
        end do
        
        ! shallow plumes: cloud base at launch point
        if ( opt == 2 ) then
            kcbase(i) = kuplaunch(i)
            kuplcl(i) = kuplaunch(i)
        end if

        ! deep plumes: cloud base at LCL
        if ( opt == 1 ) then
            do k=kuplaunch(i)-1, 1, -1
                !mse_up(i,k) = mse_up(i,k+1)
                call cal_mse2tsat(mseint(i,kuplaunch(i)), tint(i,k), qsatint(i,k), msesatint(i,k), t_up_test)
                call cal_qsat(t_up_test, pint(i,k), q_up_test)
            
                if( qint(i,kuplaunch(i)) >= q_up_test ) then
                    kuplcl(i) = k
                    kcbase(i) = k
                    !t_up(i,k) = tint(i,k)
                    !q_up(i,k) = qsatint(i,k)
                    !mse_up(i,k) = cpair*t_up(i,k) + gravit*zint(i,k) + &
                    !    (latvap-(cpliq-cpwv)*(t_up(i,k)-273.15))*q_up(i,k)
                    exit
                !else
                !    t_up(i,k) = ( mse_up(i,k)-latvap*q_up(i,k)-gravit*zint(i,k)-(cpliq-cpwv)*273.15*q_up(i,k) ) &
                !        / (cpair-(cpliq-cpwv)*q_up(i,k))
                !    q_up(i,k) = q_up(i,k+1)
                end if
            end do 
        end if

        if ( opt == 1) then
            ! deep plumes: use saturated environmental air
            t_up(i,kcbase(i)) = tint(i,kcbase(i))
            q_up(i,kcbase(i)) = qsatint(i,kcbase(i)) 
            mse_up(i,kcbase(i)) = msesatint(i,kcbase(i)) 
        else
            ! shallow plumes: use tpert and qpert
            t_up(i,kcbase(i) ) = tint(i,kcbase(i) ) + tpert_plume(i)
            q_up(i,kcbase(i) ) = qint(i,kcbase(i) ) + qpert_plume(i)
            mse_up(i,kcbase(i) ) = mseint(i,kcbase(i)) + tpert_plume(i)*cpair + qpert_plume(i)*latvap
        end if
        
        ! cloud properties below cloud base
        normassflx_up(i,kcbase(i)) = 1.0
        do k=nlevp, kcbase(i)+1, -1
            if (zint(i,k) > zsrf(i)) then
            normassflx_up(i,k) = ( max( 0.0, (zint(i,k)-zsrf(i)) &
                /(zint(i,kuplcl(i))-zsrf(i)) ) )**0.5
            mse_up(i,k) = mse_up(i, kuplcl(i))
            q_up(i,k) = q_up(i, kuplcl(i))
            t_up(i,k) = ( mse_up(i,k) - gravit*zint(i,k) - (latvap+(cpliq-cpwv)*273.15)*q_up(i,k) )/ &
                (cpair-(cpliq-cpwv)*q_up(i,k))
        else
            normassflx_up(i,k) = 0.0
            mse_up(i,k) = mseint(i,k)
            q_up(i,k) = qint(i,k)
            t_up(i,k) = tint(i,k)
        end if

        end do
    end do  ! loop for icol

end subroutine cal_launchtocldbase


! ==============================================================================
! calculate updraft properties (new scheme of MZhang grouup)
! ==============================================================================
subroutine cal_mse_up( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        flag_plume, rho, rhoint, z, zint, dz, p, pint, t, tint, q, qint, qsat, qsatint, &
        mse, mseint, msesat, msesatint, kuplaunch, kupbase, &
!in/output
        ent_rate_dp_up, det_rate_dp_up, ent_rate_sh_up, det_rate_sh_up, &
        ent_org, det_org, ent_turb, det_turb, cldrad, & 
        bs_xc, w_up_init, &
        mse_up, t_up, q_up, qliq_up, qice_up, mseqi, condrate, rainrate, snowrate, precrate, &
        normassflx_up, w_up, w_up_mid, buoy, buoy_mid, kuptop, zuptop, &
        trig)

     use buoysort, only : cal_buoysort, cal_fracmix, cal_entdet

!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    integer, intent(in) :: flag_plume   ! 1: shallow, 2: deep
    real(r8), dimension(ncol, nlev),  intent(in) :: rho   ! [kg/m3]
    real(r8), dimension(ncol, nlevp), intent(in) :: rhoint   ! [kg/m3]
    real(r8), dimension(ncol, nlev),  intent(in) :: z     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: zint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: dz    ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: p     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: pint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: t     ! [K]
    real(r8), dimension(ncol, nlevp), intent(in) :: tint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: q     ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: qsat   ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qsatint! [kg/kg]
    real(r8), dimension(ncol, nlev),  intent(in) :: mse    ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in):: mseint ! [J/kg]
    real(r8), dimension(ncol, nlev), intent(in) :: msesat ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in):: msesatint ! [J/kg]
    integer , dimension(ncol), intent(in) :: kuplaunch ! [1]
    integer , dimension(ncol), intent(in) :: kupbase    ! [1]

    real(r8), dimension(ncol, nlev), intent(inout) :: ent_rate_dp_up ! [1]
    real(r8), dimension(ncol, nlev), intent(inout) :: det_rate_dp_up ! [1]
    real(r8), dimension(ncol, nlev), intent(inout) :: ent_rate_sh_up ! [1]
    real(r8), dimension(ncol, nlev), intent(inout) :: det_rate_sh_up ! [1]
    real(r8), dimension(ncol), intent(inout) :: w_up_init ! [1]
!output
    real(r8), dimension(ncol, nlevp), intent(out) :: w_up ! [J/kg]
    real(r8), dimension(ncol, nlev),  intent(out) :: w_up_mid ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(out) :: buoy ! [J/kg]
    real(r8), dimension(ncol, nlev),  intent(out) :: buoy_mid  ! [J/kg]

    integer , dimension(ncol), intent(out) :: kuptop
    real(r8), dimension(ncol), intent(out) :: zuptop  ! [m]
!input/output
    real(r8), dimension(ncol, nlevp), intent(inout) :: mse_up  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: t_up  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: q_up  ! 
    real(r8), dimension(ncol, nlevp), intent(inout) :: qliq_up  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: qice_up  ! [kg/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: mseqi    ! 
    real(r8), dimension(ncol, nlev ), intent(inout) :: condrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: rainrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: snowrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: precrate  ! [m2/kg]

    real(r8), dimension(ncol, nlevp), intent(inout) :: normassflx_up  ! [#]
    integer , dimension(ncol), intent(inout) :: trig     ! [1]
    real(r8), dimension(ncol, nlev ), intent(inout)  :: ent_org    ! organized entrainment rate (1/m)
    real(r8), dimension(ncol, nlev ), intent(inout)  :: det_org    ! organized detrainment rate (1/m)
    real(r8), dimension(ncol, nlev ), intent(inout)  :: ent_turb   ! turbulent entrainment rate (1/m)
    real(r8), dimension(ncol, nlev ), intent(inout)  :: det_turb   ! turbulent detrainment rate (1/m)
    real(r8), dimension(ncol, nlev ), intent(inout)  :: cldrad     ! cloud radius [m]
    real(r8), dimension(ncol, nlev ), intent(inout) :: bs_xc
!local
    real(r8), dimension(ncol, nlev )  :: frezrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev )  :: normassflx_up_mid  ! [#]
    real(r8), dimension(ncol, nlev )  :: cldsr      ! cloud size ratio: 2R/H [#]
    real(r8), dimension(ncol)  :: cldh              ! cloud height [m]
    real(r8), dimension(ncol)  :: cldradinit        ! initial cloud radius [m]
   
    real(r8) :: cldh_bak, ent_bak, det_bak
    integer  :: ktop_tmp
    real(r8) :: tmp_t_up, tmp_q_up, tmp_buoy, tmp_zuptop_max, tmp_crt

    real(r8) :: xc, tv, tv_up,  w2, qw, Fp, Fi, Ek
    real(r8) :: nom, denom, ent_rate, det_rate, tmp
    real(r8) :: ent1,ent2,det1,det2

    real(r8) :: bs_p0, bs_wue, bs_rle, bs_scaleh, bs_cridis, bs_thetal_e, bs_thetal_up

    integer :: i,j,k, iteration, itercldh, iminmse
    integer :: ngbuoy

    bs_p0 = 1000.e2_r8
    bs_rle = 0.1_r8

    !intialize output.
    qliq_up = 0.0
    qice_up = 0.0
    mseqi = 0.0
    frezrate = 0.0
    condrate = 0.0
    rainrate = 0.0
    snowrate = 0.0
    precrate = 0.0
    
    w_up = 0._r8
    w_up_mid = 0._r8
    buoy = 0._r8
    buoy_mid = 0._r8
    normassflx_up_mid = 0.0_r8
    cldrad = 0.0_r8
    cldradinit = 0.0
    cldsr  = 0.0
    cldh = 0.0_r8

    kuptop = nlev
    zuptop = 0._r8
    ngbuoy = 0

    ent_rate_dp_up = 0._r8
    det_rate_dp_up = 0._r8
    ent_rate_sh_up = 0._r8
    det_rate_sh_up = 0._r8
    ent_org = 0.0
    det_org = 0.0
    ent_turb = 0.0
    det_turb = 0.0

    bs_xc = 0._r8

    itercldh = 0
    iteration = 0

    ent_rate = 0.0
    det_rate = 0.0


    do i=1, ncol
        if ( trig(i) < 1 ) cycle

        ! firstly, get the undiluted cloud top height, and initialize cldh
!MZ 7/16/18
        do k=kupbase(i)-1, 1, -1
       !! do k=kupbase(i)-1, limcnv-1, -1
            call cal_mse2tsat( mse_up(i,kupbase(i)), tint(i,k), &
                qsatint(i,k), msesatint(i,k), tmp_t_up )
            call cal_qsat(tmp_t_up, pint(i,k), tmp_q_up)

            tv = tint(i,k)*(1+tveps*qint(i,k))
            tv_up = tmp_t_up*(1+tveps*tmp_q_up )
            tmp_buoy = gravit*(tv_up-tv)/tv 
            if ( tmp_buoy <= 0. ) then
                tmp_zuptop_max = zint(i,k)
                exit
            end if

        end do
        
        ktop_tmp = max(1, k)  
        tmp_zuptop_max = zint(i, ktop_tmp)        
        
        ! initialize cloud base properties
        k = kupbase(i)
        tv = tint(i,k)*(1 + tveps*qint(i,k) )
        tv_up = t_up(i,k)*(1 + tveps*q_up(i,k) )
        buoy(i,k)  = gravit*(tv_up-tv)/tv 
        w_up(i,k) = w_up_init(i)
        normassflx_up(i,k) = 1.0
        if ( fixcldrad0 > 0 ) then
            cldradinit(i) = fixcldrad0
        else
            cldradinit(i) = trig_c2 * (trig_eps0**(-0.4)) * (w_up(i,k)**1.2)
        end if
        iminmse = minloc(mse(i,:), 1)
        if (fixcldsr > 0) then
            cldh(i) = 2 * cldradinit(i) / fixcldsr
        else
            cldh(i) = 0.5 * ( tmp_zuptop_max - zint(i,kupbase(i)) )
        end if
        cldh_bak = cldh(i)


!MZ 7/16/18        

        do itercldh = 1, cldhiteration, 1

!MZ 7/16/18 
            do k=kupbase(i)-1, 1, -1
            !!do k=kupbase(i)-1, limcnv-1, -1
            
                do iteration = 1, maxiteration, 1
                    
                    ! w, buoyancy, and mass flux at mid-layer
                    if (iteration == 1) then
                        w_up_mid(i,k) = w_up(i,k+1)
                        buoy_mid(i,k) = buoy(i,k+1)
                        normassflx_up_mid(i,k) = normassflx_up(i,k+1)
                    else
                        w_up_mid(i,k) = 0.5 * ( w_up(i,k)+w_up(i,k+1) )
                        buoy_mid(i,k) = 0.5 * ( buoy(i,k)+buoy(i,k+1) )    
                    end if                        

                    ! cloud radius, cloud size ratio
                    cldrad(i,k) = cldradinit(i)*sqrt(normassflx_up_mid(i,k)* &
                        rhoint(i,kupbase(i))*w_up(i,kupbase(i))/rho(i,k)/max(wupmin,w_up_mid(i,k)) )
                    cldsr(i,k) = 2*cldrad(i,k) / cldh(i)

                    ! organized entrainment/detrainment
                    if (flagorgent == 1) then
                        tmp = orgent_beta0 * cldsr(i,k)/sqrt(1+cldsr(i,k)*cldsr(i,k)) * &
                            sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        !if (iteration > 1 .and. rhoint(i,k)*buoy(i,k) < rhoint(i,k+1)*buoy(i,k+1)) then
                        !if (iteration > 1 .and. buoy(i,k) < buoy(i,k+1)) then
                        !if ( buoy_mid(i,k) < 0 ) then
                        if ( k < iminmse ) then
                            ent_org(i,k) = 0.0
                            det_org(i,k) = tmp
                        else
                            ent_org(i,k) = tmp
                            det_org(i,k) = 0.0                        
                        end if
                    end if
                    if (flagorgent == 5) then
                        tmp = orgent_beta0 * cldsr(i,k)/sqrt(1+cldsr(i,k)*cldsr(i,k)) * &
                            sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        if ( buoy_mid(i,k) <= 0 ) then
                            ent_org(i,k) = 0.0
                            det_org(i,k) = tmp
                        else
                            ent_org(i,k) = tmp
                            det_org(i,k) = 0.0                        
                        end if
                    end if
                    if (flagorgent == 6) then
                        if (abs(buoy_mid(i,k)) > 1e-15) then
                            tmp = orgent_beta0 * cldsr(i,k)/sqrt(1+cldsr(i,k)*cldsr(i,k)) * &
                                sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        else
                            tmp = 0.0
                        end if
                        if ( buoy_mid(i,k) <= 0 ) then
                            ent_org(i,k) = 0. !0.0004
                            det_org(i,k) = tmp * org_enhance &
                                * (max(0.0, pint(i,kupbase(i))-p(i,k)) &
                                / ( max(pint(i,kupbase(i))-p(i,ktop_tmp), 0.0) &
                                + 10.0))**org_shape
                        else
                            ent_org(i,k) = tmp * org_enhance & 
                                * (max(0.0, p(i,k)-p(i,ktop_tmp)) &
                                /( max(pint(i,kupbase(i))-p(i,ktop_tmp), 0.0) &
                                + 10.0))**org_shape
                            det_org(i,k) = 0.0                        
                        end if

                    end if
                    if (flagorgent == 8) then
                        if (abs(buoy_mid(i,k)) > 1e-15) then
                            tmp = orgent_beta0 * cldsr(i,k)/sqrt(1+cldsr(i,k)*cldsr(i,k)) * &
                                sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        else
                            tmp = 0.0
                        end if
                        if ( rhoint(i,k)*buoy(i,k) - rhoint(i,k+1)*buoy(i,k+1) <= 0 ) then
                            ent_org(i,k) = 0. !0.0004
                            det_org(i,k) = tmp * org_enhance &
                                * (max(0.0, pint(i,kupbase(i))-p(i,k)) &
                                / ( max(pint(i,kupbase(i))-p(i,ktop_tmp), 0.0) &
                                + 10.0))**org_shape
                        else
                            ent_org(i,k) = tmp * org_enhance & 
                                * (max(0.0, p(i,k)-p(i,ktop_tmp)) &
                                /( max(pint(i,kupbase(i))-p(i,ktop_tmp), 0.0) &
                                + 10.0))**org_shape
                            det_org(i,k) = 0.0                        
                        end if
                    end if
                    if (flagorgent == 7) then
                        if (abs(buoy_mid(i,k)) > 1e-15) then
                            tmp = orgent_beta0 * cldsr(i,k)/sqrt(1+cldsr(i,k)*cldsr(i,k)) * &
                                sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        else
                            tmp = 0.0
                        end if
                        if (flag_plume == 1) then
                            ! shallow
                            tmp_crt = rhoint(i,k)*buoy(i,k) - rhoint(i,k+1)*buoy(i,k+1)
                            !if ( tmp_crt < 0.0 .or. pint(i, kupbase(i)) - p(i,k) >= 20000.0 ) then
                            !!if ( tmp_crt < 0.0 ) then
                            !    ent_org(i,k) = 0.0
                            !    !det_org(i,k) = tmp * ((w_up_init(i)/2.0)) &
                            !    !    * exp(-(pint(i,kupbase(i)) - p(i,k)) / 20000.0)
                            !    det_org(i,k) = tmp * 0.25 * exp(-(p(i,k) - 20000.0) / 40000.0)
                            !else
                            !    det_org(i,k) = 0.0
                            !    !ent_org(i,k) = tmp * ((w_up_init(i)/2.0)) &
                            !    !    * exp(-(pint(i,kupbase(i)) - p(i,k)) / 20000.0)
                            !    ent_org(i,k) = tmp * 0.5 * exp(-(pint(i,kupbase(i)) - p(i,k)) / 20000.0)
                            !end if
                            !!org_enhance = 0.5 * (w_up_init(i)/2.0) ** (2.0)
                            !!org_shape = 2.0
                        else
                            ! deep 
                            tmp_crt = buoy_mid(i,k)
                            !if (tmp_crt < 0.0) then
                            !    ent_org(i,k) = 0.0
                            !    det_org(i,k) = tmp * 0.5 * exp(-(p(i,k) - 20000.0)/20000.0)
                            !else
                            !    det_org(i,k) = 0.0
                            !    ent_org(i,k) = tmp * 0.25 * exp(-(pint(i,kupbase(i)) - p(i,k) )/40000.0)
                            !end if

                            !org_enhance = 0.25
                            !org_shape = -2
                        end if

                        if ( tmp_crt < 0.0 ) then
                            ent_org(i,k) = 0. !0.0004
                            det_org(i,k) = tmp * org_enhance &
                                * (max(0.0, pint(i,kupbase(i))-p(i,k)) &
                                /( max(pint(i,kupbase(i))-p(i,ktop_tmp), 0.0) + 10.0))**org_shape
                        else
                            ent_org(i,k) = tmp * org_enhance & 
                                * (max(0.0, p(i,k)-p(i,ktop_tmp)) &
                                /( max(pint(i,kupbase(i))-p(i,ktop_tmp), 0.0) + 10.0))**org_shape
                            det_org(i,k) = 0.0                        
                        end if
                    end if

                    if (flagorgent == 2) then
                        tmp = 0.8*sqrt( cldsr(i,k)*cldsr(i,k) + cldsr(i,k)*sqrt(cldsr(i,k)*cldsr(i,k)+6) )
                        tmp = tmp*sqrt(abs(cos(3.14159265*(zint(i,k)-zint(i,kupbase(i)))/cldh(i))))
                        tmp = tmp*sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        if ( zint(i,k)-zint(i,kupbase(i)) > cldh(i)*0.5 ) then
                            ent_org(i,k) = 0.0
                            det_org(i,k) = tmp
                        else
                            ent_org(i,k) = tmp
                            det_org(i,k) = 0.0                        
                        end if
                    end if
                    if (flagorgent == 3) then
                        tmp = orgent_beta0 * cldsr(i,k)/sqrt(1+cldsr(i,k)*cldsr(i,k)) * &
                            sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        ent_org(i,k) = tmp 
                        det_org(i,k) = 0.0
                    end if
                    if (flagorgent == 4) then
                        tmp = 0.8*sqrt( cldsr(i,k)*cldsr(i,k) + cldsr(i,k)*sqrt(cldsr(i,k)*cldsr(i,k)+6) )
                        ! tmp = tmp*sqrt(abs(cos(3.14159265*(zint(i,k)-zint(i,kupbase(i)))/cldh(i))))
                        tmp = tmp*sqrt(abs(buoy_mid(i,k))*cldh(i)) / ( 2*cldrad(i,k)*max(wupmin,w_up_mid(i,k)) )
                        ent_org(i,k) = tmp
                        det_org(i,k) = 0.0
                    end if
                    ent_org(i,k) = min(max_ent_rate, max(0.0, ent_org(i,k)))
                    det_org(i,k) = min(max_det_rate, max(0.0, det_org(i,k)))

                    ! treat organized ent/det as deep plumes
                    ent_rate_dp_up(i,k) = ent_org(i,k)
                    det_rate_dp_up(i,k) = det_org(i,k)

                    
                    ! turbulent entrainment/detrainment
                    bs_scaleh = tmp_zuptop_max-zint(i, kupbase(i) ) 
                    bs_cridis = bs_rle*bs_scaleh
                    bs_thetal_e = t(i,k)*( bs_p0/p(i,k) )**(rair/cpair)
                    
                    if (flagbuoysort == 1) then
                        if (iteration == 1 .or. flagturbent == 1) then
                            bs_wue = w_up(i,k+1)
                            bs_thetal_up = ( t_up(i,k+1)-latvap*qliq_up(i,k+1)/cpair- &
                                (latice+latvap)*qice_up(i,k+1)/cpair ) * ( bs_p0/pint(i,k+1) )**(rair/cpair)
                            !bs_thetal_up =  t_up(i,k+1)  * ( bs_p0/pint(i,k+1) )**(rair/cpair)

                            call cal_buoysort(flagbspdf, bs_cridis, z(i,k), p(i,k), rho(i,k), &
                                bs_thetal_e, q(i,k), bs_thetal_up, q_up(i,k+1)+qliq_up(i,k+1)+qice_up(i,k+1), &
                                bs_wue, bs_xc(i,k), ent_turb(i,k), det_turb(i,k) )
                            !call cal_buoysort(flagbspdf, bs_cridis, z(i,k), p(i,k), rho(i,k), &
                            !    bs_thetal_e, q(i,k), bs_thetal_up, q_up(i,k+1), &
                            !    bs_wue, bs_xc(i,k), ent_turb(i,k), det_turb(i,k) )
                        else
                            bs_wue = 0.5*(w_up(i,k+1) + w_up(i,k))
                            bs_thetal_up = 0.5*( ( t_up(i,k+1)-latvap*qliq_up(i,k+1)/cpair- &
                                (latice+latvap)*qice_up(i,k+1)/cpair ) * ( bs_p0/pint(i,k+1) )**(rair/cpair) &
                                + ( t_up(i,k)-latvap*qliq_up(i,k)/cpair- &
                                (latice+latvap)*qice_up(i,k)/cpair ) * ( bs_p0/pint(i,k) )**(rair/cpair) )

                            call cal_buoysort(flagbspdf, bs_cridis, z(i,k), p(i,k), rho(i,k), &
                                bs_thetal_e, q(i,k), bs_thetal_up, 0.5*( q_up(i,k)+qliq_up(i,k)+qice_up(i,k) + &
                                q_up(i,k+1)+qliq_up(i,k+1)+qice_up(i,k+1) ), &
                                bs_wue, bs_xc(i,k), ent_turb(i,k), det_turb(i,k) )
                        end if
                    
                        ! new PDF for xc 
                        if (flagbspdf == 2) then
                            ent_turb(i,k) = ent_turb(i,k) * ratio_ent_rad/cldrad(i,k)
                            det_turb(i,k) = det_turb(i,k) * ratio_ent_rad/cldrad(i,k)
                        end if
                    end if  ! old buoysort code from CAM UW

                    if (flagbuoysort == 2) then  ! new buoysort code
                        call cal_fracmix(t(i,k), q(i,k), t_up(i,k+1), q_up(i,k+1)+qliq_up(i,k+1)+qice_up(i,k+1), &
                            p(i,k), w_up(i,k+1), bs_cridis, bs_xc(i,k))
                        !write(*,*) "fracmix:"
                        !write(*,"(8E10.3)") t(i,k), q(i,k), t_up(i,k+1), q_up(i,k+1)+qliq_up(i,k+1)+qice_up(i,k+1), &
                        !   p(i,k), w_up(i,k+1), bs_cridis, bs_xc(i,k)

                        call cal_entdet(flagbspdf, bs_xc(i,k), ent_turb(i,k), det_turb(i,k))
                        ent_turb(i,k) = ent_turb(i,k) * ratio_ent_rad/cldrad(i,k)
                        det_turb(i,k) = det_turb(i,k) * ratio_ent_rad/cldrad(i,k)
                    end if
                    
                    ! enhanced turbulent ent/det by w
                    if (turb_enhance > 0) then
                        ent_turb(i,k) = ent_turb(i,k) * turb_enhance
                        det_turb(i,k) = det_turb(i,k) * turb_enhance
                        !ent_turb(i,k) = ent_turb(i,k) * turb_enhance/max(wupmin,w_up_mid(i,k))
                        !det_turb(i,k) = det_turb(i,k) * turb_enhance/max(wupmin,w_up_mid(i,k))
                    end if

                    ent_turb(i,k) = min(max_ent_rate, max(0.0, ent_turb(i,k)))
                    det_turb(i,k) = min(max_det_rate, max(0.0, det_turb(i,k)))
                    
                    ! treat turbulent ent/det as shallow plumes
                    ent_rate_sh_up(i,k) = ent_turb(i,k)
                    det_rate_sh_up(i,k) = det_turb(i,k)

                    ! total entrainment and detrainment
                    if (flagtotent == 1) then  ! deep (org) only
                        ent_rate = max( 0.0, min( max_ent_rate, &
                            ent_rate_dp_up(i,k) ) )
                        det_rate = max( 0.0, min( max_det_rate, &
                            det_rate_dp_up(i,k) ) ) 
                    end if
                    if (flagtotent == 2) then  ! shallow (turb) only
                        ent_rate = max( 0.0, min( max_ent_rate, &
                            ent_rate_sh_up(i,k) ) )
                        det_rate = max( 0.0, min( max_det_rate, &
                            det_rate_sh_up(i,k) ) ) 
                    end if
                    if (flagtotent == 3) then  ! deep (i.e. organized) + shallow (i.e. turbulence)
                        ent_rate = max( 0.0, min( max_ent_rate, &
                            ent_rate_dp_up(i,k) + ent_rate_sh_up(i,k) ) )
                        det_rate = max( 0.0, min( max_det_rate, &
                            det_rate_dp_up(i,k) + det_rate_sh_up(i,k) ) ) 
                    end if

                    ! final output from iteration
                    if (flagmeaniter == 1) then
                        if (iteration == maxiteration-2) then
                            ent1 = ent_rate
                            det1 = det_rate
                        end if
                        if (iteration == maxiteration-1) then
                            ent2 = ent_rate
                            det2 = det_rate
                        end if
                        if (iteration == maxiteration) then
                            ent_rate = (ent1+ent2)/2.0
                            det_rate = (det1+det2)/2.0
                        end if
                    end if
                    
                    ! inner iteration loop 
                    if (iteration > 1) then
                        if (abs(ent_bak-ent_rate)<1e-5 .and. abs(det_bak-det_rate)<1e-5) then
                            ent_rate = ent_bak
                            det_rate = det_bak
                            exit
                        end if
                    end if
                    ent_bak = ent_rate
                    det_bak = det_rate

                    ! get w**2(i,k)
                    w2 = w_up(i,k+1)*w_up(i,k+1) + 2*dz(i,k)*(orgent_a*buoy_mid(i,k) / &
                        (1+cldsr(i,k)*cldsr(i,k)) - ent_rate*w_up_mid(i,k)*w_up_mid(i,k) )
                    w_up(i,k)  = sqrt(max(w2, 1e-15))

                    ! alias of nominater and denominater
                    nom   = 1.0/dz(i,k) - 0.5*ent_rate
                    denom = max(1e-15, 1.0/dz(i,k) + 0.5*ent_rate)


                    ! normalized mass flux
                    normassflx_up(i,k) = normassflx_up(i,k+1) &
                        * exp( (ent_rate-det_rate)*dz(i,k) )
                    if (abs(normassflx_up(i,k)-normassflx_up(i,k+1))<1e-6) then
                        normassflx_up_mid(i,k) = 0.5*(normassflx_up(i,k)+normassflx_up(i,k+1))
                    else
                        normassflx_up_mid(i,k) = (normassflx_up(i,k)-normassflx_up(i,k+1)) / &
                            (log(normassflx_up(i,k))-log(normassflx_up(i,k+1)))
                    end if

                    ! moisture static energy
                    mse_up(i,k) =  1./denom*( &
                          ent_rate * mse(i,k) &
                        + nom * mse_up(i,k+1) &
                        + mseqi(i,k) / max(1e-15, normassflx_up_mid(i,k)) )
                    
                    if (ctopflag == 3) then
                        mse_up(i,k) = max( mseint(i,k), mse_up(i,k) )
                    end if
                    
                    ! in-cloud temperature and moisture
                    ! ---- method 1: Taylor expanding ----
                    if (mse2tsatflag == 1) then
                        call cal_mse2tsat(mse_up(i,k), tint(i,k), &
                            qsatint(i,k), msesatint(i,k), t_up(i,k) )
                        call cal_qsat(t_up(i,k), pint(i,k), q_up(i,k))
                    end if
                    ! ---- method 2: bi-section ----
                    if (mse2tsatflag == 2) then
                        call mse2tsat( mse_up(i,k), zint(i,k), pint(i,k), t_up(i,k), q_up(i,k) )
                    end if

                    ! condensation
                    condrate(i,k) = normassflx_up_mid(i,k)/rho(i,k) *( &
                          ent_rate * q(i,k)   &
                        + nom   * q_up(i,k+1) &
                        - denom * q_up(i,k)   )

                    ! negative condensation adjustment
                    if (condrate(i,k) < 0) then
                        condrate(i,k) = 0.0
                        q_up(i,k) = 1.0/denom * ( &
                              ent_rate * q(i,k) &
                            + nom * q_up(i,k+1) )
                        t_up(i,k) = ( mse_up(i,k) - gravit*zint(i,k) &
                            - (latvap+(cpliq-cpwv)*273.15)*q_up(i,k) ) &
                            / max(1e-15, cpair-(cpliq-cpwv)*q_up(i,k))
                    end if

                    ! phase conversion rate
                    Fp = max(0.0, 1.0 - exp(-(z(i,k) - zint(i,kupbase(i)) - rain_z0)/rain_zp) )
                    Fi = 0.0
                    if (cloud_t1 < t_up(i,k) .and. t_up(i,k) < cloud_t2) then
                        Fi = (cloud_t2-t_up(i,k)) / (cloud_t2-cloud_t1)
                    else if (t_up(i,k) <= cloud_t1) then
                        Fi = 1.0
                    end if

                    frezrate(i,k) = Fi * condrate(i,k)
                    rainrate(i,k) = (1.0-Fi) * Fp * condrate(i,k)
                    snowrate(i,k) = Fi * Fp * condrate(i,k)
                    precrate(i,k) = rainrate(i,k) + snowrate(i,k)

                    ! in-cloud liquid and ice water
                    qliq_up(i,k) =  1.0/denom*( &
                          nom * qliq_up(i,k+1) &
                        + rho(i,k)*(condrate(i,k)-frezrate(i,k)-rainrate(i,k)) / &
                            max(1e-15, normassflx_up_mid(i,k)) ) 
                    qice_up(i,k) =  1./denom*( &
                          nom * qice_up(i,k+1) &
                        + rho(i,k)*(frezrate(i,k)-snowrate(i,k)) / &
                            max(1e-15, normassflx_up_mid(i,k)) )

                    ! contribution of freezing to moisture static energy
                    if (iteration < maxiteration .and. mseqiflag > 0) then
                        mseqi(i,k) = latice*rho(i,k)*frezrate(i,k)
                    end if

                    ! buoyancy at upper interfacial layer
                    tv = tint(i,k)*(1+tveps*qint(i,k))
                    tv_up = t_up(i,k)*( 1+tveps*q_up(i,k) )
                    
                    if (buoyflag == 1) then
                        buoy(i,k) = gravit* ( (tv_up-tv)/tv ) 
                    end if
                    if (buoyflag == 2) then
                        buoy(i,k) = gravit* ( (tv_up-tv)/tv - qliq_up(i,k) - qice_up(i,k) ) 
                    end if

                end do   ! loop of iteration for Qi
                
                if (ctopflag == 1) then 
                    if (buoy(i,k) < 0.0 .or. w_up(i,k) < wupmin) then
                        exit
                    end if
                end if
                
                if (ctopflag >= 2) then
                    if (w_up(i,k) < wupmin) then
                        exit
                    end if
                end if

                kuptop(i) = k  !!MZMZ

            end do  ! loop for levels
          
!MZ
            ktop_tmp = max(1, k)    ! update for calculating the shapes of ent_org and det_org
            !!ktop_tmp = max(limcnv-1, k)    ! update for calculating the shapes of ent_org and det_org

            if (fixcldsr < 0) then
                if (k == kupbase(i)-1) then
                    cldh(i) = 0.0
                    exit
                else
                    cldh(i) = max( 1.0, min( tmp_zuptop_max, zint(i,k+1) ) - zint(i,kupbase(i)) )
                    
                    if (abs(cldh(i)-cldh_bak) < 1.0) then
                        exit
                    else
                        cldh_bak = cldh(i)
                    end if
                end if
            end if

        end do   ! iteration for cloud height

!MZ 7/16/18        
        if (k>=1) then
        !!if (k>= limcnv) then
            mse_up(i,1:k) = mseint(i,1:k)
            t_up(i,1:k) = tint(i,1:k)
            q_up(i,1:k) = qint(i,1:k)
            qliq_up(i,1:k) = 0._r8
            qice_up(i,1:k) = 0._r8
            w_up(i,1:k) = 0._r8
            buoy(i,1:k) = 0._r8
            det_rate_dp_up(i,1:k) = 0._r8
            det_rate_sh_up(i,1:k) = 0._r8
            det_org(i,1:k) = 0.0
            det_turb(i,1:k) = 0.0
            ent_rate_dp_up(i,1:k) = 0.0
            ent_rate_sh_up(i,1:k) = 0.0
            det_org(i,1:k) = 0.0
            det_turb(i,1:k) = 0.0
            normassflx_up(i,1:k) = 0._r8

            condrate(i,1:k) = 0._r8
            rainrate(i,1:k) = 0._r8
            snowrate(i,1:k) = 0._r8
            precrate(i,1:k) = 0._r8
            mseqi(i,1:k) = 0._r8
            qliq_up(i,1:k) = 0._r8
            qice_up(i,1:k) = 0._r8


            kuptop(i) = k+1
            if ( kuptop(i) /= nlev ) then
                zuptop(i) = zint( i, kuptop(i) )
            end if

!not penetrating more than one level
            if ( k == kupbase(i)-1 ) then
                trig(i) = -11
            end if
        else
!MZ?
            k = 1
            !!k = limcnv
            mse_up(i,k) = mseint(i,k)
            t_up(i,k) = tint(i,k)
            q_up(i,k) = qint(i,k)
            qliq_up(i,k) = 0._r8
            qice_up(i,k) = 0._r8
            w_up(i,k) = 0._r8
            buoy(i,k) = 0._r8
            det_rate_dp_up(i,k) = 0._r8
            det_rate_sh_up(i,k) = 0._r8
            det_org(i,k) = 0.0
            det_turb(i,k) = 0.0
            ent_rate_dp_up(i,k) = 0.0
            ent_rate_sh_up(i,k) = 0.0
            det_org(i,k) = 0.0
            det_turb(i,k) = 0.0
            normassflx_up(i,k) = 0._r8

            condrate(i,k) = 0._r8
            rainrate(i,k) = 0._r8
            snowrate(i,k) = 0._r8
            precrate(i,k) = 0._r8
            mseqi(i,k) = 0._r8
            qliq_up(i,k) = 0._r8
            qice_up(i,k) = 0._r8

!MZ
            kuptop(i) = 2
            !!kuptop(i) = limcnv

            if ( kuptop(i) /= nlev ) then
                zuptop(i) = zint( i, kuptop(i) )
            end if

        end if

    end do

end subroutine cal_mse_up

! ==============================================================================
! calculate updraft properties (old scheme from CS2010 for GRE and NSJ)
! ==============================================================================
subroutine cal_mse_up_old( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        ent_opt, rho, z, zint, dz, p, pint, t, tint, q, qint, qsat, qsatint, &
        mse, mseint, msesat, msesatint, kuplaunch, kupbase, &
!in/output
        ent_rate_dp_up, det_rate_dp_up, ent_rate_sh_up, det_rate_sh_up, bs_xc, w_up_init, &
        mse_up, t_up, q_up, qliq_up, qice_up, mseqi, condrate, rainrate, snowrate, precrate, &
        normassflx_up, w_up, w_up_mid, buoy, buoy_mid, kuptop, zuptop, &
        trig)

     use buoysort, only : cal_buoysort

!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    integer, intent(in) :: ent_opt
    real(r8), dimension(ncol, nlev),  intent(in) :: rho   ! [kg/m3]
    real(r8), dimension(ncol, nlev),  intent(in) :: z     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: zint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: dz    ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: p     ! [m]
    real(r8), dimension(ncol, nlevp), intent(in) :: pint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: t     ! [K]
    real(r8), dimension(ncol, nlevp), intent(in) :: tint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: q     ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qint  ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: qsat   ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qsatint! [kg/kg]
    real(r8), dimension(ncol, nlev),  intent(in) :: mse    ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in):: mseint ! [J/kg]
    real(r8), dimension(ncol, nlev), intent(in) :: msesat ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in):: msesatint ! [J/kg]
    integer , dimension(ncol), intent(in) :: kuplaunch ! [1]
    integer , dimension(ncol), intent(in) :: kupbase    ! [1]

    real(r8), dimension(ncol, nlev), intent(inout) :: ent_rate_dp_up ! [1]
    real(r8), dimension(ncol, nlev), intent(inout) :: det_rate_dp_up ! [1]
    real(r8), dimension(ncol, nlev), intent(inout) :: ent_rate_sh_up ! [1]
    real(r8), dimension(ncol, nlev), intent(inout) :: det_rate_sh_up ! [1]
    real(r8), dimension(ncol), intent(inout) :: w_up_init ! [1]
!output
    real(r8), dimension(ncol, nlevp), intent(out) :: w_up ! [J/kg]
    real(r8), dimension(ncol, nlev),  intent(out) :: w_up_mid ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(out) :: buoy ! [J/kg]
    real(r8), dimension(ncol, nlev),  intent(out) :: buoy_mid  ! [J/kg]

    integer , dimension(ncol), intent(out) :: kuptop
    real(r8), dimension(ncol), intent(out) :: zuptop  ! [m]
!input/output
    real(r8), dimension(ncol, nlevp), intent(inout) :: mse_up  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: t_up  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: q_up  ! 
    real(r8), dimension(ncol, nlevp), intent(inout) :: qliq_up  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: qice_up  ! [kg/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: mseqi    ! 
    real(r8), dimension(ncol, nlev ), intent(inout) :: condrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: rainrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: snowrate  ! [m2/kg]
    real(r8), dimension(ncol, nlev ), intent(inout) :: precrate  ! [m2/kg]

    real(r8), dimension(ncol, nlevp), intent(inout) :: normassflx_up  ! [J/kg]
    integer , dimension(ncol), intent(inout) :: trig     ! [1]
!local
    real(r8), dimension(ncol, nlev )  :: frezrate  ! [m2/kg]

    real(r8), dimension(ncol, nlevp)  :: ent_rate_dp_up_int ! [1]
    real(r8), dimension(ncol, nlevp)  :: ent_rate_sh_up_int ! [1]
    real(r8), dimension(ncol, nlevp)  :: det_rate_sh_up_int ! [1]

    real(r8) :: tmp_t_up, tmp_q_up, tmp_buoy, tmp_zuptop_max

    real(r8) :: tv, tv_up,  w2, qw, Fp, Fi, Ek, tmp
    real(r8) :: nom, denom, ent_rate, det_rate, massflxmid

    real(r8) :: bs_p0, bs_wue, bs_rle, bs_scaleh, bs_cridis, bs_thetalint, bs_thetal_up
    real(r8), dimension(ncol, nlev ), intent(inout) :: bs_xc

    integer :: i,j,k, iteration
    integer :: ngbuoy

    bs_p0 = 1000.e2_r8
    bs_rle = 0.1_r8

    !intialize output.
    qliq_up = 0.0
    qice_up = 0.0
    mseqi = 0.0
    frezrate = 0.0
    condrate = 0.0
    rainrate = 0.0
    snowrate = 0.0
    precrate = 0.0
    
    w_up = 0._r8
    w_up_mid = 0._r8
    buoy = 0._r8
    buoy_mid = 0._r8
    kuptop = nlev
    zuptop = 0._r8
    ngbuoy = 0

    ent_rate_dp_up_int = 0._r8
    det_rate_dp_up = 0._r8
    ent_rate_sh_up = 0._r8
    det_rate_sh_up = 0._r8
    ent_rate_sh_up_int = 0._r8
    det_rate_sh_up_int = 0._r8
    bs_xc = 0._r8

! ------------------------
! cloud base diagram
! ------------------------  kupbase-1
!
!        ----------         kupbase-1
!
! ------------------------  kupbase (*)
!
!        ----------         kupbase
!
! ------------------------  kupbase+1
!
!            ...
!
! ------------------------  nlev
!
!        ----------         nlev
!
! ------------------------  nlevp

    do i=1, ncol
        if ( trig(i) < 1 ) cycle

        !do k=nlevp, kupbase(i)+1, -1
            !tv = tint(i,k)*(1+tveps*qint(i,k))
            !tv_up = t_up(i,k)*(1 + tveps*q_up(i,k) )
            !buoy(i,k) = gravit*(tv_up-tv)/tv
            !if ( k<=nlev ) then
                !buoy_mid(i,k) = 0.5*( buoy(i,k)+buoy(i,k+1) )
            !end if
        !end do

        do k=kupbase(i)-1, 1, -1
            call cal_mse2tsat( mse_up(i,kupbase(i)), tint(i,k), &
                qsatint(i,k), msesatint(i,k), tmp_t_up )
            call cal_qsat(tmp_t_up, pint(i,k), tmp_q_up)

            tv = tint(i,k)*(1+tveps*qint(i,k))
            tv_up = tmp_t_up*(1+tveps*tmp_q_up )
            tmp_buoy = gravit*(tv_up-tv)/tv 
            if ( tmp_buoy <= 0. ) then
                tmp_zuptop_max = zint(i,k)
                exit
            end if
        end do
        if ( k<1 ) tmp_zuptop_max = zint(i, 1)

        k = kupbase(i)

        tv = tint(i,k)*(1 + tveps*qint(i,k) )
        tv_up = t_up(i,k)*(1 + tveps*q_up(i,k) )
        buoy(i,k) = gravit*(tv_up-tv)/tv 

        w_up(i, kupbase(i) )  = w_up_init(i)


        !if ( buoy(i,k)<0. ) then
            !trig(i) = -10
            !cycle
        !end if

        tmp = max(w_up(i,k), wupmin)
        if ( ent_opt == 2 ) then
            ent_rate_dp_up_int(i,k) = greg_ce*greg_ent_a*buoy(i,k)/tmp/tmp
        else if ( ent_opt == 3 ) then
            ent_rate_dp_up_int(i,k) = nsj_coef/tmp
        else
            ent_rate_dp_up_int(i,k) = 0
        end if
        ent_rate_dp_up_int(i,k) = max(0.0, min( max_ent_rate,  ent_rate_dp_up_int(i,k)))

        !write(*,*) k
        !write(*,"(10f20.10)") tv, tv_up, tint(i,k), qint(i,k), buoy(i,k), tv_up, tv, ent_rate_dp_up_int(i,k)

#ifdef SCMDIAG 
        !write(*,"(i3,10f15.6)") k, mse_up(i,k), ent_rate_up_l, buoy(i,k), w_up(i,k)
        !write(*,"(a3,10a15)") 'L', 'normassflx_up', 'normassflx_up' &
            !, 'mse_up(L)', 'mse_up(H)', 'msesat', 'ent_rate_up_l', 'dz', 'buoy' &
            !, 'ent_rate_up', 'w_up'
#endif

        do k=kupbase(i)-1, 1, -1

!            write(*,*) k
        
            do iteration = 1, maxiteration, 1
                if (iteration == 1) then
                    w_up_mid(i,k) = w_up(i,k+1)
                    buoy_mid(i,k) = buoy(i,k+1)
                    ent_rate_dp_up(i,k) = ent_rate_dp_up_int(i,k+1)
                    ent_rate_sh_up(i,k) = ent_rate_sh_up_int(i,k+1)
                    det_rate_sh_up(i,k) = det_rate_sh_up_int(i,k+1)
                else
                    w_up_mid(i,k) = 0.5 * ( w_up(i,k)+w_up(i,k+1) )
                    buoy_mid(i,k) = 0.5 * ( buoy(i,k)+buoy(i,k+1) )
                    
                    if (entratemidflag == 1) then
                        ent_rate_dp_up(i,k) = 0.5 * ( ent_rate_dp_up_int(i,k) + ent_rate_dp_up_int(i,k+1) )
                        ent_rate_sh_up(i,k) = 0.5 * ( ent_rate_sh_up_int(i,k) + ent_rate_sh_up_int(i,k+1) )
                        det_rate_sh_up(i,k) = 0.5 * ( det_rate_sh_up_int(i,k) + det_rate_sh_up_int(i,k+1) )
                    end if
                    if (entratemidflag == 2) then
                        tmp = max(wupmin, w_up_mid(i,k))
                        if ( ent_opt == 2 ) then
                            ent_rate_dp_up(i,k) = &
                                greg_ce*greg_ent_a*buoy_mid(i,k)/tmp/tmp
                        else if ( ent_opt == 3 ) then
                            ent_rate_dp_up(i,k) = nsj_coef/tmp
                        else
                            ent_rate_dp_up(i,k) = 0
                        end if
                    end if
                end if
                ent_rate_dp_up(i,k) = max(0.0, &
                    min( max_ent_rate,  ent_rate_dp_up(i,k)))

                if ( ent_opt == 2 ) then
                    w2 = ( 2*greg_ent_a*(1-greg_ce)*buoy_mid(i,k) + &
                           w_up(i,k+1)*w_up(i,k+1)/dz(i,k) ) / &
                           (1.0/dz(i,k) + 1.0/greg_z0)
                else if ( ent_opt == 3 ) then
                    w2 = w_up(i,k+1)*w_up(i,k+1) + 2*dz(i,k)*( &
                            nsj_ent_a*buoy_mid(i,k)-nsj_coef*w_up(i,k+1) )
                else
                    w2 = w_up(i,k+1)**2
                end if                
                w_up(i,k)  = sqrt(max(w2, 0.0))

                ent_rate = ent_rate_dp_up(i,k) + ent_rate_sh_up(i,k)
                det_rate = det_rate_dp_up(i,k) + det_rate_sh_up(i,k)

!flux form
                !normassflx_up(i,k) = normassflx_up(i,k+1)*exp(ent_rate_dp_up(i,k)*dz(i,k) )
                !Ek = ( normassflx_up(i,k) - normassflx_up(i,k+1) ) / dz(i,k)
                !mse_up(i,k) = ( normassflx_up(i,k+1)*mse_up(i,k+1) &
                                !+ Ek*mse(i,k)*dz(i,k) + mseqi(i,k)*dz(i,k) ) &
                                !/normassflx_up(i,k)
                !write(*,*) "old"
                !write(*,"(2i2,10f20.10)") k, iteration, ent_rate, normassflx_up(i,k), &
                    !mse_up(i,k), mse(i,k), mse_up(i,k+1)
!scalar form
                normassflx_up(i,k) = normassflx_up(i,k+1) &
                    *exp( (ent_rate-det_rate)*dz(i,k) )
                denom = 1. + 0.5*ent_rate*dz(i,k)
                mse_up(i,k) =  1./denom*( &
                      ent_rate*dz(i,k)*mse(i,k) &
                    + ( 1 - 0.5*ent_rate*dz(i,k) )*mse_up(i,k+1) &
                    + dz(i,k)/( 0.5*( normassflx_up(i,k)+normassflx_up(i,k+1) ) )*mseqi(i,k) )
                !write(*,*) "new"
                !write(*,"(2i2,10f20.10)") k, iteration, ent_rate, normassflx_up(i,k), &
                    !mse_up(i,k), mse(i,k), mse_up(i,k+1)

                
            !----method 1: Taylor expanding ------------------------------------------------
                if (mse2tsatflag == 1) then
                    call cal_mse2tsat(mse_up(i,k), tint(i,k), &
                        qsatint(i,k), msesatint(i,k), t_up(i,k) )
                    call cal_qsat(t_up(i,k), pint(i,k), q_up(i,k))
                end if
            !----method 2: bi-section ------------------------------------------------------
                if (mse2tsatflag == 2) then
                    call mse2tsat( mse_up(i,k), zint(i,k), pint(i,k), t_up(i,k), q_up(i,k) )
                end if
            !-------------------------------------------------------------------------------

                Fp = max(0.0, 1.0 - exp(-(z(i,k) - zint(i,kupbase(i)) - rain_z0)/rain_zp) )
!                fp = 1._r8
                Fi = 0.0
                if (cloud_t1 < t_up(i,k) .and. t_up(i,k) < cloud_t2) then
                    Fi = (cloud_t2-t_up(i,k)) / (cloud_t2-cloud_t1)
                else if (t_up(i,k) <= cloud_t1) then
                    Fi = 1.0
                end if

!flux form
                !condrate(i,k) = 1.0/rho(i,k)*( Ek*q(i,k) &
                    !-( normassflx_up(i,k)*q_up(i,k) &
                    !-normassflx_up(i,k+1)*q_up(i,k+1) )/dz(i,k) )
                !write(*,*) "old cond"
                !write(*,"(2i2,10f20.10)") k, iteration, condrate(i,k)
!scalar form
                condrate(i,k) = 1./dz(i,k)/rho(i,k) &
                    *( 0.5*( normassflx_up(i,k)+normassflx_up(i,k+1) ) ) &
                    *( -denom*q_up(i,k) + ent_rate*dz(i,k)*q(i,k) &
                       +( 1 - 0.5*ent_rate*dz(i,k) )*q_up(i,k+1) )
                !write(*,*) "new cond"
                !write(*,"(2i2,10f20.10)") k, iteration, condrate(i,k)


                frezrate(i,k) = Fi * condrate(i,k)
                rainrate(i,k) = (1.0-Fi) * Fp * condrate(i,k)
                snowrate(i,k) = Fi * Fp * condrate(i,k)
                precrate(i,k) = rainrate(i,k) + snowrate(i,k)


!flux form fi*rate
                !qliq_up(i,k) = (normassflx_up(i,k+1)*( qliq_up(i,k+1) ) + &
                    !rho(i,k)*( condrate(i,k)-frezrate(i,k)-rainrate(i,k) )*dz(i,k) )/normassflx_up(i,k)
                !qice_up(i,k) = (normassflx_up(i,k+1)*( qice_up(i,k+1) ) + &
                    !rho(i,k)*( frezrate(i,k)-snowrate(i,k) )*dz(i,k) )/normassflx_up(i,k)
                !write(*,*) "old liq and ice rate"
                !write(*,"(2i2,10f20.10)") k, iteration, qliq_up(i,k), qice_up(i,k)
!flux form fi*qw
                !qw = (normassflx_up(i,k+1)*( qliq_up(i,k+1)+qice_up(i,k+1) ) + &
                    !rho(i,k)*(1.0-Fp)*condrate(i,k)*dz(i,k) )/normassflx_up(i,k)
                !qliq_up(i,k) = (1.0-Fi) * qw
                !qice_up(i,k) = Fi * qw
                !write(*,*) "old liq and ice qw"
                !write(*,"(2i2,10f20.10)") k, iteration, qliq_up(i,k), qice_up(i,k)
!sclar form fi*rate
                qliq_up(i,k) =  1./denom*( &
                    + ( 1 - 0.5*ent_rate*dz(i,k) )*qliq_up(i,k+1) &
                    + dz(i,k)/( 0.5*( normassflx_up(i,k)+normassflx_up(i,k+1) ) ) &
                      *rho(i,k)*( condrate(i,k)-frezrate(i,k)-rainrate(i,k) ) )
                qice_up(i,k) =  1./denom*( &
                    + ( 1 - 0.5*ent_rate*dz(i,k) )*qice_up(i,k+1) &
                    + dz(i,k)/( 0.5*( normassflx_up(i,k)+normassflx_up(i,k+1) ) ) &
                      *rho(i,k)*( frezrate(i,k)-snowrate(i,k) ) )
                !write(*,*) "new liq and ice rate"
                !write(*,"(2i2,10f20.10)") k, iteration, qliq_up(i,k), qice_up(i,k)
!scalar form fi*qw
                !qw =  1./denom*( &
                    !+ ( 1 - 0.5*ent_rate*dz(i,k) )*( qliq_up(i,k+1)+qice_up(i,k+1)  )&
                    !+ dz(i,k)/( 0.5*( normassflx_up(i,k)+normassflx_up(i,k+1) ) ) &
                      !*rho(i,k)*(1.0-Fp)*condrate(i,k) )
                !qliq_up(i,k) = (1.0-Fi) * qw
                !qice_up(i,k) = Fi * qw
                !write(*,*) "new liq and ice qw"
                !write(*,"(2i2,10f20.10)") k, iteration, qliq_up(i,k), qice_up(i,k)

!MJO
                qliq_up(i,k) = max( 0., qliq_up(i,k) )
                qice_up(i,k) = max( 0., qice_up(i,k) )

                if (iteration < maxiteration) then
                    if (mseqiflag > 0) then
                        mseqi(i,k) = latice*rho(i,k)*frezrate(i,k)
                    else
                        mseqi(i,k) = 0
                    end if
                end if

                tv = tint(i,k)*(1+tveps*qint(i,k))
                tv_up = t_up(i,k)*( 1+tveps*q_up(i,k) )
                
                if (buoyflag == 1) then
                    buoy(i,k) = gravit* ( (tv_up-tv)/tv ) 
                end if
                if (buoyflag == 2) then
                    buoy(i,k) = gravit* ( (tv_up-tv)/tv - qliq_up(i,k) - qice_up(i,k) ) 
                end if

                tmp = max(wupmin, w_up(i,k))
                if ( ent_opt == 2 ) then
                    ent_rate_dp_up_int(i,k) = greg_ce*greg_ent_a*buoy(i,k)/tmp/tmp
                else if ( ent_opt == 3 ) then
                    ent_rate_dp_up_int(i,k) = nsj_coef/tmp
                else
                    ent_rate_dp_up_int(i,k) = 0.0
                end if
                ent_rate_dp_up_int(i,k) = max(0.0, min( max_ent_rate,  ent_rate_dp_up_int(i,k)))                


                if ( bsflag == 1 ) then

                    bs_scaleh = tmp_zuptop_max-zint(i, kupbase(i) ) 
                    bs_cridis = bs_rle*bs_scaleh
                    bs_wue = w_up(i,k)
                    bs_thetalint = tint(i,k)*( bs_p0/pint(i,k) )**(rair/cpair)
                    bs_thetal_up = ( t_up(i,k)-latvap*qliq_up(i,k)/cpair-latice*qice_up(i,k)/cpair  ) &
                        *( bs_p0/pint(i,k) )**(rair/cpair)
                    ! Haiyang
                    call cal_buoysort(flagbspdf, bs_cridis, zint(i,k), pint(i,k), rho(i,k), &
                        bs_thetalint, qint(i,k), bs_thetal_up, q_up(i,k)+qliq_up(i,k)+qice_up(i,k), &
                        bs_wue, bs_xc(i,k), ent_rate_sh_up_int(i,k), det_rate_sh_up_int(i,k) )

                end if

            end do   ! loop of iteration

            
            if (ctopflag == 1) then 
                if (buoy(i,k) < 0.0 .or. w_up(i,k)<wupmin) then
                    exit
                end if
            end if
            
            if (ctopflag >= 2) then
                if (w_up(i,k) < wupmin) then
                    exit
                end if
            end if

            if (negcondflag == 0) then
                if ( condrate(i,k)<0 ) then
                    condrate(i,k) = 0.
                    frezrate(i,k) = 0.
                    rainrate(i,k) = 0.
                    snowrate(i,k) = 0.
                    precrate(i,k) = 0.
                    qliq_up(i,k) =  0.
                    qice_up(i,k) =  0.
                    exit
                end if
            end if

        end do  ! loop for levels

        
        if (k>=1) then
            mse_up(i,k) = mseint(i,k)
            t_up(i,k) = tint(i,k)
            q_up(i,k) = qint(i,k)
            qliq_up(i,k) = 0._r8
            qice_up(i,k) = 0._r8
            w_up(i,k) = 0._r8
            buoy(i,k) = 0._r8
            ent_rate_dp_up_int(i,k) = 0._r8
            det_rate_dp_up(i,k) = 0._r8
            normassflx_up(i,k) = 0._r8

            condrate(i,k) = 0._r8
            rainrate(i,k) = 0._r8
            snowrate(i,k) = 0._r8
            precrate(i,k) = 0._r8
            mseqi(i,k) = 0._r8
            qliq_up(i,k) = 0._r8
            qice_up(i,k) = 0._r8

            ent_rate_dp_up_int(i,kupbase(i) ) = 0._r8

            kuptop(i) = k+1

!not penetrating more than one level
            if ( k == kupbase(i)-1 ) then
                trig(i) = -11
            end if
        else
            k = 1
            mse_up(i,k) = mseint(i,k)
            t_up(i,k) = tint(i,k)
            q_up(i,k) = qint(i,k)
            qliq_up(i,k) = 0._r8
            qice_up(i,k) = 0._r8
            w_up(i,k) = 0._r8
            buoy(i,k) = 0._r8
            ent_rate_dp_up_int(i,k) = 0._r8
            det_rate_dp_up(i,k) = 0._r8
            normassflx_up(i,k) = 0._r8

            condrate(i,k) = 0._r8
            rainrate(i,k) = 0._r8
            snowrate(i,k) = 0._r8
            precrate(i,k) = 0._r8
            mseqi(i,k) = 0._r8
            qliq_up(i,k) = 0._r8
            qice_up(i,k) = 0._r8

            ent_rate_dp_up_int(i,kupbase(i) ) = 0._r8

            kuptop(i) = 2

        end if

    end do

end subroutine cal_mse_up_old


! ==============================================================================
! calculate downdraft 
! ==============================================================================
subroutine cal_mse_dn( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        ent_opt, kuptop, trig, dz, zint, p, pint, rho, t, twet, twetint, lvmid, &
        qint, dseint, accuprec, evaprate, buoy_mid, dn_frac, &
!output
        dse_dn, q_dn, normassflx_dn )

!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    integer, intent(in) :: ent_opt
    integer , dimension(ncol), intent(in) :: trig     ! [1]
    integer , dimension(ncol), intent(in) :: kuptop    ! [1]
    real(r8), dimension(ncol, nlev),  intent(in) :: dz    ! [m]
    real(r8), dimension(ncol, nlevp),  intent(in) :: zint    ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: p     ! [Pa]
    real(r8), dimension(ncol, nlevp),  intent(in) :: pint     ! [Pa]
    real(r8), dimension(ncol, nlev),  intent(in) :: rho   ! [kg/m3]
    real(r8), dimension(ncol, nlev),  intent(in) :: t     ! [K]
    real(r8), dimension(ncol, nlev), intent(in)  :: twet  ! [K]
    real(r8), dimension(ncol, nlevp), intent(in)  :: twetint  ! [K]
    real(r8), dimension(ncol, nlev), intent(in)  :: lvmid  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qint  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: dseint  ! [J/kg]
    real(r8), dimension(ncol, nlev), intent(in)  :: accuprec ! [#]
    real(r8), dimension(ncol, nlev), intent(in)  :: evaprate ! [1/m]
    real(r8), dimension(ncol, nlev), intent(in)  :: buoy_mid
    real(r8), intent(in)  :: dn_frac ! [#]

!output
    real(r8), dimension(ncol, nlevp), intent(inout) :: dse_dn  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: q_dn  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: normassflx_dn  ! [kg/m2/s]

!local
    real(r8) :: fac
    integer  :: i,j,k, kdntop
    real(r8), dimension(ncol, nlevp) :: qswetint, dsewetint

    dsewetint = cpair * twetint + gravit * zint
    call cal_qsat2d(twetint(:,:), pint(:,:), qswetint(:,:))

    dse_dn = dseint
    q_dn = qint
    normassflx_dn = 0._r8


    do i=1, ncol
        if ( trig(i) < 1 ) cycle
        ! downdraft starts at the layer lower than the neutral buoyancy layer
        kdntop = nlev
        do k = kuptop(i), nlev
            if (buoy_mid(i,k) > 0) then
                kdntop = k
                exit
            end if
        end do
        kdntop = min(kdntop+1, nlev)
        
        normassflx_dn(i,kdntop) = -1.0_r8 * dn_frac
        q_dn(i,kdntop) = qint(i,kdntop)
        dse_dn(i,kdntop) = dseint(i,kdntop)

        do k=kdntop, nlev
            normassflx_dn(i,k+1) = normassflx_dn(i,k) + dn_be*rho(i,k) * &
                min(0.0_r8, twet(i,k)-t(i,k)) * accuprec(i,k) * dz(i,k)
            if (zpbltop > 0) then
                fac = min(1.0_r8, max(0.0_r8, (zint(i,k+1)-zint(i,nlevp))/zpbltop ))
            else
                fac = 1.0_r8
            end if

            if(plume_model == 'zyx2')then
              fac = 1.0_r8
              if(zint(i,k+1) < 500._r8)then
                fac = min(1.0_r8, max(0.0_r8, (zint(i,k+1)-zint(i,nlevp))/500._r8))
              endif
            endif

            normassflx_dn(i,k+1) = normassflx_dn(i,k+1) * fac
            q_dn(i,k+1) = qswetint(i,k+1)
            dse_dn(i,k+1) = dsewetint(i,k+1)

            !if (abs(normassflx_dn(i,k+1)) < 1.0e-3) then
            !    dse_dn(i,k+1) = dse_dn(i,k)
            !    q_dn(i,k+1) = q_dn(i,k)
            !else
            !    dse_dn(i,k+1) = (normassflx_dn(i,k)*dse_dn(i,k) + &
            !        lvmid(i,k)*evaprate(i,k)*dz(i,k)) / normassflx_dn(i,k+1)
            !    q_dn(i,k+1) = (normassflx_dn(i,k)*q_dn(i,k) - &
            !        evaprate(i,k)*dz(i,k)) / normassflx_dn(i,k+1)
            !end if
        end do
        
        ! normassflx_dn(i,nlev+1) = 0._r8

    end do


end subroutine cal_mse_dn

! ==============================================================================
! calculate downdraft new with evap
! ==============================================================================
subroutine cal_mse_dn_evap( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        ent_opt, kuptop, trig, dz, zint, p, pint, rho, t, twet, twetint, lvmid, &
        qint, dseint, accuprec, evaprate, buoy_mid, dn_frac, &
!output
        dse_dn, q_dn, normassflx_dn , &
!MZ
        q,kuplaunch,precrate,surfprec,ent_rate_dn, det_rate_dn)


!---------------------------------------------------------------------------------
! revised by Minghua Zhang on 2018-07-14
!
!---------------------------------------------------------------------------------

!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    integer, intent(in) :: ent_opt
    integer , dimension(ncol), intent(in) :: trig     ! [1]
    integer , dimension(ncol), intent(in) :: kuptop    ! [1]
    real(r8), dimension(ncol, nlev),  intent(in) :: dz    ! [m]
    real(r8), dimension(ncol, nlevp),  intent(in) :: zint    ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: p     ! [Pa]
    real(r8), dimension(ncol, nlevp),  intent(in) :: pint     ! [Pa]
    real(r8), dimension(ncol, nlev),  intent(in) :: rho   ! [kg/m3]
    real(r8), dimension(ncol, nlev),  intent(in) :: t     ! [K]
    real(r8), dimension(ncol, nlev), intent(in)  :: twet  ! [K]
    real(r8), dimension(ncol, nlevp), intent(in)  :: twetint  ! [K]
    real(r8), dimension(ncol, nlev), intent(in)  :: lvmid  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qint  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: dseint  ! [J/kg]
!MZ
    !real(r8), dimension(ncol, nlev), intent(in)  :: accuprec ! [#]
    !real(r8), dimension(ncol, nlev), intent(in)  :: evaprate ! [1/m]
    real(r8), dimension(ncol, nlev), intent(inout)  :: accuprec ! [#]
    real(r8), dimension(ncol, nlev), intent(out)  :: evaprate ! [1/m]
    real(r8), dimension(ncol, nlev), intent(in)  :: buoy_mid
    real(r8), intent(in)  :: dn_frac ! [#]

!output
    real(r8), dimension(ncol, nlevp), intent(inout) :: dse_dn  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: q_dn  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(inout) :: normassflx_dn  ! [kg/m2/s]

!MZ
    real(r8), dimension(ncol, nlev),  intent(in) :: q     ! [kg/kg]
    real(r8), dimension(ncol, nlev),  intent(inout) :: precrate ! [m2/kg]
    integer, dimension(ncol) :: kuplaunch ! [1] cloud launching level
    real(r8), dimension(ncol), intent(out) :: surfprec  ! [#]
!output
    real(r8), dimension(ncol, nlev), intent(out) :: ent_rate_dn! [#]
    real(r8), dimension(ncol, nlev), intent(out) :: det_rate_dn! it is rate*M

!local
    real(r8) :: fac
    real(r8) :: qsat_tmp  !MZ
    integer  :: i,j,k, kdntop
    real(r8), dimension(ncol, nlevp) :: qswetint, dsewetint

    dsewetint = cpair * twetint + gravit * zint
    call cal_qsat2d(twetint(:,:), pint(:,:), qswetint(:,:))

    dse_dn = dseint
    q_dn = qint
    normassflx_dn = 0._r8
    evaprate      = 0._r8
    det_rate_dn  = 0._r8
    ent_rate_dn  = 0._r8

    dsewetint = cpair*twetint + grav*zint

!return !! Disable downdraft

    do i=1, ncol
        if ( trig(i) < 1 ) cycle
        ! downdraft starts at the layer lower than the neutral buoyancy layer
        kdntop = nlev
        do k = kuptop(i), nlev
            if (buoy_mid(i,k) > 0) then
                kdntop = k
                exit
            end if
        end do
        kdntop = min(kdntop+1, nlev)
        
        normassflx_dn(i,kdntop) = -1.0_r8 * dn_frac
        q_dn(i,kdntop) = qswetint(i,kdntop) !qint(i,kdntop)
        dse_dn(i,kdntop) = dsewetint(i,kdntop)  !dseint(i,kdntop)

        normassflx_dn(i,:) = 0.0_r8
        do k=kdntop, nlev

         ! if (z(i,k) > zsrf(i)) then
!MZ
!        write(*,*)'normassflx_dn',k,normassflx_dn(i,k)
            accuprec(i,k)  = accuprec(i,k-1) + &
                                rho(i,k)*precrate(i,k)*dz(i,k)
!                                rho(i,k)*(precrate(i,k)-evaprate(i,k-1))*dz(i,k)
            !accuprec(i,k) = max(0._r8,accuprec(i,k))


            call cal_qsat( twet(i,k), p(i,k), qsat_tmp )


                evaprate(i,k) = min( dn_ae*max( 0._r8, qsat_tmp-q(i,k) ) * &
                        accuprec(i,k) / dn_vt / rho(i,k), accuprec(i,k)/rho(i,k)/dz(i,k) ) &
                        * evap_enhance * ((p(i,k)-p(i,kuptop(i)))/(p(i,nlev)-p(i,kuptop(i))+10.0))**evap_shape
                evaprate(i,k) = min(evaprate(i,k), accuprec(i,k)/rho(i,k)/dz(i,k))

                accuprec(i,k) = max(0.0, accuprec(i,k) - evaprate(i,k)*rho(i,k)*dz(i,k))
           ! end if
        end do
        surfprec(i) = accuprec(i,nlev)
    end do

    return

            !!evaprate(i,k) = min( dn_ae*max( 0._r8, qsat_tmp-q(i,k) ) * &
            !!           accuprec(i,k) / dn_vt / rho(i,k), accuprec(i,k)/rho(i,k)/dz(i,k))
            evaprate(i,k) = - 0.2_r8*accuprec(i,k)/rho(i,k)/dz(i,k)

! height scaled!
            !!evaprate(i,k) = - evaprate(i,k) * normassflx_dn(i,k) * (zint(i,kuptop(i))-zint(i,k))/5.e3_r8

            dse_dn(i,k+1) = dsewetint(i,k+1) + (dseint(i,k+1) - dsewetint(i,k+1))*0.5_r8
            q_dn(i,k+1)   = qint(i,k+1) + (qswetint(i,k+1) - qint(i,k+1))*0.5_r8

!!MZ the assumption below can be improved 
            !if (zpbltop > 0) then
            if (zint(i,k) > zpbltop ) then

       !here the idea is to give saturated downdraft and evap to obtain the entraniment

                ent_rate_dn(i,k) = -(q_dn(i,k+1)-q_dn(i,k))/dz(i,k)*normassflx_dn(i,k) - evaprate(i,k)*rho(i,k)
                ent_rate_dn(i,k) = ent_rate_dn(i,k) /max(qint(i,k) - q_dn(i,k+1), 1.0e-6)
                ent_rate_dn(i,k) = min(2.e-3_r8, max(ent_rate_dn(i,k),0._r8) )

!               ent_rate_dn(i,k) = - dn_be*rho(i,k) * &
!                  min(0.0, twet(i,k)-t(i,k)) * accuprec(i,k)*dz(i,k) 

                ent_rate_dn(i,k) = 0._r8 !!!
!MZ empirical limit here
               !ent_rate_dn(i,k) =  - min(2.e-3_r8, ent_rate_dn(i,k)) * normassflx_dn(i,k)

               det_rate_dn(i,k) = 0._r8

               normassflx_dn(i,k+1) = normassflx_dn(i,k) - ent_rate_dn(i,k) + det_rate_dn(i,k)

            else

               ent_rate_dn(i,k) = 0._r8
               
               det_rate_dn(i,k) = - normassflx_dn(i,k)*dz(i,k)/(zint(i,k)-zint(i,nlevp))

               normassflx_dn(i,k+1) = normassflx_dn(i,k) - ent_rate_dn(i,k) + det_rate_dn(i,k)

            !!   evaprate(i,k)  = - (normassflx_dn(i,k+1)*q_dn(i,k+1) - normassflx_dn(i,k)*q_dn(i,k)) &
            !!        /dz(i,k) + det_rate_dn(i,k)*0.5_r8*(q_dn(i,k+1)+q_dn(i,k))
            !!   evaprate(i,k)  = evaprate(i,k)/rho(i,k)

            !!   if(evaprate(i,k) < 0)then  ! condensation in downdraft
!                   write(*,*) 'Condensation in downdraft happened!',evaprate,normassflx_dn(i,k+1)
            !!       precrate(i,k) = precrate(i,k) - evaprate(i,k)
            !!       accuprec(i,k)  = accuprec(i,k-1) + &
            !!                    rho(i,k)*(precrate(i,k)-evaprate(i,k-1))*dz(i,k)
            !!       evaprate(i,k) = 0._r8
            !!   endif
            endif   ! entrainment and detrainment separation

!!!!
            !!dse_dn(i,k+1) = dse_dn(i,k) -  200._r8
            !!q_dn(i,k+1) = q_dn(i,k) - 1.0e-4_r8
            !!q_dn(i,k+1) = max(q_dn(i,k+1),1.0e-6_r8)

               !dse_dn(i,k+1) = (normassflx_dn(i,k)*dse_dn(i,k) + &
               !   lvmid(i,k)*evaprate(i,k)*dz(i,k)*rho(i,k) + &
               !   - ent_rate_dn(i,k)*dseint(i,k)+det_rate_dn(i,k)*dse_dn(i,k) & 
               !   ) / normassflx_dn(i,k+1)  ! MZ with entrainment

               !q_dn(i,k+1) = (normassflx_dn(i,k)*q_dn(i,k) - &
               !   evaprate(i,k)*dz(i,k)*rho(i,k)-ent_rate_dn(i,k)*qint(i,k) &
               !   + det_rate_dn(i,k)*q_dn(i,k)) / normassflx_dn(i,k+1)

!MZ commented out 2018-07-14

          !normassflx_dn(i,k+1) = normassflx_dn(i,k) + dn_be*rho(i,k) * &
          !    min(0.0, twet(i,k)-t(i,k)) * accuprec(i,k) * dz(i,k)
          !if (zpbltop > 0) then
          !    fac = min(1.0, max(0.0, (zint(i,k+1)-zint(i,nlevp))/zpbltop ))
          !else
          !    fac = 1.0
          !end if
          !normassflx_dn(i,k+1) = normassflx_dn(i,k+1) * fac
          !q_dn(i,k+1) = qswetint(i,k+1)
          !dse_dn(i,k+1) = dsewetint(i,k+1)

            !if (abs(normassflx_dn(i,k+1)) < 1.0e-3) then
            !    dse_dn(i,k+1) = dse_dn(i,k)
            !    q_dn(i,k+1) = q_dn(i,k)
            !else
            !    dse_dn(i,k+1) = (normassflx_dn(i,k)*dse_dn(i,k) + &
            !        lvmid(i,k)*evaprate(i,k)*dz(i,k)) / normassflx_dn(i,k+1)
            !    q_dn(i,k+1) = (normassflx_dn(i,k)*q_dn(i,k) - &
            !        evaprate(i,k)*dz(i,k)) / normassflx_dn(i,k+1)
            !end if
!!        end do
        
        ! normassflx_dn(i,nlev+1) = 0._r8
!MZ
!!        surfprec(i) = accuprec(i,nlev)

!!    end do


if(i<0)then
    write(*,*)'in cal_mse_dn---'
    write(*,*)'ent_rate_dn',ent_rate_dn
    write(*,*)'det_rate_dn',ent_rate_dn
    write(*,*)'normassflx_dn',normassflx_dn
endif

end subroutine cal_mse_dn_evap

! ==============================================================================
! calculate evaporation
! ==============================================================================
subroutine cal_evap( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        ent_opt, kuptop, trig, zsrf, z, dz, p, rho, t, twet, q, &
        precrate, &
!output
        accuprec, surfprec, evaprate )

!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    integer, intent(in) :: ent_opt
    integer , dimension(ncol), intent(in) :: kuptop    ! [1]
    integer , dimension(ncol), intent(in) :: trig     ! [1]
    real(r8), dimension(ncol), intent(in) :: zsrf    ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: z, dz    ! [m]
    real(r8), dimension(ncol, nlev),  intent(in) :: p     ! [Pa]
    real(r8), dimension(ncol, nlev),  intent(in) :: rho   ! [kg/m3]
    real(r8), dimension(ncol, nlev),  intent(in) :: t     ! [K]
    real(r8), dimension(ncol, nlev),  intent(in) :: twet  ! [K]
    real(r8), dimension(ncol, nlev),  intent(in) :: q     ! [kg/kg]
    real(r8), dimension(ncol, nlev),  intent(in) :: precrate ! [m2/kg]
!output
    real(r8), dimension(ncol, nlev), intent(out) :: accuprec  ! [#]
    real(r8), dimension(ncol, nlev), intent(out) :: evaprate  ! [m2/kg]
    real(r8), dimension(ncol), intent(out) :: surfprec  ! [#]

!local
    real(r8) :: qsat_tmp
    integer :: i,j,k

    evaprate = 0._r8
    accuprec = 0._r8
    surfprec  = 0._r8

    do i=1, ncol
        if ( trig(i) < 1 ) cycle
        do k=kuptop(i), nlev
!MZ            if (z(i,k) > zsrf(i)) then
            if (z(i,k) >= zsrf(i)) then
                accuprec(i,k)  = accuprec(i,k-1) + rho(i,k)*precrate(i,k)*dz(i,k)
                call cal_qsat( twet(i,k), p(i,k), qsat_tmp )
                evaprate(i,k) = min( dn_ae*max( 0._r8, qsat_tmp-q(i,k) ) * &
                        accuprec(i,k) / dn_vt / rho(i,k), accuprec(i,k)/rho(i,k)/dz(i,k) ) &
                        * evap_enhance * ((p(i,k)-p(i,kuptop(i)))/(p(i,nlev)-p(i,kuptop(i))+10.0))**evap_shape
                evaprate(i,k) = min(evaprate(i,k), accuprec(i,k)/rho(i,k)/dz(i,k)) 
!MZ!!
!!                evaprate(i,k) = min(evaprate(i,k), precrate(i,k))

                accuprec(i,k) = max(0.0, accuprec(i,k) - evaprate(i,k)*rho(i,k)*dz(i,k))
            end if
        end do
        surfprec(i) = accuprec(i,nlev)
    end do

 if(i<0)then
    write(*,*)'in cal_evap accuprec',accuprec
 endif
end subroutine cal_evap


! ==============================================================================
! calculate CAPE
! ==============================================================================
subroutine cal_cape( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        dz, buoy_mid, normassflx_up, kupbase, kuptop, &
!output
        cape, cwf, cin, &
!in/out
        trig)
!------------------------------------------------------
!calculate CAPE given buoyancy
!------------------------------------------------------
!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    real(r8), dimension(ncol, nlev), intent(in) :: dz  ! [m]
    real(r8), dimension(ncol, nlev), intent(in) :: buoy_mid  ! [ms-2]
    real(r8), dimension(ncol, nlevp), intent(in) :: normassflx_up  ! [ms-2]
    integer, dimension(ncol), intent(in) :: kupbase ! [1]
    integer, dimension(ncol), intent(in) :: kuptop ! [1]
!output
    real(r8), dimension(ncol), intent(out) :: cape  ! [kgm-2-s]
    real(r8), dimension(ncol), intent(out) :: cwf   ! [kgm-2-s]
!MZ added cin
    real(r8), dimension(ncol), intent(out) :: cin   ! [kgm-2-s]
!input/output
    integer, dimension(ncol), intent(inout) :: trig     ! [1]
!local
    integer :: i,j,k

!intialize output
    cape = 0._r8
    cwf  = 0._r8
    cin  = 0._r8

    do i=1, ncol
        if ( trig(i) < 1 ) cycle
        do k=kupbase(i)-1, kuptop(i), -1
            cape(i) = cape(i) + dz(i,k)*max(buoy_mid(i,k), 0._r8)
            cin(i)  = cin(i)  - dz(i,k)*min(buoy_mid(i,k), 0._r8)
            cwf(i) = cwf(i) + dz(i,k)*max(buoy_mid(i,k), 0._r8)&
                *0.5*(normassflx_up(i,k)+normassflx_up(i,k+1) )
            if ( isnan(cwf(i)) ) then
!!                write(*,*) i, normassflx_up(i,:)
            end if
        end do
    end do
end subroutine cal_cape

! ==============================================================================
! calculate the tendency due to transport
! ==============================================================================
subroutine cal_tendtransport( &
!input
#ifdef OFFLINECP
        ncol, nlev, nlevp, &
#endif
        dz, kupbase, kuptop, &
        rho, dseint, qint, dse_up, q_up, &
        normassflx_up,  &
!output
        stend, qtend, &
!in/out
        trig)
!input
#ifdef OFFLINECP
    integer, intent(in) :: ncol, nlev, nlevp
#endif
    real(r8), dimension(ncol, nlev), intent(in) :: dz       ! [m]
    integer, dimension(ncol), intent(in) :: kupbase ! [1]
    integer, dimension(ncol), intent(in) :: kuptop ! [1]
    real(r8), dimension(ncol, nlev), intent(in) :: rho      ! [kg/m3]
    real(r8), dimension(ncol, nlevp), intent(in) :: dseint  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: qint    ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: dse_up  ! [J/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: q_up  ! [kg/kg]
    real(r8), dimension(ncol, nlevp), intent(in) :: normassflx_up  ! [1]
!output
    real(r8), dimension(ncol, nlev), intent(out) :: stend   ! [J/s]
    real(r8), dimension(ncol, nlev), intent(out) :: qtend   ! [kg/kg/s]
!input/output
    integer, dimension(ncol), intent(inout) :: trig     ! [1]
!local

    integer :: i,j,k

!intialize output
    stend = 0._r8
    qtend = 0._r8

    do i=1, ncol
        if ( trig(i) < 1 ) cycle

!cloud layer transport
        do k=kupbase(i)-1, kuptop(i), -1
            stend(i,k) = -( normassflx_up(i,k)*( dse_up(i,k)-dseint(i,k) ) &
                - normassflx_up(i,k+1)*( dse_up(i,k+1)-dseint(i,k+1) ) ) / dz(i,k)/rho(i,k)
            qtend(i,k) = -( normassflx_up(i,k)*( q_up(i,k)-qint(i,k) ) &
                - normassflx_up(i,k+1)*( q_up(i,k+1)-qint(i,k+1) ) ) / dz(i,k)/rho(i,k)
        end do

        ! one layer above plume
        k = max(1, kuptop(i)-1)
        stend(i,k) = -( 0.0 &
            - normassflx_up(i,k+1)*( dse_up(i,k+1)-dseint(i,k+1) ) ) / dz(i,k)/rho(i,k)
        qtend(i,k) = -( 0.0 &
            - normassflx_up(i,k+1)*( q_up(i,k+1)-qint(i,k+1) ) ) / dz(i,k)/rho(i,k)
       
        ! one layer below plume
        k = min(kupbase(i), nlev)
        stend(i,k) = -( normassflx_up(i,k)*( dse_up(i,k)-dseint(i,k) ) &
            - 0.0 ) / dz(i,k)/rho(i,k)
        qtend(i,k) = -( normassflx_up(i,k)*( q_up(i,k)-qint(i,k) ) &
            - 0.0 ) / dz(i,k)/rho(i,k)


    end do

end subroutine cal_tendtransport

! ==============================================================================
! calculate tempeature and saturated Q given MSE and pressure with bi-section
! ==============================================================================
subroutine mse2tsat( mse, z, p, t, q)
    real(r8), intent(in) :: mse
    real(r8), intent(in) :: z
    real(r8), intent(in) :: p
    real(r8), intent(out) :: t
    real(r8), intent(out) :: q
!local
    real(r8) :: fc, fa, fb, ta, tb, tc, error, torl, tmp
    integer :: n

    torl = 0.1
    ta = 10.
    tb = 400.
    error = 100.
    tmp = 611*exp( 5417.*(1/273.-1/ta) )
    fa = cpair*ta+gravit*z+latvap*0.622*tmp/p-mse
    tmp = 611*exp( 5417.*(1/273.-1/tb) )
    fb = cpair*tb+gravit*z+latvap*0.622*tmp/p-mse
    
    n = 1
    if( fa*fb>0 ) then
        t = 0.
        q = 0.
    else
        do while( (error>torl) .and. (n<100) )
            tc = (ta+tb)/2.
            tmp = 611*exp( 5417.*(1/273.-1/tc) )
            fc = cpair*tc+gravit*z+latvap*0.622*tmp/p-mse
            if (fc*fa>0) then
                ta = tc
            else
                tb = tc
            end if
            error = abs(ta-tb)
            n = n + 1
        end do
        t = tc
    end if
    q = 0.622*611/p*exp( 5417.*(1/273.-1/tb) )

end subroutine mse2tsat

! ==============================================================================
! (T,p) --> qsat
! ==============================================================================
subroutine cal_esatqsat( t, p, esat, qsat)
    real(r8) :: t, p, esat, qsat ! K, Pa, Pa, kg/kg
    esat = min(p, 611.2*exp(17.67*(t-273.15)/(t-273.15+243.5)) ) 
    qsat = epsilo * esat / p
end subroutine cal_esatqsat
subroutine cal_qsat( t, p, qsat)
    real(r8) :: t, p, qsat
    qsat = min(1.0, epsilo * 611.2*exp(17.67*(t-273.15)/(t-273.15+243.5)) / p)
end subroutine cal_qsat

subroutine cal_qsat2d( t, p, qsat)
    real(r8) :: t(:,:), p(:,:), qsat(:,:)
    qsat = epsilo * 611.2*exp(17.67*(t-273.15)/(t-273.15+243.5)) / p
end subroutine cal_qsat2d

! ==============================================================================
! MSEsat --> T (given reference T, q, and mse)
! ==============================================================================
subroutine cal_mse2tsat(mse, Tref, qref, mseref, T)
    real(r8) :: mse, Tref, qref, mseref, T
    real(r8) :: L, gama

    L = latvap - (cpliq-cpwv)*(T-273.15)
    gama = latvap*latvap/cpair/rh2o * qref/Tref/Tref
    T = Tref + (mse-mseref)/(1+gama)/cpair
end subroutine cal_mse2tsat

! ==============================================================================
! T, RH --> wet-bulb temperture
! ==============================================================================
subroutine cal_twet2d(t, rh, twet)
    real(r8) :: t(:,:), rh(:,:), twet(:,:)
    twet = (t-273.16)*atan( 0.151977*(rh*100.+8.313659)**0.5 ) + atan(t-273.16+rh*100.) &
        - atan(rh*100.-1.676331) + 0.00391838*((rh*100.)**1.5)*atan(0.023101*rh*100.) - 4.686035 &
        + 273.16
end subroutine cal_twet2d



!===============================================================================
!MZ subroutine zyx2_conv_evap(inncol,lchnk, &
subroutine zyx2_conv_evap(lchnk, inncol, nlev,nlevp, &
     t,pmid,pdel,q, &
     tend_s, tend_s_snwprd, tend_s_snwevmlt, tend_q, &
     prdprec, cldfrc, deltat,  &
     prec, snow, ntprprd, ntsnprd, flxprec, flxsnow )

!-----------------------------------------------------------------------
! Compute tendencies due to evaporation of rain from ZM scheme
!--
! Compute the total precipitation and snow fluxes at the surface.
! Add in the latent heat of fusion for snow formation and melt, since it not dealt with
! in the Zhang-MacFarlane parameterization.
! Evaporate some of the precip directly into the environment using a Sundqvist type algorithm
!-----------------------------------------------------------------------

    ! use wv_saturation,  only: qsat
    use phys_grid, only: get_rlat_all_p

!------------------------------Arguments--------------------------------
!MZ    
    integer,intent(in) :: nlev,nlevp  ! 
    integer,intent(in) :: inncol, lchnk             ! number of columns and chunk index
    real(r8),intent(in), dimension(inncol,nlev) :: t          ! temperature (K)
    real(r8),intent(in), dimension(inncol,nlev) :: pmid       ! midpoint pressure (Pa) 
    real(r8),intent(in), dimension(inncol,nlev) :: pdel       ! layer thickness (Pa)
    real(r8),intent(in), dimension(inncol,nlev) :: q          ! water vapor (kg/kg)
    real(r8),intent(inout), dimension(inncol,nlev) :: tend_s     ! heating rate (J/kg/s)
    real(r8),intent(inout), dimension(inncol,nlev) :: tend_q     ! water vapor tendency (kg/kg/s)
    real(r8),intent(out  ), dimension(inncol,nlev) :: tend_s_snwprd ! Heating rate of snow production
    real(r8),intent(out  ), dimension(inncol,nlev) :: tend_s_snwevmlt ! Heating rate of evap/melting of snow
    


    real(r8), intent(in   ) :: prdprec(inncol,nlev)! precipitation production (kg/ks/s)
    real(r8), intent(in   ) :: cldfrc(inncol,nlev) ! cloud fraction
    real(r8), intent(in   ) :: deltat             ! time step

    real(r8), intent(inout) :: prec(inncol)        ! Convective-scale preciptn rate
    real(r8), intent(out)   :: snow(inncol)        ! Convective-scale snowfall rate
!
!---------------------------Local storage-------------------------------

    !real(r8) :: es    (inncol,nlev)    ! Saturation vapor pressure
    real(r8) :: fice   (inncol,nlev)    ! ice fraction in precip production
    real(r8) :: fsnow_conv(inncol,nlev) ! snow fraction in precip production
    real(r8) :: qs   (inncol,nlev)    ! saturation specific humidity
    real(r8),intent(out) :: flxprec(inncol,nlevp)   ! Convective-scale flux of precip at interfaces (kg/m2/s)
    real(r8),intent(out) :: flxsnow(inncol,nlevp)   ! Convective-scale flux of snow   at interfaces (kg/m2/s)
    real(r8),intent(out) :: ntprprd(inncol,nlev)    ! net precip production in layer
    real(r8),intent(out) :: ntsnprd(inncol,nlev)    ! net snow production in layer
    real(r8) :: work1                  ! temp variable (pjr)
    real(r8) :: work2                  ! temp variable (pjr)

    real(r8) :: evpvint(inncol)         ! vertical integral of evaporation
    real(r8) :: evpprec(inncol)         ! evaporation of precipitation (kg/kg/s)
    real(r8) :: evpsnow(inncol)         ! evaporation of snowfall (kg/kg/s)
    real(r8) :: snowmlt(inncol)         ! snow melt tendency in layer
    real(r8) :: flxsntm(inncol)         ! flux of snow into layer, after melting

    real(r8) :: evplimit               ! temp variable for evaporation limits
    real(r8) :: rlat(inncol)

    integer :: i,k                     ! longitude,level indices


!-----------------------------------------------------------------------

!!    write(*,*)' in yzx_conv_evap1'

! convert input precip to kg/m2/s
    prec(:inncol) = prec(:inncol)*1000._r8

! determine saturation vapor pressure
!    call qsat(t(1:inncol, 1:nlev), pmid(1:inncol, 1:nlev), &
!         es(1:inncol, 1:nlev), qs(1:inncol, 1:nlev))
    call cal_qsat2d(t(1:inncol, 1:nlev), pmid(1:inncol, 1:nlev), qs(1:inncol, 1:nlev))

! determine ice fraction in rain production (use cloud water parameterization fraction at present)
    call cldfrc_fice(inncol, t, fice, fsnow_conv)
!!    write(*,*)' in yzx_conv_evap2'

! zero the flux integrals on the top boundary
    flxprec(:inncol,1) = 0._r8
    flxsnow(:inncol,1) = 0._r8
    evpvint(:inncol)   = 0._r8

!!    write(*,*)' in yzx_conv_evap3'
    do k = 1, nlev
       do i = 1, inncol

! Melt snow falling into layer, if necessary. 
          if (t(i,k) > tmelt) then
             flxsntm(i) = 0._r8
             snowmlt(i) = flxsnow(i,k) * gravit/ pdel(i,k)
          else
             flxsntm(i) = flxsnow(i,k)
             snowmlt(i) = 0._r8
          end if
!    write(*,*)' in yzx_conv_evap4'

! relative humidity depression must be > 0 for evaporation
          evplimit = max(1._r8 - q(i,k)/qs(i,k), 0._r8)

! total evaporation depends on flux in the top of the layer
! flux prec is the net production above layer minus evaporation into environmet
          evpprec(i) = ke * (1._r8 - cldfrc(i,k)) * evplimit * sqrt(flxprec(i,k))
!**********************************************************
!!          evpprec(i) = 0.    ! turn off evaporation for now
!**********************************************************

! Don't let evaporation supersaturate layer (approx). Layer may already be saturated.
! Currently does not include heating/cooling change to qs
          evplimit   = max(0._r8, (qs(i,k)-q(i,k)) / deltat)

! Don't evaporate more than is falling into the layer - do not evaporate rain formed
! in this layer but if precip production is negative, remove from the available precip
! Negative precip production occurs because of evaporation in downdrafts.
!!$          evplimit   = flxprec(i,k) * gravit / pdel(i,k) + min(prdprec(i,k), 0.)
          evplimit   = min(evplimit, flxprec(i,k) * gravit / pdel(i,k))

! Total evaporation cannot exceed input precipitation
          evplimit   = min(evplimit, (prec(i) - evpvint(i)) * gravit / pdel(i,k))

          evpprec(i) = min(evplimit, evpprec(i))

! evaporation of snow depends on snow fraction of total precipitation in the top after melting
          if (flxprec(i,k) > 0._r8) then
!            evpsnow(i) = evpprec(i) * flxsntm(i) / flxprec(i,k)
!            prevent roundoff problems
             work1 = min(max(0._r8,flxsntm(i)/flxprec(i,k)),1._r8)
             evpsnow(i) = evpprec(i) * work1
          else
             evpsnow(i) = 0._r8
          end if

! vertically integrated evaporation
          evpvint(i) = evpvint(i) + evpprec(i) * pdel(i,k)/gravit

! net precip production is production - evaporation
          ntprprd(i,k) = prdprec(i,k) - evpprec(i)
! net snow production is precip production * ice fraction - evaporation - melting
!pjrworks ntsnprd(i,k) = prdprec(i,k)*fice(i,k) - evpsnow(i) - snowmlt(i)
!pjrwrks2 ntsnprd(i,k) = prdprec(i,k)*fsnow_conv(i,k) - evpsnow(i) - snowmlt(i)
! the small amount added to flxprec in the work1 expression has been increased from 
! 1e-36 to 8.64e-11 (1e-5 mm/day).  This causes the temperature based partitioning
! scheme to be used for small flxprec amounts.  This is to address error growth problems.
#ifdef PERGRO
          work1 = min(max(0._r8,flxsnow(i,k)/(flxprec(i,k)+8.64e-11_r8)),1._r8)
#else
          if (flxprec(i,k).gt.0._r8) then
             work1 = min(max(0._r8,flxsnow(i,k)/flxprec(i,k)),1._r8)
          else
             work1 = 0._r8
          endif
#endif
          work2 = max(fsnow_conv(i,k), work1)
          if (snowmlt(i).gt.0._r8) work2 = 0._r8
!         work2 = fsnow_conv(i,k)
          ntsnprd(i,k) = prdprec(i,k)*work2 - evpsnow(i) - snowmlt(i)
          tend_s_snwprd  (i,k) = prdprec(i,k)*work2*latice
          tend_s_snwevmlt(i,k) = - ( evpsnow(i) + snowmlt(i) )*latice

! precipitation fluxes
          flxprec(i,k+1) = flxprec(i,k) + ntprprd(i,k) * pdel(i,k)/gravit
          flxsnow(i,k+1) = flxsnow(i,k) + ntsnprd(i,k) * pdel(i,k)/gravit

! protect against rounding error
          flxprec(i,k+1) = max(flxprec(i,k+1), 0._r8)
          flxsnow(i,k+1) = max(flxsnow(i,k+1), 0._r8)
! more protection (pjr)
!         flxsnow(i,k+1) = min(flxsnow(i,k+1), flxprec(i,k+1))

! heating (cooling) and moistening due to evaporation 
! - latent heat of vaporization for precip production has already been accounted for
! - snow is contained in prec
          tend_s(i,k)   =-evpprec(i)*latvap + ntsnprd(i,k)*latice
          tend_q(i,k) = evpprec(i)
       end do
    end do
!!    write(*,*)' in yzx_conv_evap5'

! set output precipitation rates (m/s)
    prec(:inncol) = flxprec(:inncol,nlev+1) / 1000._r8
    snow(:inncol) = flxsnow(:inncol,nlev+1) / 1000._r8

!**********************************************************
!!$    tend_s(:inncol,:)   = 0.      ! turn heating off
!**********************************************************
!!    write(*,*)' in yzx_conv_evap6'

  end subroutine zyx2_conv_evap



!subroutine convtran(lchnk   , &
subroutine convtran(lchnk, inncol, nlev,nlevp, &
                    doconvtran,q       ,ncnst   ,mu      ,md      , &
                    du      ,eu      ,ed      ,dp      ,dsubcld , &
                    jt      ,mx      ,ideep   ,il1g    ,il2g    , &
                    nstep   ,fracis  ,dqdt    ,dpdry   )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Convective transport of trace species
!
! Mixing ratios may be with respect to either dry or moist air
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: P. Rasch
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r8 => shr_kind_r8
   use constituents,    only: cnst_get_type_byind
   use ppgrid
   use abortutils, only: endrun

   implicit none
!-----------------------------------------------------------------------
!
! Input arguments
!
!MZ
   integer, intent(in) :: inncol,nlev,nlevp
   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: ncnst                 ! number of tracers to transport
   logical, intent(in) :: doconvtran(ncnst)     ! flag for doing convective transport
   real(r8), intent(in) :: q(inncol,nlev,ncnst)  ! Tracer array including moisture
   real(r8), intent(in) :: mu(inncol,nlev)       ! Mass flux up
   real(r8), intent(in) :: md(inncol,nlev)       ! Mass flux down
   real(r8), intent(in) :: du(inncol,nlev)       ! Mass detraining from updraft
   real(r8), intent(in) :: eu(inncol,nlev)       ! Mass entraining from updraft
   real(r8), intent(in) :: ed(inncol,nlev)       ! Mass entraining from downdraft
   real(r8), intent(in) :: dp(inncol,nlev)       ! Delta pressure between interfaces
   real(r8), intent(in) :: dsubcld(inncol)       ! Delta pressure from cloud base to sfc
   real(r8), intent(in) :: fracis(inncol,nlev,ncnst) ! fraction of tracer that is insoluble

   integer, intent(in) :: jt(inncol)         ! Index of cloud top for each column
   integer, intent(in) :: mx(inncol)         ! Index of cloud top for each column
   integer, intent(in) :: ideep(inncol)      ! Gathering array
   integer, intent(in) :: il1g              ! Gathered min lon indices over which to operate
   integer, intent(in) :: il2g              ! Gathered max lon indices over which to operate
   integer, intent(in) :: nstep             ! Time step index

   real(r8), intent(in) :: dpdry(inncol,nlev)       ! Delta pressure between interfaces


! input/output

   real(r8), intent(out) :: dqdt(inncol,nlev,ncnst)  ! Tracer tendency array

!--------------------------Local Variables------------------------------

   integer i                 ! Work index
   integer k                 ! Work index
   integer kbm               ! Highest altitude index of cloud base
   integer kk                ! Work index
   integer kkp1              ! Work index
   integer km1               ! Work index
   integer kp1               ! Work index
   integer ktm               ! Highest altitude index of cloud top
   integer m                 ! Work index

   real(r8) cabv                 ! Mix ratio of constituent above
   real(r8) cbel                 ! Mix ratio of constituent below
   real(r8) cdifr                ! Normalized diff between cabv and cbel
   real(r8) chat(inncol,nlev)     ! Mix ratio in env at interfaces
   real(r8) cond(inncol,nlev)     ! Mix ratio in downdraft at interfaces
   real(r8) const(inncol,nlev)    ! Gathered tracer array
   real(r8) fisg(inncol,nlev)     ! gathered insoluble fraction of tracer
   real(r8) conu(inncol,nlev)     ! Mix ratio in updraft at interfaces
   real(r8) dcondt(inncol,nlev)   ! Gathered tend array
   real(r8) small                ! A small number
   real(r8) mbsth                ! Threshold for mass fluxes
   real(r8) mupdudp              ! A work variable
   real(r8) minc                 ! A work variable
   real(r8) maxc                 ! A work variable
   real(r8) fluxin               ! A work variable
   real(r8) fluxout              ! A work variable
   real(r8) netflux              ! A work variable

   real(r8) dutmp(inncol,nlev)       ! Mass detraining from updraft
   real(r8) eutmp(inncol,nlev)       ! Mass entraining from updraft
   real(r8) edtmp(inncol,nlev)       ! Mass entraining from downdraft
   real(r8) dptmp(inncol,nlev)    ! Delta pressure between interfaces
!-----------------------------------------------------------------------
!
   small = 1.e-36_r8
! mbsth is the threshold below which we treat the mass fluxes as zero (in mb/s)
   mbsth = 1.e-15_r8

! Find the highest level top and bottom levels of convection
   ktm = nlev
   kbm = nlev
   do i = il1g, il2g
      ktm = min(ktm,jt(i))
      kbm = min(kbm,mx(i))
   end do

if(i<0)then
 k=25
   write(*,*)'------ lchnk in convtran 1 ',lchnk
   write(*,*)'in convtran 1 lq m=',m,doconvtran
   write(*,*)'q(:,k,1)',q(:,k,1)
   write(*,*)'q(:,k,2)',q(:,k,2)
   write(*,*)'q(:,k,4)',q(:,k,4)
   write(*,*)'q(:,k,5)',q(:,k,5)
   write(*,*)'mu',mu(:,k)
   write(*,*)'md',md(:,k)
   write(*,*)'du',du(:,k)
   write(*,*)'eu',eu(:,k)
   write(*,*)'ed',ed(:,k)
   write(*,*)'dp',dp(:,k)
   write(*,*)'dsubcld',dsubcld(:)
   write(*,*)'fracis(2)',fracis(:,k,4)
   write(*,*)'fracis(3)',fracis(:,k,5)
   write(*,*)'jt,mx,ideep,il1g,il2g,nstep',jt,mx,ideep,il1g,il2g,nstep
   write(*,*)'dpdry',dpdry(:,k)
endif

! Loop ever each constituent
   do m = 2, ncnst
      if (doconvtran(m)) then

         if (cnst_get_type_byind(m).eq.'dry') then
            do k = 1,nlev
               do i =il1g,il2g
                  dptmp(i,k) = dpdry(i,k)
                  dutmp(i,k) = du(i,k)*dp(i,k)/dpdry(i,k)
                  eutmp(i,k) = eu(i,k)*dp(i,k)/dpdry(i,k)
                  edtmp(i,k) = ed(i,k)*dp(i,k)/dpdry(i,k)
               end do
            end do
         else
            do k = 1,nlev
               do i =il1g,il2g
                  dptmp(i,k) = dp(i,k)
                  dutmp(i,k) = du(i,k)
                  eutmp(i,k) = eu(i,k)
                  edtmp(i,k) = ed(i,k)
               end do
            end do
         endif
!        dptmp = dp

! Gather up the constituent and set tend to zero
         do k = 1,nlev
            do i =il1g,il2g
               const(i,k) = q(ideep(i),k,m)
               fisg(i,k) = fracis(ideep(i),k,m)
            end do
         end do

! From now on work only with gathered data
! Interpolate environment tracer values to interfaces
         do k = 1,nlev
            km1 = max(1,k-1)
            do i = il1g, il2g
               minc = min(const(i,km1),const(i,k))
               maxc = max(const(i,km1),const(i,k))
               if (minc < 0) then
                  cdifr = 0._r8
               else
                  cdifr = abs(const(i,k)-const(i,km1))/max(maxc,small)
               endif

! If the two layers differ significantly use a geometric averaging
! procedure
               if (cdifr > 1.E-6_r8) then
                  cabv = max(const(i,km1),maxc*1.e-12_r8)
                  cbel = max(const(i,k),maxc*1.e-12_r8)
                  chat(i,k) = log(cabv/cbel)/(cabv-cbel)*cabv*cbel

               else             ! Small diff, so just arithmetic mean
                  chat(i,k) = 0.5_r8* (const(i,k)+const(i,km1))
               end if

! Provisional up and down draft values
               conu(i,k) = chat(i,k)
               cond(i,k) = chat(i,k)

!              provisional tends
               dcondt(i,k) = 0._r8

            end do
         end do

! Do levels adjacent to top and bottom
         k = 2
         km1 = 1
         kk = nlev
         do i = il1g,il2g
            mupdudp = mu(i,kk) + dutmp(i,kk)*dptmp(i,kk)
            if (mupdudp > mbsth) then
               conu(i,kk) = (+eutmp(i,kk)*fisg(i,kk)*const(i,kk)*dptmp(i,kk))/mupdudp
            endif
            if (md(i,k) < -mbsth) then
               cond(i,k) =  (-edtmp(i,km1)*fisg(i,km1)*const(i,km1)*dptmp(i,km1))/md(i,k)
            endif
         end do

! Updraft from bottom to top
         do kk = nlev-1,1,-1
            kkp1 = min(nlev,kk+1)
            do i = il1g,il2g
               mupdudp = mu(i,kk) + dutmp(i,kk)*dptmp(i,kk)
               if (mupdudp > mbsth) then
                  conu(i,kk) = (  mu(i,kkp1)*conu(i,kkp1)+eutmp(i,kk)*fisg(i,kk)* &
                                  const(i,kk)*dptmp(i,kk) )/mupdudp
               endif
            end do
         end do

! Downdraft from top to bottom
         do k = 3,nlev
            km1 = max(1,k-1)
            do i = il1g,il2g
               if (md(i,k) < -mbsth) then
                  cond(i,k) =  (  md(i,km1)*cond(i,km1)-edtmp(i,km1)*fisg(i,km1)*const(i,km1) &
                                  *dptmp(i,km1) )/md(i,k)
               endif
            end do
         end do

         do k = ktm,nlev
            km1 = max(1,k-1)
            kp1 = min(nlev,k+1)
            do i = il1g,il2g

! version 1 hard to check for roundoff errors
!               dcondt(i,k) =
!     $                  +(+mu(i,kp1)* (conu(i,kp1)-chat(i,kp1))
!     $                    -mu(i,k)*   (conu(i,k)-chat(i,k))
!     $                    +md(i,kp1)* (cond(i,kp1)-chat(i,kp1))
!     $                    -md(i,k)*   (cond(i,k)-chat(i,k))
!     $                   )/dp(i,k)

! version 2 hard to limit fluxes
!               fluxin =  mu(i,kp1)*conu(i,kp1) + mu(i,k)*chat(i,k)
!     $                 -(md(i,k)  *cond(i,k)   + md(i,kp1)*chat(i,kp1))
!               fluxout = mu(i,k)*conu(i,k)     + mu(i,kp1)*chat(i,kp1)
!     $                 -(md(i,kp1)*cond(i,kp1) + md(i,k)*chat(i,k))

! version 3 limit fluxes outside convection to mass in appropriate layer
! these limiters are probably only safe for positive definite quantitities
! it assumes that mu and md already satify a courant number limit of 1
               fluxin =  mu(i,kp1)*conu(i,kp1)+ mu(i,k)*min(chat(i,k),const(i,km1)) &
                         -(md(i,k)  *cond(i,k) + md(i,kp1)*min(chat(i,kp1),const(i,kp1)))
               fluxout = mu(i,k)*conu(i,k) + mu(i,kp1)*min(chat(i,kp1),const(i,k)) &
                         -(md(i,kp1)*cond(i,kp1) + md(i,k)*min(chat(i,k),const(i,k)))

               netflux = fluxin - fluxout
               if (abs(netflux) < max(fluxin,fluxout)*1.e-12_r8) then
                  netflux = 0._r8
               endif
               dcondt(i,k) = netflux/dptmp(i,k)
            end do
         end do
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!DIR$ NOINTERCHANGE
         do k = kbm,nlev
            km1 = max(1,k-1)
            do i = il1g,il2g
               if (k == mx(i)) then

! version 1
!                  dcondt(i,k) = (1./dsubcld(i))*
!     $              (-mu(i,k)*(conu(i,k)-chat(i,k))
!     $               -md(i,k)*(cond(i,k)-chat(i,k))
!     $              )

! version 2
!                  fluxin =  mu(i,k)*chat(i,k) - md(i,k)*cond(i,k)
!                  fluxout = mu(i,k)*conu(i,k) - md(i,k)*chat(i,k)
! version 3
                  fluxin =  mu(i,k)*min(chat(i,k),const(i,km1)) - md(i,k)*cond(i,k)
                  fluxout = mu(i,k)*conu(i,k) - md(i,k)*min(chat(i,k),const(i,k))

                  netflux = fluxin - fluxout
                  if (abs(netflux) < max(fluxin,fluxout)*1.e-12_r8) then
                     netflux = 0._r8
                  endif
!                  dcondt(i,k) = netflux/dsubcld(i)
                  dcondt(i,k) = netflux/dptmp(i,k)
               else if (k > mx(i)) then
!                  dcondt(i,k) = dcondt(i,k-1)
                  dcondt(i,k) = 0._r8
               end if
            end do
         end do

! Initialize to zero everywhere, then scatter tendency back to full array
         dqdt(:,:,m) = 0._r8
         do k = 1,nlev
            kp1 = min(nlev,k+1)
!DIR$ CONCURRENT
            do i = il1g,il2g
               dqdt(ideep(i),k,m) = dcondt(i,k)
            end do
         end do

      end if      ! for doconvtran

   end do

if(i<0)then
   write(*,*)'in convtran done',lchnk
 k=25
   write(*,*)'3-- in convtran',doconvtran,ncnst,fracis(:,25,k)
   write(*,*)'q(:,k,1)',q(:,k,1)
   write(*,*)'q(:,k,4)',q(:,k,4)
   write(*,*)'q(:,k,5)',q(:,k,5)
endif

   return
end subroutine convtran

!=========================================================================================

!MZ subroutine momtran(lchnk, inncol, &
subroutine momtran(lchnk, inncol, nlev,nlevp, &
                    domomtran,q       ,ncnst   ,mu      ,md    , &
                    du      ,eu      ,ed      ,dp      ,dsubcld , &
                    jt      ,mx      ,ideep   ,il1g    ,il2g    , &
                    nstep   ,dqdt    ,pguall     ,pgdall, icwu, icwd, dt, seten    )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Convective transport of momentum
!
! Mixing ratios may be with respect to either dry or moist air
! 
! Method: 
! Based on the convtran subroutine by P. Rasch
! <Also include any applicable external references.> 
! 
! Author: J. Richter and P. Rasch
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r8 => shr_kind_r8
   use constituents,    only: cnst_get_type_byind
   use ppgrid
   use abortutils, only: endrun

   implicit none
!-----------------------------------------------------------------------
!
! Input arguments
!MZ   
   integer, intent(in) :: nlev,nlevp

   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: inncol                  ! number of atmospheric columns
   integer, intent(in) :: ncnst                 ! number of tracers to transport
   logical, intent(in) :: domomtran(ncnst)      ! flag for doing convective transport
   real(r8), intent(in) :: q(inncol,nlev,ncnst)  ! Wind array
   real(r8), intent(in) :: mu(inncol,nlev)       ! Mass flux up
   real(r8), intent(in) :: md(inncol,nlev)       ! Mass flux down
   real(r8), intent(in) :: du(inncol,nlev)       ! Mass detraining from updraft
   real(r8), intent(in) :: eu(inncol,nlev)       ! Mass entraining from updraft
   real(r8), intent(in) :: ed(inncol,nlev)       ! Mass entraining from downdraft
   real(r8), intent(in) :: dp(inncol,nlev)       ! Delta pressure between interfaces
   real(r8), intent(in) :: dsubcld(inncol)       ! Delta pressure from cloud base to sfc
   real(r8), intent(in)    :: dt                       !  time step in seconds : 2*delta_t

   integer, intent(in) :: jt(inncol)         ! Index of cloud top for each column
   integer, intent(in) :: mx(inncol)         ! Index of cloud top for each column
   integer, intent(in) :: ideep(inncol)      ! Gathering array
   integer, intent(in) :: il1g              ! Gathered min lon indices over which to operate
   integer, intent(in) :: il2g              ! Gathered max lon indices over which to operate
   integer, intent(in) :: nstep             ! Time step index



! input/output

   real(r8), intent(out) :: dqdt(inncol,nlev,ncnst)  ! Tracer tendency array

!--------------------------Local Variables------------------------------

   integer i                 ! Work index
   integer k                 ! Work index
   integer kbm               ! Highest altitude index of cloud base
   integer kk                ! Work index
   integer kkp1              ! Work index
   integer kkm1              ! Work index
   integer km1               ! Work index
   integer kp1               ! Work index
   integer ktm               ! Highest altitude index of cloud top
   integer m                 ! Work index
   integer ii                 ! Work index

   real(r8) cabv                 ! Mix ratio of constituent above
   real(r8) cbel                 ! Mix ratio of constituent below
   real(r8) cdifr                ! Normalized diff between cabv and cbel
   real(r8) chat(inncol,nlev)     ! Mix ratio in env at interfaces
   real(r8) cond(inncol,nlev)     ! Mix ratio in downdraft at interfaces
   real(r8) const(inncol,nlev)    ! Gathered wind array
   real(r8) conu(inncol,nlev)     ! Mix ratio in updraft at interfaces
   real(r8) dcondt(inncol,nlev)   ! Gathered tend array
   real(r8) small                ! A small number
   real(r8) mbsth                ! Threshold for mass fluxes
   real(r8) mupdudp              ! A work variable
   real(r8) minc                 ! A work variable
   real(r8) maxc                 ! A work variable
   real(r8) fluxin               ! A work variable
   real(r8) fluxout              ! A work variable
   real(r8) netflux              ! A work variable

   real(r8) momcu                ! constant for updraft pressure gradient term
   real(r8) momcd                ! constant for downdraft pressure gradient term
   real(r8) sum                  ! sum
   real(r8) sum2                  ! sum2
 
   real(r8) mududp(inncol,nlev) ! working variable
   real(r8) mddudp(inncol,nlev)     ! working variable

   real(r8) pgu(inncol,nlev)      ! Pressure gradient term for updraft
   real(r8) pgd(inncol,nlev)      ! Pressure gradient term for downdraft

   real(r8),intent(out) ::  pguall(inncol,nlev,ncnst)      ! Apparent force from  updraft PG
   real(r8),intent(out) ::  pgdall(inncol,nlev,ncnst)      ! Apparent force from  downdraft PG

   real(r8),intent(out) ::  icwu(inncol,nlev,ncnst)      ! In-cloud winds in updraft
   real(r8),intent(out) ::  icwd(inncol,nlev,ncnst)      ! In-cloud winds in downdraft

   real(r8),intent(out) ::  seten(inncol,nlev) ! Dry static energy tendency
   real(r8)                 gseten(inncol,nlev) ! Gathered dry static energy tendency

   real(r8)  mflux(inncol,nlevp,ncnst)   ! Gathered momentum flux

   real(r8)  wind0(inncol,nlev,ncnst)       !  gathered  wind before time step
   real(r8)  windf(inncol,nlev,ncnst)       !  gathered  wind after time step
   real(r8) fkeb, fket, ketend_cons, ketend, utop, ubot, vtop, vbot, gset2
   

!-----------------------------------------------------------------------
!

! Initialize outgoing fields
   pguall(:,:,:)     = 0.0_r8
   pgdall(:,:,:)     = 0.0_r8
! Initialize in-cloud winds to environmental wind
   icwu(:inncol,:,:)       = q(:inncol,:,:)
   icwd(:inncol,:,:)       = q(:inncol,:,:)

! Initialize momentum flux and  final winds
   mflux(:,:,:)       = 0.0_r8
   wind0(:,:,:)         = 0.0_r8
   windf(:,:,:)         = 0.0_r8

! Initialize dry static energy

   seten(:,:)         = 0.0_r8
   gseten(:,:)         = 0.0_r8

! Define constants for parameterization

   momcu = 0.4_r8
   momcd = 0.4_r8

   small = 1.e-36_r8
! mbsth is the threshold below which we treat the mass fluxes as zero (in mb/s)
   mbsth = 1.e-15_r8

! Find the highest level top and bottom levels of convection
   ktm = nlev
   kbm = nlev
   do i = il1g, il2g
      ktm = min(ktm,jt(i))
      kbm = min(kbm,mx(i))
   end do

! Loop ever each wind component
   do m = 1, ncnst                    !start at m = 1 to transport momentum
      if (domomtran(m)) then

! Gather up the winds and set tend to zero
         do k = 1,nlev
            do i =il1g,il2g
               const(i,k) = q(ideep(i),k,m)
                wind0(i,k,m) = const(i,k)
            end do
         end do


! From now on work only with gathered data

! Interpolate winds to interfaces

         do k = 1,nlev
            km1 = max(1,k-1)
            do i = il1g, il2g

               ! use arithmetic mean
               chat(i,k) = 0.5_r8* (const(i,k)+const(i,km1))

! Provisional up and down draft values
               conu(i,k) = chat(i,k)
               cond(i,k) = chat(i,k)

!              provisional tends
               dcondt(i,k) = 0._r8

            end do
         end do


!
! Pressure Perturbation Term
! 

      !Top boundary:  assume mu is zero 

         k=1
         pgu(:il2g,k) = 0.0_r8
         pgd(:il2g,k) = 0.0_r8

         do k=2,nlev-1
            km1 = max(1,k-1)
            kp1 = min(nlev,k+1)
            do i = il1g,il2g
            
               !interior points

               mududp(i,k) =  ( mu(i,k) * (const(i,k)- const(i,km1))/dp(i,km1) &
                           +  mu(i,kp1) * (const(i,kp1) - const(i,k))/dp(i,k))

               pgu(i,k) = - momcu * 0.5_r8 * mududp(i,k)
                           

               mddudp(i,k) =  ( md(i,k) * (const(i,k)- const(i,km1))/dp(i,km1) &
                           +  md(i,kp1) * (const(i,kp1) - const(i,k))/dp(i,k))

               pgd(i,k) = - momcd * 0.5_r8 * mddudp(i,k)


            end do
         end do

       ! bottom boundary 
       k = nlev
       km1 = max(1,k-1)
       do i=il1g,il2g

          mududp(i,k) =   mu(i,k) * (const(i,k)- const(i,km1))/dp(i,km1)
          pgu(i,k) = - momcu *  mududp(i,k)
          
          mddudp(i,k) =   md(i,k) * (const(i,k)- const(i,km1))/dp(i,km1) 

          pgd(i,k) = - momcd * mddudp(i,k)
          
       end do
       

!
! In-cloud velocity calculations
!

! Do levels adjacent to top and bottom
         k = 2
         km1 = 1
         kk = nlev
         kkm1 = max(1,kk-1)
         do i = il1g,il2g
            mupdudp = mu(i,kk) + du(i,kk)*dp(i,kk)
            if (mupdudp > mbsth) then
                 
               conu(i,kk) = (+eu(i,kk)*const(i,kk)*dp(i,kk)+pgu(i,kk)*dp(i,kk))/mupdudp
            endif
            if (md(i,k) < -mbsth) then
               cond(i,k) =  (-ed(i,km1)*const(i,km1)*dp(i,km1))-pgd(i,km1)*dp(i,km1)/md(i,k)
            endif

                        
         end do



! Updraft from bottom to top
         do kk = nlev-1,1,-1
            kkm1 = max(1,kk-1)
            kkp1 = min(nlev,kk+1)
            do i = il1g,il2g
               mupdudp = mu(i,kk) + du(i,kk)*dp(i,kk)
               if (mupdudp > mbsth) then
            
                  conu(i,kk) = (  mu(i,kkp1)*conu(i,kkp1)+eu(i,kk)* &
                                  const(i,kk)*dp(i,kk)+pgu(i,kk)*dp(i,kk))/mupdudp
               endif
            end do

         end do


! Downdraft from top to bottom
         do k = 3,nlev
            km1 = max(1,k-1)
            do i = il1g,il2g
               if (md(i,k) < -mbsth) then
                            
                  cond(i,k) =  (  md(i,km1)*cond(i,km1)-ed(i,km1)*const(i,km1) &
                                  *dp(i,km1)-pgd(i,km1)*dp(i,km1) )/md(i,k)

               endif
            end do
         end do


         sum = 0._r8
         sum2 = 0._r8


         do k = ktm,nlev
            km1 = max(1,k-1)
            kp1 = min(nlev,k+1)
            do i = il1g,il2g
               ii = ideep(i)
	
! version 1 hard to check for roundoff errors
               dcondt(i,k) =  &
                           +(mu(i,kp1)* (conu(i,kp1)-chat(i,kp1)) &
                           -mu(i,k)*   (conu(i,k)-chat(i,k))      &
                           +md(i,kp1)* (cond(i,kp1)-chat(i,kp1)) &
                           -md(i,k)*   (cond(i,k)-chat(i,k)) &
                          )/dp(i,k)

            end do
         end do

  ! dcont for bottom layer
          !
          !DIR$ NOINTERCHANGE
          do k = kbm,nlev
             km1 = max(1,k-1)
             do i = il1g,il2g
                if (k == mx(i)) then

                   ! version 1
                   dcondt(i,k) = (1._r8/dp(i,k))*   &  
                        (-mu(i,k)*(conu(i,k)-chat(i,k)) &
                        -md(i,k)*(cond(i,k)-chat(i,k)) &
                        )
                end if
             end do
          end do

! Initialize to zero everywhere, then scatter tendency back to full array
         dqdt(:,:,m) = 0._r8

         do k = 1,nlev
            do i = il1g,il2g
               ii = ideep(i)
               dqdt(ii,k,m) = dcondt(i,k)
    ! Output apparent force on the mean flow from pressure gradient
               pguall(ii,k,m) = -pgu(i,k)
               pgdall(ii,k,m) = -pgd(i,k)
               icwu(ii,k,m)   =  conu(i,k)
               icwd(ii,k,m)   =  cond(i,k)
            end do
         end do

          ! Calculate momentum flux in units of mb*m/s2 

          do k = ktm,nlev
             do i = il1g,il2g
                ii = ideep(i)
                mflux(i,k,m) = &
                     -mu(i,k)*   (conu(i,k)-chat(i,k))      &
                     -md(i,k)*   (cond(i,k)-chat(i,k))
             end do
          end do


          ! Calculate winds at the end of the time step 

          do k = ktm,nlev
             do i = il1g,il2g
                ii = ideep(i)
                km1 = max(1,k-1)
                kp1 = k+1
                windf(i,k,m) = const(i,k)    -   (mflux(i,kp1,m) - mflux(i,k,m)) * dt /dp(i,k)

             end do
          end do

       end if      ! for domomtran
   end do

 ! Need to add an energy fix to account for the dissipation of kinetic energy
    ! Formulation follows from Boville and Bretherton (2003)
    ! formulation by PJR

    do k = ktm,nlev
       km1 = max(1,k-1)
       kp1 = min(nlev,k+1)
       do i = il1g,il2g

          ii = ideep(i)

          ! calculate the KE fluxes at top and bot of layer 
          ! based on a discrete approximation to b&b eq(35) F_KE = u*F_u + v*F_v at interface
          utop = (wind0(i,k,1)+wind0(i,km1,1))/2._r8
          vtop = (wind0(i,k,2)+wind0(i,km1,2))/2._r8
          ubot = (wind0(i,kp1,1)+wind0(i,k,1))/2._r8
          vbot = (wind0(i,kp1,2)+wind0(i,k,2))/2._r8
          fket = utop*mflux(i,k,1)   + vtop*mflux(i,k,2)    ! top of layer
          fkeb = ubot*mflux(i,k+1,1) + vbot*mflux(i,k+1,2)  ! bot of layer

          ! divergence of these fluxes should give a conservative redistribution of KE
          ketend_cons = (fket-fkeb)/dp(i,k)

          ! tendency in kinetic energy resulting from the momentum transport
          ketend = ((windf(i,k,1)**2 + windf(i,k,2)**2) - (wind0(i,k,1)**2 + wind0(i,k,2)**2))*0.5_r8/dt

          ! the difference should be the dissipation
          gset2 = ketend_cons - ketend
          gseten(i,k) = gset2

       end do

    end do

    ! Scatter dry static energy to full array
    do k = 1,nlev
       do i = il1g,il2g
          ii = ideep(i)
          seten(ii,k) = gseten(i,k)

       end do
    end do

   return
end subroutine momtran

!=========================================================================================

!MZ subroutine buoyan(lchnk   ,inncol    , &
subroutine buoyan(lchnk   ,inncol    , nlev, nlevp, &
                  q       ,t       ,p       ,z       ,pf      , &
                  tp      ,qstp    ,tl      ,rl      ,cape    , &
                  pblt    ,lcl     ,lel     ,lon     ,mx      , &
                  rd      ,grav    ,cp      ,msg     , &
                  tpert   )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! <Say what the routine does> 
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author:
! This is contributed code not fully standardized by the CCM core group.
! The documentation has been enhanced to the degree that we are able.
! Reviewed:          P. Rasch, April 1996
! 
!-----------------------------------------------------------------------
   implicit none
!-----------------------------------------------------------------------
!
! input arguments
!
   integer, intent(in) :: lchnk                 ! chunk identifier
!MZ   integer, intent(in) :: inncol                  ! number of atmospheric columns
   integer, intent(in) :: inncol, nlev, nlevp                  ! number of atmospheric columns

   real(r8), intent(in) :: q(inncol,nlev)        ! spec. humidity
   real(r8), intent(in) :: t(inncol,nlev)        ! temperature
   real(r8), intent(in) :: p(inncol,nlev)        ! pressure
   real(r8), intent(in) :: z(inncol,nlev)        ! height
   real(r8), intent(in) :: pf(inncol,nlev+1)     ! pressure at interfaces
   real(r8), intent(in) :: pblt(inncol)          ! index of pbl depth
   real(r8), intent(in) :: tpert(inncol)         ! perturbation temperature by pbl processes

!
! output arguments
!
   real(r8), intent(out) :: tp(inncol,nlev)       ! parcel temperature
   real(r8), intent(out) :: qstp(inncol,nlev)     ! saturation mixing ratio of parcel
   real(r8), intent(out) :: tl(inncol)            ! parcel temperature at lcl
   real(r8), intent(out) :: cape(inncol)          ! convective aval. pot. energy.
   
   integer lcl(inncol)        !
   integer lel(inncol)        !
   integer lon(inncol)        ! level of onset of deep convection
   integer mx(inncol)         ! level of max moist static energy
!
!--------------------------Local Variables------------------------------
!
   real(r8) capeten(inncol,5)     ! provisional value of cape
!MZ
   real(r8) cinten(inncol,5)     ! provisional value of cape
   real(r8) tv(inncol,nlev)       !
   real(r8) tpv(inncol,nlev)      !
   real(r8) buoy(inncol,nlev)

   real(r8) a1(inncol)
   real(r8) a2(inncol)
   real(r8) estp(inncol)
   real(r8) pl(inncol)
   real(r8) plexp(inncol)
   real(r8) hmax(inncol)
   real(r8) hmn(inncol)
   real(r8) y(inncol)

   logical plge600(inncol)
   integer knt(inncol)
   integer lelten(inncol,5)

   real(r8) cp
   real(r8) e
   real(r8) grav

   integer i
   integer k
   integer msg
   integer n

   real(r8) rd
   real(r8) rl
#ifdef PERGRO
   real(r8) rhd
#endif
!
!-----------------------------------------------------------------------
!
   do n = 1,5
      do i = 1,inncol
         lelten(i,n) = nlev
         capeten(i,n) = 0._r8
      end do
   end do
!
   do i = 1,inncol
      lon(i) = nlev
      knt(i) = 0
      lel(i) = nlev
      mx(i) = lon(i)
      cape(i) = 0._r8
      hmax(i) = 0._r8
   end do

   tp(:inncol,:) = t(:inncol,:)
   qstp(:inncol,:) = q(:inncol,:)

!!! RBN - Initialize tv and buoy for output.
!!! tv=tv : tpv=tpv : qstp=q : buoy=0.
   tv(:inncol,:) = t(:inncol,:) *(1._r8+1.608_r8*q(:inncol,:))/ (1._r8+q(:inncol,:))
   tpv(:inncol,:) = tv(:inncol,:)
   buoy(:inncol,:) = 0._r8

!
! set "launching" level(mx) to be at maximum moist static energy.
! search for this level stops at planetary boundary layer top.
!
#ifdef PERGRO
   do k = nlev,msg + 1,-1
      do i = 1,inncol
         hmn(i) = cp*t(i,k) + grav*z(i,k) + rl*q(i,k)
!
! Reset max moist static energy level when relative difference exceeds 1.e-4
!

         rhd = (hmn(i) - hmax(i))/(hmn(i) + hmax(i))
         if (k >= nint(pblt(i)) .and. k <= lon(i) .and. rhd > -1.e-4_r8) then
            hmax(i) = hmn(i)
            mx(i) = k
         end if
      end do
   end do
#else
   do k = nlev,msg + 1,-1
      do i = 1,inncol
         hmn(i) = cp*t(i,k) + grav*z(i,k) + rl*q(i,k)
         if (k >= nint(pblt(i)) .and. k <= lon(i) .and. hmn(i) > hmax(i)) then
            hmax(i) = hmn(i)
            mx(i) = k
         end if
      end do
   end do
#endif
!
   do i = 1,inncol
      lcl(i) = mx(i)
      e = p(i,mx(i))*q(i,mx(i))/ (eps1+q(i,mx(i)))
      tl(i) = 2840._r8/ (3.5_r8*log(t(i,mx(i)))-log(e)-4.805_r8) + 55._r8
      if (tl(i) < t(i,mx(i))) then
         plexp(i) = (1._r8/ (0.2854_r8* (1._r8-0.28_r8*q(i,mx(i)))))
         pl(i) = p(i,mx(i))* (tl(i)/t(i,mx(i)))**plexp(i)
      else
         tl(i) = t(i,mx(i))
         pl(i) = p(i,mx(i))
      end if
   end do

!
! calculate lifting condensation level (lcl).
!
   do k = nlev,msg + 2,-1
      do i = 1,inncol
         if (k <= mx(i) .and. (p(i,k) > pl(i) .and. p(i,k-1) <= pl(i))) then
            lcl(i) = k - 1
         end if
      end do
   end do
!
! if lcl is above the nominal level of non-divergence (600 mbs),
! no deep convection is permitted (ensuing calculations
! skipped and cape retains initialized value of zero).
!
   do i = 1,inncol
      plge600(i) = pl(i).ge.600._r8
   end do
!
! initialize parcel properties in sub-cloud layer below lcl.
!
   do k = nlev,msg + 1,-1
      do i=1,inncol
         if (k > lcl(i) .and. k <= mx(i) .and. plge600(i)) then
            tv(i,k) = t(i,k)* (1._r8+1.608_r8*q(i,k))/ (1._r8+q(i,k))
            qstp(i,k) = q(i,mx(i))
            tp(i,k) = t(i,mx(i))* (p(i,k)/p(i,mx(i)))**(0.2854_r8* (1._r8-0.28_r8*q(i,mx(i))))
!
! buoyancy is increased by 0.5 k as in tiedtke
!
!-jjh          tpv (i,k)=tp(i,k)*(1.+1.608*q(i,mx(i)))/
!-jjh     1                     (1.+q(i,mx(i)))
            tpv(i,k) = (tp(i,k)+tpert(i))*(1._r8+1.608_r8*q(i,mx(i)))/ (1._r8+q(i,mx(i)))
            buoy(i,k) = tpv(i,k) - tv(i,k) + tiedke_add
         end if
      end do
   end do

!
! define parcel properties at lcl (i.e. level immediately above pl).
!
   do k = nlev,msg + 1,-1
      do i=1,inncol
         if (k == lcl(i) .and. plge600(i)) then
            tv(i,k) = t(i,k)* (1._r8+1.608_r8*q(i,k))/ (1._r8+q(i,k))
            qstp(i,k) = q(i,mx(i))
            tp(i,k) = tl(i)* (p(i,k)/pl(i))**(0.2854_r8* (1._r8-0.28_r8*qstp(i,k)))
!              estp(i)  =exp(21.656_r8 - 5418._r8/tp(i,k))
! use of different formulas for es has about 1 g/kg difference
! in qs at t= 300k, and 0.02 g/kg at t=263k, with the formula
! above giving larger qs.
            call qmmr_hPa(tp(i,k), p(i,k), estp(i), qstp(i,k))
            a1(i) = cp / rl + qstp(i,k) * (1._r8+ qstp(i,k) / eps1) * rl * eps1 / &
                    (rd * tp(i,k) ** 2)
            a2(i) = .5_r8* (qstp(i,k)* (1._r8+2._r8/eps1*qstp(i,k))* &
                    (1._r8+qstp(i,k)/eps1)*eps1**2*rl*rl/ &
                    (rd**2*tp(i,k)**4)-qstp(i,k)* &
                    (1._r8+qstp(i,k)/eps1)*2._r8*eps1*rl/ &
                    (rd*tp(i,k)**3))
            a1(i) = 1._r8/a1(i)
            a2(i) = -a2(i)*a1(i)**3
            y(i) = q(i,mx(i)) - qstp(i,k)
            tp(i,k) = tp(i,k) + a1(i)*y(i) + a2(i)*y(i)**2
            call qmmr_hPa(tp(i,k), p(i,k), estp(i), qstp(i,k))
!
! buoyancy is increased by 0.5 k in cape calculation.
! dec. 9, 1994
!-jjh          tpv(i,k) =tp(i,k)*(1.+1.608*qstp(i,k))/(1.+q(i,mx(i)))
!
            tpv(i,k) = (tp(i,k)+tpert(i))* (1._r8+1.608_r8*qstp(i,k)) / (1._r8+q(i,mx(i)))
            buoy(i,k) = tpv(i,k) - tv(i,k) + tiedke_add
         end if
      end do
   end do
!
! main buoyancy calculation.
!
   do k = nlev - 1,msg + 1,-1
      do i=1,inncol
         if (k < lcl(i) .and. plge600(i)) then
            tv(i,k) = t(i,k)* (1._r8+1.608_r8*q(i,k))/ (1._r8+q(i,k))
            qstp(i,k) = qstp(i,k+1)
            tp(i,k) = tp(i,k+1)* (p(i,k)/p(i,k+1))**(0.2854_r8* (1._r8-0.28_r8*qstp(i,k)))
            call qmmr_hPa(tp(i,k), p(i,k), estp(i), qstp(i,k))
            a1(i) = cp/rl + qstp(i,k)* (1._r8+qstp(i,k)/eps1)*rl*eps1/ (rd*tp(i,k)**2)
            a2(i) = .5_r8* (qstp(i,k)* (1._r8+2._r8/eps1*qstp(i,k))* &
                    (1._r8+qstp(i,k)/eps1)*eps1**2*rl*rl/ &
                    (rd**2*tp(i,k)**4)-qstp(i,k)* &
                    (1._r8+qstp(i,k)/eps1)*2._r8*eps1*rl/ &
                    (rd*tp(i,k)**3))
            a1(i) = 1._r8/a1(i)
            a2(i) = -a2(i)*a1(i)**3
            y(i) = qstp(i,k+1) - qstp(i,k)
            tp(i,k) = tp(i,k) + a1(i)*y(i) + a2(i)*y(i)**2
            call qmmr_hPa(tp(i,k), p(i,k), estp(i), qstp(i,k))
!-jjh          tpv(i,k) =tp(i,k)*(1.+1.608*qstp(i,k))/
!jt            (1.+q(i,mx(i)))
            tpv(i,k) = (tp(i,k)+tpert(i))* (1._r8+1.608_r8*qstp(i,k))/(1._r8+q(i,mx(i)))
            buoy(i,k) = tpv(i,k) - tv(i,k) + tiedke_add
         end if
      end do
   end do

!
   do k = msg + 2,nlev
      do i = 1,inncol
         if (k < lcl(i) .and. plge600(i)) then
            if (buoy(i,k+1) > 0._r8 .and. buoy(i,k) <= 0._r8) then
               knt(i) = min(5,knt(i) + 1)
               lelten(i,knt(i)) = k
            end if
         end if
      end do
   end do
!
! calculate convective available potential energy (cape).
!
   do n = 1,5
      do k = msg + 1,nlev
         do i = 1,inncol
            if (plge600(i) .and. k <= mx(i) .and. k > lelten(i,n)) then
               capeten(i,n) = capeten(i,n) + rd*buoy(i,k)*log(pf(i,k+1)/pf(i,k))
            end if
         end do
      end do
   end do
!
! find maximum cape from all possible tentative capes from
! one sounding,
! and use it as the final cape, april 26, 1995
!
   do n = 1,5
      do i = 1,inncol
         if (capeten(i,n) > cape(i)) then
            cape(i) = capeten(i,n)
            lel(i) = lelten(i,n)
         end if
      end do
   end do

! put lower bound on cape for diagnostic purposes.
!
   do i = 1,inncol
      cape(i) = max(cape(i), 0._r8)
   end do
!
   return
end subroutine buoyan

!subroutine cldprp(lchnk   ,  &
subroutine cldprp(lchnk   , inncol, nlev, nlevp,  &
                  q       ,t       ,u       ,v       ,p       , &
                  z       ,s       ,mu      ,eu      ,du      , &
                  md      ,ed      ,sd      ,qd      ,mc      , &
                  qu      ,su      ,zf      ,qst     ,hmn     , &
                  hsat    ,shat    ,ql      , &
                  cmeg    ,jb      ,lel     ,jt      ,jlcl    , &
                  mx      ,j0      ,jd      ,rl      ,il2g    , &
                  rd      ,grav    ,cp      ,msg     , &
                  pflx    ,evp     ,cu      ,rprd    ,limcnv  ,landfrac)
!----------------------------------------------------------------------- 
! 
! Purpose: 
! <Say what the routine does> 
! 
! Method: 
! may 09/91 - guang jun zhang, m.lazare, n.mcfarlane.
!             original version cldprop.
! 
! Author: See above, modified by P. Rasch
! This is contributed code not fully standardized by the CCM core group.
!
! this code is very much rougher than virtually anything else in the CCM
! there are debug statements left strewn about and code segments disabled
! these are to facilitate future development. We expect to release a
! cleaner code in a future release
!
! the documentation has been enhanced to the degree that we are able
!
!-----------------------------------------------------------------------

   implicit none

!------------------------------------------------------------------------------
!
! Input arguments
!
!MZ   integer, intent(in) :: lchnk                  ! chunk identifier
   integer, intent(in) :: lchnk, inncol, nlev, nlevp                  ! chunk identifier

   real(r8), intent(in) :: q(inncol,nlev)         ! spec. humidity of env
   real(r8), intent(in) :: t(inncol,nlev)         ! temp of env
   real(r8), intent(in) :: p(inncol,nlev)         ! pressure of env
   real(r8), intent(in) :: z(inncol,nlev)         ! height of env
   real(r8), intent(in) :: s(inncol,nlev)         ! normalized dry static energy of env
   real(r8), intent(in) :: zf(inncol,nlevp)       ! height of interfaces
   real(r8), intent(in) :: u(inncol,nlev)         ! zonal velocity of env
   real(r8), intent(in) :: v(inncol,nlev)         ! merid. velocity of env

   real(r8), intent(in) :: landfrac(inncol) ! RBN Landfrac

   integer, intent(in) :: jb(inncol)              ! updraft base level
   integer, intent(in) :: lel(inncol)             ! updraft launch level
   integer, intent(out) :: jt(inncol)              ! updraft plume top
   integer, intent(out) :: jlcl(inncol)            ! updraft lifting cond level
   integer, intent(in) :: mx(inncol)              ! updraft base level (same is jb)
   integer, intent(out) :: j0(inncol)              ! level where updraft begins detraining
   integer, intent(out) :: jd(inncol)              ! level of downdraft
   integer, intent(in) :: limcnv                 ! convection limiting level
   integer, intent(in) :: il2g                   !CORE GROUP REMOVE
   integer, intent(in) :: msg                    ! missing moisture vals (always 0)
   real(r8), intent(in) :: rl                    ! latent heat of vap
   real(r8), intent(in) :: shat(inncol,nlev)      ! interface values of dry stat energy
!
! output
!
   real(r8), intent(out) :: rprd(inncol,nlev)     ! rate of production of precip at that layer
   real(r8), intent(out) :: du(inncol,nlev)       ! detrainement rate of updraft
   real(r8), intent(out) :: ed(inncol,nlev)       ! entrainment rate of downdraft
   real(r8), intent(out) :: eu(inncol,nlev)       ! entrainment rate of updraft
   real(r8), intent(out) :: hmn(inncol,nlev)      ! moist stat energy of env
   real(r8), intent(out) :: hsat(inncol,nlev)     ! sat moist stat energy of env
   real(r8), intent(out) :: mc(inncol,nlev)       ! net mass flux
   real(r8), intent(out) :: md(inncol,nlev)       ! downdraft mass flux
   real(r8), intent(out) :: mu(inncol,nlev)       ! updraft mass flux
   real(r8), intent(out) :: pflx(inncol,nlevp)    ! precipitation flux thru layer
   real(r8), intent(out) :: qd(inncol,nlev)       ! spec humidity of downdraft
   real(r8), intent(out) :: ql(inncol,nlev)       ! liq water of updraft
   real(r8), intent(out) :: qst(inncol,nlev)      ! saturation mixing ratio of env.
   real(r8), intent(out) :: qu(inncol,nlev)       ! spec hum of updraft
   real(r8), intent(out) :: sd(inncol,nlev)       ! normalized dry stat energy of downdraft
   real(r8), intent(out) :: su(inncol,nlev)       ! normalized dry stat energy of updraft


   real(r8) rd                   ! gas constant for dry air
   real(r8) grav                 ! gravity
   real(r8) cp                   ! heat capacity of dry air

!
! Local workspace
!
!xiex
   real(r8) intsum
   real(r8) rh(inncol,nlev)
   real(r8) qsat(inncol,nlev)
   real(r8) fscale_up(inncol,nlev)
   real(r8) entrain_rates_bulk_up(inncol,nlev)
   real(r8) detrain_rates_bulk_up(inncol,nlev)

   real(r8) gamma(inncol,nlev)
   real(r8) dz(inncol,nlev)
   real(r8) iprm(inncol,nlev)
   real(r8) hu(inncol,nlev)
   real(r8) hd(inncol,nlev)
   real(r8) eps(inncol,nlev)
   real(r8) f(inncol,nlev)
   real(r8) k1(inncol,nlev)
   real(r8) i2(inncol,nlev)
   real(r8) ihat(inncol,nlev)
   real(r8) i3(inncol,nlev)
   real(r8) idag(inncol,nlev)
   real(r8) i4(inncol,nlev)
   real(r8) qsthat(inncol,nlev)
   real(r8) hsthat(inncol,nlev)
   real(r8) gamhat(inncol,nlev)
   real(r8) cu(inncol,nlev)
   real(r8) evp(inncol,nlev)
   real(r8) cmeg(inncol,nlev)
   real(r8) qds(inncol,nlev)
! RBN For c0mask
   real(r8) c0mask(inncol)

   real(r8) hmin(inncol)
   real(r8) expdif(inncol)
   real(r8) expnum(inncol)
   real(r8) ftemp(inncol)
   real(r8) eps0(inncol)
   real(r8) rmue(inncol)
   real(r8) zuef(inncol)
   real(r8) zdef(inncol)
   real(r8) epsm(inncol)
   real(r8) ratmjb(inncol)
   real(r8) est(inncol)
   real(r8) totpcp(inncol)
   real(r8) totevp(inncol)
   real(r8) alfa(inncol)
   real(r8) ql1
   real(r8) tu
   real(r8) estu
   real(r8) qstu

   real(r8) small
   real(r8) mdt

   integer khighest
   integer klowest
   integer kount
   integer i,k

   logical doit(inncol)
   logical done(inncol)
!
!------------------------------------------------------------------------------
!
   do i = 1,il2g
      ftemp(i) = 0._r8
      expnum(i) = 0._r8
      expdif(i) = 0._r8
      c0mask(i)  = c0_ocn * (1._r8-landfrac(i)) +   c0_lnd * landfrac(i) 
   end do
!
!jr Change from msg+1 to 1 to prevent blowup
!
   do k = 1,nlev
      do i = 1,il2g
         dz(i,k) = zf(i,k) - zf(i,k+1)
      end do
   end do

!
! initialize many output and work variables to zero
!
   pflx(:il2g,1) = 0
!MZ
   rprd(:,:) = 0._r8

   do k = 1,nlev
      do i = 1,il2g
         k1(i,k) = 0._r8
         i2(i,k) = 0._r8
         i3(i,k) = 0._r8
         i4(i,k) = 0._r8
         mu(i,k) = 0._r8
         f(i,k) = 0._r8
         eps(i,k) = 0._r8
         eu(i,k) = 0._r8
         du(i,k) = 0._r8
         ql(i,k) = 0._r8
         cu(i,k) = 0._r8
         evp(i,k) = 0._r8
         cmeg(i,k) = 0._r8
         qds(i,k) = q(i,k)
         md(i,k) = 0._r8
         ed(i,k) = 0._r8
         sd(i,k) = s(i,k)
         qd(i,k) = q(i,k)
         mc(i,k) = 0._r8
         qu(i,k) = q(i,k)
         su(i,k) = s(i,k)
         call qmmr_hPa(t(i,k), p(i,k), est(i), qst(i,k))
!++bee
         if ( p(i,k)-est(i) <= 0._r8 ) then
            qst(i,k) = 1.0_r8
         end if
!--bee
         gamma(i,k) = qst(i,k)*(1._r8 + qst(i,k)/eps1)*eps1*rl/(rd*t(i,k)**2)*rl/cp
         hmn(i,k) = cp*t(i,k) + grav*z(i,k) + rl*q(i,k)
         hsat(i,k) = cp*t(i,k) + grav*z(i,k) + rl*qst(i,k)
         hu(i,k) = hmn(i,k)
         hd(i,k) = hmn(i,k)
         rprd(i,k) = 0._r8
      end do
   end do
!
!jr Set to zero things which make this routine blow up
!
   do k=1,msg
      do i=1,il2g
         rprd(i,k) = 0._r8
      end do
   end do
!
! interpolate the layer values of qst, hsat and gamma to
! layer interfaces
!
   do k = 1, msg+1
      do i = 1,il2g
         hsthat(i,k) = hsat(i,k)
         qsthat(i,k) = qst(i,k)
         gamhat(i,k) = gamma(i,k)
      end do
   end do
   do i = 1,il2g
      totpcp(i) = 0._r8
      totevp(i) = 0._r8
   end do
   do k = msg + 2,nlev
      do i = 1,il2g
         if (abs(qst(i,k-1)-qst(i,k)) > 1.E-6_r8) then
            qsthat(i,k) = log(qst(i,k-1)/qst(i,k))*qst(i,k-1)*qst(i,k)/ (qst(i,k-1)-qst(i,k))
         else
            qsthat(i,k) = qst(i,k)
         end if
         hsthat(i,k) = cp*shat(i,k) + rl*qsthat(i,k)
         if (abs(gamma(i,k-1)-gamma(i,k)) > 1.E-6_r8) then
            gamhat(i,k) = log(gamma(i,k-1)/gamma(i,k))*gamma(i,k-1)*gamma(i,k)/ &
                                (gamma(i,k-1)-gamma(i,k))
         else
            gamhat(i,k) = gamma(i,k)
         end if
      end do
   end do
!
! initialize cloud top to highest plume top.
!jr changed hard-wired 4 to limcnv+1 (not to exceed nlev)
!
   jt(:) = nlev
   do i = 1,il2g
      jt(i) = max(lel(i),limcnv+1)
      jt(i) = min(jt(i),nlev)
      jd(i) = nlev
      jlcl(i) = lel(i)
      hmin(i) = 1.E6_r8
   end do
!
! find the level of minimum hsat, where detrainment starts
!

   do k = msg + 1,nlev
      do i = 1,il2g
         if (hsat(i,k) <= hmin(i) .and. k >= jt(i) .and. k <= jb(i)) then
            hmin(i) = hsat(i,k)
            j0(i) = k
         end if
      end do
   end do
   do i = 1,il2g
      j0(i) = min(j0(i),jb(i)-2)
      j0(i) = max(j0(i),jt(i)+2)
!
! Fix from Guang Zhang to address out of bounds array reference
!
      j0(i) = min(j0(i),nlev)
   end do
!
! Initialize certain arrays inside cloud
!
   do k = msg + 1,nlev
      do i = 1,il2g
         if (k >= jt(i) .and. k <= jb(i)) then
            hu(i,k) = hmn(i,mx(i)) + cp*tiedke_add
            su(i,k) = s(i,mx(i)) + tiedke_add
         end if
      end do
   end do
!
! *********************************************************
! compute taylor series for approximate eps(z) below
! *********************************************************
!
   do k = nlev - 1,msg + 1,-1
      do i = 1,il2g
         if (k < jb(i) .and. k >= jt(i)) then
            k1(i,k) = k1(i,k+1) + (hmn(i,mx(i))-hmn(i,k))*dz(i,k)
            ihat(i,k) = 0.5_r8* (k1(i,k+1)+k1(i,k))
            i2(i,k) = i2(i,k+1) + ihat(i,k)*dz(i,k)
            idag(i,k) = 0.5_r8* (i2(i,k+1)+i2(i,k))
            i3(i,k) = i3(i,k+1) + idag(i,k)*dz(i,k)
            iprm(i,k) = 0.5_r8* (i3(i,k+1)+i3(i,k))
            i4(i,k) = i4(i,k+1) + iprm(i,k)*dz(i,k)
         end if
      end do
   end do
!
! re-initialize hmin array for ensuing calculation.
!
   do i = 1,il2g
      hmin(i) = 1.E6_r8
   end do
   do k = msg + 1,nlev
      do i = 1,il2g
         if (k >= j0(i) .and. k <= jb(i) .and. hmn(i,k) <= hmin(i)) then
            hmin(i) = hmn(i,k)
            expdif(i) = hmn(i,mx(i)) - hmin(i)
         end if
      end do
   end do
!
! *********************************************************
! compute approximate eps(z) using above taylor series
! *********************************************************
!
   do k = msg + 2,nlev
      do i = 1,il2g
         expnum(i) = 0._r8
         ftemp(i) = 0._r8
         if (k < jt(i) .or. k >= jb(i)) then
            k1(i,k) = 0._r8
            expnum(i) = 0._r8
         else
            expnum(i) = hmn(i,mx(i)) - (hsat(i,k-1)*(zf(i,k)-z(i,k)) + &
                        hsat(i,k)* (z(i,k-1)-zf(i,k)))/(z(i,k-1)-z(i,k))
         end if
         if ((expdif(i) > 100._r8 .and. expnum(i) > 0._r8) .and. &
	     k1(i,k) > expnum(i)*dz(i,k)) then
            ftemp(i) = expnum(i)/k1(i,k)
            f(i,k) = ftemp(i) + i2(i,k)/k1(i,k)*ftemp(i)**2 + &
                     (2._r8*i2(i,k)**2-k1(i,k)*i3(i,k))/k1(i,k)**2* &
                     ftemp(i)**3 + (-5._r8*k1(i,k)*i2(i,k)*i3(i,k)+ &
                     5._r8*i2(i,k)**3+k1(i,k)**2*i4(i,k))/ &
                     k1(i,k)**3*ftemp(i)**4
            f(i,k) = max(f(i,k),0._r8)
            f(i,k) = min(f(i,k),0.0002_r8)
         end if
      end do
   end do
   do i = 1,il2g
      if (j0(i) < jb(i)) then
         if (f(i,j0(i)) < 1.E-6_r8 .and. f(i,j0(i)+1) > f(i,j0(i))) j0(i) = j0(i) + 1
      end if
   end do
   do k = msg + 2,nlev
      do i = 1,il2g
         if (k >= jt(i) .and. k <= j0(i)) then
            f(i,k) = max(f(i,k),f(i,k-1))
         end if
      end do
   end do
   do i = 1,il2g
      eps0(i) = f(i,j0(i))
      eps(i,jb(i)) = eps0(i)
   end do
!
! This is set to match the Rasch and Kristjansson paper
!
   do k = nlev,msg + 1,-1
      do i = 1,il2g
         if (k >= j0(i) .and. k <= jb(i)) then
            eps(i,k) = f(i,j0(i))
         end if
      end do
   end do
   do k = nlev,msg + 1,-1
      do i = 1,il2g
         if (k < j0(i) .and. k >= jt(i)) eps(i,k) = f(i,k)
      end do
   end do
!
! specify the updraft mass flux mu, entrainment eu, detrainment du
! and moist static energy hu.
! here and below mu, eu,du, md and ed are all normalized by mb
!
   do i = 1,il2g
      if (eps0(i) > 0._r8) then
         mu(i,jb(i)) = 1._r8
         eu(i,jb(i)) = mu(i,jb(i))/dz(i,jb(i))
      end if
   end do
   do k = nlev,msg + 1,-1
      do i = 1,il2g
         if (eps0(i) > 0._r8 .and. (k >= jt(i) .and. k < jb(i))) then
            zuef(i) = zf(i,k) - zf(i,jb(i))
            rmue(i) = (1._r8/eps0(i))* (exp(eps(i,k+1)*zuef(i))-1._r8)/zuef(i)
            mu(i,k) = (1._r8/eps0(i))* (exp(eps(i,k  )*zuef(i))-1._r8)/zuef(i)
            eu(i,k) = (rmue(i)-mu(i,k+1))/dz(i,k)
            du(i,k) = (rmue(i)-mu(i,k))/dz(i,k)
         end if
      end do
   end do
!xiex
!!   write(*,"(5f20.10)") mu
   !qsat = 0.622*( 611.2*exp(5417*(1/273.16-1/t)) )/p
   !rh = q/qsat
   !do k = nlev,msg + 1,-1
      !do i = 1,il2g
         !if (k >= jt(i) .and. k <= jb(i)) then
            !fscale_up(i,k) = (qsat(i,k)/qsat(i,jb(i)))**3
         !end if
      !end do
   !end do
   !entrain_rates_bulk_up = 5.e-5_r8*1._r8*(1.3-rh)*fscale_up
   !detrain_rates_bulk_up = 0.5e-5_r8*(1.6-rh)
   !do k = nlev,msg + 1,-1
      !do i = 1,il2g
         !if (k >= jt(i) .and. k < jb(i)) then
            !mu(i,k) = mu(i,k+1) + dz(i,k)*0.5 &
                !*( entrain_rates_bulk_up(i,k)-detrain_rates_bulk_up(i,k)+&
                !entrain_rates_bulk_up(i,k+1)-detrain_rates_bulk_up(i,k+1) )
         !end if
      !end do
   !end do
   !do k = nlev,msg + 1,-1
      !do i = 1,il2g
         !if (k >= jt(i) .and. k < jb(i)) then
            !mu(i,k) = exp( mu(i,k) )
         !end if
      !end do
   !end do
!!   write(*,"(5f20.10)") mu
   !eu = entrain_rates_bulk_up*mu
   !du = detrain_rates_bulk_up*mu

!
   khighest = nlevp
   klowest = 1
   do i=1,il2g
      khighest = min(khighest,lel(i))
      klowest = max(klowest,jb(i))
   end do
   do k = klowest-1,khighest,-1
      do i = 1,il2g
         if (k <= jb(i)-1 .and. k >= lel(i) .and. eps0(i) > 0._r8) then
            if (mu(i,k) < 0.02_r8) then
               hu(i,k) = hmn(i,k)
               mu(i,k) = 0._r8
               eu(i,k) = 0._r8
               du(i,k) = mu(i,k+1)/dz(i,k)
            else
               hu(i,k) = mu(i,k+1)/mu(i,k)*hu(i,k+1) + &
                         dz(i,k)/mu(i,k)* (eu(i,k)*hmn(i,k)- du(i,k)*hsat(i,k))
            end if
         end if
      end do
   end do
!
! reset cloud top index beginning from two layers above the
! cloud base (i.e. if cloud is only one layer thick, top is not reset
!
i=1
k=25
!write(*,*)'KK3 lchnk, i,rprd',lchnk,i,rprd(i,k)

   do i=1,il2g
      doit(i) = .true.
   end do
   do k=klowest-2,khighest-1,-1
      do i=1,il2g
         if (doit(i) .and. k <= jb(i)-2 .and. k >= lel(i)-1) then
  	   if (hu(i,k) <= hsthat(i,k) .and. hu(i,k+1) > hsthat(i,k+1) &
	       .and. mu(i,k) >= 0.02_r8) then
               if (hu(i,k)-hsthat(i,k) < -2000._r8) then
                  jt(i) = k + 1
                  doit(i) = .false.
               else
                  jt(i) = k
                  doit(i) = .false.
               end if
            else if (hu(i,k) > hu(i,jb(i)) .or. mu(i,k) < 0.02_r8) then
               jt(i) = k + 1
               doit(i) = .false.
            end if
         end if
      end do
   end do
   do k = nlev,msg + 1,-1
      do i = 1,il2g
         if (k >= lel(i) .and. k <= jt(i) .and. eps0(i) > 0._r8) then
            mu(i,k) = 0._r8
            eu(i,k) = 0._r8
            du(i,k) = 0._r8
            hu(i,k) = hmn(i,k)
         end if
         if (k == jt(i) .and. eps0(i) > 0._r8) then
            du(i,k) = mu(i,k+1)/dz(i,k)
            eu(i,k) = 0._r8
            mu(i,k) = 0._r8
         end if
      end do
   end do
!
! specify downdraft properties (no downdrafts if jd.ge.jb).
! scale down downward mass flux profile so that net flux
! (up-down) at cloud base in not negative.
!
i=1
k=25
!write(*,*)'KK4 lchnk, i,rprd',lchnk,i,rprd(i,k)

   do i = 1,il2g
!
! in normal downdraft strength run alfa=0.2.  In test4 alfa=0.1
!
      alfa(i) = 0.1_r8
      jt(i) = min(jt(i),jb(i)-1)
      jd(i) = max(j0(i),jt(i)+1)
      jd(i) = min(jd(i),jb(i))
      hd(i,jd(i)) = hmn(i,jd(i)-1)
      if (jd(i) < jb(i) .and. eps0(i) > 0._r8) then
         epsm(i) = eps0(i)
         md(i,jd(i)) = -alfa(i)*epsm(i)/eps0(i)
      end if
   end do
   do k = msg + 1,nlev
      do i = 1,il2g
         if ((k > jd(i) .and. k <= jb(i)) .and. eps0(i) > 0._r8) then
            zdef(i) = zf(i,jd(i)) - zf(i,k)
            md(i,k) = -alfa(i)/ (2._r8*eps0(i))*(exp(2._r8*epsm(i)*zdef(i))-1._r8)/zdef(i)
         end if
      end do
   end do
   do k = msg + 1,nlev
      do i = 1,il2g
         if ((k >= jt(i) .and. k <= jb(i)) .and. eps0(i) > 0._r8 .and. jd(i) < jb(i)) then
            ratmjb(i) = min(abs(mu(i,jb(i))/md(i,jb(i))),1._r8)
            md(i,k) = md(i,k)*ratmjb(i)
         end if
      end do
   end do

   small = 1.e-20_r8
   do k = msg + 1,nlev
      do i = 1,il2g
         if ((k >= jt(i) .and. k <= nlev) .and. eps0(i) > 0._r8) then
            ed(i,k-1) = (md(i,k-1)-md(i,k))/dz(i,k-1)
            mdt = min(md(i,k),-small)
            hd(i,k) = (md(i,k-1)*hd(i,k-1) - dz(i,k-1)*ed(i,k-1)*hmn(i,k-1))/mdt
         end if
      end do
   end do
!
! calculate updraft and downdraft properties.
!
   do k = msg + 2,nlev
      do i = 1,il2g
         if ((k >= jd(i) .and. k <= jb(i)) .and. eps0(i) > 0._r8 .and. jd(i) < jb(i)) then
            qds(i,k) = qsthat(i,k) + gamhat(i,k)*(hd(i,k)-hsthat(i,k))/ &
               (rl*(1._r8 + gamhat(i,k)))
         end if
      end do
   end do
!
   do i = 1,il2g
      done(i) = .false.
   end do
   kount = 0
   do k = nlev,msg + 2,-1
      do i = 1,il2g
         if (k == jb(i) .and. eps0(i) > 0._r8) then
            qu(i,k) = q(i,mx(i))
            su(i,k) = (hu(i,k)-rl*qu(i,k))/cp
         end if
         if (( .not. done(i) .and. k > jt(i) .and. k < jb(i)) .and. eps0(i) > 0._r8) then
            su(i,k) = mu(i,k+1)/mu(i,k)*su(i,k+1) + &
                      dz(i,k)/mu(i,k)* (eu(i,k)-du(i,k))*s(i,k)
            qu(i,k) = mu(i,k+1)/mu(i,k)*qu(i,k+1) + dz(i,k)/mu(i,k)* (eu(i,k)*q(i,k)- &
                            du(i,k)*qst(i,k))
            tu = su(i,k) - grav/cp*zf(i,k)
            call qmmr_hPa(tu, (p(i,k)+p(i,k-1))/2._r8, estu, qstu)
            if (qu(i,k) >= qstu) then
               jlcl(i) = k
               kount = kount + 1
               done(i) = .true.
            end if
         end if
      end do
      if (kount >= il2g) goto 690
   end do
690 continue
   do k = msg + 2,nlev
      do i = 1,il2g
         if ((k > jt(i) .and. k <= jlcl(i)) .and. eps0(i) > 0._r8) then
            su(i,k) = shat(i,k) + (hu(i,k)-hsthat(i,k))/(cp* (1._r8+gamhat(i,k)))
            qu(i,k) = qsthat(i,k) + gamhat(i,k)*(hu(i,k)-hsthat(i,k))/ &
                     (rl* (1._r8+gamhat(i,k)))
         end if
      end do
   end do

! compute condensation in updraft
   do k = nlev,msg + 2,-1
      do i = 1,il2g
         if (k >= jt(i) .and. k < jb(i) .and. eps0(i) > 0._r8) then
            cu(i,k) = ((mu(i,k)*su(i,k)-mu(i,k+1)*su(i,k+1))/ &
                      dz(i,k)- (eu(i,k)-du(i,k))*s(i,k))/(rl/cp)
            if (k == jt(i)) cu(i,k) = 0._r8
            cu(i,k) = max(0._r8,cu(i,k))
         end if
      end do
   end do

i=1
k=25
!write(*,*)'KK4.5 lchnk, i,rprd',lchnk,i,rprd(i,k),c0_lnd,c0_ocn,landfrac(1)


! compute condensed liquid, rain production rate
! accumulate total precipitation (condensation - detrainment of liquid)
! Note ql1 = ql(k) + rprd(k)*dz(k)/mu(k)
! The differencing is somewhat strange (e.g. du(i,k)*ql(i,k+1)) but is
! consistently applied.
!    mu, ql are interface quantities
!    cu, du, eu, rprd are midpoint quantites
   do k = nlev,msg + 2,-1
      do i = 1,il2g
         rprd(i,k) = 0._r8
         if (k >= jt(i) .and. k < jb(i) .and. eps0(i) > 0._r8 .and. mu(i,k) >= 0.0_r8) then
            if (mu(i,k) > 0._r8) then
               ql1 = 1._r8/mu(i,k)* (mu(i,k+1)*ql(i,k+1)- &
                     dz(i,k)*du(i,k)*ql(i,k+1)+dz(i,k)*cu(i,k))
               ql(i,k) = ql1/ (1._r8+dz(i,k)*c0mask(i))
            else
               ql(i,k) = 0._r8
            end if
            totpcp(i) = totpcp(i) + dz(i,k)*(cu(i,k)-du(i,k)*ql(i,k+1))
            rprd(i,k) = c0mask(i)*mu(i,k)*ql(i,k)
!if(k==25)then 
!    write(*,*)'KK5 lchnk, i,rprd',lchnk,i,rprd(i,k),c0mask(i),mu(i,k),ql(i,k)
!endif
         end if
      end do
   end do
!
   do i = 1,il2g
      qd(i,jd(i)) = qds(i,jd(i))
      sd(i,jd(i)) = (hd(i,jd(i)) - rl*qd(i,jd(i)))/cp
   end do
!
   do k = msg + 2,nlev
      do i = 1,il2g
         if (k >= jd(i) .and. k < jb(i) .and. eps0(i) > 0._r8) then
            qd(i,k+1) = qds(i,k+1)
            evp(i,k) = -ed(i,k)*q(i,k) + (md(i,k)*qd(i,k)-md(i,k+1)*qd(i,k+1))/dz(i,k)
            evp(i,k) = max(evp(i,k),0._r8)
            mdt = min(md(i,k+1),-small)
            sd(i,k+1) = ((rl/cp*evp(i,k)-ed(i,k)*s(i,k))*dz(i,k) + md(i,k)*sd(i,k))/mdt
            totevp(i) = totevp(i) - dz(i,k)*ed(i,k)*q(i,k)
         end if
      end do
   end do
   do i = 1,il2g
!*guang         totevp(i) = totevp(i) + md(i,jd(i))*q(i,jd(i)-1) -
      totevp(i) = totevp(i) + md(i,jd(i))*qd(i,jd(i)) - md(i,jb(i))*qd(i,jb(i))
   end do
!!$   if (.true.) then
   if (.false.) then
      do i = 1,il2g
         k = jb(i)
         if (eps0(i) > 0._r8) then
            evp(i,k) = -ed(i,k)*q(i,k) + (md(i,k)*qd(i,k))/dz(i,k)
            evp(i,k) = max(evp(i,k),0._r8)
            totevp(i) = totevp(i) - dz(i,k)*ed(i,k)*q(i,k)
         end if
      end do
   endif

   do i = 1,il2g
      totpcp(i) = max(totpcp(i),0._r8)
      totevp(i) = max(totevp(i),0._r8)
   end do
!
   do k = msg + 2,nlev
      do i = 1,il2g
         if (totevp(i) > 0._r8 .and. totpcp(i) > 0._r8) then
            md(i,k)  = md (i,k)*min(1._r8, totpcp(i)/(totevp(i)+totpcp(i)))
            ed(i,k)  = ed (i,k)*min(1._r8, totpcp(i)/(totevp(i)+totpcp(i)))
            evp(i,k) = evp(i,k)*min(1._r8, totpcp(i)/(totevp(i)+totpcp(i)))
         else
            md(i,k) = 0._r8
            ed(i,k) = 0._r8
            evp(i,k) = 0._r8
         end if
! cmeg is the cloud water condensed - rain water evaporated
! rprd is the cloud water converted to rain - (rain evaporated)
         cmeg(i,k) = cu(i,k) - evp(i,k)
         rprd(i,k) = rprd(i,k)-evp(i,k)
      end do
   end do
i=1
k=25
!write(*,*)'KK6 lchnk, i,rprd',lchnk,i,rprd(i,k)


! compute the net precipitation flux across interfaces
   pflx(:il2g,1) = 0._r8
   do k = 2,nlevp
      do i = 1,il2g
         pflx(i,k) = pflx(i,k-1) + rprd(i,k-1)*dz(i,k-1)
      end do
   end do
!
   do k = msg + 1,nlev
      do i = 1,il2g
         mc(i,k) = mu(i,k) + md(i,k)
      end do
   end do
!
i=1
if(i<0)then
              k=25
              write(*,"(A50/,A30/,5I10/,2(A10/,3(5I15/)),14(A10/,3(5E15.7/)) )") &
                  ' YY2YY in cldprp  at end ....', &
              'state%lchnk, pcols,pver,ncol',lchnk, ncol,nlev,ncol,limcnv, &
              'jt',jt(1:15),        &
              'jlcl',jlcl(1:15),        &
              'q(1)',q(1:15,k),        &
              'mu',mu(1:15,k),             &
              'rprd',rprd(1:15,k), &
              'dz',dz(1:15,k), &
              'eu',eu(1:15,k), &
              'du',du(1:15,k), &
              'su',su(1:15,k), &
              'qu',qu(1:15,k), &
              'cu',cu(1:15,k), &
              'md',md(1:15,k), &
              'evp',evp(1:15,k), &
              'eps0',eps0(1:15),&
              'ql',ql(1:15,k),&
              'c0mask',c0mask(1:15)
endif
   return
end subroutine cldprp

!MZsubroutine closure(lchnk   , &
subroutine closure(lchnk   , inncol,  nlev,  nlevp, &
                   q       ,t       ,p       ,z       ,s       , &
                   tp      ,qs      ,qu      ,su      ,mc      , &
                   du      ,mu      ,md      ,qd      ,sd      , &
                   qhat    ,shat    ,dp      ,qstp    ,zf      , &
                   ql      ,dsubcld ,mb      ,cape    ,tl      , &
                   lcl     ,lel     ,jt      ,mx      ,il1g    , &
                   il2g    ,rd      ,grav    ,cp      ,rl      , &
                   msg     ,capelmt )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! <Say what the routine does> 
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: G. Zhang and collaborators. CCM contact:P. Rasch
! This is contributed code not fully standardized by the CCM core group.
!
! this code is very much rougher than virtually anything else in the CCM
! We expect to release cleaner code in a future release
!
! the documentation has been enhanced to the degree that we are able
! 
!-----------------------------------------------------------------------
   use dycore,    only: dycore_is, get_resolution

   implicit none

!
!-----------------------------Arguments---------------------------------
!
   integer, intent(in) :: lchnk                 ! chunk identifier
!MZ
   integer, intent(in) :: inncol, nlev, nlevp
   real(r8), intent(inout) :: q(inncol,nlev)        ! spec humidity
   real(r8), intent(inout) :: t(inncol,nlev)        ! temperature
   real(r8), intent(inout) :: p(inncol,nlev)        ! pressure (mb)
   real(r8), intent(inout) :: mb(inncol)            ! cloud base mass flux
   real(r8), intent(in) :: z(inncol,nlev)        ! height (m)
   real(r8), intent(in) :: s(inncol,nlev)        ! normalized dry static energy
   real(r8), intent(in) :: tp(inncol,nlev)       ! parcel temp
   real(r8), intent(in) :: qs(inncol,nlev)       ! sat spec humidity
   real(r8), intent(in) :: qu(inncol,nlev)       ! updraft spec. humidity
   real(r8), intent(in) :: su(inncol,nlev)       ! normalized dry stat energy of updraft
   real(r8), intent(in) :: mc(inncol,nlev)       ! net convective mass flux
   real(r8), intent(in) :: du(inncol,nlev)       ! detrainment from updraft
   real(r8), intent(in) :: mu(inncol,nlev)       ! mass flux of updraft
   real(r8), intent(in) :: md(inncol,nlev)       ! mass flux of downdraft
   real(r8), intent(in) :: qd(inncol,nlev)       ! spec. humidity of downdraft
   real(r8), intent(in) :: sd(inncol,nlev)       ! dry static energy of downdraft
   real(r8), intent(in) :: qhat(inncol,nlev)     ! environment spec humidity at interfaces
   real(r8), intent(in) :: shat(inncol,nlev)     ! env. normalized dry static energy at intrfcs
   real(r8), intent(in) :: dp(inncol,nlev)       ! pressure thickness of layers
   real(r8), intent(in) :: qstp(inncol,nlev)     ! spec humidity of parcel
   real(r8), intent(in) :: zf(inncol,nlev+1)     ! height of interface levels
   real(r8), intent(in) :: ql(inncol,nlev)       ! liquid water mixing ratio

   real(r8), intent(in) :: cape(inncol)          ! available pot. energy of column
   real(r8), intent(in) :: tl(inncol)
   real(r8), intent(in) :: dsubcld(inncol)       ! thickness of subcloud layer

   integer, intent(in) :: lcl(inncol)        ! index of lcl
   integer, intent(in) :: lel(inncol)        ! index of launch leve
   integer, intent(in) :: jt(inncol)         ! top of updraft
   integer, intent(in) :: mx(inncol)         ! base of updraft
!
!--------------------------Local variables------------------------------
!
   real(r8) dtpdt(inncol,nlev)
   real(r8) dqsdtp(inncol,nlev)
   real(r8) dtmdt(inncol,nlev)
   real(r8) dqmdt(inncol,nlev)
   real(r8) dboydt(inncol,nlev)
   real(r8) thetavp(inncol,nlev)
   real(r8) thetavm(inncol,nlev)

   real(r8) dtbdt(inncol),dqbdt(inncol),dtldt(inncol)
   real(r8) beta
   real(r8) capelmt
   real(r8) cp
   real(r8) dadt(inncol)
   real(r8) debdt
   real(r8) dltaa
   real(r8) eb
   real(r8) grav

   integer i
   integer il1g
   integer il2g
   integer k, kmin, kmax
   integer msg

   real(r8) rd
   real(r8) rl
! change of subcloud layer properties due to convection is
! related to cumulus updrafts and downdrafts.
! mc(z)=f(z)*mb, mub=betau*mb, mdb=betad*mb are used
! to define betau, betad and f(z).
! note that this implies all time derivatives are in effect
! time derivatives per unit cloud-base mass flux, i.e. they
! have units of 1/mb instead of 1/sec.
!
   do i = il1g,il2g
      mb(i) = 0._r8
      eb = p(i,mx(i))*q(i,mx(i))/ (eps1+q(i,mx(i)))
      dtbdt(i) = (1._r8/dsubcld(i))* (mu(i,mx(i))*(shat(i,mx(i))-su(i,mx(i)))+ &
                  md(i,mx(i))* (shat(i,mx(i))-sd(i,mx(i))))
      dqbdt(i) = (1._r8/dsubcld(i))* (mu(i,mx(i))*(qhat(i,mx(i))-qu(i,mx(i)))+ &
                 md(i,mx(i))* (qhat(i,mx(i))-qd(i,mx(i))))
      debdt = eps1*p(i,mx(i))/ (eps1+q(i,mx(i)))**2*dqbdt(i)
      dtldt(i) = -2840._r8* (3.5_r8/t(i,mx(i))*dtbdt(i)-debdt/eb)/ &
                 (3.5_r8*log(t(i,mx(i)))-log(eb)-4.805_r8)**2
   end do
!
!   dtmdt and dqmdt are cumulus heating and drying.
!
   do k = msg + 1,nlev
      do i = il1g,il2g
         dtmdt(i,k) = 0._r8
         dqmdt(i,k) = 0._r8
      end do
   end do
!
   do k = msg + 1,nlev - 1
      do i = il1g,il2g
         if (k == jt(i)) then
            dtmdt(i,k) = (1._r8/dp(i,k))*(mu(i,k+1)* (su(i,k+1)-shat(i,k+1)- &
                          rl/cp*ql(i,k+1))+md(i,k+1)* (sd(i,k+1)-shat(i,k+1)))
            dqmdt(i,k) = (1._r8/dp(i,k))*(mu(i,k+1)* (qu(i,k+1)- &
                         qhat(i,k+1)+ql(i,k+1))+md(i,k+1)*(qd(i,k+1)-qhat(i,k+1)))
         end if
      end do
   end do
!
   beta = 0._r8
   do k = msg + 1,nlev - 1
      do i = il1g,il2g
         if (k > jt(i) .and. k < mx(i)) then
            dtmdt(i,k) = (mc(i,k)* (shat(i,k)-s(i,k))+mc(i,k+1)* (s(i,k)-shat(i,k+1)))/ &
                         dp(i,k) - rl/cp*du(i,k)*(beta*ql(i,k)+ (1-beta)*ql(i,k+1))
!          dqmdt(i,k)=(mc(i,k)*(qhat(i,k)-q(i,k))
!     1                +mc(i,k+1)*(q(i,k)-qhat(i,k+1)))/dp(i,k)
!     2                +du(i,k)*(qs(i,k)-q(i,k))
!     3                +du(i,k)*(beta*ql(i,k)+(1-beta)*ql(i,k+1))

            dqmdt(i,k) = (mu(i,k+1)* (qu(i,k+1)-qhat(i,k+1)+cp/rl* (su(i,k+1)-s(i,k)))- &
                          mu(i,k)* (qu(i,k)-qhat(i,k)+cp/rl*(su(i,k)-s(i,k)))+md(i,k+1)* &
                         (qd(i,k+1)-qhat(i,k+1)+cp/rl*(sd(i,k+1)-s(i,k)))-md(i,k)* &
                         (qd(i,k)-qhat(i,k)+cp/rl*(sd(i,k)-s(i,k))))/dp(i,k) + &
                          du(i,k)* (beta*ql(i,k)+(1-beta)*ql(i,k+1))
         end if
      end do
   end do
!
   do k = msg + 1,nlev
      do i = il1g,il2g
         if (k >= lel(i) .and. k <= lcl(i)) then
            thetavp(i,k) = tp(i,k)* (1000._r8/p(i,k))** (rd/cp)*(1._r8+1.608_r8*qstp(i,k)-q(i,mx(i)))
            thetavm(i,k) = t(i,k)* (1000._r8/p(i,k))** (rd/cp)*(1._r8+0.608_r8*q(i,k))
            dqsdtp(i,k) = qstp(i,k)* (1._r8+qstp(i,k)/eps1)*eps1*rl/(rd*tp(i,k)**2)
!
! dtpdt is the parcel temperature change due to change of
! subcloud layer properties during convection.
!
            dtpdt(i,k) = tp(i,k)/ (1._r8+rl/cp* (dqsdtp(i,k)-qstp(i,k)/tp(i,k)))* &
                        (dtbdt(i)/t(i,mx(i))+rl/cp* (dqbdt(i)/tl(i)-q(i,mx(i))/ &
                         tl(i)**2*dtldt(i)))
!
! dboydt is the integrand of cape change.
!
            dboydt(i,k) = ((dtpdt(i,k)/tp(i,k)+1._r8/(1._r8+1.608_r8*qstp(i,k)-q(i,mx(i)))* &
                          (1.608_r8 * dqsdtp(i,k) * dtpdt(i,k) -dqbdt(i))) - (dtmdt(i,k)/t(i,k)+0.608_r8/ &
                          (1._r8+0.608_r8*q(i,k))*dqmdt(i,k)))*grav*thetavp(i,k)/thetavm(i,k)
         end if
      end do
   end do
!
   do k = msg + 1,nlev
      do i = il1g,il2g
         if (k > lcl(i) .and. k < mx(i)) then
            thetavp(i,k) = tp(i,k)* (1000._r8/p(i,k))** (rd/cp)*(1._r8+0.608_r8*q(i,mx(i)))
            thetavm(i,k) = t(i,k)* (1000._r8/p(i,k))** (rd/cp)*(1._r8+0.608_r8*q(i,k))
!
! dboydt is the integrand of cape change.
!
            dboydt(i,k) = (dtbdt(i)/t(i,mx(i))+0.608_r8/ (1._r8+0.608_r8*q(i,mx(i)))*dqbdt(i)- &
                          dtmdt(i,k)/t(i,k)-0.608_r8/ (1._r8+0.608_r8*q(i,k))*dqmdt(i,k))* &
                          grav*thetavp(i,k)/thetavm(i,k)
         end if
      end do
   end do

!
! buoyant energy change is set to 2/3*excess cape per 3 hours
!
   dadt(il1g:il2g)  = 0._r8
   kmin = minval(lel(il1g:il2g))
   kmax = maxval(mx(il1g:il2g)) - 1
   do k = kmin, kmax
      do i = il1g,il2g
         if ( k >= lel(i) .and. k <= mx(i) - 1) then
            dadt(i) = dadt(i) + dboydt(i,k)* (zf(i,k)-zf(i,k+1))
         endif
      end do
   end do
   do i = il1g,il2g
      dltaa = -1._r8* (cape(i)-capelmt)
      if (dadt(i) /= 0._r8) mb(i) = max(dltaa/tau/dadt(i),0._r8)
   end do

if(i<0)then
    write(*,*)' in zm_conv closure'
    write(*,*)'dltaa',dltaa
    write(*,*)'tau',tau
    write(*,*)'dadt',dadt
    write(*,*)'mb',mb
    write(*,*)'dboydt',dboydt
endif

!
   return
end subroutine closure

!MZ subroutine q1q2_pjr(lchnk   , &
subroutine q1q2_pjr(lchnk   , inncol, nlev, nlevp,&
                    dqdt    ,dsdt    ,q       ,qs      ,qu      , &
                    su      ,du      ,qhat    ,shat    ,dp      , &
                    mu      ,md      ,sd      ,qd      ,ql      , &
                    dsubcld ,jt      ,mx      ,il1g    ,il2g    , &
                    cp      ,rl      ,msg     ,          &
!xiex
                    dsdtcond, dqdtcond, dsdttranup, dqdttranup, &
                    dsdttrandn, dqdttrandn, &
                    dl      ,evp     ,cu      )


   implicit none

!----------------------------------------------------------------------- 
! 
! Purpose: 
! <Say what the routine does> 
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: phil rasch dec 19 1995
! 
!-----------------------------------------------------------------------


   real(r8), intent(in) :: cp

!MZ   integer, intent(in) :: lchnk             ! chunk identifier
   integer, intent(in) :: lchnk , inncol, nlev, nlevp            ! chunk identifier
   integer, intent(in) :: il1g
   integer, intent(in) :: il2g
   integer, intent(in) :: msg

   real(r8), intent(in) :: q(inncol,nlev)
   real(r8), intent(in) :: qs(inncol,nlev)
   real(r8), intent(in) :: qu(inncol,nlev)
   real(r8), intent(in) :: su(inncol,nlev)
   real(r8), intent(in) :: du(inncol,nlev)
   real(r8), intent(in) :: qhat(inncol,nlev)
   real(r8), intent(in) :: shat(inncol,nlev)
   real(r8), intent(in) :: dp(inncol,nlev)
   real(r8), intent(in) :: mu(inncol,nlev)
   real(r8), intent(in) :: md(inncol,nlev)
   real(r8), intent(in) :: sd(inncol,nlev)
   real(r8), intent(in) :: qd(inncol,nlev)
   real(r8), intent(in) :: ql(inncol,nlev)
   real(r8), intent(in) :: evp(inncol,nlev)
   real(r8), intent(in) :: cu(inncol,nlev)
   real(r8), intent(in) :: dsubcld(inncol)

!xiex
   real(r8),intent(out) :: dsdtcond(inncol,nlev),dqdtcond(inncol,nlev)
   real(r8),intent(out) :: dsdttranup(inncol,nlev),dqdttranup(inncol,nlev)
   real(r8),intent(out) :: dsdttrandn(inncol,nlev),dqdttrandn(inncol,nlev)

   real(r8),intent(out) :: dqdt(inncol,nlev),dsdt(inncol,nlev)
   real(r8),intent(out) :: dl(inncol,nlev)
   integer kbm
   integer ktm
   integer jt(inncol)
   integer mx(inncol)
!
! work fields:
!
   integer i
   integer k

   real(r8) emc
   real(r8) rl
!-------------------------------------------------------------------
   do k = msg + 1,nlev
      do i = il1g,il2g
         dsdt(i,k) = 0._r8
         dqdt(i,k) = 0._r8
         dl(i,k) = 0._r8
      end do
   end do

!xiex
   dqdtcond = 0._r8
   dsdtcond = 0._r8
   dqdttranup = 0._r8
   dsdttranup = 0._r8
   dqdttrandn = 0._r8
   dsdttrandn = 0._r8

!
! find the highest level top and bottom levels of convection
!
   ktm = nlev
   kbm = nlev
   do i = il1g, il2g
      ktm = min(ktm,jt(i))
      kbm = min(kbm,mx(i))
   end do

   do k = ktm,nlev-1
      do i = il1g,il2g
         emc = -cu (i,k)               &         ! condensation in updraft
               +evp(i,k)                         ! evaporating rain in downdraft

!xiex
         dsdtcond(i,k) = -rl/cp*emc
         dqdtcond(i,k) = emc
         dsdttranup(i,k) = &
                       ( mu(i,k+1)* (su(i,k+1)-shat(i,k+1)) &
                        -mu(i,k)*   (su(i,k)-shat(i,k)) &
                       )/dp(i,k)
         dqdttranup(i,k) = &
                    ( mu(i,k+1)* (qu(i,k+1)-qhat(i,k+1)) &
                     -mu(i,k)*   (qu(i,k)-qhat(i,k)) &
                    )/dp(i,k)
         dsdttrandn(i,k) = &
                       ( &
                         md(i,k+1)* (sd(i,k+1)-shat(i,k+1)) &
                        -md(i,k)*   (sd(i,k)-shat(i,k)) &
                       )/dp(i,k)
         dqdttrandn(i,k) = &
                    ( &
                      md(i,k+1)* (qd(i,k+1)-qhat(i,k+1)) &
                     -md(i,k)*   (qd(i,k)-qhat(i,k)) &
                    )/dp(i,k)


         dsdt(i,k) = -rl/cp*emc &
                     + (+mu(i,k+1)* (su(i,k+1)-shat(i,k+1)) &
                        -mu(i,k)*   (su(i,k)-shat(i,k)) &
                        +md(i,k+1)* (sd(i,k+1)-shat(i,k+1)) &
                        -md(i,k)*   (sd(i,k)-shat(i,k)) &
                       )/dp(i,k)

         dqdt(i,k) = emc + &
                    (+mu(i,k+1)* (qu(i,k+1)-qhat(i,k+1)) &
                     -mu(i,k)*   (qu(i,k)-qhat(i,k)) &
                     +md(i,k+1)* (qd(i,k+1)-qhat(i,k+1)) &
                     -md(i,k)*   (qd(i,k)-qhat(i,k)) &
                    )/dp(i,k)

         dl(i,k) = du(i,k)*ql(i,k+1)

      end do
   end do

!
!DIR$ NOINTERCHANGE!
   do k = kbm,nlev
      do i = il1g,il2g
         if (k == mx(i)) then

!xiex
            dsdttranup(i,k) = (1._r8/dsubcld(i))* &
                        (-mu(i,k)* (su(i,k)-shat(i,k)) &
                        )
            dqdttranup(i,k) = (1._r8/dsubcld(i))* &
                        (-mu(i,k)*(qu(i,k)-qhat(i,k)) &
                        )
            dsdttrandn(i,k) = (1._r8/dsubcld(i))* &
                        ( &
                         -md(i,k)* (sd(i,k)-shat(i,k)) &
                        )
            dqdttrandn(i,k) = (1._r8/dsubcld(i))* &
                        ( &
                         -md(i,k)*(qd(i,k)-qhat(i,k)) &
                        )

            dsdt(i,k) = (1._r8/dsubcld(i))* &
                        (-mu(i,k)* (su(i,k)-shat(i,k)) &
                         -md(i,k)* (sd(i,k)-shat(i,k)) &
                        )
            dqdt(i,k) = (1._r8/dsubcld(i))* &
                        (-mu(i,k)*(qu(i,k)-qhat(i,k)) &
                         -md(i,k)*(qd(i,k)-qhat(i,k)) &
                        )
         else if (k > mx(i)) then
!xiex
            dsdttranup(i,k) = dsdt(i,k-1)
            dqdttranup(i,k) = dqdt(i,k-1)

            dsdt(i,k) = dsdt(i,k-1)
            dqdt(i,k) = dqdt(i,k-1)
         end if
      end do
   end do
!
   return
end subroutine q1q2_pjr

!MZ subroutine buoyan_dilute(lchnk   ,inncol    , &
subroutine buoyan_dilute(lchnk   ,inncol    , nlev, nlevp, &
                  q       ,t       ,p       ,z       ,pf      , &
                  tp      ,qstp    ,tl      ,rl      ,cape    , &
                  pblt    ,lcl     ,lel     ,lon     ,mx      , &
                  rd      ,grav    ,cp      ,msg     , &
                  tpert, buoy, cin   )  !MZ
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Calculates CAPE the lifting condensation level and the convective top
! where buoyancy is first -ve.
! 
! Method: Calculates the parcel temperature based on a simple constant
! entraining plume model. CAPE is integrated from buoyancy.
! 09/09/04 - Simplest approach using an assumed entrainment rate for 
!            testing (dmpdp). 
! 08/04/05 - Swap to convert dmpdz to dmpdp  
!
! SCAM Logical Switches - DILUTE:RBN - Now Disabled 
! ---------------------
! switch(1) = .T. - Uses the dilute parcel calculation to obtain tendencies.
! switch(2) = .T. - Includes entropy/q changes due to condensate loss and freezing.
! switch(3) = .T. - Adds the PBL Tpert for the parcel temperature at all levels.
! 
! References:
! Raymond and Blythe (1992) JAS 
! 
! Author:
! Richard Neale - September 2004
! 
!-----------------------------------------------------------------------
   implicit none
!-----------------------------------------------------------------------
!
! input arguments
!
!MZ   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: lchnk, nlev, nlevp                 ! chunk identifier
   integer, intent(in) :: inncol                  ! number of atmospheric columns

   real(r8), intent(in) :: q(inncol,nlev)        ! spec. humidity
   real(r8), intent(in) :: t(inncol,nlev)        ! temperature
   real(r8), intent(in) :: p(inncol,nlev)        ! pressure
   real(r8), intent(in) :: z(inncol,nlev)        ! height
   real(r8), intent(in) :: pf(inncol,nlev+1)     ! pressure at interfaces
   real(r8), intent(in) :: pblt(inncol)          ! index of pbl depth
   real(r8), intent(in) :: tpert(inncol)         ! perturbation temperature by pbl processes

!
! output arguments
!
   real(r8), intent(out) :: tp(inncol,nlev)       ! parcel temperature
   real(r8), intent(out) :: qstp(inncol,nlev)     ! saturation mixing ratio of parcel (only above lcl, just q below).
!MZ
   real(r8),intent(out) ::  buoy(inncol,nlev)

   real(r8), intent(out) :: tl(inncol)            ! parcel temperature at lcl
   real(r8), intent(out) :: cape(inncol)          ! convective aval. pot. energy.
!MZ
   real(r8), intent(out) :: cin(inncol) 
   integer lcl(inncol)        !
   integer lel(inncol)        !
   integer lon(inncol)        ! level of onset of deep convection
   integer mx(inncol)         ! level of max moist static energy
!
!--------------------------Local Variables------------------------------
!
   real(r8) capeten(inncol,5)     ! provisional value of cape
   real(r8) tv(inncol,nlev)       !
   real(r8) tpv(inncol,nlev)      !
   !MZ real(r8) buoy(inncol,nlev)

   real(r8) a1(inncol)
   real(r8) a2(inncol)
   real(r8) estp(inncol)
   real(r8) pl(inncol)
   real(r8) plexp(inncol)
   real(r8) hmax(inncol)
   real(r8) hmn(inncol)
   real(r8) y(inncol)

   logical plge600(inncol)
   integer knt(inncol)
   integer lelten(inncol,5)

   real(r8) cp
   real(r8) e
   real(r8) grav

   integer i
   integer k
   integer msg
   integer n

   real(r8) rd
   real(r8) rl
#ifdef PERGRO
   real(r8) rhd
#endif
!
!-----------------------------------------------------------------------
!
   do n = 1,5
      do i = 1,inncol
         lelten(i,n) = nlev
         capeten(i,n) = 0._r8
      end do
   end do
!
   do i = 1,inncol
      lon(i) = nlev
      knt(i) = 0
      lel(i) = nlev
      mx(i) = lon(i)
      cape(i) = 0._r8
      cin(i) = 0._r8
      hmax(i) = 0._r8
   end do

   tp(:inncol,:) = t(:inncol,:)
   qstp(:inncol,:) = q(:inncol,:)

!!! RBN - Initialize tv and buoy for output.
!!! tv=tv : tpv=tpv : qstp=q : buoy=0.
   tv(:inncol,:) = t(:inncol,:) *(1._r8+1.608_r8*q(:inncol,:))/ (1._r8+q(:inncol,:))
   tpv(:inncol,:) = tv(:inncol,:)
   buoy(:inncol,:) = 0._r8

!
! set "launching" level(mx) to be at maximum moist static energy.
! search for this level stops at planetary boundary layer top.
!
#ifdef PERGRO
   do k = nlev,msg + 1,-1
      do i = 1,inncol
         hmn(i) = cp*t(i,k) + grav*z(i,k) + rl*q(i,k)
!
! Reset max moist static energy level when relative difference exceeds 1.e-4
!
         rhd = (hmn(i) - hmax(i))/(hmn(i) + hmax(i))
         if (k >= nint(pblt(i)) .and. k <= lon(i) .and. rhd > -1.e-4_r8) then
            hmax(i) = hmn(i)
            mx(i) = k
         end if
      end do
   end do
#else
   do k = nlev,msg + 1,-1
      do i = 1,inncol
         hmn(i) = cp*t(i,k) + grav*z(i,k) + rl*q(i,k)
         if (k >= nint(pblt(i)) .and. k <= lon(i) .and. hmn(i) > hmax(i)) then
            hmax(i) = hmn(i)
            mx(i) = k
         end if
      end do
   end do
#endif

! LCL dilute calculation - initialize to mx(i)
! Determine lcl in parcel_dilute and get pl,tl after parcel_dilute
! Original code actually sets LCL as level above wher condensate forms.
! Therefore in parcel_dilute lcl(i) will be at first level where qsmix < qtmix.

   do i = 1,inncol ! Initialise LCL variables.
      lcl(i) = mx(i)
      tl(i) = t(i,mx(i))
      pl(i) = p(i,mx(i))
   end do

!
! main buoyancy calculation.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! DILUTE PLUME CALCULATION USING ENTRAINING PLUME !!!
!!!   RBN 9/9/04   !!!

!MZ   call parcel_dilute(lchnk, inncol, msg, mx, p, t, q, tpert, tp, tpv, qstp, pl, tl, lcl)
   call parcel_dilute(lchnk, inncol, nlev, nlevp,msg, mx, p, t, q, tpert, tp, tpv, qstp, pl, tl, lcl)


! If lcl is above the nominal level of non-divergence (600 mbs),
! no deep convection is permitted (ensuing calculations
! skipped and cape retains initialized value of zero).
!
   do i = 1,inncol
      plge600(i) = pl(i).ge.600._r8 ! Just change to always allow buoy calculation.
   end do

!
! Main buoyancy calculation.
!
   do k = nlev,msg + 1,-1
      do i=1,inncol
         if (k <= mx(i) .and. plge600(i)) then   ! Define buoy from launch level to cloud top.
            tv(i,k) = t(i,k)* (1._r8+1.608_r8*q(i,k))/ (1._r8+q(i,k))
            buoy(i,k) = tpv(i,k) - tv(i,k) + tiedke_add  ! +0.5K or not?
         else
            qstp(i,k) = q(i,k)
            tp(i,k)   = t(i,k)            
            tpv(i,k)  = tv(i,k)
         endif
      end do
   end do



!-------------------------------------------------------------------------------

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
   do k = msg + 2,nlev
      do i = 1,inncol
         if (k < lcl(i) .and. plge600(i)) then
            if (buoy(i,k+1) > 0._r8 .and. buoy(i,k) <= 0._r8) then
               knt(i) = min(5,knt(i) + 1)
               lelten(i,knt(i)) = k
            end if
         end if
      end do
   end do
!
! calculate convective available potential energy (cape).
!
   do n = 1,5
      do k = msg + 1,nlev
         do i = 1,inncol
            if (plge600(i) .and. k <= mx(i) .and. k > lelten(i,n)) then
               capeten(i,n) = capeten(i,n) + rd*buoy(i,k)*log(pf(i,k+1)/pf(i,k))
            end if
         end do
      end do
   end do
!
! find maximum cape from all possible tentative capes from
! one sounding,
! and use it as the final cape, april 26, 1995
!
   do n = 1,5
      do i = 1,inncol
         if (capeten(i,n) > cape(i)) then
            cape(i) = capeten(i,n)
            lel(i) = lelten(i,n)
         end if
      end do
   end do
!
!MZ
   do i = 1,inncol
    do k= lel(i),mx(i)-1
           cin(i) = cin(i) - rd*min(buoy(i,k),0._r8)*log(pf(i,k+1)/pf(i,k))
     enddo
   enddo
!
! put lower bound on cape for diagnostic purposes.
!
   do i = 1,inncol
      cape(i) = max(cape(i), 0._r8)
   end do
!
   return
end subroutine buoyan_dilute

!MZ subroutine parcel_dilute (lchnk, inncol, msg, klaunch, p, t, q, tpert, tp, tpv, qstp, pl, tl, lcl)
subroutine parcel_dilute (lchnk, inncol, nlev, nlevp,msg, klaunch, p, t, q, tpert, tp, tpv, qstp, pl, tl, lcl)

! Routine  to determine 
!   1. Tp   - Parcel temperature
!   2. qstp - Saturated mixing ratio at the parcel temperature.

!--------------------
implicit none
!--------------------

!MZ integer, intent(in) :: lchnk
integer, intent(in) :: lchnk,  nlev, nlevp
integer, intent(in) :: inncol
integer, intent(in) :: msg

integer, intent(in), dimension(inncol) :: klaunch(inncol)

real(r8), intent(in), dimension(inncol,nlev) :: p
real(r8), intent(in), dimension(inncol,nlev) :: t
real(r8), intent(in), dimension(inncol,nlev) :: q
real(r8), intent(in), dimension(inncol) :: tpert ! PBL temperature perturbation.

real(r8), intent(inout), dimension(inncol,nlev) :: tp    ! Parcel temp.
real(r8), intent(inout), dimension(inncol,nlev) :: qstp  ! Parcel water vapour (sat value above lcl).
real(r8), intent(inout), dimension(inncol) :: tl         ! Actual temp of LCL.
real(r8), intent(inout), dimension(inncol) :: pl          ! Actual pressure of LCL. 

integer, intent(inout), dimension(inncol) :: lcl ! Lifting condesation level (first model level with saturation).

real(r8), intent(out), dimension(inncol,nlev) :: tpv   ! Define tpv within this routine.

!--------------------

! Have to be careful as s is also dry static energy.


! If we are to retain the fact that CAM loops over grid-points in the internal
! loop then we need to dimension sp,atp,mp,xsh2o with inncol.


real(r8) tmix(inncol,nlev)        ! Tempertaure of the entraining parcel.
real(r8) qtmix(inncol,nlev)       ! Total water of the entraining parcel.
real(r8) qsmix(inncol,nlev)       ! Saturated mixing ratio at the tmix.
real(r8) smix(inncol,nlev)        ! Entropy of the entraining parcel.
real(r8) xsh2o(inncol,nlev)       ! Precipitate lost from parcel.
real(r8) ds_xsh2o(inncol,nlev)    ! Entropy change due to loss of condensate.
real(r8) ds_freeze(inncol,nlev)   ! Entropy change sue to freezing of precip.

real(r8) mp(inncol)    ! Parcel mass flux.
real(r8) qtp(inncol)   ! Parcel total water.
real(r8) sp(inncol)    ! Parcel entropy.

real(r8) sp0(inncol)    ! Parcel launch entropy.
real(r8) qtp0(inncol)   ! Parcel launch total water.
real(r8) mp0(inncol)    ! Parcel launch relative mass flux.

real(r8) lwmax      ! Maximum condesate that can be held in cloud before rainout.
real(r8) dmpdp      ! Parcel fractional mass entrainment rate (/mb).
!real(r8) dmpdpc     ! In cloud parcel mass entrainment rate (/mb).
real(r8) dmpdz      ! Parcel fractional mass entrainment rate (/m)
real(r8) dpdz,dzdp  ! Hydrstatic relation and inverse of.
real(r8) senv       ! Environmental entropy at each grid point.
real(r8) qtenv      ! Environmental total water "   "   ".
real(r8) penv       ! Environmental total pressure "   "   ".
real(r8) tenv       ! Environmental total temperature "   "   ".
real(r8) new_s      ! Hold value for entropy after condensation/freezing adjustments.
real(r8) new_q      ! Hold value for total water after condensation/freezing adjustments.
real(r8) dp         ! Layer thickness (center to center)
real(r8) tfguess    ! First guess for entropy inversion - crucial for efficiency!
real(r8) tscool     ! Super cooled temperature offset (in degC) (eg -35).

real(r8) qxsk, qxskp1        ! LCL excess water (k, k+1)
real(r8) dsdp, dqtdp, dqxsdp ! LCL s, qt, p gradients (k, k+1)
real(r8) slcl,qtlcl,qslcl    ! LCL s, qt, qs values.

integer rcall       ! Number of ientropy call for errors recording
integer nit_lheat     ! Number of iterations for condensation/freezing loop.
integer i,k,ii   ! Loop counters.

!======================================================================
!    SUMMARY
!
!  9/9/04 - Assumes parcel is initiated from level of maxh (klaunch)
!           and entrains at each level with a specified entrainment rate.
!
! 15/9/04 - Calculates lcl(i) based on k where qsmix is first < qtmix.          
!
!======================================================================
!
! Set some values that may be changed frequently.
!

nit_lheat = 2 ! iterations for ds,dq changes from condensation freezing.

!MZ 
!2018-08-06 this parameter determines the height of plume
! default dmpdz=-1.e-3_r8        ! Entrainment rate. (-ve for /m)
dmpdz=-0.5e-3_r8

!dmpdpc = 3.e-2_r8   ! In cloud entrainment rate (/mb).
lwmax = 1.e-3_r8    ! Need to put formula in for this.
tscool = 0.0_r8   ! Temp at which water loading freezes in the cloud.

qtmix=0._r8
smix=0._r8

qtenv = 0._r8
senv = 0._r8
tenv = 0._r8
penv = 0._r8

qtp0 = 0._r8
sp0  = 0._r8
mp0 = 0._r8

qtp = 0._r8
sp = 0._r8
mp = 0._r8

new_q = 0._r8
new_s = 0._r8

! **** Begin loops ****

do k = nlev, msg+1, -1
   do i=1,inncol 

! Initialize parcel values at launch level.

      if (k == klaunch(i)) then 
         qtp0(i) = q(i,k)   ! Parcel launch total water (assuming subsaturated) - OK????.
         sp0(i)  = entropy(t(i,k),p(i,k),qtp0(i))  ! Parcel launch entropy.
         mp0(i)  = 1._r8       ! Parcel launch relative mass (i.e. 1 parcel stays 1 parcel for dmpdp=0, undilute). 
         smix(i,k)  = sp0(i)
         qtmix(i,k) = qtp0(i)
         tfguess = t(i,k)
         rcall = 1
         call ientropy (rcall,i,lchnk,smix(i,k),p(i,k),qtmix(i,k),tmix(i,k),qsmix(i,k),tfguess)
      end if

! Entraining levels
      
      if (k < klaunch(i)) then 

! Set environmental values for this level.                 
         
         dp = (p(i,k)-p(i,k+1)) ! In -ve mb as p decreasing with height - difference between center of layers.
         qtenv = 0.5_r8*(q(i,k)+q(i,k+1))         ! Total water of environment.
         tenv  = 0.5_r8*(t(i,k)+t(i,k+1)) 
         penv  = 0.5_r8*(p(i,k)+p(i,k+1))

         senv  = entropy(tenv,penv,qtenv)  ! Entropy of environment.   

! Determine fractional entrainment rate /pa given value /m.

         dpdz = -(penv*grav)/(rgas*tenv) ! in mb/m since  p in mb.
         dzdp = 1._r8/dpdz                  ! in m/mb
         dmpdp = dmpdz*dzdp              ! /mb Fractional entrainment

! Sum entrainment to current level
! entrains q,s out of intervening dp layers, in which linear variation is assumed
! so really it entrains the mean of the 2 stored values.

         sp(i)  = sp(i)  - dmpdp*dp*senv 
         qtp(i) = qtp(i) - dmpdp*dp*qtenv 
         mp(i)  = mp(i)  - dmpdp*dp
            
! Entrain s and qt to next level.

         smix(i,k)  = (sp0(i)  +  sp(i)) / (mp0(i) + mp(i))
         qtmix(i,k) = (qtp0(i) + qtp(i)) / (mp0(i) + mp(i))

! Invert entropy from s and q to determine T and saturation-capped q of mixture.
! t(i,k) used as a first guess so that it converges faster.

         tfguess = tmix(i,k+1)
         rcall = 2
         call ientropy(rcall,i,lchnk,smix(i,k),p(i,k),qtmix(i,k),tmix(i,k),qsmix(i,k),tfguess)   

!
! Determine if this is lcl of this column if qsmix <= qtmix.
! FIRST LEVEL where this happens on ascending.

         if (qsmix(i,k) <= qtmix(i,k) .and. qsmix(i,k+1) > qtmix(i,k+1)) then
            lcl(i) = k
            qxsk   = qtmix(i,k) - qsmix(i,k)
            qxskp1 = qtmix(i,k+1) - qsmix(i,k+1)
            dqxsdp = (qxsk - qxskp1)/dp
            pl(i)  = p(i,k+1) - qxskp1/dqxsdp    ! pressure level of actual lcl.
            dsdp   = (smix(i,k)  - smix(i,k+1))/dp
            dqtdp  = (qtmix(i,k) - qtmix(i,k+1))/dp
            slcl   = smix(i,k+1)  +  dsdp* (pl(i)-p(i,k+1))  
            qtlcl  = qtmix(i,k+1) +  dqtdp*(pl(i)-p(i,k+1))

            tfguess = tmix(i,k)
            rcall = 3
            call ientropy (rcall,i,lchnk,slcl,pl(i),qtlcl,tl(i),qslcl,tfguess)

!            write(iulog,*)' '
!            write(iulog,*)' p',p(i,k+1),pl(i),p(i,lcl(i))
!            write(iulog,*)' t',tmix(i,k+1),tl(i),tmix(i,lcl(i))
!            write(iulog,*)' s',smix(i,k+1),slcl,smix(i,lcl(i))
!            write(iulog,*)'qt',qtmix(i,k+1),qtlcl,qtmix(i,lcl(i))
!            write(iulog,*)'qs',qsmix(i,k+1),qslcl,qsmix(i,lcl(i))

         endif
!         
      end if !  k < klaunch

 
   end do ! Levels loop
end do ! Columns loop

!!!!!!!!!!!!!!!!!!!!!!!!!!END ENTRAINMENT LOOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!! Could stop now and test with this as it will provide some estimate of buoyancy
!! without the effects of freezing/condensation taken into account for tmix.

!! So we now have a profile of entropy and total water of the entraining parcel
!! Varying with height from the launch level klaunch parcel=environment. To the 
!! top allowed level for the existence of convection.

!! Now we have to adjust these values such that the water held in vaopor is < or 
!! = to qsmix. Therefore, we assume that the cloud holds a certain amount of
!! condensate (lwmax) and the rest is rained out (xsh2o). This, obviously 
!! provides latent heating to the mixed parcel and so this has to be added back 
!! to it. But does this also increase qsmix as well? Also freezing processes
 

xsh2o = 0._r8
ds_xsh2o = 0._r8
ds_freeze = 0._r8

!!!!!!!!!!!!!!!!!!!!!!!!!PRECIPITATION/FREEZING LOOP!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Iterate solution twice for accuracy



do k = nlev, msg+1, -1
   do i=1,inncol    
      
! Initialize variables at k=klaunch
      
      if (k == klaunch(i)) then

! Set parcel values at launch level assume no liquid water.            

         tp(i,k)    = tmix(i,k)
         qstp(i,k)  = q(i,k) 
         tpv(i,k)   =  (tp(i,k) + tpert(i)) * (1._r8+1.608_r8*qstp(i,k)) / (1._r8+qstp(i,k))
         
      end if

      if (k < klaunch(i)) then
            
! Initiaite loop if switch(2) = .T. - RBN:DILUTE - TAKEN OUT BUT COULD BE RETURNED LATER.

! Iterate nit_lheat times for s,qt changes.

         do ii=0,nit_lheat-1            

! Rain (xsh2o) is excess condensate, bar LWMAX (Accumulated loss from qtmix).

            xsh2o(i,k) = max (0._r8, qtmix(i,k) - qsmix(i,k) - lwmax)

! Contribution to ds from precip loss of condensate (Accumulated change from smix).(-ve)                     
                     
            ds_xsh2o(i,k) = ds_xsh2o(i,k+1) - cpliq * log (tmix(i,k)/tfreez) * max(0._r8,(xsh2o(i,k)-xsh2o(i,k+1)))
!
! Entropy of freezing: latice times amount of water involved divided by T.
!
 
            if (tmix(i,k) <= tfreez+tscool .and. ds_freeze(i,k+1) == 0._r8) then ! One off freezing of condensate. 
               ds_freeze(i,k) = (latice/tmix(i,k)) * max(0._r8,qtmix(i,k)-qsmix(i,k)-xsh2o(i,k)) ! Gain of LH
            end if
            
            if (tmix(i,k) <= tfreez+tscool .and. ds_freeze(i,k+1) /= 0._r8) then ! Continual freezing of additional condensate.
               ds_freeze(i,k) = ds_freeze(i,k+1)+(latice/tmix(i,k)) * max(0._r8,(qsmix(i,k+1)-qsmix(i,k)))
            end if
            
! Adjust entropy and accordingly to sum of ds (be careful of signs).

            new_s = smix(i,k) + ds_xsh2o(i,k) + ds_freeze(i,k) 

! Adjust liquid water and accordingly to xsh2o.

            new_q = qtmix(i,k) - xsh2o(i,k)

! Invert entropy to get updated Tmix and qsmix of parcel.

            tfguess = tmix(i,k)
            rcall =4
            call ientropy (rcall,i,lchnk,new_s, p(i,k), new_q, tmix(i,k), qsmix(i,k), tfguess)
            
         end do  ! Iteration loop for freezing processes.

! tp  - Parcel temp is temp of mixture.
! tpv - Parcel v. temp should be density temp with new_q total water. 

         tp(i,k)    = tmix(i,k)

! tpv = tprho in the presence of condensate (i.e. when new_q > qsmix)

         if (new_q > qsmix(i,k)) then  ! Super-saturated so condensate present - reduces buoyancy.
            qstp(i,k) = qsmix(i,k)
         else                          ! Just saturated/sub-saturated - no condensate virtual effects.
            qstp(i,k) = new_q
         end if

         tpv(i,k) = (tp(i,k)+tpert(i))* (1._r8+1.608_r8*qstp(i,k)) / (1._r8+ new_q) 

      end if ! k < klaunch
      
   end do ! Loop for columns
   
end do  ! Loop for vertical levels.


return
end subroutine parcel_dilute

!-----------------------------------------------------------------------------------------
real(r8) function entropy(TK,p,qtot)
!-----------------------------------------------------------------------------------------
!
! TK(K),p(mb),qtot(kg/kg)
! from Raymond and Blyth 1992
!
     real(r8), intent(in) :: p,qtot,TK
     real(r8) :: qv,qst,e,est,L,eref,pref

pref = 1000.0_r8           ! mb
eref = 6.106_r8            ! sat p at tfreez (mb)

L = rl - (cpliq - cpwv)*(TK-tfreez)         ! T IN CENTIGRADE

! Replace call to satmixutils.

call qmmr_hPa(TK, p, est, qst)

qv = min(qtot,qst)                         ! Partition qtot into vapor part only.
e = qv*p / (eps1 +qv)

entropy = (cpres + qtot*cpliq)*log( TK/tfreez) - rgas*log( (p-e)/pref ) + &
        L*qv/TK - qv*rh2o*log(qv/qst)
! 
return
end FUNCTION entropy

!
!-----------------------------------------------------------------------------------------
   SUBROUTINE ientropy (rcall,icol,lchnk,s,p,qt,T,qst,Tfg)
!-----------------------------------------------------------------------------------------
!
! p(mb), Tfg/T(K), qt/qv(kg/kg), s(J/kg). 
! Inverts entropy, pressure and total water qt 
! for T and saturated vapor mixing ratio
! 

     use phys_grid, only: get_rlon_p, get_rlat_p

     integer, intent(in) :: icol, lchnk, rcall
     real(r8), intent(in)  :: s, p, Tfg, qt
     real(r8), intent(out) :: qst, T
     real(r8) :: qv,Ts,dTs,fs1,fs2,est
     real(r8) :: pref,eref,L,e
     real(r8) :: this_lat,this_lon
     integer :: LOOPMAX,i

LOOPMAX = 100                   !* max number of iteration loops 

! Values for entropy
pref = 1000.0_r8           ! mb ref pressure.
eref = 6.106_r8           ! sat p at tfreez (mb)

! Invert the entropy equation -- use Newton's method

Ts = Tfg                  ! Better first guess based on Tprofile from conv.

converge: do i=0, LOOPMAX

   L = rl - (cpliq - cpwv)*(Ts-tfreez) 

   call qmmr_hPa(Ts, p, est, qst)
   qv = min(qt,qst) 
   e = qv*p / (eps1 +qv)  ! Bolton (eq. 16)
   fs1 = (cpres + qt*cpliq)*log( Ts/tfreez ) - rgas*log( (p-e)/pref ) + &
        L*qv/Ts - qv*rh2o*log(qv/qst) - s
   
   L = rl - (cpliq - cpwv)*(Ts-1._r8-tfreez)         

   call qmmr_hPa(Ts-1._r8, p, est, qst)
   qv = min(qt,qst) 
   e = qv*p / (eps1 +qv)
   fs2 = (cpres + qt*cpliq)*log( (Ts-1._r8)/tfreez ) - rgas*log( (p-e)/pref ) + &
        L*qv/(Ts-1._r8) - qv*rh2o*log(qv/qst) - s 
   
   dTs = fs1/(fs2 - fs1)
   Ts  = Ts+dTs
   if (abs(dTs).lt.0.001_r8) exit converge
   if (i .eq. LOOPMAX - 1) then
      this_lat = get_rlat_p(lchnk, icol)*57.296_r8
      this_lon = get_rlon_p(lchnk, icol)*57.296_r8
      write(iulog,*) '*** ZM_CONV: IENTROPY: Failed and about to exit, info follows ****'
      write(iulog,100) 'ZM_CONV: IENTROPY. Details: call#,lchnk,icol= ',rcall,lchnk,icol, &
       ' lat: ',this_lat,' lon: ',this_lon, &
       ' P(mb)= ', p, ' Tfg(K)= ', Tfg, ' qt(g/kg) = ', 1000._r8*qt, &
       ' qst(g/kg) = ', 1000._r8*qst,', s(J/kg) = ',s
      call endrun('**** ZM_CONV IENTROPY: Tmix did not converge ****')
   end if
enddo converge

! Replace call to satmixutils.

call qmmr_hPa(Ts, p, est, qst)

qv = min(qt,qst)                             !       /* check for saturation */
T = Ts 

 100    format (A,I1,I4,I4,7(A,F6.2))

return
end SUBROUTINE ientropy

! Wrapper for qmmr that does translation between Pa and hPa
! qmmr uses Pa internally, so get qmmr right, need to pass in Pa.
! Afterward, set es back to hPa.
subroutine qmmr_hPa(t, p, es, qm)
  ! use wv_saturation, only: qmmr

  ! Inputs
  real(r8), intent(in) :: t    ! Temperature (K)
  real(r8), intent(in) :: p    ! Pressure (hPa)
  ! Outputs
  real(r8), intent(out) :: es  ! Saturation vapor pressure (hPa)
  real(r8), intent(out) :: qm  ! Saturation mass mixing ratio
                               ! (vapor mass over dry mass, kg/kg)

  !call qmmr(t, p*100._r8, es, qm)
    
    call cal_esatqsat(t, p*100.0, es, qm)

  es = es*0.01_r8

end subroutine qmmr_hPa

subroutine zyx2_convi(limcnv_in, no_deep_pbl_in)

   use dycore,       only: dycore_is, get_resolution

   integer, intent(in)           :: limcnv_in       ! top interface level limit for convection
   logical, intent(in), optional :: no_deep_pbl_in  ! no_deep_pbl = .true. eliminates ZM convection entirely within PBL 

   ! local variables
   character(len=32)   :: hgrid           ! horizontal grid specifier

   ! Initialization of ZM constants
   limcnv = limcnv_in
   tfreez = tmelt
   eps1   = epsilo
   rl     = latvap
   cpres  = cpair
   rgrav  = 1.0_r8/gravit
   rgas   = rair
   grav   = gravit
   cp     = cpres

   if ( present(no_deep_pbl_in) )  then
      no_deep_pbl = no_deep_pbl_in
   else
      no_deep_pbl = .false.
   endif

   ! tau=4800. were used in canadian climate center. however, in echam3 t42, 
   ! convection is too weak, thus adjusted to 2400.

   hgrid = get_resolution()
   tau = 3600._r8

   if ( masterproc ) then
      write(iulog,*) 'tuning parameters zm_convi: tau',tau
      write(iulog,*) 'tuning parameters zm_convi: c0_lnd',c0_lnd, ', c0_ocn', c0_ocn 
      write(iulog,*) 'tuning parameters zm_convi: ke',ke
      write(iulog,*) 'tuning parameters zm_convi: no_deep_pbl',no_deep_pbl
   endif

   if (masterproc) write(iulog,*)'**** ZM: DILUTE Buoyancy Calculation ****'

end subroutine zyx2_convi

end module zyx2_conv

