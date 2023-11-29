#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <math.h>
//#include "cuda/common.h"
#include <gptl.h>
///#include <string>
//#include <iostream>
//using namespace std;

#define mix 1024
#define mkx 30
#define ncnst 25

#define min(x,y) (((x)<(y))?(x):(y))
#define max(x,y) (((x)>(y))?(x):(y))

extern double uwshcu_mp_xlv_;
extern double uwshcu_mp_xlf_;
extern double uwshcu_mp_xls_;
extern double uwshcu_mp_cp_;
extern double uwshcu_mp_zvir_;
extern double uwshcu_mp_r_;
extern double uwshcu_mp_g_;
extern double uwshcu_mp_ep2_;
extern double uwshcu_mp_p00_;
extern double uwshcu_mp_rovcp_;

extern double uwshcu_mp_rpen_;

extern char constituents_mp_cnst_type_[25][3];
extern double wv_saturation_mp_estbl_[250];// ! table values of saturation vapor pressure;

extern  double wv_saturation_mp_tmin_;//! min temperature (K) for table
extern  double wv_saturation_mp_tmax_;//! max temperature (K) for table
extern  double wv_saturation_mp_ttrice_;//! transition range from es over H2O to es over ice
extern  double wv_saturation_mp_pcf_[6];//! polynomial coeffs -> es transition water to ice
extern  double wv_saturation_mp_epsqs_;//! Ratio of h2o to dry air molecular weights
extern  double wv_saturation_mp_rgasv_;//! Gas constant for water vapor
extern  double wv_saturation_mp_hlatf_;//! Latent heat of vaporization
extern  double wv_saturation_mp_hlatv_;//! Latent heat of fusion
extern  double wv_saturation_mp_cp_;//! specific heat of dry air
extern  double wv_saturation_mp_tmelt_;//! Melting point of water (K)
extern  bool   wv_saturation_mp_icephs_;//! false => saturation vapor press over water only

extern double constituents_mp_qmin_[ncnst];
extern int modal_aero_data_mp_numptr_amode_[3];



// extern "C"{
// 	//extern __device__ void __constituents_MOD_cnst_get_ind(char* name,int ind);
// 	 extern void  __uwshcu_MOD_findsp_(int *lchnk,int *iend,double *qv0_in,double *t0_in,double *p0_in,double *tw0_in,double *qw0_in);
// 	//extern __device__ void __wv_saturation_MOD_vqsatd(double *t,double *p,double *es,double *qs,double *gam,int len);
// 	//extern __device__ double __error _function_MOD_erfc(double a);
// 	//extern __device__ void __abortutils_MOD_endrun(char* a);
// 	//extern __device__ char* __constituents_MOD_cnst_get_type_byind(int ind);
// 	}
double estblf__(double td);
void findsp__(int lchnk,int ,double *q,double *t,double *p,double *tsp,double *qsp);
__device__ double estblf(double td);
__device__ double exnf(double pressure);
__device__ void slope(double *sslop, int mkxtemp, double *field, double *p0);
__device__ int qsat(double *t,double *p,double *es,double *qs,double *gam,int len);
__device__ void conden(double p,double thl,double qt,double *th,double *qv,double *ql,double *qi,double *rvls,int *id_check);
__device__ void getbuoy(double pbot,double thv0bot,double ptop,double thv0top,double thvubot,double thvutop,double *plfc,double *cin);
__device__ double single_cin(double pbot,double thv0bot,double ptop,double thv0top,double thvubot,double thvutop);
__device__ void roots(double a,double b,double c,double *r1,double *r2,int *status);
__device__ double qsinvert(double qt,double thl,double psfc);
__device__ double compute_alpha(double del_CIN,double ke);
__device__ double compute_mumin2(double mulcl,double rmaxfrac,double mulow);
__device__ double compute_ppen(double wtwb,double D,double bogbot,double bogtop,double rho0j,double dpen);
__device__ void fluxbelowinv(double cbmf,double *ps0,int mkxtemp,int kinv,double dt,double xsrc,double xmean,double xtopin,double xbotin,double *xflx);
__device__ void positive_moisture_single(double xlv,double xls,int mkxtemp,double dt,double qvmin,double qlmin,double qimin,double *dp,double *qv, 
							  			 double *ql,double *qi,double *s,double *qvten,double *qlten,double *qiten,double *sten );
//__device__ double eerfc(double x);


__device__  double wv_saturation_mp_tmin__;//! min temperature (K) for table
__device__  double wv_saturation_mp_tmax__;//! max temperature (K) for table
__device__  double wv_saturation_mp_ttrice__;//! transition range from es over H2O to es over ice

//__device__  double wv_saturation_mp_pcf__[6];//! polynomial coeffs -> es transition water to ice

__device__  double wv_saturation_mp_epsqs__;//! Ratio of h2o to dry air molecular weights
__device__  double wv_saturation_mp_rgasv__;//! Gas constant for water vapor
__device__  double wv_saturation_mp_hlatf__;//! Latent heat of vaporization
__device__  double wv_saturation_mp_hlatv__;//! Latent heat of fusion
__device__  double wv_saturation_mp_cp__;//! specific heat of dry air
__device__  double wv_saturation_mp_tmelt__;//! Melting point of water (K)
__device__  bool   wv_saturation_mp_icephs__;//! false => saturation vapor press over water only

//extern double error_function_mp_erfc_;



__device__  double wv_saturation_mp_estbl__[250];// ! table values of saturation vapor pressure

__device__ double xlv;
__device__ double xlf;
__device__ double xls;//xlv + xlf;
__device__ double cp;
__device__ double zvir;
__device__ double r;
__device__ double g;
__device__ double ep2;
__device__ double p00;
__device__ double rovcp;//r/cp;
__device__ double rpen;
//__device__ double tw0_in[mix*mkx];
//__device__ double qw0_in[mix*mkx];
//
// Internal Output Variables	
//lifei改
//__device__ double  qtten_out[mix*mkx];           //  !  Tendency of qt [ kg/kg/s ]
//__device__ double  slten_out[mix*mkx];           //  !  Tendency of sl [ J/kg/s ]
//__device__ double ufrc_out[mix*(mkx+1)];         //  !  Updraft fractional area at the interfaces [ fraction ]
//__device__ double uflx_out[mix*(mkx+1)];         //  !  Updraft/pen.entrainment zonal momentum flux [ m/s/m2/s ]
//__device__ double vflx_out[mix*(mkx+1)];         //  !  Updraft/pen.entrainment meridional momentum flux [ m/s/m2/s ]
//__device__ double fer_out[mix*mkx];              //  !  Fractional lateral entrainment rate [ 1/Pa ]
//__device__ double fdr_out[mix*mkx];              //  !  Fractional lateral detrainment rate [ 1/Pa ]
//__device__ double cinh_out[mix];                 //  !  Convective INhibition upto LFC (CIN) [ J/kg ]
//__device__ double trflx_out[mix*(mkx+1)*ncnst];  //  !  Updraft/pen.entrainment tracer flux [ #/m2/s, kg/kg/m2/s ] 

///__device__ double  ufrcinvbase_out[mix];//                            !  Cumulus updraft fraction at the PBL top [ fraction ]
///__device__ double  ufrclcl_out[mix];//                                !  Cumulus updraft fraction at the LCL ( or PBL top when LCL is below PBL top ) [ fraction ]
///__device__ double  winvbase_out[mix];//                               !  Cumulus updraft velocity at the PBL top [ m/s ]
///__device__ double  wlcl_out[mix];//                                   !  Cumulus updraft velocity at the LCL ( or PBL top when LCL is below PBL top ) [ m/s ]
///__device__ double  plcl_out[mix];//                                   !  LCL of source air [ Pa ]
///__device__ double  pinv_out[mix];//                                   !  PBL top pressure [ Pa ]
// __device__ double  plfc_out[mix];//                                   !  LFC of source air [ Pa ]
// __device__ double  pbup_out[mix];//                                   !  Highest interface level of positive buoyancy [ Pa ]
// __device__ double  ppen_out[mix];//                                   !  Highest interface evel where Cu w = 0 [ Pa ]
// __device__ double  qtsrc_out[mix];//                                  !  Sourse air qt [ kg/kg ]
// __device__ double  thlsrc_out[mix];//                                 !  Sourse air thl [ K ]
// __device__ double  thvlsrc_out[mix];//                                !  Sourse air thvl [ K ]
// __device__ double  emfkbup_out[mix];//                                !  Penetrative downward mass flux at 'kbup' interface [ kg/m2/s ]
// __device__ double  cinlclh_out[mix];//                                !  Convective INhibition upto LCL (CIN) [ J/kg = m2/s2 ]
// __device__ double  tkeavg_out[mix];//                                 !  Average tke over the PBL [ m2/s2 ]
// __device__ double  cbmflimit_out[mix];//                              !  Cloud base mass flux limiter [ kg/m2/s ]
// __device__ double  zinv_out[mix];//                                   !  PBL top height [ m ]
// __device__ double  rcwp_out[mix];//                                   !  Layer mean Cumulus LWP+IWP [ kg/m2 ] 
// __device__ double  rlwp_out[mix];//                                   !  Layer mean Cumulus LWP [ kg/m2 ] 
// __device__ double  riwp_out[mix];//                                   !  Layer mean Cumulus IWP [ kg/m2 ] 


///__device__ double  wu_out[mix*(mkx+1)];//                               !  Updraft vertical velocity ( defined from the release level to 'kpen-1' interface )
///__device__ double  qtu_out[mix*(mkx+1)];//                              !  Updraft qt [ kg/kg ]
///__device__ double  thlu_out[mix*(mkx+1)];//                             !  Updraft thl [ K ]
///__device__ double  thvu_out[mix*(mkx+1)];//                             !  Updraft thv [ K ]
///__device__ double  uu_out[mix*(mkx+1)];//                               !  Updraft zonal wind [ m/s ] 
///__device__ double  vu_out[mix*(mkx+1)];//                               !  Updraft meridional wind [ m/s ]
///__device__ double  qtu_emf_out[mix*(mkx+1)];//                          !  Penetratively entrained qt [ kg/kg ]   
///__device__ double  thlu_emf_out[mix*(mkx+1)];//                         !  Penetratively entrained thl [ K ]
///__device__ double  uu_emf_out[mix*(mkx+1)];//                           !  Penetratively entrained u [ m/s ]
///__device__ double  vu_emf_out[mix*(mkx+1)];//                           !  Penetratively entrained v [ m/s ]
///__device__ double  uemf_out[mix*(mkx+1)];//                             !  Net upward mass flux including penetrative entrainment (umf+emf) [ kg/m2/s ]
//__device__ double  tru_out[mix*(mkx+1)*ncnst];//                        !  Updraft tracers [ #, kg/kg ]   
//__device__ double  tru_emf_out[mix*(mkx+1)*ncnst];//                    !  Penetratively entrained tracers [ #, kg/kg ]


///__device__ double  dwten_out[mix*mkx];//
///__device__ double  diten_out[mix*mkx];//                      
///__device__ double  flxrain_out[mix*(mkx+1)];//  
///__device__ double  flxsnow_out[mix*(mkx+1)];//        
///__device__ double  ntraprd_out[mix*mkx];//               
///__device__ double  ntsnprd_out[mix*mkx];//             



///__device__ double  excessu_arr_out[mix*mkx];//
///__device__ double  excess0_arr_out[mix*mkx];//
///__device__ double  xc_arr_out[mix*mkx];//
///__device__ double  aquad_arr_out[mix*mkx];//
///__device__ double  bquad_arr_out[mix*mkx];//
///__device__ double  cquad_arr_out[mix*mkx];//
///__device__ double  bogbot_arr_out[mix*mkx];// 
///__device__ double  bogtop_arr_out[mix*mkx];//



__device__ double  exit_UWCu[mix];//
__device__ double  exit_conden[mix];//
__device__ double  exit_klclmkx[mix];//
__device__ double  exit_klfcmkx[mix];//
__device__ double  exit_ufrc[mix];//
__device__ double  exit_wtw[mix];//
__device__ double  exit_drycore[mix];//
__device__ double  exit_wu[mix];//
__device__ double  exit_cufilter[mix];//
__device__ double  exit_kinv1[mix];//
__device__ double  exit_rei[mix];//

///__device__ double  limit_shcu[mix];//
///__device__ double  limit_negcon[mix];//
///__device__ double  limit_ufrc[mix];//
///__device__ double  limit_ppen[mix];//
///__device__ double  limit_emf[mix];//
///__device__ double  limit_cinlcl[mix];//
///__device__ double  limit_cin[mix];//
///__device__ double  limit_cbmf[mix];//
///__device__ double  limit_rei[mix];//
///__device__ double  ind_delcin[mix];//

//!----- Some diagnostic internal output variables

///__device__ double  qtu_s[mix*(mkx+1)];//
///__device__ double  thlu_s[mix*(mkx+1)];//
///__device__ double  thvu_s[mix*(mkx+1)];//
///__device__ double  uu_s[mix*(mkx+1)];//
///__device__ double  vu_s[mix*(mkx+1)];//
///__device__ double  qtu_emf_s[mix*(mkx+1)];// 
///__device__ double  thlu_emf_s[mix*(mkx+1)];//  
///__device__ double  uu_emf_s[mix*(mkx+1)];//   
///__device__ double  vu_emf_s[mix*(mkx+1)];//
///__device__ double  uemf_s[mix*(mkx+1)];//   
//__device__ double  tru_s[mix*(mkx+1)*ncnst];//
//__device__ double  tru_emf_s[mix*(mkx+1)*ncnst];//   

///__device__ double  dwten_s[mix*mkx];//
///__device__ double  diten_s[mix*mkx];//
///__device__ double  flxrain_s[mix*(mkx+1)];//  
///__device__ double  flxsnow_s[mix*(mkx+1)];//  
///__device__ double  ntraprd_s[mix*mkx];//    
///__device__ double  ntsnprd_s[mix*mkx];//  

///__device__ double  excessu_arr_s[mix*mkx];//
///__device__ double  excess0_arr_s[mix*mkx];//
///__device__ double  xc_arr_s[mix*mkx];//
///__device__ double  aquad_arr_s[mix*mkx];//
///__device__ double  bquad_arr_s[mix*mkx];//
///__device__ double  cquad_arr_s[mix*mkx];//
///__device__ double  bogbot_arr_s[mix*mkx];//
///__device__ double  bogtop_arr_s[mix*mkx];//


//__device__ double tr0_s[mix*mkx*ncnst];


__global__ void compute_uwshcu(
	double uwshcu_mp_xlv_,
	double uwshcu_mp_xlf_,
	double uwshcu_mp_xls_,
	double uwshcu_mp_cp_,
	double uwshcu_mp_zvir_,
	double uwshcu_mp_r_,
	double uwshcu_mp_g_,
	double uwshcu_mp_ep2_,
	double uwshcu_mp_p00_,
	double uwshcu_mp_rovcp_,
	double uwshcu_mp_rpen_,
	double wv_saturation_mp_tmin_,
	double wv_saturation_mp_tmax_,
	double wv_saturation_mp_ttrice_,
//	double *wv_saturation_mp_pcf_d,
	double wv_saturation_mp_epsqs_,
	double wv_saturation_mp_rgasv_,
	double wv_saturation_mp_hlatf_,
	double wv_saturation_mp_hlatv_,
	double wv_saturation_mp_cp_,
	double wv_saturation_mp_tmelt_,
	double wv_saturation_mp_icephs_,
	double *wv_saturation_mp_estbl_d,

	int mixtemp  ,int mkxtemp    ,int iendtemp      ,int ncnsttemp   , double ztodt ,
	double *ps0_in  ,double *zs0_in    ,double *p0_in        ,double *z0_in    ,double *dp0_in  , 
	double *u0_in   ,double *v0_in     ,double *qv0_in       ,double *ql0_in   ,double *qi0_in  ,  
	double *t0_in   ,double *s0_in     ,double *tr0_in       ,                 
	double *tke_in  ,double *cldfrct_in,double *concldfrct_in,double *pblh_in  ,double *cush_inout ,  
	double *umf_out  ,double *slflx_out  ,double *qtflx_out     ,                  
	double *flxprc1_out  ,double *flxsnow1_out  ,		            
	double *qvten_out,double *qlten_out  ,double *qiten_out     ,                 
	double *sten_out ,double *uten_out   ,double *vten_out      ,double *trten_out ,        
	double *qrten_out,double *qsten_out  ,double *precip_out    ,double *snow_out  ,double *evapc_out, 
	double *cufrc_out,double *qcu_out    ,double *qlu_out       ,double *qiu_out   ,        
	double *cbmf_out ,double *qc_out     ,double *rliq_out      ,                
	double *cnt_out  ,double *cnb_out    ,int lchnk ,double *dpdry0_in , double *wu_out,double *qmin, int *numptr_amode, double *qw0_in)
{

	double    wu_s[mkx+1];//                                     !  Same as above but for implicit CIN
	double    evapc[mkx];//                                    !  Tendency of evaporation of precipitation [ kg/kg/s ]
	double    slflx[mkx+1];//                                  !  Updraft/pen.entrainment liquid static energy flux [ J/kg * kg/m2/s ]
	double    qtflx[mkx+1];//                                  !  Updraft/pen.entrainment total water flux [ kg/kg * kg/m2/s ]
	double    uflx[mkx+1];//                                   !  Updraft/pen.entrainment flux of zonal momentum [ m/s/m2/s ]
	double    vflx[mkx+1];//                                   !  Updraft/pen.entrainment flux of meridional momentum [ m/s/m2/s ]
	double    cufrc[mkx];//                                    !  Shallow cumulus cloud fraction at the layer mid-point [ fraction ]

	double    trflx_d[mkx+1];//                                !  Adjustive downward flux of tracers to prevent negative tracers
	double    trflx_u[mkx+1];//                                !  Adjustive upward   flux of tracers to prevent negative tracers

	double    thlu_emf[mkx+1];//                               !  Penetrative downdraft liquid potential temperature at entraining interfaces [ K ]
	double    qtu_emf[mkx+1];//                                !  Penetrative downdraft total water at entraining interfaces [ kg/kg ]
	double    uu_emf[mkx+1];//                                 !  Penetrative downdraft zonal wind at entraining interfaces [ m/s ]
	double    vu_emf[mkx+1];//                                 !  Penetrative downdraft meridional wind at entraining interfaces [ m/s ]

	double    uemf[mkx+1];//                                   !  Net updraft mass flux at the interface ( emf + umf ) [ kg/m2/s ]
	double    comsub[mkx];//                                   !  Compensating subsidence at the layer mid-point ( unit of mass flux, umf ) [ kg/m2/s ]

	double    qc_l[mkx];//                                     !  Tendency due to detrained 'cloud water' (without rain-snow contribution) [ kg/kg/s ]
	double    qc_i[mkx];//                                     !  Tendency due to detrained 'cloud ice' (without rain-snow contribution) [ kg/kg/s ]

	double    qrten[mkx];//                                    !  Tendency of rain water specific humidity [ kg/kg/s ]
	double    qsten[mkx];//                                    !  Tendency of snow specific humidity [ kg/kg/s ]

	double    qv0_star[mkx];//                                 !  Environmental water vapor specific humidity [ kg/kg ]
	double    ql0_star[mkx];//                                 !  Environmental liquid water specific humidity [ kg/kg ]
	double    qi0_star[mkx];//                                 !  Environmental ice specific humidity [ kg/kg ]
///__device__ double    t0_star[mkx];//                                  !  Environmental temperature [ K ]
	double    s0_star[mkx];//                                  !  Environmental dry static energy [ J/kg ]


	double    thv0bot[mkx];//                                  !  Environmental virtual potential temperature at the bottom of each layer [ K ]
	double    thv0top[mkx];//                                  !  Environmental virtual potential temperature at the top of each layer [ K ]
	double    thvl0bot[mkx];//                                 !  Environmental liquid virtual potential temperature at the bottom of each layer [ K ]
	double    thvl0top[mkx];//                                 !  Environmental liquid virtual potential temperature at the top of each layer [ K ]

	double    dwten[mkx];//                                    !  Detrained water tendency from cumulus updraft [ kg/kg/s ]
	double    diten[mkx];//                                    !  Detrained ice   tendency from cumulus updraft [ kg/kg/s ]

	double    thvu[mkx+1];//                                   !  Updraft virtual potential temperature at the interface [ m/s ]
	double    rei[mkx];//                                      !  Updraft fractional mixing rate with the environment [ 1/Pa ]

	double    flxrain[mkx+1];//                                !  Downward rain flux at each interface [ kg/m2/s ]
	double    flxsnow[mkx+1];//                                !  Downward snow flux at each interface [ kg/m2/s ]
	double    ntraprd[mkx];//                                  !  Net production ( production - evaporation +  melting ) rate of rain in each layer [ kg/kg/s ]
	double    ntsnprd[mkx];//                                  !  Net production ( production - evaporation + freezing ) rate of snow in each layer [ kg/kg/s ]

	double    cldfrct[mkx];//[mkx];//                                  !  Total cloud fraction at the previous time step [ fraction ]
	double    concldfrct[mkx];//[mkx];//                               !  Total convective cloud fraction at the previous time step [ fraction ]

	double    qlten_sink[mkx];//                               !  Liquid condensate tendency by compensating subsidence/upwelling [ kg/kg/s ]
	double    qiten_sink[mkx];//                               !  Ice    condensate tendency by compensating subsidence/upwelling [ kg/kg/s ]
	double    nlten_sink[mkx];//                               !  Liquid droplets # tendency by compensating subsidence/upwelling [ kg/kg/s ]
	double    niten_sink[mkx];//                               !  Ice    droplets # tendency by compensating subsidence/upwelling [ kg/kg/s ]

 double umf_s[mkx+1], slflx_s[mkx+1] , qtflx_s[mkx+1];/// , ufrc_s[mkx+1]  , uflx_s[mkx+1] , vflx_s[mkx+1];
 double trten_s[mkx*ncnst];
//__device__ double trflx_s[mix*(mkx+1)*ncnst];

 double tr0[mkx*ncnst];//[mkx*ncnst];//                                !  Environmental tracers [ #, kg/kg ]
//!----- Variables for temporary storages

 double qv0_o[mkx], ql0_o[mkx], qi0_o[mkx], t0_o[mkx], s0_o[mkx], u0_o[mkx], v0_o[mkx];
 double qt0_o[mkx]    , thl0_o[mkx]   , thvl0_o[mkx]   ,                         
//qvten_o[mix*mkx] , qlten_o[mix*mkx]  , qiten_o[mix*mkx]   , qrten_o[mix*mkx]   , qsten_o[mix*mkx] ,   
//sten_o[mix*mkx]   , uten_o[mix*mkx]   , vten_o[mix*mkx]    , qcu_o[mix*mkx]     , qlu_o[mix*mkx]   ,   
//qiu_o[mix*mkx]    , cufrc_o[mix*mkx]  , evapc_o[mix*mkx]   ,                         
thv0bot_o[mkx], thv0top_o[mkx], thvl0bot_o[mkx], thvl0top_o[mkx],             
ssthl0_o[mkx] , ssqt0_o[mkx]  , ssu0_o[mkx]    , ssv0_o[mkx];     //qc_o[mix*mkx]    ,   
//qtten_o[mix*mkx]  , slten_o[mix*mkx];  

//double umf_o[mix*(mkx+1)]    , slflx_o[mix*(mkx+1)]  , qtflx_o[mix*(mkx+1)]   , ufrc_o[mix*(mkx+1)]; 
//double cush_o[mix]   , precip_o[mix] , snow_o[mix]    , rliq_o[mix], cbmf_o[mix], cnt_o[mix], cnb_o[mix];
//double uflx_o[mix*(mkx+1)]   , vflx_o[mix*(mkx+1)];

 double tr0_o[mkx*ncnst];
 double sstr0_o[mkx*ncnst];  //trten_o[mix*mkx*ncnst], 
//double trflx_o[(mkx+1)*ncnst];

 double sstr0[mkx*ncnst];//                              !  Linear slope of environmental tracers [ #/Pa, kg/kg/Pa ]


 double trten[mkx*ncnst];//                              !  Tendency of tracers [ #/s, kg/kg/s ]
 double trflx[(mkx+1)*ncnst];//                          !  Flux of tracers due to convection [ # * kg/m2/s, kg/kg * kg/m2/s ]

 double tru[(mkx+1)*ncnst];//                              !  Updraft tracers [ #, kg/kg ]

 double tru_emf[(mkx+1)*ncnst];//                          !  Penetrative Downdraft tracers at entraining interfaces [ #, kg/kg ] 

	//!----- Variables for implicit CIN computation
 double qv0_s[mkx]  , ql0_s[mkx]   , qi0_s[mkx]   , s0_s[mkx]    , u0_s[mkx]    ,      
           v0_s[mkx]   , t0_s[mkx]    , qt0_s[mkx]    , qvten_s[mkx] , //, thl0_s[mkx]  , thvl0_s[mkx]
           qlten_s[mkx], qiten_s[mkx] , qrten_s[mkx] , qsten_s[mkx] , sten_s[mkx]  , evapc_s[mkx] , 
           uten_s[mkx] , vten_s[mkx]  , cufrc_s[mkx] , qcu_s[mkx]   , qlu_s[mkx]   , qiu_s[mkx]   , 
           qc_s[mkx]; ///    , qtten_s[mix*mkx] , slten_s[mix*mkx]; fer_s[mix*mkx]  , fdr_s[mix*mkx]   ,



 double    qvten[mkx];//                                    !  Tendency of water vapor specific humidity [ kg/kg/s ]
 double    qlten[mkx];//                                    !  Tendency of liquid water specific humidity [ kg/kg/s ]
 double    qiten[mkx];//                                    !  Tendency of ice specific humidity [ kg/kg/s ]
 double    sten[mkx];//                                     !  Tendency of dry static energy [ J/kg ]
 double    uten[mkx];//                                     !  Tendency of zonal wind [ m/s2 ]
 double    vten[mkx];//                                     !  Tendency of meridional wind [ m/s2 ]
/////qsat使用//////////////////////////////////////////////
	//  wv_saturation_mp_tmin__ = 173.1599999999999965894;
	//  wv_saturation_mp_tmax__ = 375.1600000000000250111;
	//  wv_saturation_mp_ttrice__ = 20.0000000000000000000;
	//  wv_saturation_mp_epsqs__ = 0.6219705862045155076;
	//  wv_saturation_mp_rgasv__ = 461.5046398201599231470;
	//  wv_saturation_mp_hlatf__ = 333700.0000000000000000000;
	//  wv_saturation_mp_hlatv__ = 2501000.0000000000000000000;
	//  wv_saturation_mp_cp__ = 1004.6399999999999863576;
	//  wv_saturation_mp_tmelt__ = 273.1499999999999772626;
	//  wv_saturation_mp_icephs__ = false;
//////////////////////////////////////////////////////////

	// printf("lifeilifeilifei\n");
	////////////////////////////////////////////////
	//One-dimensional variables at each grid point//
	////////////////////////////////////////////////

    //1. Input variables
	double    ps0[mkx+1];//[mkx+1];//                                    !  Environmental pressure at the interfaces [ Pa ]
    double    zs0[mkx+1];//[mkx+1];//                                    !  Environmental height at the interfaces [ m ]
    double    p0[mkx];//[mkx];//                                       !  Environmental pressure at the layer mid-point [ Pa ]
    double    z0[mkx];//[mkx];//                                       !  Environmental height at the layer mid-point [ m ]
    double    dp0[mkx];//[mkx];//                                      !  Environmental layer pressure thickness [ Pa ] > 0.
    double    dpdry0[mkx];//[mkx];//                                   !  Environmental dry layer pressure thickness [ Pa ]
    double    u0[mkx];//[mkx];//                                       !  Environmental zonal wind [ m/s ]
    double    v0[mkx];//[mkx];//                                       !  Environmental meridional wind [ m/s ]
    double    tke[mkx+1];//[mkx+1];//                                    !  Turbulent kinetic energy at the interfaces [ m2/s2 ]
    // double    cldfrct[mkx];//[mkx];//                                  !  Total cloud fraction at the previous time step [ fraction ]
    // double    concldfrct[mkx];//[mkx];//                               !  Total convective cloud fraction at the previous time step [ fraction ]
    double    qv0[mkx];//[mkx];//                                      !  Environmental water vapor specific humidity [ kg/kg ]
    double    ql0[mkx];//[mkx];//                                      !  Environmental liquid water specific humidity [ kg/kg ]
    double    qi0[mkx];//[mkx];//                                      !  Environmental ice specific humidity [ kg/kg ]
    double    t0[mkx];//[mkx];//                                       !  Environmental temperature [ K ]
    double    s0[mkx];//[mkx];//                                       !  Environmental dry static energy [ J/kg ]
    double    pblh;   //                                       !  Height of PBL [ m ]
    double    cush;   //                                       !  Convective scale height [ m ]
 //   double    tr0[mkx*ncnst];//[mkx*ncnst];//                                !  Environmental tracers [ #, kg/kg ]

	//2. Environmental variables directly derived from the input variables
	double    qt0[mkx];//                                      !  Environmental total specific humidity [ kg/kg ]
    double    thl0[mkx];//                                     !  Environmental liquid potential temperature [ K ]
    double    thvl0[mkx];//                                    !  Environmental liquid virtual potential temperature [ K ]
    double    ssqt0[mkx];//                                    !  Linear internal slope of environmental total specific humidity [ kg/kg/Pa ]
    double    ssthl0[mkx];//                                   !  Linear internal slope of environmental liquid potential temperature [ K/Pa ]
    double    ssu0[mkx];//                                     !  Linear internal slope of environmental zonal wind [ m/s/Pa ]
    double    ssv0[mkx];//                                     !  Linear internal slope of environmental meridional wind [ m/s/Pa ]
    // double    thv0bot[mkx];//                                  !  Environmental virtual potential temperature at the bottom of each layer [ K ]
    // double    thv0top[mkx];//                                  !  Environmental virtual potential temperature at the top of each layer [ K ]
    // double    thvl0bot[mkx];//                                 !  Environmental liquid virtual potential temperature at the bottom of each layer [ K ]
    // double    thvl0top[mkx];//                                 !  Environmental liquid virtual potential temperature at the top of each layer [ K ]
    double    exn0[mkx];//                                     !  Exner function at the layer mid points [ no ]
    double    exns0[mkx+1];//                                  !  Exner function at the interfaces [ no ]
 //   double    sstr0[mkx*ncnst];//                              !  Linear slope of environmental tracers [ #/Pa, kg/kg/Pa ]

   //! 2-1. For preventing negative condensate at the provisional time step

    // double    qv0_star[mkx];//                                 !  Environmental water vapor specific humidity [ kg/kg ]
    // double    ql0_star[mkx];//                                 !  Environmental liquid water specific humidity [ kg/kg ]
    // double    qi0_star[mkx];//                                 !  Environmental ice specific humidity [ kg/kg ]
    // double    t0_star[mkx];//                                  !  Environmental temperature [ K ]
    // double    s0_star[mkx];//                                  !  Environmental dry static energy [ J/kg ]

   //! 3. Variables associated with cumulus convection

    double    umf[mkx+1];//                                    !  Updraft mass flux at the interfaces [ kg/m2/s ]
    double    emf[mkx+1];//                                    !  Penetrative entrainment mass flux at the interfaces [ kg/m2/s ]
    // double    qvten[mkx];//                                    !  Tendency of water vapor specific humidity [ kg/kg/s ]
    // double    qlten[mkx];//                                    !  Tendency of liquid water specific humidity [ kg/kg/s ]
    // double    qiten[mkx];//                                    !  Tendency of ice specific humidity [ kg/kg/s ]
    // double    sten[mkx];//                                     !  Tendency of dry static energy [ J/kg ]
    // double    uten[mkx];//                                     !  Tendency of zonal wind [ m/s2 ]
    // double    vten[mkx];//                                     !  Tendency of meridional wind [ m/s2 ]
    // double    qrten[mkx];//                                    !  Tendency of rain water specific humidity [ kg/kg/s ]
    // double    qsten[mkx];//                                    !  Tendency of snow specific humidity [ kg/kg/s ]
    double    precip; //                                       !  Precipitation rate ( rain + snow) at the surface [ m/s ]
    double    snow ;  //                                       !  Snow rate at the surface [ m/s ]
    // double    evapc[mkx];//                                    !  Tendency of evaporation of precipitation [ kg/kg/s ]
    // double    slflx[mkx+1];//                                  !  Updraft/pen.entrainment liquid static energy flux [ J/kg * kg/m2/s ]
    // double    qtflx[mkx+1];//                                  !  Updraft/pen.entrainment total water flux [ kg/kg * kg/m2/s ]
    // double    uflx[mkx+1];//                                   !  Updraft/pen.entrainment flux of zonal momentum [ m/s/m2/s ]
    // double    vflx[mkx+1];//                                   !  Updraft/pen.entrainment flux of meridional momentum [ m/s/m2/s ]
    // double    cufrc[mkx];//                                    !  Shallow cumulus cloud fraction at the layer mid-point [ fraction ]
    double    qcu[mkx];//                                      !  Condensate water specific humidity within convective updraft [ kg/kg ]
    double    qlu[mkx];//                                      !  Liquid water specific humidity within convective updraft [ kg/kg ]
    double    qiu[mkx];//                                      !  Ice specific humidity within convective updraft [ kg/kg ]
    // double    dwten[mkx];//                                    !  Detrained water tendency from cumulus updraft [ kg/kg/s ]
    // double    diten[mkx];//                                    !  Detrained ice   tendency from cumulus updraft [ kg/kg/s ]
    double    fer[mkx];//                                      !  Fractional lateral entrainment rate [ 1/Pa ]
    double    fdr[mkx];//                                      !  Fractional lateral detrainment rate [ 1/Pa ]
    double    uf[mkx];//                                       !  Zonal wind at the provisional time step [ m/s ]
    double    vf[mkx];//                                       !  Meridional wind at the provisional time step [ m/s ]
    double    qc[mkx];//                                       !  Tendency due to detrained 'cloud water + cloud ice' (without rain-snow contribution) [ kg/kg/s ]
    // double    qc_l[mkx];//                                     !  Tendency due to detrained 'cloud water' (without rain-snow contribution) [ kg/kg/s ]
    // double    qc_i[mkx];//                                     !  Tendency due to detrained 'cloud ice' (without rain-snow contribution) [ kg/kg/s ]
    double    qc_lm;
    double    qc_im;
    double    nc_lm;
    double    nc_im;
    double    ql_emf_kbup;
    double    qi_emf_kbup;
    double    nl_emf_kbup;
    double    ni_emf_kbup;
    double    qlten_det;
    double    qiten_det;
    double    rliq;        //                                  !  Vertical integral of qc [ m/s ] 
    double    cnt;         //                                  !  Cumulus top  interface index, cnt = kpen [ no ]
    double    cnb;         //                                  !  Cumulus base interface index, cnb = krel - 1 [ no ] 
    double    qtten[mkx];//                                    !  Tendency of qt [ kg/kg/s ]
    double    slten[mkx];//                                    !  Tendency of sl [ J/kg/s ]
    double    ufrc[mkx+1];//                                   !  Updraft fractional area [ fraction ]
  //  double    trten[mkx*ncnst];//                              !  Tendency of tracers [ #/s, kg/kg/s ]
  //  double    trflx[(mkx+1)*ncnst];//                          !  Flux of tracers due to convection [ # * kg/m2/s, kg/kg * kg/m2/s ]
///    double    trflx_d[mkx+1];//                                !  Adjustive downward flux of tracers to prevent negative tracers
///    double    trflx_u[mkx+1];//                                !  Adjustive upward   flux of tracers to prevent negative tracers
    double    trmin;             //                            !  Minimum concentration of tracers allowed
    double    pdelx, dum; 
    
   // !----- Variables used for the calculation of condensation sink associated with compensating subsidence
   // !      In the current code, this 'sink' tendency is simply set to be zero.

///    double    uemf[mix*(mkx+1)];//                                   !  Net updraft mass flux at the interface ( emf + umf ) [ kg/m2/s ]
///    double    comsub[mix*mkx];//                                   !  Compensating subsidence at the layer mid-point ( unit of mass flux, umf ) [ kg/m2/s ]
    // double    qlten_sink[mkx];//                               !  Liquid condensate tendency by compensating subsidence/upwelling [ kg/kg/s ]
    // double    qiten_sink[mkx];//                               !  Ice    condensate tendency by compensating subsidence/upwelling [ kg/kg/s ]
    // double    nlten_sink[mkx];//                               !  Liquid droplets # tendency by compensating subsidence/upwelling [ kg/kg/s ]
    // double    niten_sink[mkx];//                               !  Ice    droplets # tendency by compensating subsidence/upwelling [ kg/kg/s ]
    double    thlten_sub, qtten_sub;//                         !  Tendency of conservative scalars by compensating subsidence/upwelling
    double    qlten_sub, qiten_sub;//                          !  Tendency of ql0, qi0             by compensating subsidence/upwelling
    double    nlten_sub, niten_sub;//                          !  Tendency of nl0, ni0             by compensating subsidence/upwelling
    double    thl_prog, qt_prog;//                             !  Prognosed 'thl, qt' by compensating subsidence/upwelling 

   // !----- Variables describing cumulus updraft

    double    wu[mkx+1];//                                     !  Updraft vertical velocity at the interface [ m/s ]
    double    thlu[mkx+1];//                                   !  Updraft liquid potential temperature at the interface [ K ]
    double    qtu[mkx+1];//                                    !  Updraft total specific humidity at the interface [ kg/kg ]
    double    uu[mkx+1];//                                     !  Updraft zonal wind at the interface [ m/s ]
    double    vu[mkx+1];//                                     !  Updraft meridional wind at the interface [ m/s ]
    // double    thvu[mkx+1];//                                   !  Updraft virtual potential temperature at the interface [ m/s ]
    // double    rei[mkx];//                                      !  Updraft fractional mixing rate with the environment [ 1/Pa ]
  //  double    tru[(mkx+1)*ncnst];//                              !  Updraft tracers [ #, kg/kg ]

	//!----- Variables describing conservative scalars of entraining downdrafts  at the 
    //!      entraining interfaces, i.e., 'kbup <= k < kpen-1'. At the other interfaces,
    //!      belows are simply set to equal to those of updraft for simplicity - but it
    //!      does not influence numerical calculation.

///    double    thlu_emf[mix*(mkx+1)];//                               !  Penetrative downdraft liquid potential temperature at entraining interfaces [ K ]
///    double    qtu_emf[mix*(mkx+1)];//                                !  Penetrative downdraft total water at entraining interfaces [ kg/kg ]
///    double    uu_emf[mix*(mkx+1)];//                                 !  Penetrative downdraft zonal wind at entraining interfaces [ m/s ]
///    double    vu_emf[mix*(mkx+1)];//                                 !  Penetrative downdraft meridional wind at entraining interfaces [ m/s ]
   // double    tru_emf[(mkx+1)*ncnst];//                          !  Penetrative Downdraft tracers at entraining interfaces [ #, kg/kg ]    

    //!----- Variables associated with evaporations of convective 'rain' and 'snow'

    // double    flxrain[mkx+1];//                                !  Downward rain flux at each interface [ kg/m2/s ]
    // double    flxsnow[mkx+1];//                                !  Downward snow flux at each interface [ kg/m2/s ]
    // double    ntraprd[mkx];//                                  !  Net production ( production - evaporation +  melting ) rate of rain in each layer [ kg/kg/s ]
    // double    ntsnprd[mkx];//                                  !  Net production ( production - evaporation + freezing ) rate of snow in each layer [ kg/kg/s ]
    double    flxsntm;//                                       !  Downward snow flux at the top of each layer after melting [ kg/m2/s ]
    double    snowmlt;//                                       !  Snow melting tendency [ kg/kg/s ]
    double    subsat;//                                        !  Sub-saturation ratio (1-qv/qs) [ no unit ]
    double    evprain;//                                       !  Evaporation rate of rain [ kg/kg/s ]
    double    evpsnow;//                                       !  Evaporation rate of snow [ kg/kg/s ]
    double    evplimit;//                                      !  Limiter of 'evprain + evpsnow' [ kg/kg/s ]
    double    evplimit_rain;//                                 !  Limiter of 'evprain' [ kg/kg/s ]
    double    evplimit_snow;//                                 !  Limiter of 'evpsnow' [ kg/kg/s ]
    double    evpint_rain;//                                   !  Vertically-integrated evaporative flux of rain [ kg/m2/s ]
    double    evpint_snow;//                                   !  Vertically-integrated evaporative flux of snow [ kg/m2/s ]
    double    kevp;//                                          !  Evaporative efficiency [ complex unit ]

   // !----- Other internal variables

    int     kk, mm, k, i, m, kp1, km1;
    int     iter_scaleh, iter_xc;
    int     id_check, status;
    int     klcl;//                                          !  Layer containing LCL of source air
    int     kinv;//                                          !  Inversion layer with PBL top interface as a lower interface
    int     krel;//                                          !  Release layer where buoyancy sorting mixing occurs for the first time
    int     klfc;//                                          !  LFC layer of cumulus source air
    int     kbup;//                                          !  Top layer in which cloud buoyancy is positive at the top interface
    int     kpen;//                                          !  Highest layer with positive updraft vertical velocity - top layer cumulus can reach
    bool     id_exit;//   
    bool     forcedCu;//                                      !  If 'true', cumulus updraft cannot overcome the buoyancy barrier just above the PBL top.
    double    thlsrc, qtsrc, usrc, vsrc, thvlsrc;//            !  Updraft source air properties
    double    PGFc, uplus, vplus;//
    double    trsrc[ncnst], tre[ncnst];//
    double    plcl, plfc, prel, wrel;//
    double    frc_rasn;//
    double    ee2, ud2, wtw, wtwb, wtwh;//
    double    xc, xc_2;//                                       
    double    cldhgt, scaleh, tscaleh, cridis, rle, rkm;//
    double    rkfre, sigmaw, epsvarw, tkeavg, dpsum, dpi, thvlmin;//
    double    thlxsat, qtxsat, thvxsat, x_cu, x_en, thv_x0, thv_x1;//
    double    thj, qvj, qlj, qij, thvj, tj, thv0j, rho0j, rhos0j, qse;// 
    double    cin, cinlcl;//
    double    pe, dpe, exne, thvebot, thle, qte, ue, ve, thlue, qtue, wue;
    double    mu, mumin0, mumin1, mumin2, mulcl, mulclstar;
    double    cbmf, wcrit, winv, wlcl, ufrcinv, ufrclcl, rmaxfrac;
    double    criqc, exql, exqi, ppen;
    double    thl0top, thl0bot, qt0bot, qt0top, thvubot, thvutop;
    double    thlu_top, qtu_top, qlu_top, qiu_top, qlu_mid, qiu_mid, exntop;
    double    thl0lcl, qt0lcl, thv0lcl, thv0rel, rho0inv, autodet;
    double    aquad, bquad, cquad, xc1, xc2, excessu, excess0, xsat, xs1, xs2;
    double    bogbot, bogtop, delbog, drage, expfac, rbuoy, rdrag;
    double    rcwp, rlwp, riwp, qcubelow, qlubelow, qiubelow;
    double    rainflx, snowflx;                     
    double    es[1];                               
    double    qs[1];                               
    double    gam[1];   //                                     !  (L/cp)*dqs/dT
    //!-----------------------------!
    double    qsat_arglf[1];
    double    pelf[1];
    //!-----------------------------!
    double    qsat_arg;             
    double    xsrc, xmean, xtop, xbot, xflx[mkx+1];//
    double    tmp1, tmp2;

    //!----- Some diagnostic internal output variables
    
///   double  excessu_arr[mkx];// 
    
    
///   double  excess0_arr[mkx];//
   
   
///   double  xc_arr[mkx];//
    
 
///   double  aquad_arr[mkx];//
    
    
///   double  bquad_arr[mkx];//
   
   
///   double  cquad_arr[mkx];//
    
    
///   double  bogbot_arr[mkx];//
   
 
///   double  bogtop_arr[mkx];//


	double  ufrcinvbase_s, ufrclcl_s, winvbase_s, wlcl_s, plcl_s, pinv_s, plfc_s, 
                qtsrc_s, thlsrc_s, thvlsrc_s, emfkbup_s, cinlcl_s, pbup_s, ppen_s, cbmflimit_s, 
                tkeavg_s, zinv_s, rcwp_s, rlwp_s, riwp_s; 
    double  ufrcinvbase, winvbase, pinv, zinv, emfkbup, cbmflimit, rho0rel;  

	// //!----- Variables for implicit CIN computation
    // double qv0_s[mkx]  , ql0_s[mkx]   , qi0_s[mkx]   , s0_s[mkx]    , u0_s[mkx]    ,      
    //        v0_s[mkx]   , t0_s[mkx]    , qt0_s[mkx]    , qvten_s[mkx] , //, thl0_s[mkx]  , thvl0_s[mkx]
    //        qlten_s[mkx], qiten_s[mkx] , qrten_s[mkx] , qsten_s[mkx] , sten_s[mkx]  , evapc_s[mkx] , 
    //        uten_s[mkx] , vten_s[mkx]  , cufrc_s[mkx] , qcu_s[mkx]   , qlu_s[mkx]   , qiu_s[mkx]   , 
    //        fer_s[mkx]  , fdr_s[mkx]   , qc_s[mkx]    , qtten_s[mkx] , slten_s[mkx]; 
  
    double cush_s , precip_s, snow_s  , cin_s, rliq_s, cbmf_s, cnt_s, cnb_s;
    double cin_i,cin_f,del_CIN,ke,alpha,thlj;
    double cinlcl_i,cinlcl_f,del_cinlcl;
    int iter;


	double trsrc_o[ncnst];
	
    double tkeavg_o , thvlmin_o, qtsrc_o  , thvlsrc_o, thlsrc_o ,    
                                        usrc_o   , vsrc_o   , plcl_o   , plfc_o   ,               
                                        thv0lcl_o, cinlcl_o; 
    int kinv_o   , klcl_o   , klfc_o;  


    int ixnumliq, ixnumice;



	xlv = uwshcu_mp_xlv_;
	xlf = uwshcu_mp_xlf_;
	xls = uwshcu_mp_xls_;
	cp = uwshcu_mp_cp_;
	zvir = uwshcu_mp_zvir_;
	r = uwshcu_mp_r_;
	g = uwshcu_mp_g_;
	ep2 = uwshcu_mp_ep2_;
	p00 = uwshcu_mp_p00_;
	rovcp = uwshcu_mp_rovcp_;
	rpen = uwshcu_mp_rpen_;
	
	wv_saturation_mp_tmin__ = wv_saturation_mp_tmin_;
	wv_saturation_mp_tmax__ = wv_saturation_mp_tmax_;
	wv_saturation_mp_ttrice__ = wv_saturation_mp_ttrice_;
	
	wv_saturation_mp_epsqs__ = wv_saturation_mp_epsqs_;
	wv_saturation_mp_rgasv__ = wv_saturation_mp_rgasv_;
	wv_saturation_mp_hlatf__ = wv_saturation_mp_hlatf_;
	wv_saturation_mp_hlatv__ = wv_saturation_mp_hlatv_;
	wv_saturation_mp_cp__ = wv_saturation_mp_cp_;
	wv_saturation_mp_tmelt__ = wv_saturation_mp_tmelt_;
	wv_saturation_mp_icephs__ = wv_saturation_mp_icephs_;
	 int tempp;
	// for(tempp = 0;tempp < 6;++tempp)
	// 	wv_saturation_mp_pcf__[tempp] = wv_saturation_mp_pcf_d[tempp];
	for(tempp = 0;tempp < 250;++tempp)
		wv_saturation_mp_estbl__[tempp] = wv_saturation_mp_estbl_d[tempp];
		//  xlv   = 2501000.00000000;
		 //  xlf   = 333700.000000000;
		 //  xls   = xlv + xlf;
		 //  cp    = 1004.64000000000;
		 //  zvir  = 0.607793072824156;
		 //  r     = 287.042311365049;
		 //  g     = 9.80616000000000;
		 //  ep2   = 0.621970586204516;
		 //  p00   = 1.0e5;
		//  rovcp = r/cp;
	
		//  uwshcu_mp_rpen_=10.0000000000;
		int iend = iendtemp;
		 double ntot_amode = 3;
	
		double dt = ztodt;


		//! ------------------ !
    //!                    !
    //! Define Parameters  !
    //!                    !
    //! ------------------ !
	//! ------------------------ !
    //! Iterative xc calculation !
    //! ------------------------ !
	int niter_xc = 2;
	//////////lifei///////////////
//////*	int ntot_amode = 8;
//////*	int numptr_amode[8];
//////*	double erfc,qmin[ncnst];
	//! ----------------------------------------------------------- !
    //! Choice of 'CIN = cin' (.true.) or 'CIN = cinlcl' (.false.). !
    //! ----------------------------------------------------------- !
/////double erfc;
    bool use_CINcin = true;

    //! --------------------------------------------------------------- !
    //! Choice of 'explicit' ( 1 ) or 'implicit' ( 2 )  CIN.            !
    //!                                                                 !
    //! When choose 'CIN = cinlcl' above,  it is recommended not to use ! 
    //! implicit CIN, i.e., do 'NOT' choose simultaneously :            !
    //!            [ 'use_CINcin=.false. & 'iter_cin=2' ]               !
    //! since 'cinlcl' will be always set to zero whenever LCL is below !
    //! the PBL top interface in the current code. So, averaging cinlcl !
    //! of two iter_cin steps is likely not so good. Except that,   all !
    //! the other combinations of  'use_CINcin'  & 'iter_cin' are OK.   !
    //!                                                                 !
    //! Feb 2007, Bundy: Note that use_CINcin = .false. will try to use !
    //!           a variable (del_cinlcl) that is not currently set     !
    //!                                                                 !
    //! --------------------------------------------------------------- !

    int iter_cin = 2;

    //! ---------------------------------------------------------------- !
    //! Choice of 'self-detrainment' by negative buoyancy in calculating !
    //! cumulus updraft mass flux at the top interface in each layer.    !
    //! ---------------------------------------------------------------- !

    bool use_self_detrain = false;
    
    //! --------------------------------------------------------- !
    //! Cumulus momentum flux : turn-on (.true.) or off (.false.) !
    //! --------------------------------------------------------- !

    bool use_momenflx = true;

    //! ----------------------------------------------------------------------------------------- !
   // ! Penetrative Entrainment : Cumulative ( .true. , original ) or Non-Cumulative ( .false. )  !
   // ! This option ( .false. ) is designed to reduce the sensitivity to the vertical resolution. !
   // ! ----------------------------------------------------------------------------------------- !

    bool use_cumpenent = true;

    //! --------------------------------------------------------------------------------------------------------------- !
    //! Computation of the grid-mean condensate tendency.                                                               !
    //!     use_expconten = .true.  : explcitly compute tendency by condensate detrainment and compensating subsidence  !
    //!     use_expconten = .false. : use the original proportional condensate tendency equation. ( original )          !
    //! --------------------------------------------------------------------------------------------------------------- !

    bool use_expconten = true;

    //! --------------------------------------------------------------------------------------------------------------- !
    //! Treatment of reserved condensate                                                                                !
    //!     use_unicondet = .true.  : detrain condensate uniformly over the environment ( original )                    !
    //!     use_unicondet = .false. : detrain condensate into the pre-existing stratus                                  !
    //! --------------------------------------------------------------------------------------------------------------- !

    bool use_unicondet = false;

    //! ----------------------- !
    //! For lateral entrainment !
    //! ----------------------- !

    rle = 0.1;//            !  For critical stopping distance for lateral entrainment [no unit]
  //rkm = 16.0_r8)//        !  Determine the amount of air that is involved in buoyancy-sorting [no unit] 
    rkm = 14.0;//           !  Determine the amount of air that is involved in buoyancy-sorting [no unit]

    rkfre = 1.0;//       !  Vertical velocity variance as fraction of  tke. 
    rmaxfrac = 0.10;//   !  Maximum allowable 'core' updraft fraction
    mumin1 = 0.906;//    !  Normalized CIN ('mu') corresponding to 'rmaxfrac' at the PBL top
                   //                  !  obtaind by inverting 'rmaxfrac = 0.5*erfc(mumin1)'.
                   //                  !  [ rmaxfrac:mumin1 ] = [ 0.05:1.163, 0.075:1.018, 0.1:0.906, 0.15:0.733, 0.2:0.595, 0.25:0.477 ] 
    rbuoy = 1.0;//       !  For nonhydrostatic pressure effects on updraft [no unit]
    rdrag = 1.0;//       !  Drag coefficient [no unit]

    epsvarw = 5.0e-4;//  !  Variance of w at PBL top by meso-scale component [m2/s2]          
    PGFc = 0.7;//        !  This is used for calculating vertical variations cumulus  
               //        !  'u' & 'v' by horizontal PGF during upward motion [no unit]

   // ! ---------------------------------------- !
   // ! Bulk microphysics controlling parameters !
   // ! --------------------------------------------------------------------------- ! 
   // ! criqc    : Maximum condensate that can be hold by cumulus updraft [kg/kg]   !
   // ! frc_rasn : Fraction of precipitable condensate in the expelled cloud water  !
   // !            from cumulus updraft. The remaining fraction ('1-frc_rasn')  is  !
   // !            'suspended condensate'.                                          !
   // !                0 : all expelled condensate is 'suspended condensate'        ! 
   // !                1 : all expelled condensate is 'precipitable condensate'     !
   // ! kevp     : Evaporative efficiency                                           !
   // ! noevap_krelkpen : No evaporation from 'krel' to 'kpen' layers               ! 
   // ! --------------------------------------------------------------------------- !    

    criqc    = 0.7e-3; 
    frc_rasn = 1.0;    
    kevp     = 2.0e-6;  
    bool noevap_krelkpen = false;

    //!--------------------------------------------!
    //!character*3, public :: cnst_type(ncnst)
    //!------------------------!
    //!                        !
    //! Start Main Calculation !
    //!                        !
    // //!------------------------!
     ixnumliq = 4;
     ixnumice = 5;
	//__constituents_MOD_cnst_get_ind( "NUMLIQ", ixnumliq );
    //__constituents_MOD_cnst_get_ind( "NUMICE", ixnumice );
////erfc = 1.0;
	int j;//index
	// for(j=0;j<ncnst;++j)
    // 	qmin[j] = 1.0;
   // !call cnst_get_ind( 'NUMLIQ', ixnumliq )
   // !call cnst_get_ind( 'NUMICE', ixnumice )

   // ! ------------------------------------------------------- !
   // ! Initialize output variables defined for all grid points !
   // ! ------------------------------------------------------- !
   // !---------------------------lifei------------------------------!

   // !---------------------------------------------------------!

   //!cnst_type(:ncnst) = 'wet'
//    for(j=0;j<8;++j)
//    numptr_amode[j] = 4;

   i=blockDim.x * blockIdx.x + threadIdx.x;

   if((i>=0)&&(i<iend))
   {
//printf("lifeilifei\n");
   	// for(j=0;j<mkx;++j)
   	// //for(i=0;i<iend;++i)
	//    {
	// 	   tw0_in[j*iend+i] = 0.0;
	// 	   qw0_in[j*iend+i] = 1.0;
	//    }

   //for(i=0;i<iend;++i)
   //{
		precip_out[i]            = 0.0;
		snow_out[i]              = 0.0;
//		cinh_out[i]              = -1.0;
///   	    cinlclh_out[i]           = -1.0;
        cbmf_out[i]              = 0.0;
		rliq_out[i]              = 0.0;
   	 	//cnt_out[i]               = real(mkx, r8)
		cnt_out[i]               = (double)mkx;
    	cnb_out[i]               = 0.0;
///		ufrcinvbase_out[i]       = 0.0;
///    	ufrclcl_out[i]           = 0.0;
///    	winvbase_out[i]          = 0.0;
///    	wlcl_out[i]              = 0.0;
///    	plcl_out[i]              = 0.0;
///    	pinv_out[i]              = 0.0;
    	// plfc_out[i]              = 0.0;
    	// pbup_out[i]              = 0.0;
    	// ppen_out[i]              = 0.0;
    	// qtsrc_out[i]             = 0.0;
    	// thlsrc_out[i]            = 0.0;
    	// thvlsrc_out[i]           = 0.0;
    	// emfkbup_out[i]           = 0.0;
    	// cbmflimit_out[i]         = 0.0;
    	// tkeavg_out[i]            = 0.0;
    	// zinv_out[i]              = 0.0;
    	// rcwp_out[i]              = 0.0;
    	// rlwp_out[i]              = 0.0;
    	// riwp_out[i]              = 0.0;
		exit_UWCu[i]             = 0.0; 
    	exit_conden[i]           = 0.0; 
    	exit_klclmkx[i]          = 0.0; 
    	exit_klfcmkx[i]          = 0.0; 
    	exit_ufrc[i]             = 0.0; 
    	exit_wtw[i]              = 0.0; 
    	exit_drycore[i]          = 0.0; 
    	exit_wu[i]               = 0.0; 
    	exit_cufilter[i]         = 0.0; 
    	exit_kinv1[i]            = 0.0; 
    	exit_rei[i]              = 0.0; 

///    	limit_shcu[i]            = 0.0; 
///    	limit_negcon[i]          = 0.0; 
///    	limit_ufrc[i]            = 0.0;
///    	limit_ppen[i]            = 0.0;
///    	limit_emf[i]             = 0.0;
///    	limit_cinlcl[i]          = 0.0;
///    	limit_cin[i]             = 0.0;
///    	limit_cbmf[i]            = 0.0;
///    	limit_rei[i]             = 0.0;

///    	ind_delcin[i]            = 0.0;
  // }
   for(j=0;j<mkx+1;++j)
   //	for(i=0;i<iend;++i)
	   {
		umf_out[j*iend+i]         = 0.0;
		slflx_out[j*iend+i]       = 0.0;
		qtflx_out[j*iend+i]       = 0.0;
		flxprc1_out[j*iend+i]     = 0.0;
		flxsnow1_out[j*iend+i]    = 0.0;

//		ufrc_out[j*iend+i]        = 0.0;
//		uflx_out[j*iend+i]        = 0.0;
//		vflx_out[j*iend+i]        = 0.0;

		wu_out[j*iend+i]          = 0.0;
///		qtu_out[j*iend+i]         = 0.0;
///		thlu_out[j*iend+i]        = 0.0;
///		thvu_out[j*iend+i]        = 0.0;
///		uu_out[j*iend+i]          = 0.0;
///		vu_out[j*iend+i]          = 0.0;
///		qtu_emf_out[j*iend+i]     = 0.0;
///		thlu_emf_out[j*iend+i]    = 0.0;
///		uu_emf_out[j*iend+i]      = 0.0;
///		vu_emf_out[j*iend+i]      = 0.0;
///		uemf_out[j*iend+i]        = 0.0;
///		flxrain_out[j*iend+i]     = 0.0;  
///		flxsnow_out[j*iend+i]     = 0.0;
	   }

	for(j=0;j<mkx;++j)
   	// for(i=0;i<iend;++i)
	   {
        qvten_out[j*iend+i]        = 0.0;
    	qlten_out[j*iend+i]        = 0.0;
    	qiten_out[j*iend+i]        = 0.0;
    	sten_out[j*iend+i]         = 0.0;
    	uten_out[j*iend+i]         = 0.0;
    	vten_out[j*iend+i]         = 0.0;
    	qrten_out[j*iend+i]        = 0.0;
    	qsten_out[j*iend+i]        = 0.0;
		evapc_out[j*iend+i]        = 0.0;
		cufrc_out[j*iend+i]        = 0.0;
		qcu_out[j*iend+i]          = 0.0;
		qlu_out[j*iend+i]          = 0.0;
		qiu_out[j*iend+i]          = 0.0;
//		fer_out[j*iend+i]          = 0.0;
//		fdr_out[j*iend+i]          = 0.0;
		qc_out[j*iend+i]           = 0.0;
//		qtten_out[j*iend+i]        = 0.0;
//		slten_out[j*iend+i]        = 0.0;
///		dwten_out[j*iend+i]        = 0.0;
///		diten_out[j*iend+i]        = 0.0;

	
///		excessu_arr_out[j*iend+i]  = 0.0;
///		excess0_arr_out[j*iend+i]  = 0.0;
///		xc_arr_out[j*iend+i]       = 0.0;
///		aquad_arr_out[j*iend+i]    = 0.0;
///		bquad_arr_out[j*iend+i]    = 0.0;
///		cquad_arr_out[j*iend+i]    = 0.0;
///		bogbot_arr_out[j*iend+i]   = 0.0;
///		bogtop_arr_out[j*iend+i]   = 0.0;
	   }


   // for(i=0;i<iend;++i)
//	 {
///	    ntraprd_out[(mkx-1)*iend+i] = 0.0;
///	    ntsnprd_out[(mkx-1)*iend+i] = 0.0;
//	 }
	   

	for(k=0;k<ncnst;++k)
	 for(j=0;j<mkx;++j)
//	  for(i=0;i<iend;++i)
	   {
		trten_out[k*mkx*iend+j*iend+i] = 0.0;
	   }
	for(k=0;k<ncnst;++k)
	 for(j=0;j<mkx+1;++j)
//	  for(i=0;i<iend;++i)
	   {
//		trflx_out[k*(mkx+1)*iend+j*iend+i] = 0.0;
//		tru_out[k*(mkx+1)*iend+j*iend+i] = 0.0;
//		tru_emf_out[k*(mkx+1)*iend+j*iend+i] = 0.0;
	   }
}

// if(i == 7)
// {
// 	printf("-------------------- At sub. compute_uwshcu -----------------------\n");
// 	printf("1st: trten_out(8,1,17) =%15.13f\n",trten_out[16*mkx*iend+0*iend+7]);
// 	printf("dpdry0_in(8,1) =%15.13f\n",dpdry0_in[0*iend+7]);
// 	printf("-------------------------------------------------------------------\n");
// }
	  // !--------------------------------------------------------------!
	  // !                                                              !
	  // ! Start the column i loop where i is a horozontal column index !
	  // !                                                              !
	  // !--------------------------------------------------------------!
	  //! Compute wet-bulb temperature and specific humidity
	  //! for treating evaporation of precipitation.
	// if(i==0)
	// {
	// findsp( lchnk, iend, qv0_in, t0_in, p0_in, tw0_in, qw0_in );
	// }
//	 if(i==0) printf("qw0_in[8]=%e\n",qw0_in[0*iend+8]);

  // for(i=0;i<iend;++i)
  if((i>=0)&&(i<iend))
	{
		//printf("%d, %d\n",i,id_check);
		id_exit = false;
		//! -------------------------------------------- !
		//! Define 1D input variables at each grid point !
		//! -------------------------------------------- !
		for(j=0;j<mkx+1;++j)
		{
			ps0[j]       = ps0_in[j*iend+i];
			zs0[j]       = zs0_in[j*iend+i];
			tke[j]       = tke_in[j*iend+i];
		}
		for(j=0;j<mkx;++j)
		{
			p0[j]         = p0_in[j*iend+i];
			z0[j]         = z0_in[j*iend+i];
			dp0[j]        = dp0_in[j*iend+i];
			dpdry0[j]     = dpdry0_in[j*iend+i];
			u0[j]         = u0_in[j*iend+i];
			v0[j]         = v0_in[j*iend+i];
			qv0[j]        = qv0_in[j*iend+i];
			ql0[j]        = ql0_in[j*iend+i];
			qi0[j]        = qi0_in[j*iend+i];
			t0[j]         = t0_in[j*iend+i];
			s0[j]         = s0_in[j*iend+i];
			
			cldfrct[j]    = cldfrct_in[j*iend+i];
			concldfrct[j] = concldfrct_in[j*iend+i];
		}	

		pblh             = pblh_in[i];
		cush             = cush_inout[i];

		for(k=0;k<ncnst;++k)
			for(j=0;j<mkx;++j)
		   		tr0[k*mkx+j] = tr0_in[k*mkx*iend+j*iend+i];
//	__syncthreads();

     // ! --------------------------------------------------------- !
     // ! Compute other basic thermodynamic variables directly from ! 
     // ! the input variables at each grid point                    !
     // ! --------------------------------------------------------- !

	 //!----- 1. Compute internal environmental variables
	 for(j=0;j<mkx;++j)
		exn0[j]   = pow((p0[j]/p00),rovcp);
	for(j=0;j<mkx+1;++j)
		exns0[j] = pow((ps0[j]/p00),rovcp);
	 for(j=0;j<mkx;++j)
		qt0[j]    = (qv0[j] + ql0[j] + qi0[j]);
		//printf("t0[j]=%f,ql0[j]=%f,qi0[j]=%f,exn0[j]=%f\n",t0[j],ql0[j],qi0[j],exn0[j]);
	for(j=0;j<mkx;++j)
		thl0[j]   = (t0[j] - xlv*ql0[j]/cp - xls*qi0[j]/cp)/exn0[j];
	for(j=0;j<mkx;++j)
		thvl0[j]  = (1.0 + zvir*qt0[j])*thl0[j];
	
	// !----- 2. Compute slopes of environmental variables in each layer
	// !         Dimension of ssthl0(:mkx) is implicit.
	slope(ssthl0,mkxtemp,thl0,p0); 
	slope(ssqt0,mkxtemp,qt0,p0);
	slope(ssu0,mkxtemp,u0,p0);
	slope(ssv0,mkxtemp,v0,p0);
	
	double temp_sstr0[mkx],temp_tr0[mkx];
	for(k=0;k<ncnst;++k)
	{
		for(j=0;j<mkx;++j)
			temp_tr0[j] = tr0[k*mkx+j];
		slope(temp_sstr0,mkxtemp,temp_tr0,p0);
		for(j=0;j<mkx;++j)
			sstr0[k*mkx+j] = temp_sstr0[j];
	}

	//!----- 3. Compute "thv0" and "thvl0" at the top/bottom interfaces in each layer
	//!         There are computed from the reconstructed thl, qt at the top/bottom.

	for(j=0;j<mkx;++j)
	{
		thl0bot = thl0[j] + ssthl0[j]*(ps0[j] - p0[j]);
        qt0bot  = qt0[j]  + ssqt0[j] *(ps0[j] - p0[j]);
		//if(i==220)	printf("thl0bot=%e,qt0bot=%e\n",thl0bot,qt0bot);
		conden(ps0[j],thl0bot,qt0bot,&thj,&qvj,&qlj,&qij,&qse,&id_check);
//if(i==220) printf("thj=%e,qvj=%e,qlj=%e,qij=%e,qse=%e\n",thj,qvj,qlj,qij,qse);

		if(id_check == 1)
		{
			exit_conden[i] = 1.0;
            id_exit = true;
            goto lable333;
		}
		thv0bot[j]  = thj*(1.0 + zvir*qvj - qlj - qij);
		thvl0bot[j] = thl0bot*(1.0 + zvir*qt0bot);

		thl0top = thl0[j] + ssthl0[j]*(ps0[j+1] - p0[j]);
		qt0top  =  qt0[j] + ssqt0[j]*(ps0[j+1] - p0[j]);
	//	if(i==220) printf("thl0top=%e,qt0top=%e\n",thl0top,qt0top);
		conden(ps0[j+1],thl0top,qt0top,&thj,&qvj,&qlj,&qij,&qse,&id_check);
		if( id_check == 1 ) 
		{
			exit_conden[i] = 1.0;
			id_exit = true;
			goto lable333;
		}
		thv0top[j]  = thj*(1.0 + zvir*qvj - qlj - qij);
		thvl0top[j] = thl0top*(1.0 + zvir*qt0top);
	//	if(i==22) printf("j=%d,thv0top[j]=%e,thvl0top[j]=%e\n",j,thv0top[j],thvl0top[j]);
	}
	
	//! ------------------------------------------------------------ !
	//! Save input and related environmental thermodynamic variables !
	//! for use at "iter_cin=2" when "del_CIN >= 0"                  !
	//! ------------------------------------------------------------ !

	for(j=0;j<mkx;++j)
	{
		qv0_o[j]          = qv0[j];
		ql0_o[j]          = ql0[j];
		qi0_o[j]          = qi0[j];
		t0_o[j]           = t0[j];
		s0_o[j]           = s0[j];
		u0_o[j]           = u0[j];
		v0_o[j]           = v0[j];
		qt0_o[j]          = qt0[j];
		thl0_o[j]         = thl0[j];
		thvl0_o[j]        = thvl0[j];
		ssthl0_o[j]       = ssthl0[j];
		ssqt0_o[j]        = ssqt0[j];
		thv0bot_o[j]      = thv0bot[j];
		thv0top_o[j]      = thv0top[j];
		thvl0bot_o[j]     = thvl0bot[j];
		thvl0top_o[j]     = thvl0top[j];
		ssu0_o[j]         = ssu0[j]; 
		ssv0_o[j]         = ssv0[j];
	}
 	for(k=0;k<ncnst;++k)
		for(j=0;j<mkx;++j)
		{
			tr0_o[k*mkx+j] = tr0[k*mkx+j];
	  		sstr0_o[k*mkx+j]   = sstr0[k*mkx+j];
		}
	
	//! ---------------------------------------------- !
	//! Initialize output variables at each grid point !
	//! ---------------------------------------------- !
	for(j=0;j<mkx;++j)
	{
		qvten[j]         = 0.0;
		qlten[j]         = 0.0;
		qiten[j]         = 0.0;
		sten[j]          = 0.0;
		uten[j]          = 0.0;
		vten[j]          = 0.0;
		qrten[j]         = 0.0;
		qsten[j]         = 0.0;
		dwten[j]         = 0.0;
		diten[j]         = 0.0;
		evapc[j]         = 0.0;
		cufrc[j]         = 0.0;
		qcu[j]           = 0.0;
		qlu[j]           = 0.0;
		qiu[j]           = 0.0;
		fer[j]           = 0.0;
		fdr[j]           = 0.0;
		qc[j]            = 0.0;
		qc_l[j]          = 0.0;
		qc_i[j]          = 0.0;
		qtten[j]         = 0.0;
		slten[j]         = 0.0;  
///		excessu_arr[j]   = 0.0;
///		excess0_arr[j]   = 0.0;
///		xc_arr[j]        = 0.0;
///		aquad_arr[j]     = 0.0;
///		bquad_arr[j]     = 0.0;
///		cquad_arr[j]     = 0.0;
///		bogbot_arr[j]    = 0.0;
///		bogtop_arr[j]    = 0.0;


		comsub[j]        = 0.0;
		qlten_sink[j]    = 0.0;
		qiten_sink[j]    = 0.0; 
		nlten_sink[j]    = 0.0;
		niten_sink[j]    = 0.0; 
	}
	
	for(j=0;j<mkx+1;++j)
	{
		uemf[j]         = 0.0;
		umf[j]          = 0.0;
		emf[j]          = 0.0;
		slflx[j]        = 0.0;
		qtflx[j]        = 0.0;
		uflx[j]         = 0.0;
		vflx[j]         = 0.0;
		ufrc[j]         = 0.0;  

		thlu[j]         = 0.0;
		qtu[j]          = 0.0;
		uu[j]           = 0.0;
		vu[j]           = 0.0;
		wu[j]           = 0.0;
		thvu[j]         = 0.0;
		thlu_emf[j]     = 0.0;
		qtu_emf[j]      = 0.0;
		uu_emf[j]       = 0.0;
		vu_emf[j]       = 0.0;
	}
	
	for(k=0;k<ncnst;++k)
		for(j=0;j<mkx;++j)
		{
			trten[k*mkx+j]   = 0.0;
		}
	for(k=0;k<ncnst;++k)
		for(j=0;j<mkx+1;++j)
		{
			trflx[k*(mkx+1)+j]   = 0.0;
			tru[k*(mkx+1)+j]     = 0.0;
			tru_emf[k*(mkx+1)+j] = 0.0;
		}

	precip   = 0.0;
	snow     = 0.0;
	
	cin      = 0.0;
	cbmf     = 0.0;
	
	rliq     = 0.0;
	cnt      = (double)mkx;  //real(mkx, r8)
	cnb      = 0.0;
	
	ufrcinvbase    = 0.0;
	ufrclcl        = 0.0;
	winvbase       = 0.0;
	wlcl           = 0.0;
	emfkbup        = 0.0; 
	cbmflimit      = 0.0;

	//!-----------------------------------------------! 
    //! Below 'iter' loop is for implicit CIN closure !
    //!-----------------------------------------------!

    //! ----------------------------------------------------------------------------- ! 
    //! It is important to note that this iterative cin loop is located at the outest !
    //! shell of the code. Thus, source air properties can also be changed during the !
    //! iterative cin calculation, because cumulus convection induces non-zero fluxes !
    //! even at interfaces below PBL top height through 'fluxbelowinv' subroutine.    !
    //! ----------------------------------------------------------------------------- !
	
	for(iter=1;iter<=iter_cin;++iter)
	{

	      //if(i==220)
		//printf("iter = %d\n",iter);
		//! ---------------------------------------------------------------------- ! 
		//! Cumulus scale height                                                   ! 
		//! In contrast to the premitive code, cumulus scale height is iteratively !
		//! calculated at each time step, and at each iterative cin step.          !
		//! It is not clear whether I should locate below two lines within or  out !
		//! of the iterative cin loop.                                             !
		//! ---------------------------------------------------------------------- !

		tscaleh = cush;                        
		cush    = -1.0;

	//	int aaa;
		//! ----------------------------------------------------------------------- !
		//! Find PBL top height interface index, 'kinv-1' where 'kinv' is the layer !
		//! index with PBLH in it. When PBLH is exactly at interface, 'kinv' is the !
		//! layer index having PBLH as a lower interface.                           !
		//! In the previous code, I set the lower limit of 'kinv' by 2  in order to !
		//! be consistent with the other parts of the code. However in the modified !
		//! code, I allowed 'kinv' to be 1 & if 'kinv = 1', I just exit the program !
		//! without performing cumulus convection. This new approach seems to be    !
		//! more reasonable: if PBL height is within 'kinv=1' layer, surface is STL !
		//! interface (bflxs <= 0) and interface just above the surface should be   !
		//! either non-turbulent (Ri>0.19) or stably turbulent (0<=Ri<0.19 but this !
		//! interface is identified as a base external interface of upperlying CL.  !
		//! Thus, when 'kinv=1', PBL scheme guarantees 'bflxs <= 0'.  For this case !
		//! it is reasonable to assume that cumulus convection does not happen.     !
		//! When these is SBCL, PBL height from the PBL scheme is likely to be very !
		//! close at 'kinv-1' interface, but not exactly, since 'zi' information is !
		//! changed between two model time steps. In order to ensure correct identi !
		//! fication of 'kinv' for general case including SBCL, I imposed an offset !
		//! of 5 [m] in the below 'kinv' finding block.                             !
		//! ----------------------------------------------------------------------- !

		for(j=mkx-1;j>=1;j--)
		{
			if((pblh + 5.0 - zs0[j])*(pblh + 5.0 - zs0[j+1]) < 0.0 )
			{
				kinv = j+1;
				goto lable15;
			}
		}
		kinv = 1;
lable15:

	//	aaa=1;//continue;
//if(i == 0 || i==1 || i==2)

		if(kinv<=1)
		{
		   exit_kinv1[i] = 1.0;
           id_exit = true;
		//   printf("1\n");
           goto lable333;
		}
		//! From here, it must be 'kinv >= 2'.

		//! -------------------------------------------------------------------------- !
		//! Find PBL averaged tke ('tkeavg') and minimum 'thvl' ('thvlmin') in the PBL !
		//! In the current code, 'tkeavg' is obtained by averaging all interfacial TKE !
		//! within the PBL. However, in order to be conceptually consistent with   PBL !
		//! scheme, 'tkeavg' should be calculated by considering surface buoyancy flux.!
		//! If surface buoyancy flux is positive ( bflxs >0 ), surface interfacial TKE !
		//! should be included in calculating 'tkeavg', while if bflxs <= 0,   surface !
		//! interfacial TKE should not be included in calculating 'tkeavg'.   I should !
		//! modify the code when 'bflxs' is available as an input of cumulus scheme.   !
		//! 'thvlmin' is a minimum 'thvl' within PBL obtained by comparing top &  base !
		//! interface values of 'thvl' in each layers within the PBL.                  !
		//! -------------------------------------------------------------------------- !

		dpsum    = 0.0;
		tkeavg   = 0.0;
		thvlmin  = 1000.0;
		for(k=0;k<=kinv-1;++k) //! Here, 'k' is an interfacial layer index. 
		{
		  if(k == 0) 
              dpi = ps0[0] - p0[0];
          else if(k == (kinv-1))
              dpi = p0[kinv-1-1] - ps0[kinv-1];
          else
              dpi = p0[k-1] - p0[k];
           
          dpsum  = dpsum  + dpi;  
          tkeavg = tkeavg + dpi*tke[k]; 
//if(i==220) printf("k=%d,tke[k]=%10.9e\n",k,tke[k]);
          if( k != 0 ) thvlmin = min(thvlmin,min(thvl0bot[k-1],thvl0top[k-1]));
		}
		tkeavg  = tkeavg/dpsum;

		//! ------------------------------------------------------------------ !
		//! Find characteristics of cumulus source air: qtsrc,thlsrc,usrc,vsrc !
		//! Note that 'thlsrc' was con-cocked using 'thvlsrc' and 'qtsrc'.     !
		//! 'qtsrc' is defined as the lowest layer mid-point value;   'thlsrc' !
		//! is from 'qtsrc' and 'thvlmin=thvlsrc'; 'usrc' & 'vsrc' are defined !
		//! as the values just below the PBL top interface.                    !
		//! ------------------------------------------------------------------ !

		qtsrc   = qt0[0];                     
		thvlsrc = thvlmin; 
		thlsrc  = thvlsrc / ( 1.0 + zvir * qtsrc );  
		usrc    = u0[kinv-1-1] + ssu0[kinv-1-1] * ( ps0[kinv-1] - p0[kinv-1-1] );             
		vsrc    = v0[kinv-1-1] + ssv0[kinv-1-1] * ( ps0[kinv-1] - p0[kinv-1-1] ); 
		for(k=0;k<ncnst;++k)
			trsrc[k] = tr0[k*mkx+0];

	   //! ------------------------------------------------------------------ !
       //! Find LCL of the source air and a layer index containing LCL (klcl) !
       //! When the LCL is exactly at the interface, 'klcl' is a layer index  ! 
       //! having 'plcl' as the lower interface similar to the 'kinv' case.   !
       //! In the previous code, I assumed that if LCL is located within the  !
       //! lowest model layer ( 1 ) or the top model layer ( mkx ), then  no  !
       //! convective adjustment is performed and just exited.   However, in  !
       //! the revised code, I relaxed the first constraint and  even though  !
       //! LCL is at the lowest model layer, I allowed cumulus convection to  !
       //! be initiated. For this case, cumulus convection should be started  !
       //! from the PBL top height, as shown in the following code.           !
       //! When source air is already saturated even at the surface, klcl is  !
       //! set to 1.                                                          !
       //! ------------------------------------------------------------------ !

	   plcl = qsinvert(qtsrc,thlsrc,ps0[0]);
	// if(i==220)  printf("plcl=%e\n",plcl);
	   for(k=0;k<=mkx;++k)
	   {
		   if(ps0[k] < plcl)
			{
				klcl = k;
				goto lable25;
			}
	   }
	   klcl = mkx;
lable25:
		//	aaa=1;//continue;
	   klcl = max(1,klcl);

	   if(plcl < 30000.0)
	   {
		//! if( klcl .eq. mkx ) then 
		   exit_klclmkx[i] = 1.0;
		   id_exit = true;
		  // printf("2\n");
		   goto lable333;
	   }

	   //! ------------------------------------------------------------- !
       //! Calculate environmental virtual potential temperature at LCL, !
       //!'thv0lcl' which is solely used in the 'cin' calculation. Note  !
       //! that 'thv0lcl' is calculated first by calculating  'thl0lcl'  !
       //! and 'qt0lcl' at the LCL, and performing 'conden' afterward,   !
       //! in fully consistent with the other parts of the code.         !
       //! ------------------------------------------------------------- !

	   thl0lcl = thl0[klcl-1] + ssthl0[klcl-1] * ( plcl - p0[klcl-1] );
       qt0lcl  = qt0[klcl-1]  + ssqt0[klcl-1]  * ( plcl - p0[klcl-1] );
	   conden(plcl,thl0lcl,qt0lcl,&thj,&qvj,&qlj,&qij,&qse,&id_check);
	   if(id_check == 1)
	   {
		   exit_conden[i] = 1.0;
		   id_exit = true;
		   goto lable333;
	   }
	   thv0lcl = thj * ( 1.0 + zvir * qvj - qlj - qij );
	  
	//    ! ------------------------------------------------------------------------ !
    //    ! Compute Convective Inhibition, 'cin' & 'cinlcl' [J/kg]=[m2/s2] TKE unit. !
    //    !                                                                          !
    //    ! 'cin' (cinlcl) is computed from the PBL top interface to LFC (LCL) using ! 
    //    ! piecewisely reconstructed environmental profiles, assuming environmental !
    //    ! buoyancy profile within each layer ( or from LCL to upper interface in   !
    //    ! each layer ) is simply a linear profile. For the purpose of cin (cinlcl) !
    //    ! calculation, we simply assume that lateral entrainment does not occur in !
    //    ! updrafting cumulus plume, i.e., cumulus source air property is conserved.!
    //    ! Below explains some rules used in the calculations of cin (cinlcl).   In !
    //    ! general, both 'cin' and 'cinlcl' are calculated from a PBL top interface !
    //    ! to LCL and LFC, respectively :                                           !
    //    ! 1. If LCL is lower than the PBL height, cinlcl = 0 and cin is calculated !
    //    !    from PBL height to LFC.                                               !
    //    ! 2. If LCL is higher than PBL height,   'cinlcl' is calculated by summing !
    //    !    both positive and negative cloud buoyancy up to LCL using 'single_cin'!
    //    !    From the LCL to LFC, however, only negative cloud buoyancy is counted !
    //    !    to calculate final 'cin' upto LFC.                                    !
    //    ! 3. If either 'cin' or 'cinlcl' is negative, they are set to be zero.     !
    //    ! In the below code, 'klfc' is the layer index containing 'LFC' similar to !
    //    ! 'kinv' and 'klcl'.                                                       !
    //    ! ------------------------------------------------------------------------ !
		
		cin    = 0.0;
        cinlcl = 0.0;
        plfc   = 0.0;
        klfc   = mkx;

		// ! ------------------------------------------------------------------------- !
        // ! Case 1. LCL height is higher than PBL interface ( 'pLCL <= ps0(kinv-1)' ) !
        // ! ------------------------------------------------------------------------- !
		
		if(klcl >= kinv)
		{
			for(k=kinv;k<=mkx-1;++k)
			{
				if(k < klcl)
				{
				   thvubot = thvlsrc;
                   thvutop = thvlsrc;  
                   cin     = cin + single_cin(ps0[k-1],thv0bot[k-1],ps0[k],thv0top[k-1],thvubot,thvutop);
				}
				else if(k == klcl)
				{
				   //!----- Bottom to LCL
				   thvubot = thvlsrc;
                   thvutop = thvlsrc;
                   cin     = cin + single_cin(ps0[k-1],thv0bot[k-1],plcl,thv0lcl,thvubot,thvutop);
                   if( cin < 0.0 )
				   {
///					   limit_cinlcl[i] = 1.0;
				   }
					   cinlcl  = max(cin,0.0);
					   cin     = cinlcl;
					   //!----- LCL to Top
					   thvubot = thvlsrc;
					   conden(ps0[k],thlsrc,qtsrc,&thj,&qvj,&qlj,&qij,&qse,&id_check);
					   if( id_check == 1 ) 
					   {
							exit_conden[i] = 1.0;
							id_exit = true;
							goto lable333;
					   }
					   thvutop = thj * ( 1.0 + zvir*qvj - qlj - qij );
					   getbuoy(plcl,thv0lcl,ps0[k],thv0top[k-1],thvubot,thvutop,&plfc,&cin);
					   if(plfc > 0.0)
					   {
						   klfc = k;
						   goto lable35;
					   }
                       
					
				}
				else
				{
						thvubot = thvutop;
						conden(ps0[k],thlsrc,qtsrc,&thj,&qvj,&qlj,&qij,&qse,&id_check);
						if(id_check == 1)
						{
							exit_conden[i] = 1.0;
							id_exit = true;
							goto lable333;
						}
						thvutop = thj * ( 1.0 + zvir*qvj - qlj - qij );
						getbuoy(ps0[k-1],thv0bot[k-1],ps0[k],thv0top[k-1],thvubot,thvutop,&plfc,&cin);
						if(plfc > 0.0)
						{
							klfc = k;
							goto lable35;
						}
				} 
			}
		}
		
			// ! ----------------------------------------------------------------------- !
			// ! Case 2. LCL height is lower than PBL interface ( 'pLCL > ps0(kinv-1)' ) !
			// ! ----------------------------------------------------------------------- !

		else
		{
			cinlcl = 0.0;
			for(k=kinv;k<=mkx-1;++k)
			{
				conden(ps0[k-1],thlsrc,qtsrc,&thj,&qvj,&qlj,&qij,&qse,&id_check);
				if(id_check == 1)
				{
					exit_conden[i] = 1.0;
					id_exit = true;
					goto lable333;
				}
				thvubot = thj * ( 1.0 + zvir*qvj - qlj - qij );
				conden(ps0[k],thlsrc,qtsrc,&thj,&qvj,&qlj,&qij,&qse,&id_check);
				if(id_check == 1)
				{
					exit_conden[i] = 1.0;
					id_exit = true;
					goto lable333;
				}
				thvutop = thj * ( 1.0 + zvir*qvj - qlj - qij );
		//	if(i==22)	printf("thv0top[k-1]=%33.32f\n",thv0top[k-1]);
				getbuoy(ps0[k-1],thv0bot[k-1],ps0[k],thv0top[k-1],thvubot,thvutop,&plfc,&cin);
				if(plfc > 0.0)
				{
					klfc = k;
					goto lable35;
				}
			}
		}//! End of CIN case selection

lable35:	//	 aaa=1;//continue;

///		if(cin < 0.0) limit_cin[i] = 1.0;
		cin = max(0.0,cin);
		if(klfc >= mkx)
		{
			klfc = mkx;
        // !!! write(iulog,*) 'klfc >= mkx'
           exit_klfcmkx[i] = 1.0;
           id_exit = true;
		 //  printf("10\n");
           goto lable333;
		}

		// ! ---------------------------------------------------------------------- !
		// ! In order to calculate implicit 'cin' (or 'cinlcl'), save the initially !
		// ! calculated 'cin' and 'cinlcl', and other related variables. These will !
		// ! be restored after calculating implicit CIN.                            !
		// ! ---------------------------------------------------------------------- !
	
		if(iter == 1)
		{
			cin_i       = cin;
			cinlcl_i    = cinlcl;
			ke          = rbuoy / ( rkfre * tkeavg + epsvarw ); 
			kinv_o      = kinv;     
			klcl_o      = klcl;     
			klfc_o      = klfc;    
			plcl_o      = plcl;    
			plfc_o      = plfc;     
			tkeavg_o    = tkeavg;   
			thvlmin_o   = thvlmin;
			qtsrc_o     = qtsrc;    
			thvlsrc_o   = thvlsrc;  
			thlsrc_o    = thlsrc;                   
			usrc_o      = usrc;     
			vsrc_o      = vsrc;     
			thv0lcl_o   = thv0lcl;
			for(k=0;k<ncnst;++k)
			{
				trsrc_o[k] = trsrc[k];
			}  
//if(i == 220) printf("rbuoy=%e,rkfre=%e,tkeavg=%e,epsvarw=%e\n",rbuoy,rkfre,tkeavg,epsvarw);
		}

		// ! Modification : If I impose w = max(0.1_r8, w) up to the top interface of
		// !                klfc, I should only use cinlfc.  That is, if I want to
		// !                use cinlcl, I should not impose w = max(0.1_r8, w).
		// !                Using cinlcl is equivalent to treating only 'saturated'
		// !                moist convection. Note that in this sense, I should keep
		// !                the functionality of both cinlfc and cinlcl.
		// !                However, the treatment of penetrative entrainment level becomes
		// !                ambiguous if I choose 'cinlcl'. Thus, the best option is to use
		// !                'cinlfc'. 

		// ! -------------------------------------------------------------------------- !
		// ! Calculate implicit 'cin' by averaging initial and final cins.    Note that !
		// ! implicit CIN is adopted only when cumulus convection stabilized the system,!
		// ! i.e., only when 'del_CIN >0'. If 'del_CIN<=0', just use explicit CIN. Note !
		// ! also that since 'cinlcl' is set to zero whenever LCL is below the PBL top, !
		// ! (see above CIN calculation part), the use of 'implicit CIN=cinlcl'  is not !
		// ! good. Thus, when using implicit CIN, always try to only use 'implicit CIN= !
		// ! cin', not 'implicit CIN=cinlcl'. However, both 'CIN=cin' and 'CIN=cinlcl'  !
		// ! are good when using explicit CIN.                                          !
		// ! -------------------------------------------------------------------------- !
//  if(i==22)
// printf("cin02=%e\n",cin);
		if(iter != 1)
		{
			cin_f = cin;
            cinlcl_f = cinlcl;
			if(use_CINcin)
			{
				del_CIN = cin_f - cin_i;
			}
			else
			{
				del_CIN = cinlcl_f - cinlcl_i;
			}
	// if(i==220)
	//  printf("del_CIN=%e\n",del_CIN);
			if(del_CIN > 0.0)
			{
				// ! -------------------------------------------------------------- ! 
				// ! Calculate implicit 'cin' and 'cinlcl'. Note that when we chose !
				// ! to use 'implicit CIN = cin', choose 'cinlcl = cinlcl_i' below: !
				// ! because iterative CIN only aims to obtain implicit CIN,  once  !
				// ! we obtained 'implicit CIN=cin', it is good to use the original !
				// ! profiles information for all the other variables after that.   !
				// ! Note 'cinlcl' will be explicitly used in calculating  'wlcl' & !
				// ! 'ufrclcl' after calculating 'winv' & 'ufrcinv'  at the PBL top !
				// ! interface later, after calculating 'cbmf'.                     !
				// ! -------------------------------------------------------------- !
				alpha = compute_alpha( del_CIN, ke );
                cin   = cin_i + alpha * del_CIN;
				if(use_CINcin)
					cinlcl = cinlcl_i;
				else
					cinlcl = cinlcl_i + alpha * del_cinlcl;

			//    ! ----------------------------------------------------------------- !
            //    ! Restore the original values from the previous 'iter_cin' step (1) !
            //    ! to compute correct tendencies for (n+1) time step by implicit CIN !
            //    ! ----------------------------------------------------------------- !

				kinv      = kinv_o;    
				klcl      = klcl_o;     
				klfc      = klfc_o;    
				plcl      = plcl_o;    
				plfc      = plfc_o;     
				tkeavg    = tkeavg_o;   
				thvlmin   = thvlmin_o;
				qtsrc     = qtsrc_o;    
				thvlsrc   = thvlsrc_o;  
				thlsrc    = thlsrc_o;
				usrc      = usrc_o;     
				vsrc      = vsrc_o;     
				thv0lcl   = thv0lcl_o;

				for(k=0;k<ncnst;++k)
				{
					trsrc[k] = trsrc_o[k];
				}

				for(j=0;j<mkx;++j)
				{
					qv0[j]            = qv0_o[j];
					ql0[j]            = ql0_o[j];
					qi0[j]            = qi0_o[j];
					t0[j]             = t0_o[j];
					s0[j]             = s0_o[j];
					u0[j]             = u0_o[j];
					v0[j]             = v0_o[j];
					qt0[j]            = qt0_o[j];
					thl0[j]           = thl0_o[j];
					thvl0[j]          = thvl0_o[j];
					ssthl0[j]         = ssthl0_o[j];
					ssqt0[j]          = ssqt0_o[j];
					thv0bot[j]        = thv0bot_o[j];
					thv0top[j]        = thv0top_o[j];
					thvl0bot[j]       = thvl0bot_o[j];
					thvl0top[j]       = thvl0top_o[j];
					ssu0[j]           = ssu0_o[j]; 
					ssv0[j]           = ssv0_o[j]; 
				}
				for(k=0;k<ncnst;++k)
					for(j=0;j<mkx;++j)
					{
						tr0[k*mkx+j]   = tr0_o[k*mkx+j];
						sstr0[k*mkx+j] = sstr0_o[k*mkx+j];
					}

			//    ! ------------------------------------------------------ !
            //    ! Initialize all fluxes, tendencies, and other variables ! 
            //    ! in association with cumulus convection.                !
            //    ! ------------------------------------------------------ !
			for(j=0;j<mkx+1;++j)
			{
				umf[j]          = 0.0;
				emf[j]          = 0.0;
				slflx[j]        = 0.0;
				qtflx[j]        = 0.0;
				uflx[j]         = 0.0;
				vflx[j]         = 0.0;

				ufrc[j]         = 0.0;

				thlu[j]         = 0.0;
				qtu[j]          = 0.0;
				uu[j]           = 0.0;
				vu[j]           = 0.0;
				wu[j]           = 0.0;
				thvu[j]         = 0.0;
				thlu_emf[j]     = 0.0;
				qtu_emf[j]      = 0.0;
				uu_emf[j]       = 0.0;
				vu_emf[j]       = 0.0;
			}
			for(j=0;j<mkx;++j)
			{
				qvten[j]         = 0.0;
				qlten[j]         = 0.0;
				qiten[j]         = 0.0;
				sten[j]          = 0.0;
				uten[j]          = 0.0;
				vten[j]          = 0.0;
				qrten[j]         = 0.0;
				qsten[j]         = 0.0;
				dwten[j]         = 0.0;
				diten[j]         = 0.0;

				evapc[j]         = 0.0;
				cufrc[j]         = 0.0;
				qcu[j]           = 0.0;
				qlu[j]           = 0.0;
				qiu[j]           = 0.0;
				fer[j]           = 0.0;
				fdr[j]           = 0.0;
				qc[j]            = 0.0;
				qc_l[j]          = 0.0;
				qc_i[j]          = 0.0;
				qtten[j]         = 0.0;
				slten[j]         = 0.0;
				
///				excessu_arr[j]   = 0.0;
///				excess0_arr[j]   = 0.0;
///				xc_arr[j]        = 0.0;
///				aquad_arr[j]     = 0.0;
///				bquad_arr[j]     = 0.0;
///				cquad_arr[j]     = 0.0;
///				bogbot_arr[j]    = 0.0;
///				bogtop_arr[j]    = 0.0;
			}

			for(k=0;k<ncnst;++k)
			{
				for(j=0;j<mkx+1;++j)
				{
					trflx[k*(mkx+1)+j]   = 0.0;
					tru[k*(mkx+1)+j]     = 0.0;
			   		tru_emf[k*(mkx+1)+j] = 0.0;
				}
			}
			for(k=0;k<ncnst;++k)
			{
				for(j=0;j<mkx;++j)
				{
					trten[k*mkx+j]    = 0.0;
				}
			}
			
			precip              = 0.0;
			snow                = 0.0;
			
			rliq                = 0.0;
			cbmf                = 0.0;
			cnt                 = (double)mkx;	//real(mkx, r8)
			cnb                 = 0.0;

			// ! -------------------------------------------------- !
			// ! Below are diagnostic output variables for detailed !
			// ! analysis of cumulus scheme.                        !
			// ! -------------------------------------------------- ! 

			ufrcinvbase         = 0.0;
			ufrclcl             = 0.0;
			winvbase            = 0.0;
			wlcl                = 0.0;
			emfkbup             = 0.0; 
			cbmflimit           = 0.0;
			}
			else	//! When 'del_CIN < 0', use explicit CIN instead of implicit CIN.
			{
				// ! ----------------------------------------------------------- ! 
				// ! Identifier showing whether explicit or implicit CIN is used !
				// ! ----------------------------------------------------------- !

///				ind_delcin[i] = 1.0;

				// ! --------------------------------------------------------- !
				// ! Restore original output values of "iter_cin = 1" and exit !
				// ! --------------------------------------------------------- !

				for(j=0;j<mkx+1;++j)
				{
					umf_out[j*iend+i]         = umf_s[j];
//!wxc zmh
					wu_out[j*iend+i]          = wu[j];

					slflx_out[j*iend+i]       = slflx_s[j];  
					qtflx_out[j*iend+i]       = qtflx_s[j];
				}
				for(j=0;j<mkx;++j)
				{
					qvten_out[j*iend+i]        = qvten_s[j];
					qlten_out[j*iend+i]        = qlten_s[j];  
					qiten_out[j*iend+i]        = qiten_s[j];
					sten_out[j*iend+i]         = sten_s[j];
					uten_out[j*iend+i]         = uten_s[j];  
					vten_out[j*iend+i]         = vten_s[j];
					qrten_out[j*iend+i]        = qrten_s[j];
					qsten_out[j*iend+i]        = qsten_s[j];

					evapc_out[j*iend+i]        = evapc_s[j];
					cufrc_out[j*iend+i]        = cufrc_s[j]; 

					qcu_out[j*iend+i]          = qcu_s[j];    
					qlu_out[j*iend+i]          = qlu_s[j];  
					qiu_out[j*iend+i]          = qiu_s[j]; 

					qc_out[j*iend+i]           = qc_s[j]; 
				}	

				precip_out[i]            = precip_s;
				snow_out[i]              = snow_s;

				cush_inout[i]            = cush_s;
 
				cbmf_out[i]              = cbmf_s;
 
				rliq_out[i]              = rliq_s;
				cnt_out[i]               = cnt_s;
				cnb_out[i]               = cnb_s;

				for(k=0;k<ncnst;++k)
				{
					for(j=0;j<mkx;++j)
					{
						trten_out[k*mkx*iend+j*iend+i]   = trten_s[k*mkx+j];
					}
				}
// if(i == 7)
// {
// 	printf("-------------------- At sub. compute_uwshcu -----------------------\n");
// 	printf("1st: trten_out(8,1,17) =%15.13f\n",trten_out[16*mkx*iend+0*iend+7]);
// 	printf("dpdry0_in(8,1) =%15.13f\n",dpdry0_in[0*iend+7]);
// 	printf("-------------------------------------------------------------------\n");
// }
// !======================== zhh debug 2012-02-09 =======================     
// !              if (i==8) then
// !                 print*, '2nd: trten_out(8,1,17) =', trten_out(8,1,17)
// !                 print*, '-------------------------------------------------------------------'
// !              end if
// !======================== zhh debug 2012-02-09 =======================  

			//    ! ------------------------------------------------------------------------------ ! 
            //    ! Below are diagnostic output variables for detailed analysis of cumulus scheme. !
            //    ! The order of vertical index is reversed for this internal diagnostic output.   !
            //    ! ------------------------------------------------------------------------------ !

			int temp_j;
			for(j=mkx-1;j>=0;j--)
			{
				temp_j = mkx-j-1;

//				fer_out[j*iend+i]      = fer_s[temp_j*iend+i];  
//				fdr_out[j*iend+i]      = fdr_s[temp_j*iend+i]; 

//				qtten_out[j*iend+i]    = qtten_s[temp_j*iend+i];
//				slten_out[j*iend+i]    = slten_s[temp_j*iend+i];

///				dwten_out[j*iend+i]    = dwten_s[temp_j*iend+i];
///				diten_out[j*iend+i]    = diten_s[temp_j*iend+i];

///				ntraprd_out[j*iend+i]  = ntraprd_s[temp_j*iend+i];
///				ntsnprd_out[j*iend+i]  = ntsnprd_s[temp_j*iend+i];

///				excessu_arr_out[j*iend+i]  = excessu_arr_s[temp_j*iend+i];
///				excess0_arr_out[j*iend+i]  = excess0_arr_s[temp_j*iend+i];
///				xc_arr_out[j*iend+i]       = xc_arr_s[temp_j*iend+i];
///				aquad_arr_out[j*iend+i]    = aquad_arr_s[temp_j*iend+i];
///				bquad_arr_out[j*iend+i]    = bquad_arr_s[temp_j*iend+i];
///				cquad_arr_out[j*iend+i]    = cquad_arr_s[temp_j*iend+i];
///				bogbot_arr_out[j*iend+i]   = bogbot_arr_s[temp_j*iend+i];
///				bogtop_arr_out[j*iend+i]   = bogtop_arr_s[temp_j*iend+i];
			}
			for(j=mkx;j>=0;j--)
			{
				temp_j = mkx-j;

//				ufrc_out[j*iend+i]     = ufrc_s[temp_j];
//				uflx_out[j*iend+i]     = uflx_s[temp_j];  
//				vflx_out[j*iend+i]     = vflx_s[temp_j];  

///				wu_out[j*iend+i]       = wu_s[temp_j*iend+i];
///				qtu_out[j*iend+i]      = qtu_s[temp_j*iend+i];
///				thlu_out[j*iend+i]     = thlu_s[temp_j*iend+i];
///				thvu_out[j*iend+i]     = thvu_s[temp_j*iend+i];
///				uu_out[j*iend+i]       = uu_s[temp_j*iend+i];
///				vu_out[j*iend+i]       = vu_s[temp_j*iend+i];
///				qtu_emf_out[j*iend+i]  = qtu_emf_s[temp_j*iend+i];
///				thlu_emf_out[j*iend+i] = thlu_emf_s[temp_j*iend+i];
///				uu_emf_out[j*iend+i]   = uu_emf_s[temp_j*iend+i];
///				vu_emf_out[j*iend+i]   = vu_emf_s[temp_j*iend+i];
///				uemf_out[j*iend+i]     = uemf_s[temp_j*iend+i];
	
///				flxrain_out[j*iend+i]  = flxrain_s[temp_j*iend+i];
///				flxsnow_out[j*iend+i]  = flxsnow_s[temp_j*iend+i];
			}
			 
//			cinh_out[i]              = cin_s;
///			cinlclh_out[i]           = cinlcl_s;

///			ufrcinvbase_out[i]       = ufrcinvbase_s;
///			ufrclcl_out[i]           = ufrclcl_s; 
///			winvbase_out[i]          = winvbase_s;
///			wlcl_out[i]              = wlcl_s;
///			plcl_out[i]              = plcl_s;
///			pinv_out[i]              = pinv_s;    
			// plfc_out[i]              = plfc_s;    
			// pbup_out[i]              = pbup_s;
			// ppen_out[i]              = ppen_s;    
			// qtsrc_out[i]             = qtsrc_s;
			// thlsrc_out[i]            = thlsrc_s;
			// thvlsrc_out[i]           = thvlsrc_s;
			// emfkbup_out[i]           = emfkbup_s;
			// cbmflimit_out[i]         = cbmflimit_s;
			// tkeavg_out[i]            = tkeavg_s;
			// zinv_out[i]              = zinv_s;
			// rcwp_out[i]              = rcwp_s;
			// rlwp_out[i]              = rlwp_s;
			// riwp_out[i]              = riwp_s;


			for(k=0;k<ncnst;k++)
				for(j=mkx;j>=0;j--)
				{
					temp_j = mkx-j;

//					trflx_out[k*(mkx+1)*iend+j*iend+i]   = trflx_s[k*(mkx+1)*iend+temp_j*iend+i];  
//					tru_out[k*(mkx+1)*iend+j*iend+i]     = tru_s[k*(mkx+1)*iend+temp_j*iend+i];
//					tru_emf_out[k*(mkx+1)*iend+j*iend+i] = tru_emf_s[k*(mkx+1)*iend+temp_j*iend+i];
				}

			id_exit = false;
		//	printf("3\n");
			goto lable333;  

			}
		}

		// ! ------------------------------------------------------------------ !
		// ! Define a release level, 'prel' and release layer, 'krel'.          !
		// ! 'prel' is the lowest level from which buoyancy sorting occurs, and !
		// ! 'krel' is the layer index containing 'prel' in it, similar to  the !
		// ! previous definitions of 'kinv', 'klcl', and 'klfc'.    In order to !
		// ! ensure that only PBL scheme works within the PBL,  if LCL is below !
		// ! PBL top height, then 'krel = kinv', while if LCL is above  PBL top !
		// ! height, then 'krel = klcl'.   Note however that regardless of  the !
		// ! definition of 'krel', cumulus convection induces fluxes within PBL !
		// ! through 'fluxbelowinv'.  We can make cumulus convection start from !
		// ! any level, even within the PBL by appropriately defining 'krel'  & !
		// ! 'prel' here. Then it must be accompanied by appropriate definition !
		// ! of source air properties, CIN, and re-setting of 'fluxbelowinv', & !
		// ! many other stuffs.                                                 !
		// ! Note that even when 'prel' is located above the PBL top height, we !
		// ! still have cumulus convection between PBL top height and 'prel':   !
		// ! we simply assume that no lateral mixing occurs in this range.      !
		// ! ------------------------------------------------------------------ !
	
			if(klcl < kinv)
			{
				krel    = kinv;
				prel    = ps0[krel-1];
				thv0rel = thv0bot[krel-1]; 
			}
			else
			{
				krel    = klcl;
          	    prel    = plcl; 
           		thv0rel = thv0lcl;
			}
			
			// ! --------------------------------------------------------------------------- !
			// ! Calculate cumulus base mass flux ('cbmf'), fractional area ('ufrcinv'), and !
			// ! and mean vertical velocity (winv) of cumulus updraft at PBL top interface.  !
			// ! Also, calculate updraft fractional area (ufrclcl) and vertical velocity  at !
			// ! the LCL (wlcl). When LCL is below PBLH, cinlcl = 0 and 'ufrclcl = ufrcinv', !
			// ! and 'wlcl = winv.                                                           !
			// ! Only updrafts strong enough to overcome CIN can rise over PBL top interface.! 
			// ! Thus,  in order to calculate cumulus mass flux at PBL top interface, 'cbmf',!
			// ! we need to know 'CIN' ( the strength of potential energy barrier ) and      !
			// ! 'sigmaw' ( a standard deviation of updraft vertical velocity at the PBL top !
			// ! interface, a measure of turbulentce strength in the PBL ).   Naturally, the !
			// ! ratio of these two variables, 'mu' - normalized CIN by TKE- is key variable !
			// ! controlling 'cbmf'.  If 'mu' becomes large, only small fraction of updrafts !
			// ! with very strong TKE can rise over the PBL - both 'cbmf' and 'ufrc' becomes !
			// ! small, but 'winv' becomes large ( this can be easily understood by PDF of w !
			// ! at PBL top ).  If 'mu' becomes small, lots of updraft can rise over the PBL !
			// ! top - both 'cbmf' and 'ufrc' becomes large, but 'winv' becomes small. Thus, !
			// ! all of the key variables associated with cumulus convection  at the PBL top !
			// ! - 'cbmf', 'ufrc', 'winv' where 'cbmf = rho*ufrc*winv' - are a unique functi !
			// ! ons of 'mu', normalized CIN. Although these are uniquely determined by 'mu',! 
			// ! we usually impose two comstraints on 'cbmf' and 'ufrc': (1) because we will !
			// ! simply assume that subsidence warming and drying of 'kinv-1' layer in assoc !
			// ! iation with 'cbmf' at PBL top interface is confined only in 'kinv-1' layer, !
			// ! cbmf must not be larger than the mass within the 'kinv-1' layer. Otherwise, !
			// ! instability will occur due to the breaking of stability con. If we consider !
			// ! semi-Lagrangian vertical advection scheme and explicitly consider the exten !
			// ! t of vertical movement of each layer in association with cumulus mass flux, !
			// ! we don't need to impose this constraint. However,  using a  semi-Lagrangian !
			// ! scheme is a future research subject. Note that this constraint should be ap !
			// ! plied for all interfaces above PBL top as well as PBL top interface.   As a !
			// ! result, this 'cbmf' constraint impose a 'lower' limit on mu - 'mumin0'. (2) !
			// ! in order for mass flux parameterization - rho*(w'a')= M*(a_c-a_e) - to   be !
			// ! valid, cumulus updraft fractional area should be much smaller than 1.    In !
			// ! current code, we impose 'rmaxfrac = 0.1 ~ 0.2'   through the whole vertical !
			// ! layers where cumulus convection occurs. At the PBL top interface,  the same !
			// ! constraint is made by imposing another lower 'lower' limit on mu, 'mumin1'. !
			// ! After that, also limit 'ufrclcl' to be smaller than 'rmaxfrac' by 'mumin2'. !
			// ! --------------------------------------------------------------------------- !
			
			// ! --------------------------------------------------------------------------- !
			// ! Calculate normalized CIN, 'mu' satisfying all the three constraints imposed !
			// ! on 'cbmf'('mumin0'), 'ufrc' at the PBL top - 'ufrcinv' - ( by 'mumin1' from !
			// ! a parameter sentence), and 'ufrc' at the LCL - 'ufrclcl' ( by 'mumin2').    !
			// ! Note that 'cbmf' does not change between PBL top and LCL  because we assume !
			// ! that buoyancy sorting does not occur when cumulus updraft is unsaturated.   !
			// ! --------------------------------------------------------------------------- !
			
			if( use_CINcin )      
          		wcrit = sqrt( 2.0 * cin * rbuoy );   
      		else
           		wcrit = sqrt( 2.0 * cinlcl * rbuoy );
			sigmaw = sqrt( rkfre * tkeavg + epsvarw );
		
       		mu = wcrit/sigmaw/1.4142;
			  
			if(mu >= 3.0)
			{
				//!!! write(iulog,*) 'mu >= 3'
				id_exit = true;
			//	printf("4\n");
				goto lable333;
			}
			
			rho0inv = ps0[kinv-1]/(r*thv0top[kinv-1-1]*exns0[kinv-1]);
			// if(i==220) printf("ps0[kinv-1]=%f  r=%f  thv0top[kinv-1-1]=%35.33f  exns0[kinv-1]=%f\n",ps0[kinv-1],
			//  		r,thv0top[kinv-1-1],exns0[kinv-1]);

		//	if(i==220) printf("rho0inv=%f,sigmaw=%f,mu=%f\n",rho0inv,sigmaw,mu);
		
			cbmf = (rho0inv*sigmaw/2.5066)*exp((-1)*pow(mu,2));//exp(-mu**2)
			//! 1. 'cbmf' constraint
			cbmflimit = 0.9*dp0[kinv-1-1]/g/dt;
			mumin0 = 0.0;
			if( cbmf > cbmflimit ) mumin0 = sqrt(-log(2.5066*cbmflimit/rho0inv/sigmaw));
			//! 2. 'ufrcinv' constraint
			mu = max(max(mu,mumin0),mumin1);
			//! 3. 'ufrclcl' constraint
			mulcl = sqrt(2.0*cinlcl*rbuoy)/1.4142/sigmaw;
			mulclstar = sqrt(max(0.0,2.0*pow((exp((-1)*pow(mu,2))/2.5066),2)*(1.0/pow(erfc(mu),2)-0.25/pow(rmaxfrac,2))));//!!!erfc(mu)__error_function_MOD_erfc(mu)
			//mulclstar = sqrt(max(0.0,2.0*(exp((-1)*pow(mu,2))/2.5066)**2*(1.0/erfc**2-0.25/rmaxfrac**2))) //!!!erfc(mu)
			if((mulcl > 1.0e-8) && (mulcl > mulclstar))
			{
				mumin2 = compute_mumin2(mulcl,rmaxfrac,mu);
				if(mu > mumin2)
				{
					//!!!    write(iulog,*) 'Critical error in mu calculation in UW_ShCu'
				    //__abortutils_MOD_endrun("");
					printf("error------------------------------\n");
					printf("error------------------------------\n");
					printf("error------------------------------\n");
				//	return;
				}
				mu = max(mu,mumin2);
///				if( mu == mumin2 ) limit_ufrc[i] = 1.0;
			}
///			if( mu == mumin0 ) limit_cbmf[i] = 1.0;
///			if( mu == mumin1 ) limit_ufrc[i] = 1.0;

			// ! ------------------------------------------------------------------- !    
			// ! Calculate final ['cbmf','ufrcinv','winv'] at the PBL top interface. !
			// ! Note that final 'cbmf' here is obtained in such that 'ufrcinv' and  !
			// ! 'ufrclcl' are smaller than ufrcmax with no instability.             !
			// ! ------------------------------------------------------------------- !

			cbmf = (rho0inv*sigmaw/2.5066)*exp((-1)*pow(mu,2));   
			winv = sigmaw*(2.0/2.5066)*exp((-1)*pow(mu,2))/erfc(mu);//__error_function_MOD_erfc(mu); //!!!erfc(mu)
			ufrcinv = cbmf/winv/rho0inv;
//if(i==22) printf("cbmf=%e\n",cbmf);	
			// ! ------------------------------------------------------------------- !
			// ! Calculate ['ufrclcl','wlcl'] at the LCL. When LCL is below PBL top, !
			// ! it automatically becomes 'ufrclcl = ufrcinv' & 'wlcl = winv', since !
			// ! it was already set to 'cinlcl=0' if LCL is below PBL top interface. !
			// ! Note 'cbmf' at the PBL top is the same as 'cbmf' at the LCL.  Note  !
			// ! also that final 'cbmf' here is obtained in such that 'ufrcinv' and  !
			// ! 'ufrclcl' are smaller than ufrcmax and there is no instability.     !
			// ! By construction, it must be 'wlcl > 0' but for assurance, I checked !
			// ! this again in the below block. If 'ufrclcl < 0.1%', just exit.      !
			// ! ------------------------------------------------------------------- !

			wtw = winv * winv - 2.0 * cinlcl * rbuoy;
			if(wtw <= 0.0)
			{
				//!!! write(iulog,*) 'wlcl < 0 at the LCL'
				exit_wtw[i] = 1.0;
				id_exit = true;
				//printf("5\n");
				goto lable333;
			}
			wlcl = sqrt(wtw);
			ufrclcl = cbmf/wlcl/rho0inv;
       		wrel = wlcl;
			if(ufrclcl <= 0.0001)
			{
				//!!! write(iulog,*) 'ufrclcl <= 0.0001' 
				exit_ufrc[i] = 1.0;
				id_exit = true;
			//	printf("6\n");
				goto lable333;
			}
			ufrc[krel-1] = ufrclcl;

			// ! ----------------------------------------------------------------------- !
			// ! Below is just diagnostic output for detailed analysis of cumulus scheme !
			// ! ----------------------------------------------------------------------- !
			
			ufrcinvbase        = ufrcinv;
			winvbase           = winv;
			for(m=kinv-1;m<=krel-1;++m)
			{
				umf[m] = cbmf;
				wu[m]  = winv;
			}
			//if(i==220) printf("cbmf=%f\n",cbmf);
			// ! -------------------------------------------------------------------------- ! 
			// ! Define updraft properties at the level where buoyancy sorting starts to be !
			// ! happening, i.e., by definition, at 'prel' level within the release layer.  !
			// ! Because no lateral entrainment occurs upto 'prel', conservative scalars of ! 
			// ! cumulus updraft at release level is same as those of source air.  However, ! 
			// ! horizontal momentums of source air are modified by horizontal PGF forcings ! 
			// ! from PBL top interface to 'prel'.  For this case, we should add additional !
			// ! horizontal momentum from PBL top interface to 'prel' as will be done below !
			// ! to 'usrc' and 'vsrc'. Note that below cumulus updraft properties - umf, wu,!
			// ! thlu, qtu, thvu, uu, vu - are defined all interfaces not at the layer mid- !
			// ! point. From the index notation of cumulus scheme, wu(k) is the cumulus up- !
			// ! draft vertical velocity at the top interface of k layer.                   !
			// ! Diabatic horizontal momentum forcing should be treated as a kind of 'body' !
			// ! forcing without actual mass exchange between convective updraft and        !
			// ! environment, but still taking horizontal momentum from the environment to  !
			// ! the convective updrafts. Thus, diabatic convective momentum transport      !
			// ! vertically redistributes environmental horizontal momentum.                !
			// ! -------------------------------------------------------------------------- !

			emf[krel-1]  = 0.0;
			umf[krel-1]  = cbmf;
			wu[krel-1]   = wrel;
			thlu[krel-1] = thlsrc;
			qtu[krel-1]  = qtsrc;
			conden(prel,thlsrc,qtsrc,&thj,&qvj,&qlj,&qij,&qse,&id_check);
			if(id_check == 1)
			{
				exit_conden[i] = 1.0;
				id_exit = true;
				goto lable333;
			}
			thvu[krel-1] = thj * ( 1.0 + zvir*qvj - qlj - qij );

			uplus = 0.0;
			vplus = 0.0;
			if(krel == kinv)
			{
				uplus = PGFc * ssu0[kinv-1] * ( prel - ps0[kinv-1] );
				vplus = PGFc * ssv0[kinv-1] * ( prel - ps0[kinv-1] );
			}
			else
			{
				for(k=kinv;k<=max(krel-1,kinv);++k)
				{
					uplus = uplus + PGFc * ssu0[k-1] * ( ps0[k] - ps0[k-1] );
					vplus = vplus + PGFc * ssv0[k-1] * ( ps0[k] - ps0[k-1] );
				}
			 	uplus = uplus + PGFc * ssu0[krel-1] * ( prel - ps0[krel-1] );
			 	vplus = vplus + PGFc * ssv0[krel-1] * ( prel - ps0[krel-1] );
			}
			uu[krel-1] = usrc + uplus;
			vu[krel-1] = vsrc + vplus; 
			for(k=0;k<ncnst;++k)
			{
				tru[k*(mkx+1)+(krel-1)] = trsrc[k];
			}
		
			// ! -------------------------------------------------------------------------- !
			// ! Define environmental properties at the level where buoyancy sorting occurs !
			// ! ('pe', normally, layer midpoint except in the 'krel' layer). In the 'krel' !
			// ! layer where buoyancy sorting starts to occur, however, 'pe' is defined     !
			// ! differently because LCL is regarded as lower interface for mixing purpose. !
			// ! -------------------------------------------------------------------------- !
			pe      = 0.5 * ( prel + ps0[krel] );
			dpe     = prel - ps0[krel];
			exne    = exnf(pe);
			thvebot = thv0rel;
			thle    = thl0[krel-1] + ssthl0[krel-1] * ( pe - p0[krel-1] );
       		qte     = qt0[krel-1]  + ssqt0[krel-1]  * ( pe - p0[krel-1] );
       		ue      = u0[krel-1]   + ssu0[krel-1]   * ( pe - p0[krel-1] );
       		ve      = v0[krel-1]   + ssv0[krel-1]   * ( pe - p0[krel-1] );
			for(k=0;k<ncnst;++k)
			{
				tre[k] = tr0[k*mkx+(krel-1)] + sstr0[k*mkx+(krel-1)] * ( pe - p0[krel-1] );
			}
		
			// !-------------------------! 
			// ! Buoyancy-Sorting Mixing !
			// !-------------------------!------------------------------------------------ !
			// !                                                                           !
			// !  In order to complete buoyancy-sorting mixing at layer mid-point, and so  ! 
			// !  calculate 'updraft mass flux, updraft w velocity, conservative scalars'  !
			// !  at the upper interface of each layer, we need following 3 information.   ! 
			// !                                                                           !
			// !  1. Pressure where mixing occurs ('pe'), and temperature at 'pe' which is !
			// !     necessary to calculate various thermodynamic coefficients at pe. This !
			// !     temperature is obtained by undiluted cumulus properties lifted to pe. ! 
			// !  2. Undiluted updraft properties at pe - conservative scalar and vertical !
			// !     velocity -which are assumed to be the same as the properties at lower !
			// !     interface only for calculation of fractional lateral entrainment  and !
			// !     detrainment rate ( fer(k) and fdr(k) [Pa-1] ), respectively.    Final !
			// !     values of cumulus conservative scalars and w at the top interface are !
			// !     calculated afterward after obtaining fer(k) & fdr(k).                 !
			// !  3. Environmental properties at pe.                                       !
			// ! ------------------------------------------------------------------------- !
			
			// ! ------------------------------------------------------------------------ ! 
			// ! Define cumulus scale height.                                             !
			// ! Cumulus scale height is defined as the maximum height cumulus can reach. !
			// ! In case of premitive code, cumulus scale height ('cush')  at the current !
			// ! time step was assumed to be the same as 'cush' of previous time step.    !
			// ! However, I directly calculated cush at each time step using an iterative !
			// ! method. Note that within the cumulus scheme, 'cush' information is  used !
			// ! only at two places during buoyancy-sorting process:                      !
			// ! (1) Even negatively buoyancy mixtures with strong vertical velocity      !
			// !     enough to rise up to 'rle*scaleh' (rle = 0.1) from pe are entrained  !
			// !     into cumulus updraft,                                                !  
			// ! (2) The amount of mass that is involved in buoyancy-sorting mixing       !
			// !      process at pe is rei(k) = rkm/scaleh/rho*g [Pa-1]                   !
			// ! In terms of (1), I think critical stopping distance might be replaced by !
			// ! layer thickness. In future, we will use rei(k) = (0.5*rkm/z0(k)/rho/g).  !
			// ! In the premitive code,  'scaleh' was largely responsible for the jumping !
			// ! variation of precipitation amount.                                       !
			// ! ------------------------------------------------------------------------ !
			
			// for(m=0;m<=mkx;++m)
			// {
			// 	if(i==220)
			// 		printf("m=%d,umf=%33.32f\n",m,umf[m]);
			// }
			scaleh = tscaleh;
			if(tscaleh < 0.0) scaleh = 1000.0;

			// ! Save time : Set iter_scaleh = 1. This will automatically use 'cush' from the previous time step
			// !             at the first implicit iteration. At the second implicit iteration, it will use
			// !             the updated 'cush' by the first implicit cin. So, this updating has an effect of
			// !             doing one iteration for cush calculation, which is good. 
			// !             So, only this setting of 'iter_scaleh = 1' is sufficient-enough to save computation time.
			// ! OK

			for(iter_scaleh=1;iter_scaleh<=3;++iter_scaleh)
			{
				
				// ! ---------------------------------------------------------------- !
				// ! Initialization of 'kbup' and 'kpen'                              !
				// ! ---------------------------------------------------------------- !
				// ! 'kbup' is the top-most layer in which cloud buoyancy is positive !
				// ! both at the top and bottom interface of the layer. 'kpen' is the !
				// ! layer upto which cumulus panetrates ,i.e., cumulus w at the base !
				// ! interface is positive, but becomes negative at the top interface.!
				// ! Here, we initialize 'kbup' and 'kpen'. These initializations are !  
				// ! not trivial but important, expecially   in calculating turbulent !
				// ! fluxes without confliction among several physics as explained in !
				// ! detail in the part of turbulent fluxes calculation later.   Note !
				// ! that regardless of whether 'kbup' and 'kpen' are updated or  not !
				// ! during updraft motion,  penetrative entrainments are dumped down !
				// ! across the top interface of 'kbup' later.      More specifically,!
				// ! penetrative entrainment heat and moisture fluxes are  calculated !
				// ! from the top interface of 'kbup' layer  to the base interface of !
				// ! 'kpen' layer. Because of this, initialization of 'kbup' & 'kpen' !
				// ! influence the convection system when there are not updated.  The !  
				// ! below initialization of 'kbup = krel' assures  that  penetrative !
				// ! entrainment fluxes always occur at interfaces above the PBL  top !
				// ! interfaces (i.e., only at interfaces k >=kinv ), which seems  to !
				// ! be attractable considering that the most correct fluxes  at  the !
				// ! PBL top interface can be ontained from the 'fluxbelowinv'  using !
				// ! reconstructed PBL height.                                        ! 
				// ! The 'kbup = krel'(after going through the whole buoyancy sorting !
				// ! proces during updraft motion) implies that cumulus updraft  from !
				// ! the PBL top interface can not reach to the LFC,so that 'kbup' is !
				// ! not updated during upward. This means that cumulus updraft   did !
				// ! not fully overcome the buoyancy barrier above just the PBL top.  !
				// ! If 'kpen' is not updated either ( i.e., cumulus cannot rise over !
				// ! the top interface of release layer),penetrative entrainment will !
				// ! not happen at any interfaces.  If cumulus updraft can rise above !
				// ! the release layer but cannot fully overcome the buoyancy barrier !
				// ! just above PBL top interface, penetratve entrainment   occurs at !
				// ! several above interfaces, including the top interface of release ! 
				// ! layer. In the latter case, warming and drying tendencies will be !
				// ! be initiated in 'krel' layer. Note current choice of 'kbup=krel' !
				// ! is completely compatible with other flux physics without  double !
				// ! or miss counting turbulent fluxes at any interface. However, the !
				// ! alternative choice of 'kbup=krel-1' also has itw own advantage - !
				// ! when cumulus updraft cannot overcome buoyancy barrier just above !
				// ! PBL top, entrainment warming and drying are concentrated in  the !
				// ! 'kinv-1' layer instead of 'kinv' layer for this case. This might !
				// ! seems to be more dynamically reasonable, but I will choose the   !
				// ! 'kbup = krel' choice since it is more compatible  with the other !
				// ! parts of the code, expecially, when we chose ' use_emf=.false. ' !
				// ! as explained in detail in turbulent flux calculation part.       !
				// ! ---------------------------------------------------------------- ! 

				kbup    = krel;
				kpen    = krel;

				// ! ------------------------------------------------------------ !
				// ! Since 'wtw' is continuously updated during vertical motion,  !
				// ! I need below initialization command within this 'iter_scaleh'!
				// ! do loop. Similarily, I need initializations of environmental !
				// ! properties at 'krel' layer as below.                         !
				// ! ------------------------------------------------------------ !

				wtw     = wlcl * wlcl;
				pe      = 0.5 * ( prel + ps0[krel] );
     			dpe     = prel - ps0[krel];
				exne    = exnf(pe);
				thvebot = thv0rel;
				thle    = thl0[krel-1] + ssthl0[krel-1] * ( pe - p0[krel-1] );
				qte     = qt0[krel-1]  + ssqt0[krel-1]  * ( pe - p0[krel-1] );
				ue      = u0[krel-1]   + ssu0[krel-1]   * ( pe - p0[krel-1] );
				ve      = v0[krel-1]   + ssv0[krel-1]   * ( pe - p0[krel-1] );
				for(k=0;k<ncnst;++k)
				{
					tre[k] = tr0[k*mkx+(krel-1)]  + sstr0[k*mkx+(krel-1)] * ( pe - p0[krel-1] );
				}
				// ! ----------------------------------------------------------------------- !
				// ! Cumulus rises upward from 'prel' ( or base interface of  'krel' layer ) !
				// ! until updraft vertical velocity becomes zero.                           !
				// ! Buoyancy sorting is performed via two stages. (1) Using cumulus updraft !
				// ! properties at the base interface of each layer,perform buoyancy sorting !
				// ! at the layer mid-point, 'pe',  and update cumulus properties at the top !
				// ! interface, and then  (2) by averaging updated cumulus properties at the !
				// ! top interface and cumulus properties at the base interface,   calculate !
				// ! cumulus updraft properties at pe that will be used  in buoyancy sorting !
				// ! mixing - thlue, qtue and, wue.  Using this averaged properties, perform !
				// ! buoyancy sorting again at pe, and re-calculate fer(k) and fdr(k). Using !
				// ! this recalculated fer(k) and fdr(k),  finally calculate cumulus updraft !
				// ! properties at the top interface - thlu, qtu, thvu, uu, vu. In the below,!
				// ! 'iter_xc = 1' performs the first stage, while 'iter_xc= 2' performs the !
				// ! second stage. We can increase the number of iterations, 'nter_xc'.as we !
				// ! want, but a sample test indicated that about 3 - 5 iterations  produced !
				// ! satisfactory converent solution. Finally, identify 'kbup' and 'kpen'.   !
				// ! ----------------------------------------------------------------------- !
				
				for(k=krel;k<=mkx-1;++k) //! Here, 'k' is a layer index.
				{
					km1 = k-1;

					thlue = thlu[km1];
					qtue  = qtu[km1];    
					wue   = wu[km1];
					wtwb  = wtw; 

					for(iter_xc=1;iter_xc<=niter_xc;++iter_xc)
					{
						wtw = wu[km1] * wu[km1];

						// ! ---------------------------------------------------------------- !
						// ! Calculate environmental and cumulus saturation 'excess' at 'pe'. !
						// ! Note that in order to calculate saturation excess, we should use ! 
						// ! liquid water temperature instead of temperature  as the argument !
						// ! of "qsat". But note normal argument of "qsat" is temperature.    !
						// ! ---------------------------------------------------------------- !

						conden(pe,thle,qte,&thj,&qvj,&qlj,&qij,&qse,&id_check);
						if(id_check == 1)
						{
							exit_conden[i] = 1.0;
							id_exit = true;
							goto lable333;
						}
						thv0j    = thj * ( 1.0 + zvir*qvj - qlj - qij );
						rho0j    = pe / ( r * thv0j * exne );
						qsat_arg = thle*exne;

						//!------------------------!
						qsat_arglf[0] = qsat_arg;
						pelf[0]       = pe;
						status   =  qsat(qsat_arglf,pelf,es,qs,gam,1);
						excess0  = qte - qs[0];

						conden(pe,thlue,qtue,&thj,&qvj,&qlj,&qij,&qse,&id_check);
						if(id_check == 1)
						{
							exit_conden[i] = 1.0;
              				id_exit = true;
              				goto lable333;
						}
						// ! ----------------------------------------------------------------- !
						// ! Detrain excessive condensate larger than 'criqc' from the cumulus ! 
						// ! updraft before performing buoyancy sorting. All I should to do is !
						// ! to update 'thlue' &  'que' here. Below modification is completely !
						// ! compatible with the other part of the code since 'thule' & 'qtue' !
						// ! are used only for buoyancy sorting. I found that as long as I use !
						// ! 'niter_xc >= 2',  detraining excessive condensate before buoyancy !
						// ! sorting has negligible influence on the buoyancy sorting results. !   
						// ! ----------------------------------------------------------------- !
						if((qlj + qij) > criqc)
						{
							exql  = ( ( qlj + qij ) - criqc ) * qlj / ( qlj + qij );
							exqi  = ( ( qlj + qij ) - criqc ) * qij / ( qlj + qij );
							qtue  = qtue - exql - exqi;
							thlue = thlue + (xlv/cp/exne)*exql + (xls/cp/exne)*exqi; 
						}
						conden(pe,thlue,qtue,&thj,&qvj,&qlj,&qij,&qse,&id_check);
						if(id_check == 1)
						{
							exit_conden[i] = 1.0;
							id_exit = true;
							goto lable333;
						}
						thvj     = thj * ( 1.0 + zvir * qvj - qlj - qij );
         				tj       = thj * exne; //! This 'tj' is used for computing thermo. coeffs. below
						qsat_arg = thlue*exne;

						//!------------------------!
						qsat_arglf[0] = qsat_arg;
						pelf[0] = pe;
						status = qsat(qsat_arglf,pelf,es,qs,gam,1);
						excessu  = qtue - qs[0];

						// ! ------------------------------------------------------------------- !
						// ! Calculate critical mixing fraction, 'xc'. Mixture with mixing ratio !
						// ! smaller than 'xc' will be entrained into cumulus updraft.  Both the !
						// ! saturated updrafts with 'positive buoyancy' or 'negative buoyancy + ! 
						// ! strong vertical velocity enough to rise certain threshold distance' !
						// ! are kept into the updraft in the below program. If the core updraft !
						// ! is unsaturated, we can set 'xc = 0' and let the cumulus  convection !
						// ! still works or we may exit.                                         !
						// ! Current below code does not entrain unsaturated mixture. However it !
						// ! should be modified such that it also entrain unsaturated mixture.   !
						// ! ------------------------------------------------------------------- !
			  
						// ! ----------------------------------------------------------------- !
						// ! cridis : Critical stopping distance for buoyancy sorting purpose. !
						// !          scaleh is only used here.                                !
						// ! ----------------------------------------------------------------- !

						cridis = rle*scaleh;                 //! Original code
						//! cridis = 1._r8*(zs0(k) - zs0(k-1))  ! New code

						// ! ---------------- !
						// ! Buoyancy Sorting !
						// ! ---------------- !                   
			  
						// ! ----------------------------------------------------------------- !
						// ! Case 1 : When both cumulus and env. are unsaturated or saturated. !
						// ! ----------------------------------------------------------------- !

						if(((excessu <= 0.0) && (excess0 <= 0.0)) || ((excessu >= 0.0) && (excess0 >= 0.0)))
						{
							xc = min(1.0,max(0.0,1.0-2.0*rbuoy*g*cridis/pow(wue,2.0)*(1.0-thvj/thv0j)));
							// ! Below 3 lines are diagnostic output not influencing
							// ! numerical calculations.
							aquad = 0.0;
							bquad = 0.0;
							cquad = 0.0;
						}
						else
						{
							// ! -------------------------------------------------- !
							// ! Case 2 : When either cumulus or env. is saturated. !
							// ! -------------------------------------------------- !
							xsat    = excessu / ( excessu - excess0 );
							thlxsat = thlue + xsat * ( thle - thlue );
							qtxsat  = qtue  + xsat * ( qte - qtue );
							conden(pe,thlxsat,qtxsat,&thj,&qvj,&qlj,&qij,&qse,&id_check);
							if(id_check == 1)
							{
								exit_conden[i] = 1.0;
								id_exit = true;
								goto lable333;
							}
							thvxsat = thj * ( 1.0 + zvir * qvj - qlj - qij );
							// ! -------------------------------------------------- !
							// ! kk=1 : Cumulus Segment, kk=2 : Environment Segment !
							// ! -------------------------------------------------- !
							for(kk=1;kk<=2;++kk)
							{
								if(kk == 1)
								{
									thv_x0 = thvj;
									thv_x1 = ( 1.0 - 1.0/xsat ) * thvj + ( 1.0/xsat ) * thvxsat;
								}
								else
								{
									thv_x1 = thv0j;
                       				thv_x0 = ( xsat / ( xsat - 1.0 ) ) * thv0j + ( 1.0/( 1.0 - xsat ) ) * thvxsat;
								}
								aquad =  pow(wue,2);
                  				bquad =  2.0*rbuoy*g*cridis*(thv_x1 - thv_x0)/thv0j - 2.0*pow(wue,2);
                   				cquad =  2.0*rbuoy*g*cridis*(thv_x0 -  thv0j)/thv0j + pow(wue,2);
								if(kk == 1)
								{
									if(( pow(bquad,2)-4.0*aquad*cquad ) >= 0.0)
									{
										roots(aquad,bquad,cquad,&xs1,&xs2,&status);
										x_cu = min(1.0,max(0.0,min(xsat,min(xs1,xs2))));
									}
									else
										x_cu = xsat;
								}
								else
								{
									if((pow(bquad,2)-4.0*aquad*cquad) >= 0.0)
									{
										roots(aquad,bquad,cquad,&xs1,&xs2,&status);
                             			x_en = min(1.0,max(0.0,max(xsat,min(xs1,xs2))));
									}
									else
										x_en = 1.0;
								}
							} 
							if(x_cu == xsat)
								xc = max(x_cu, x_en);
							else
								xc = x_cu;
						}

						// ! ------------------------------------------------------------------------ !
          				// ! Compute fractional lateral entrainment & detrainment rate in each layers.!
          				// ! The unit of rei(k), fer(k), and fdr(k) is [Pa-1].  Alternative choice of !
          				// ! 'rei(k)' is also shown below, where coefficient 0.5 was from approximate !
          				// ! tuning against the BOMEX case.                                           !
          				// ! In order to prevent the onset of instability in association with cumulus !
          				// ! induced subsidence advection, cumulus mass flux at the top interface  in !
          				// ! any layer should be smaller than ( 90% of ) total mass within that layer.!
          				// ! I imposed limits on 'rei(k)' as below,  in such that stability condition ! 
          				// ! is always satisfied.                                                     !
          				// ! Below limiter of 'rei(k)' becomes negative for some cases, causing error.!
          				// ! So, for the time being, I came back to the original limiter.             !
          				// ! ------------------------------------------------------------------------ !
						ee2    = pow(xc,2);
						ud2    = 1.0 - 2.0*xc + pow(xc,2);
						//! rei(k) = ( rkm / scaleh / g / rho0j )        ! Default.
						rei[k-1] = ( 0.5 * rkm / z0[k-1] / g /rho0j ); //! Alternative.
						if((dp0[k-1]/g/dt/umf[km1] + 1.0)>=0.0)
						{
							if( xc > 0.5 ) rei[k-1] = min(rei[k-1],0.9*log(dp0[k-1]/g/dt/umf[km1] + 1.0)/dpe/(2.0*xc-1.0));
						}
						else 
						{
							if( xc > 0.5 ) rei[k-1] = rei[k-1];
						}
						fer[k-1] = rei[k-1] * ee2;
						fdr[k-1] = rei[k-1] * ud2;

						// ! ------------------------------------------------------------------------------ !
						// ! Iteration Start due to 'maxufrc' constraint [ ****************************** ] ! 
						// ! ------------------------------------------------------------------------------ !
			  
						// ! -------------------------------------------------------------------------- !
						// ! Calculate cumulus updraft mass flux and penetrative entrainment mass flux. !
						// ! Note that  non-zero penetrative entrainment mass flux will be asigned only !
						// ! to interfaces from the top interface of 'kbup' layer to the base interface !
						// ! of 'kpen' layer as will be shown later.                                    !
						// ! -------------------------------------------------------------------------- !
						
						umf[k] = umf[km1] * exp( dpe * ( fer[k-1] - fdr[k-1] ) );
						emf[k] = 0.0; 

						// ! --------------------------------------------------------- !
						// ! Compute cumulus updraft properties at the top interface.  !
						// ! Also use Tayler expansion in order to treat limiting case !
						// ! --------------------------------------------------------- !

						if(fer[k-1]*dpe < 1.0e-4 )
						{
							thlu[k] = thlu[km1] + ( thle + ssthl0[k-1] * dpe / 2.0 - thlu[km1] ) * fer[k-1] * dpe;
							qtu[k]  =  qtu[km1] + ( qte  +  ssqt0[k-1] * dpe / 2.0 -  qtu[km1] ) * fer[k-1] * dpe;
							uu[k]   =   uu[km1] + ( ue   +   ssu0[k-1] * dpe / 2.0 -   uu[km1] ) * fer[k-1] * dpe - PGFc * ssu0[k-1] * dpe;
							vu[k]   =   vu[km1] + ( ve   +   ssv0[k-1] * dpe / 2.0 -   vu[km1] ) * fer[k-1] * dpe - PGFc * ssv0[k-1] * dpe;
							for(m=0;m<ncnst;++m)
							{
							   tru[m*(mkx+1)+k]  =  tru[m*(mkx+1)+km1] + ( tre[m]  + sstr0[m*mkx+(k-1)] * dpe / 2.0  -  tru[m*(mkx+1)+km1] ) * fer[k-1] * dpe;
							}
						}
						else
						{
							thlu[k] = ( thle + ssthl0[k-1] / fer[k-1] - ssthl0[k-1] * dpe / 2.0 ) -    
									  ( thle + ssthl0[k-1] * dpe / 2.0 - thlu[km1] + ssthl0[k-1] / fer[k-1] ) * exp(-fer[k-1] * dpe);
							qtu[k]  = ( qte  +  ssqt0[k-1] / fer[k-1] -  ssqt0[k-1] * dpe / 2.0 ) -        
									  ( qte  +  ssqt0[k-1] * dpe / 2.0 -  qtu[km1] +  ssqt0[k-1] / fer[k-1] ) * exp(-fer[k-1] * dpe);
							uu[k] =   ( ue + ( 1.0 - PGFc ) * ssu0[k-1] / fer[k-1] - ssu0[k-1] * dpe / 2.0 ) -
									  ( ue +     ssu0[k-1] * dpe / 2.0 -   uu[km1] + ( 1.0 - PGFc ) * ssu0[k-1] / fer[k-1] ) * exp(-fer[k-1] * dpe);
							vu[k] =   ( ve + ( 1.0 - PGFc ) * ssv0[k-1] / fer[k-1] - ssv0[k-1] * dpe / 2.0 ) -
									  ( ve +     ssv0[k-1] * dpe / 2.0 -   vu[km1] + ( 1.0 - PGFc ) * ssv0[k-1] / fer[k-1] ) * exp(-fer[k-1] * dpe);
							for(m=0;m<ncnst;++m)
							{
								tru[m*(mkx+1)+k]  = ( tre[m]  + sstr0[m*mkx+(k-1)] / fer[k-1] - sstr0[m*mkx+(k-1)] * dpe / 2.0 ) -  
													( tre[m]  + sstr0[m*mkx+(k-1)] * dpe / 2.0 - tru[m*(mkx+1)+km1] + sstr0[m*mkx+(k-1)] / fer[k-1] ) * exp(-fer[k-1] * dpe);
							}
						}
						// printf("umf=");
						// if(i==220)
						// for(m=0;m<mkx+1;++m)
						// {
						// 	printf("i=%d,m=%d,%f\n",i,m,umf[m]);
						// }
						// printf("\n");
						// !------------------------------------------------------------------- !
						// ! Expel some of cloud water and ice from cumulus  updraft at the top !
						// ! interface.  Note that this is not 'detrainment' term  but a 'sink' !
						// ! term of cumulus updraft qt ( or one part of 'source' term of  mean !
						// ! environmental qt ). At this stage, as the most simplest choice, if !
						// ! condensate amount within cumulus updraft is larger than a critical !
						// ! value, 'criqc', expels the surplus condensate from cumulus updraft !
						// ! to the environment. A certain fraction ( e.g., 'frc_sus' ) of this !
						// ! expelled condesnate will be in a form that can be suspended in the !
						// ! layer k where it was formed, while the other fraction, '1-frc_sus' ! 
						// ! will be in a form of precipitatble (e.g.,can potentially fall down !
						// ! across the base interface of layer k ). In turn we should describe !
						// ! subsequent falling of precipitable condensate ('1-frc_sus') across !
						// ! the base interface of the layer k, &  evaporation of precipitating !
						// ! water in the below layer k-1 and associated evaporative cooling of !
						// ! the later, k-1, and falling of 'non-evaporated precipitating water !
						// ! ( which was initially formed in layer k ) and a newly-formed preci !
						// ! pitable water in the layer, k-1', across the base interface of the !
						// ! lower layer k-1.  Cloud microphysics should correctly describe all !
						// ! of these process.  In a near future, I should significantly modify !
						// ! this cloud microphysics, including precipitation-induced downdraft !
						// ! also.                                                              !
						// ! ------------------------------------------------------------------ !

						conden(ps0[k],thlu[k],qtu[k],&thj,&qvj,&qlj,&qij,&qse,&id_check);
						if(id_check == 1)
						{
							exit_conden[i] = 1.0;
							id_exit = true;
							goto lable333;
						}
						if((qlj + qij) > criqc)
						{
							exql    = ( ( qlj + qij ) - criqc ) * qlj / ( qlj + qij );
               				exqi    = ( ( qlj + qij ) - criqc ) * qij / ( qlj + qij );
							// ! ---------------------------------------------------------------- !
              				// ! It is very important to re-update 'qtu' and 'thlu'  at the upper ! 
               				// ! interface after expelling condensate from cumulus updraft at the !
               				// ! top interface of the layer. As mentioned above, this is a 'sink' !
               				// ! of cumulus qt (or equivalently, a 'source' of environmentasl qt),!
               				// ! not a regular convective'detrainment'.                           !
               				// ! ---------------------------------------------------------------- !
							qtu[k]  = qtu[k] - exql - exqi;
							thlu[k] = thlu[k] + (xlv/cp/exns0[k])*exql + (xls/cp/exns0[k])*exqi;
							// ! ---------------------------------------------------------------- !
							// ! Expelled cloud condensate into the environment from the updraft. ! 
							// ! After all the calculation later, 'dwten' and 'diten' will have a !
							// ! unit of [ kg/kg/s ], because it is a tendency of qt. Restoration !
							// ! of 'dwten' and 'diten' to this correct unit through  multiplying !
							// ! 'umf(k)*g/dp0(k)' will be performed later after finally updating !
							// ! 'umf' using a 'rmaxfrac' constraint near the end of this updraft !
							// ! buoyancy sorting loop.                                           !
							// ! ---------------------------------------------------------------- !
							dwten[k-1] = exql;   
							diten[k-1] = exqi;
						}
						else
						{
							dwten[k-1] = 0.0;
               				diten[k-1] = 0.0;
						}
						// ! ----------------------------------------------------------------- ! 
						// ! Update 'thvu(k)' after detraining condensate from cumulus updraft.!
						// ! ----------------------------------------------------------------- ! 
						conden(ps0[k],thlu[k],qtu[k],&thj,&qvj,&qlj,&qij,&qse,&id_check);
						if(id_check == 1)
						{
							exit_conden[i] = 1.0;
							id_exit = true;
							goto lable333;
						}
						thvu[k] = thj * ( 1.0 + zvir * qvj - qlj - qij );

						// ! ----------------------------------------------------------- ! 
						// ! Calculate updraft vertical velocity at the upper interface. !
						// ! In order to calculate 'wtw' at the upper interface, we use  !
						// ! 'wtw' at the lower interface. Note  'wtw'  is continuously  ! 
						// ! updated as cumulus updraft rises.                           !
						// ! ----------------------------------------------------------- !
						
						bogbot = rbuoy * ( thvu[km1] / thvebot  - 1.0 ); //! Cloud buoyancy at base interface
						bogtop = rbuoy * ( thvu[k] / thv0top[k-1] - 1.0 ); //! Cloud buoyancy at top  interface

						delbog = bogtop - bogbot;
						drage  = fer[k-1] * ( 1.0 + rdrag );
						expfac = exp(-2.0*drage*dpe);

						wtwb = wtw;
						if(drage*dpe > 1.0e-3)
						{
							wtw = wtw*expfac + (delbog + (1.0-expfac)*(bogbot + delbog/(-2.0*drage*dpe)))/(rho0j*drage);
						}
						else
						{
							wtw = wtw + dpe * ( bogbot + bogtop ) / rho0j;
						}

						// ! Force the plume rise at least to klfc of the undiluted plume.
						// ! Because even the below is not complete, I decided not to include this.
				
						// ! if( k .le. klfc ) then
						// !     wtw = max( 1.e-2_r8, wtw )
						// ! endif 

						// ! -------------------------------------------------------------- !
						// ! Repeat 'iter_xc' iteration loop until 'iter_xc = niter_xc'.    !
						// ! Also treat the case even when wtw < 0 at the 'kpen' interface. !
						// ! -------------------------------------------------------------- ! 

						if(wtw > 0.0)
						{
							thlue = 0.5 * ( thlu[km1] + thlu[k] );
              				qtue  = 0.5 * ( qtu[km1]  +  qtu[k] );         
              				wue   = 0.5 *   sqrt( max( wtwb + wtw, 0.0 ) );
						}
						else
						{
							goto lable111;
						}	
						

					} //! End of 'iter_xc' loop 
lable111:
					// aaa=1;//continue;

					// ! --------------------------------------------------------------------------- ! 
          			// ! Add the contribution of self-detrainment  to vertical variations of cumulus !
          			// ! updraft mass flux. The reason why we are trying to include self-detrainment !
          			// ! is as follows.  In current scheme,  vertical variation of updraft mass flux !
          			// ! is not fully consistent with the vertical variation of updraft vertical w.  !
          			// ! For example, within a given layer, let's assume that  cumulus w is positive !
         			// ! at the base interface, while negative at the top interface. This means that !
          			// ! cumulus updraft cannot reach to the top interface of the layer. However,    !
          			// ! cumulus updraft mass flux at the top interface is not zero according to the !
         			// ! vertical tendency equation of cumulus mass flux.   Ideally, cumulus updraft ! 
          			// ! mass flux at the top interface should be zero for this case. In order to    !
          			// ! assures that cumulus updraft mass flux goes to zero when cumulus updraft    ! 
          			// ! vertical velocity goes to zero, we are imposing self-detrainment term as    !
          			// ! below by considering layer-mean cloud buoyancy and cumulus updraft vertical !
          			// ! velocity square at the top interface. Use of auto-detrainment term will  be !
          			// ! determined by setting 'use_self_detrain=.true.' in the parameter sentence.  !
         			// ! --------------------------------------------------------------------------- !

					if(use_self_detrain)
					{
						autodet = min( 0.5*g*(bogbot+bogtop)/(max(wtw,0.0)+1.0e-4), 0.0 ); 
						umf[k]  = umf[k] * exp( 0.637*(dpe/rho0j/g) * autodet );
					}     
				 	if( umf[k] == 0.0 ) wtw = -1.0;
					 
					// ! -------------------------------------- !
					// ! Below block is just a dignostic output !
					// ! -------------------------------------- ! 

///					excessu_arr[k-1] = excessu;
///					excess0_arr[k-1] = excess0;
///					xc_arr[k-1]      = xc;
///					aquad_arr[k-1]   = aquad;
///					bquad_arr[k-1]   = bquad;
///					cquad_arr[k-1]   = cquad;
///					bogbot_arr[k-1]  = bogbot;
///					bogtop_arr[k-1]  = bogtop;

					// ! ------------------------------------------------------------------- !
					// ! 'kbup' is the upper most layer in which cloud buoyancy  is positive ! 
					// ! both at the base and top interface.  'kpen' is the upper most layer !
					// ! up to cumulus can reach. Usually, 'kpen' is located higher than the !
					// ! 'kbup'. Note we initialized these by 'kbup = krel' & 'kpen = krel'. !
					// ! As explained before, it is possible that only 'kpen' is updated,    !
					// ! while 'kbup' keeps its initialization value. For this case, current !
					// ! scheme will simply turns-off penetrative entrainment fluxes and use ! 
					// ! normal buoyancy-sorting fluxes for 'kbup <= k <= kpen-1' interfaces,!
					// ! in order to describe shallow continental cumulus convection.        !
					// ! ------------------------------------------------------------------- !

					// ! if( bogbot .gt. 0._r8 .and. bogtop .gt. 0._r8 ) then 
					// ! if( bogtop .gt. 0._r8 ) then 
					if((bogtop > 0.0) && (wtw > 0.0))
					{
						kbup = k;
					}

					if(wtw <= 0.0)
					{
						kpen = k;
						goto lable45;	
					}

					wu[k] = sqrt(wtw);
					if(wu[k] > 100.0)
					{
						exit_wu[i] = 1.0;
              			id_exit = true;
					//	printf("7\n");
              			goto lable333;
					}
					
					// ! ---------------------------------------------------------------------------- !
					// ! Iteration end due to 'rmaxfrac' constraint [ ***************************** ] ! 
					// ! ---------------------------------------------------------------------------- !
		  
					// ! ---------------------------------------------------------------------- !
					// ! Calculate updraft fractional area at the upper interface and set upper ! 
					// ! limit to 'ufrc' by 'rmaxfrac'. In order to keep the consistency  among !
					// ! ['ufrc','umf','wu (or wtw)'], if ufrc is limited by 'rmaxfrac', either !
					// ! 'umf' or 'wu' should be changed. Although both 'umf' and 'wu (wtw)' at !
					// ! the current upper interface are used for updating 'umf' & 'wu'  at the !
					// ! next upper interface, 'umf' is a passive variable not influencing  the !
					// ! buoyancy sorting process in contrast to 'wtw'. This is a reason why we !
					// ! adjusted 'umf' instead of 'wtw'. In turn we updated 'fdr' here instead !
					// ! of 'fer',  which guarantees  that all previously updated thermodynamic !
					// ! variables at the upper interface before applying 'rmaxfrac' constraint !
					// ! are already internally consistent,  even though 'ufrc'  is  limited by !
					// ! 'rmaxfrac'. Thus, we don't need to go through interation loop again.If !
					// ! If we update 'fer' however, we should go through above iteration loop. !
					// ! ---------------------------------------------------------------------- !

					rhos0j  = ps0[k] / ( r * 0.5 * ( thv0bot[k] + thv0top[k-1] ) * exns0[k] );
					ufrc[k] = umf[k] / ( rhos0j * wu[k] );
					if( ufrc[k] > rmaxfrac )
					{
///						limit_ufrc[i] = 1.0; 
              			ufrc[k] = rmaxfrac;
              			umf[k]  = rmaxfrac * rhos0j * wu[k];
              			fdr[k-1]  = fer[k-1] - log( umf[k] / umf[km1] ) / dpe;
					}
					// ! ------------------------------------------------------------ !
					// ! Update environmental properties for at the mid-point of next !
					// ! upper layer for use in buoyancy sorting.                     !
					// ! ------------------------------------------------------------ ! 

					pe      = p0[k];
					dpe     = dp0[k];
					exne    = exn0[k];
					thvebot = thv0bot[k];
					thle    = thl0[k];
					qte     = qt0[k];
					ue      = u0[k];
					ve      = v0[k];
					for(m=0;m<ncnst;++m)
					{
						tre[m] = tr0[m*mkx+k];
					} 
				}  //! End of cumulus updraft loop from the 'krel' layer to 'kpen' layer.

				// ! ------------------------------------------------------------------------------- !
				// ! Up to this point, we finished all of buoyancy sorting processes from the 'krel' !
				// ! layer to 'kpen' layer: at the top interface of individual layers, we calculated !
				// ! updraft and penetrative mass fluxes [ umf(k) & emf(k) = 0 ], updraft fractional !
				// ! area [ ufrc(k) ],  updraft vertical velocity [ wu(k) ],  updraft  thermodynamic !
				// ! variables [thlu(k),qtu(k),uu(k),vu(k),thvu(k)]. In the layer,we also calculated !
				// ! fractional entrainment-detrainment rate [ fer(k), fdr(k) ], and detrainment ten !
				// ! dency of water and ice from cumulus updraft [ dwten(k), diten(k) ]. In addition,!
				// ! we updated and identified 'krel' and 'kpen' layer index, if any.  In the 'kpen' !
				// ! layer, we calculated everything mentioned above except the 'wu(k)' and 'ufrc(k)'!
				// ! since a real value of updraft vertical velocity is not defined at the kpen  top !
				// ! interface (note 'ufrc' at the top interface of layer is calculated from 'umf(k)'!
				// ! and 'wu(k)'). As mentioned before, special treatment is required when 'kbup' is !
				// ! not updated and so 'kbup = krel'.                                               !
				// ! ------------------------------------------------------------------------------- !
				
				// ! ------------------------------------------------------------------------------ !
				// ! During the 'iter_scaleh' iteration loop, non-physical ( with non-zero values ) !
				// ! values can remain in the variable arrays above (also 'including' in case of wu !
				// ! and ufrc at the top interface) the 'kpen' layer. This can happen when the kpen !
				// ! layer index identified from the 'iter_scaleh = 1' iteration loop is located at !
				// ! above the kpen layer index identified from   'iter_scaleh = 3' iteration loop. !
				// ! Thus, in the following calculations, we should only use the values in each     !
				// ! variables only up to finally identified 'kpen' layer & 'kpen' interface except ! 
				// ! 'wu' and 'ufrc' at the top interface of 'kpen' layer.    Note that in order to !
				// ! prevent any problems due to these non-physical values, I re-initialized    the !
				// ! values of [ umf(kpen:mkx), emf(kpen:mkx), dwten(kpen+1:mkx), diten(kpen+1:mkx),! 
				// ! fer(kpen:mkx), fdr(kpen+1:mkx), ufrc(kpen:mkx) ] to be zero after 'iter_scaleh'!
				// ! do loop.                                                                       !
				// ! ------------------------------------------------------------------------------ !

lable45:
		// aaa=1;//continue;

				// ! ------------------------------------------------------------------------------ !
      			// ! Calculate 'ppen( < 0 )', updarft penetrative distance from the lower interface !
       			// ! of 'kpen' layer. Note that bogbot & bogtop at the 'kpen' layer either when fer !
       			// ! is zero or non-zero was already calculated above.                              !
       			// ! It seems that below qudarature solving formula is valid only when bogbot < 0.  !
       			// ! Below solving equation is clearly wrong ! I should revise this !               !
       			// ! ------------------------------------------------------------------------------ ! 

				if(drage == 0.0)
				{
					aquad =  ( bogtop - bogbot ) / ( ps0[kpen] - ps0[kpen-1] );
           			bquad =  2.0 * bogbot;
           			cquad = (-1)*pow(wu[kpen-1],2) * rho0j;
					roots(aquad,bquad,cquad,&xc1,&xc2,&status);
					if(status == 0)
					{
						if((xc1 <= 0.0) && (xc2 <= 0.0))
						{
							ppen = max( xc1, xc2 );
                   			ppen = min( 0.0,max( -dp0[kpen-1], ppen ) );  
						}
						else if((xc1 > 0.0) && (xc2 > 0.0))
						{
							ppen = -dp0[kpen-1];
           					//!!!        write(iulog,*) 'Warning : UW-Cumulus penetrates upto kpen interface'
						}
						else
						{
							ppen = min( xc1, xc2 );
                   			ppen = min( 0.0,max( -dp0[kpen-1], ppen ) );
						}
					}
					else
					{
						ppen = -dp0[kpen-1];
						//!!!       write(iulog,*) 'Warning : UW-Cumulus penetrates upto kpen interface'
					}
				}
				else
				{
					ppen = compute_ppen(wtwb,drage,bogbot,bogtop,rho0j,dp0[kpen-1]);
				}
///				if( (ppen == -dp0[kpen-1]) || (ppen == 0.0) ) limit_ppen[i] = 1.0;

				// ! -------------------------------------------------------------------- !
				// ! Re-calculate the amount of expelled condensate from cloud updraft    !
				// ! at the cumulus top. This is necessary for refined calculations of    !
				// ! bulk cloud microphysics at the cumulus top. Note that ppen < 0._r8   !
				// ! In the below, I explicitly calculate 'thlu_top' & 'qtu_top' by       !
				// ! using non-zero 'fer(kpen)'.                                          !    
				// ! -------------------------------------------------------------------- !

				if( fer[kpen-1]*(-ppen) < 1.0e-4 )
				{
					thlu_top = thlu[kpen-1] + ( thl0[kpen-1] + ssthl0[kpen-1] * (-ppen) / 2.0 - thlu[kpen-1] ) * fer[kpen-1] * (-ppen);
					qtu_top  =  qtu[kpen-1] + (  qt0[kpen-1] +  ssqt0[kpen-1] * (-ppen) / 2.0  - qtu[kpen-1] ) * fer[kpen-1] * (-ppen);
				}
				else
				{
					thlu_top = ( thl0[kpen-1] + ssthl0[kpen-1] / fer[kpen-1] - ssthl0[kpen-1] * (-ppen) / 2.0 ) - 
							   ( thl0[kpen-1] + ssthl0[kpen-1] * (-ppen) / 2.0 - thlu[kpen-1] + ssthl0[kpen-1] / fer[kpen-1] ) * exp(-fer[kpen-1] * (-ppen));
					 qtu_top = ( qt0[kpen-1]  +  ssqt0[kpen-1] / fer[kpen-1] -  ssqt0[kpen-1] * (-ppen) / 2.0 ) - 
							   ( qt0[kpen-1]  +  ssqt0[kpen-1] * (-ppen) / 2.0 -  qtu[kpen-1] +  ssqt0[kpen-1] / fer[kpen-1] ) * exp(-fer[kpen-1] * (-ppen));
				}

				conden(ps0[kpen-1]+ppen,thlu_top,qtu_top,&thj,&qvj,&qlj,&qij,&qse,&id_check);
				if(id_check == 1)
				{
					exit_conden[i] = 1.0;
           			id_exit = true;
           			goto lable333;
				}
				exntop = pow(((ps0[kpen-1]+ppen)/p00),rovcp);
				if((qlj + qij) > criqc)
				{
					//printf("qlj=%e, qij=%f, criqc=%e\n",qlj,qij,criqc);
					dwten[kpen-1] = ( ( qlj + qij ) - criqc ) * qlj / ( qlj + qij );
					diten[kpen-1] = ( ( qlj + qij ) - criqc ) * qij / ( qlj + qij );
					qtu_top  = qtu_top - dwten[kpen-1] - diten[kpen-1];
					thlu_top = thlu_top + (xlv/cp/exntop)*dwten[kpen-1] + (xls/cp/exntop)*diten[kpen-1]; 
				}
				else
				{
					dwten[kpen-1] = 0.0;
					diten[kpen-1] = 0.0;
				}

				// ! ----------------------------------------------------------------------- !
				// ! Calculate cumulus scale height as the top height that cumulus can reach.!
				// ! ----------------------------------------------------------------------- !

				rhos0j = ps0[kpen-1]/(r*0.5*(thv0bot[kpen-1]+thv0top[kpen-2])*exns0[kpen-1]);  
				cush   = zs0[kpen-1] - ppen/rhos0j/g;
				scaleh = cush;
			} //! End of 'iter_scaleh' loop.   
//if(i==220) printf("rhos0j=%e,cush=%e,scaleh=%e\n",rhos0j,cush,scaleh);
			// ! -------------------------------------------------------------------- !   
			// ! The 'forcedCu' is logical identifier saying whether cumulus updraft  !
			// ! overcome the buoyancy barrier just above the PBL top. If it is true, !
			// ! cumulus did not overcome the barrier -  this is a shallow convection !
			// ! with negative cloud buoyancy, mimicking  shallow continental cumulus !
			// ! convection. Depending on 'forcedCu' parameter, treatment of heat  &  !
			// ! moisture fluxes at the entraining interfaces, 'kbup <= k < kpen - 1' !
			// ! will be set up in a different ways, as will be shown later.          !
			// ! -------------------------------------------------------------------- !

			if(kbup == krel)
			{
				forcedCu = true;
///				limit_shcu[i] = 1.0;
			}
			else
			{
				forcedCu = false;
///				limit_shcu[i] = 0.0;
			}

			// ! ------------------------------------------------------------------ !
			// ! Filtering of unerasonable cumulus adjustment here.  This is a very !
			// ! important process which should be done cautiously. Various ways of !
			// ! filtering are possible depending on cases mainly using the indices !
			// ! of key layers - 'klcl','kinv','krel','klfc','kbup','kpen'. At this !
			// ! stage, the followings are all possible : 'kinv >= 2', 'klcl >= 1', !
			// ! 'krel >= kinv', 'kbup >= krel', 'kpen >= krel'. I must design this !
			// ! filtering very cautiously, in such that none of  realistic cumulus !
			// ! convection is arbitrarily turned-off. Potentially, I might turn-off! 
			// ! cumulus convection if layer-mean 'ql > 0' in the 'kinv-1' layer,in !
			// ! order to suppress cumulus convection growing, based at the Sc top. ! 
			// ! This is one of potential future modifications. Note that ppen < 0. !
			// ! ------------------------------------------------------------------ !

			cldhgt = ps0[kpen-1] + ppen;
		//	printf("%f/n",umf[5]);
			if(forcedCu)
			{
				//!!! write(iulog,*) 'forcedCu - did not overcome initial buoyancy barrier'
				exit_cufilter[i] = 1.0;
				id_exit = true;
			//	printf("8\n");
				goto lable333;
			}
			// ! Limit 'additional shallow cumulus' for DYCOMS simulation.
			// ! if( cldhgt.ge.88000._r8 ) then
			// !     id_exit = .true.
			// !     go to 333
			// ! end if

			// ! ------------------------------------------------------------------------------ !
			// ! Re-initializing some key variables above the 'kpen' layer in order to suppress !
			// ! the influence of non-physical values above 'kpen', in association with the use !
			// ! of 'iter_scaleh' loop. Note that umf, emf,  ufrc are defined at the interfaces !
			// ! (0:mkx), while 'dwten','diten', 'fer', 'fdr' are defined at layer mid-points.  !
			// ! Initialization of 'fer' and 'fdr' is for correct writing purpose of diagnostic !
			// ! output. Note that we set umf(kpen)=emf(kpen)=ufrc(kpen)=0, in consistent  with !
			// ! wtw < 0  at the top interface of 'kpen' layer. However, we still have non-zero !
			// ! expelled cloud condensate in the 'kpen' layer.                                 !
			// ! ------------------------------------------------------------------------------ !

			//  for(m=0;m<mkx+1;++m)
			//  printf("umf=%f\n",umf[m]);
			for(m=kpen;m<=mkx;++m)
			{
				umf[m]     = 0.0;
				emf[m]     = 0.0;
				ufrc[m]    = 0.0;
			}
			for(m=kpen;m<mkx;++m)
			{
				dwten[m] = 0.0;
				diten[m] = 0.0;
				fer[m]   = 0.0;
				fdr[m]   = 0.0;
			}

			// ! ------------------------------------------------------------------------ !
			// ! Calculate downward penetrative entrainment mass flux, 'emf(k) < 0',  and !
			// ! thermodynamic properties of penetratively entrained airs at   entraining !
			// ! interfaces. emf(k) is defined from the top interface of the  layer  kbup !
			// ! to the bottom interface of the layer 'kpen'. Note even when  kbup = krel,!
			// ! i.e.,even when 'kbup' was not updated in the above buoyancy  sorting  do !
			// ! loop (i.e., 'kbup' remains as the initialization value),   below do loop !
			// ! of penetrative entrainment flux can be performed without  any conceptual !
			// ! or logical problems, because we have already computed all  the variables !
			// ! necessary for performing below penetrative entrainment block.            !
			// ! In the below 'do' loop, 'k' is an interface index at which non-zero 'emf'! 
			// ! (penetrative entrainment mass flux) is calculated. Since cumulus updraft !
			// ! is negatively buoyant in the layers between the top interface of 'kbup'  !
			// ! layer (interface index, kbup) and the top interface of 'kpen' layer, the !
			// ! fractional lateral entrainment, fer(k) within these layers will be close !
			// ! to zero - so it is likely that only strong lateral detrainment occurs in !
			// ! thses layers. Under this situation,we can easily calculate the amount of !
			// ! detrainment cumulus air into these negatively buoyanct layers by  simply !
			// ! comparing cumulus updraft mass fluxes between the base and top interface !
			// ! of each layer: emf(k) = emf(k-1)*exp(-fdr(k)*dp0(k))                     !
			// !                       ~ emf(k-1)*(1-rei(k)*dp0(k))                       !
			// !                emf(k-1)-emf(k) ~ emf(k-1)*rei(k)*dp0(k)                  !
			// ! Current code assumes that about 'rpen~10' times of these detrained  mass !
			// ! are penetratively re-entrained down into the 'k-1' interface. And all of !
			// ! these detrained masses are finally dumped down into the top interface of !
			// ! 'kbup' layer. Thus, the amount of penetratively entrained air across the !
			// ! top interface of 'kbup' layer with 'rpen~10' becomes too large.          !
			// ! Note that this penetrative entrainment part can be completely turned-off !
			// ! and we can simply use normal buoyancy-sorting involved turbulent  fluxes !
			// ! by modifying 'penetrative entrainment fluxes' part below.                !
			// ! ------------------------------------------------------------------------ !
			
			// ! -----------------------------------------------------------------------!
			// ! Calculate entrainment mass flux and conservative scalars of entraining !
			// ! free air at interfaces of 'kbup <= k < kpen - 1'                       !
			// ! ---------------------------------------------------------------------- !
			
			for(k=0;k<mkx+1;++k)
			{
				thlu_emf[k] = thlu[k];
				qtu_emf[k]  = qtu[k];
				uu_emf[k]   = uu[k];
				vu_emf[k]   = vu[k];
			}
			for(m=0;m<ncnst;++m)
				for(k=0;k<mkx+1;++k)
					tru_emf[m*(mkx+1)+k] = tru[m*(mkx+1)+k];
			//printf("kpen=%d kbup=%d\n",kpen,kbup);
			for(k=kpen-1;k>=kbup;k--) //! Here, 'k' is an interface index at which
								      //! penetrative entrainment fluxes are calculated. 
			{
				rhos0j = ps0[k] / ( r * 0.5 * ( thv0bot[k] + thv0top[k-1] ) * exns0[k] );	
				if( k == kpen - 1 )
				{

					// ! ------------------------------------------------------------------------ ! 
					// ! Note that 'ppen' has already been calculated in the above 'iter_scaleh'  !
					// ! loop assuming zero lateral entrainmentin the layer 'kpen'.               !
					// ! ------------------------------------------------------------------------ !       
					
					// ! -------------------------------------------------------------------- !
					// ! Calculate returning mass flux, emf ( < 0 )                           !
					// ! Current penetrative entrainment rate with 'rpen~10' is too large and !
					// ! future refinement is necessary including the definition of 'thl','qt'! 
					// ! of penetratively entrained air.  Penetratively entrained airs across !
					// ! the 'kpen-1' interface is assumed to have the properties of the base !
					// ! interface of 'kpen' layer. Note that 'emf ~ - umf/ufrc = - w * rho'. !
					// ! Thus, below limit sets an upper limit of |emf| to be ~ 10cm/s, which !
					// ! is very loose constraint. Here, I used more restricted constraint on !
					// ! the limit of emf, assuming 'emf' cannot exceed a net mass within the !
					// ! layer above the interface. Similar to the case of warming and drying !
					// ! due to cumulus updraft induced compensating subsidence,  penetrative !
					// ! entrainment induces compensating upwelling -     in order to prevent !  
					// ! numerical instability in association with compensating upwelling, we !
					// ! should similarily limit the amount of penetrative entrainment at the !
					// ! interface by the amount of masses within the layer just above the    !
					// ! penetratively entraining interface.                                  !
					// ! -------------------------------------------------------------------- !

///					if( ( umf[k]*ppen*rei[kpen-1]*rpen ) < -0.1*rhos0j )         limit_emf[i] = 1.0;
///					if( ( umf[k]*ppen*rei[kpen-1]*rpen ) < -0.9*dp0[kpen-1]/g/dt ) limit_emf[i] = 1.0;
					
					emf[k] = max( max( umf[k]*ppen*rei[kpen-1]*rpen, -0.1*rhos0j), -0.9*dp0[kpen-1]/g/dt);
					thlu_emf[k] = thl0[kpen-1] + ssthl0[kpen-1] * ( ps0[k] - p0[kpen-1] );
					qtu_emf[k]  = qt0[kpen-1]  + ssqt0[kpen-1]  * ( ps0[k] - p0[kpen-1] );
					uu_emf[k]   = u0[kpen-1]   + ssu0[kpen-1]   * ( ps0[k] - p0[kpen-1] );     
					vu_emf[k]   = v0[kpen-1]   + ssv0[kpen-1]   * ( ps0[k] - p0[kpen-1] );
					for(m=0;m<ncnst;++m)
					{
						tru_emf[m*(mkx+1)+k] = tr0[m*mkx+(kpen-1)] + sstr0[m*mkx+(kpen-1)] * ( ps0[k] - p0[kpen-1] );
					}
				}
				else	//! if(k.lt.kpen-1).
				{
					// ! --------------------------------------------------------------------------- !
					// ! Note we are coming down from the higher interfaces to the lower interfaces. !
					// ! Also note that 'emf < 0'. So, below operation is a summing not subtracting. !
					// ! In order to ensure numerical stability, I imposed a modified correct limit  ! 
					// ! of '-0.9*dp0(k+1)/g/dt' on emf(k).                                          !
					// ! --------------------------------------------------------------------------- !
					if( use_cumpenent )   //! Original Cumulative Penetrative Entrainment
					{
///						if( ( emf[k+1]-umf[k]*dp0[k]*rei[k]*rpen ) < -0.1*rhos0j )        limit_emf[i] = 1;
///                 	if( ( emf[k+1]-umf[k]*dp0[k]*rei[k]*rpen ) < -0.9*dp0[k]/g/dt ) limit_emf[i] = 1;    
						emf[k] = max(max(emf[k+1]-umf[k]*dp0[k]*rei[k]*rpen, -0.1*rhos0j), -0.9*dp0[k]/g/dt );
						if(fabs(emf[k]) > fabs(emf[k+1]))
						{
							thlu_emf[k] = ( thlu_emf[k+1] * emf[k+1] + thl0[k] * ( emf[k] - emf[k+1] ) ) / emf[k];
							qtu_emf[k]  = ( qtu_emf[k+1]  * emf[k+1] + qt0[k]  * ( emf[k] - emf[k+1] ) ) / emf[k];
							uu_emf[k]   = ( uu_emf[k+1]   * emf[k+1] + u0[k]   * ( emf[k] - emf[k+1] ) ) / emf[k];
							vu_emf[k]   = ( vu_emf[k+1]   * emf[k+1] + v0[k]   * ( emf[k] - emf[k+1] ) ) / emf[k];
							for(m=0;m<ncnst;++m)
							{
								tru_emf[m*(mkx+1)+k] = ( tru_emf[m*(mkx+1)+(k+1)] * emf[k+1] + tr0[m*mkx+k] * ( emf[k] - emf[k+1] ) ) / emf[k];
							}
						}
						else
						{
							thlu_emf[k] = thl0[k];
                     		qtu_emf[k]  =  qt0[k];
                     		uu_emf[k]   =   u0[k];
                     		vu_emf[k]   =   v0[k];
							for(m=0;m<ncnst;++m)
							{
								tru_emf[m*(mkx+1)+k] = tr0[m*mkx+k];
							}
						}
					}
					else	//! Alternative Non-Cumulative Penetrative Entrainment
					{

///						if( ( -umf[k]*dp0[k]*rei[k]*rpen ) < -0.1*rhos0j )        limit_emf[i] = 1;
///                 	if( ( -umf[k]*dp0[k]*rei[k]*rpen ) < -0.9*dp0[k]/g/dt ) 	 limit_emf[i] = 1;
						emf[k] = max(max(-umf[k]*dp0[k]*rei[k]*rpen, -0.1*rhos0j), -0.9*dp0[k]/g/dt );
						thlu_emf[k] = thl0[k];
						qtu_emf[k]  =  qt0[k];
						uu_emf[k]   =   u0[k];
						vu_emf[k]   =   v0[k];  
						for(m=0;m<ncnst;++m)
						{
							tru_emf[m*(mkx+1)+k] = tr0[m*mkx+k];
						}
					}
				}
				
				// ! ---------------------------------------------------------------------------- !
				// ! In this GCM modeling framework,  all what we should do is to calculate  heat !
				// ! and moisture fluxes at the given geometrically-fixed height interfaces -  we !
				// ! don't need to worry about movement of material height surface in association !
				// ! with compensating subsidence or unwelling, in contrast to the bulk modeling. !
				// ! In this geometrically fixed height coordinate system, heat and moisture flux !
				// ! at the geometrically fixed height handle everything - a movement of material !
				// ! surface is implicitly treated automatically. Note that in terms of turbulent !
				// ! heat and moisture fluxes at model interfaces, both the cumulus updraft  mass !
				// ! flux and penetratively entraining mass flux play the same role -both of them ! 
				// ! warms and dries the 'kbup' layer, cools and moistens the 'kpen' layer,   and !
				// ! cools and moistens any intervening layers between 'kbup' and 'kpen' layers.  !
				// ! It is important to note these identical roles on turbulent heat and moisture !
				// ! fluxes of 'umf' and 'emf'.                                                   !
				// ! When 'kbup' is a stratocumulus-topped PBL top interface,  increase of 'rpen' !
				// ! is likely to strongly diffuse stratocumulus top interface,  resulting in the !
				// ! reduction of cloud fraction. In this sense, the 'kbup' interface has a  very !
				// ! important meaning and role : across the 'kbup' interface, strong penetrative !
				// ! entrainment occurs, thus any sharp gradient properties across that interface !
				// ! are easily diffused through strong mass exchange. Thus, an initialization of ! 
				// ! 'kbup' (and also 'kpen') should be done very cautiously as mentioned before. ! 
				// ! In order to prevent this stron diffusion for the shallow cumulus convection  !
				// ! based at the Sc top, it seems to be good to initialize 'kbup = krel', rather !
				// ! that 'kbup = krel-1'.                                                        !
				// ! ---------------------------------------------------------------------------- !
				
			}
//if((i == 220) && (iter == 1)) printf("umf111=%e\n",umf[6]);
			// if(i==220)
			// for(m=0;m<mkx+1;++m)
			// printf("m=%d,emf=%e\n",m,emf[m]);
			// !------------------------------------------------------------------ !
			// !                                                                   ! 
			// ! Compute turbulent heat, moisture, momentum flux at all interfaces !
			// !                                                                   !
			// !------------------------------------------------------------------ !
			// ! It is very important to note that in calculating turbulent fluxes !
			// ! below, we must not double count turbulent flux at any interefaces.!
			// ! In the below, turbulent fluxes at the interfaces (interface index !
			// ! k) are calculated by the following 4 blocks in consecutive order: !
			// !                                                                   !
			// ! (1) " 0 <= k <= kinv - 1 "  : PBL fluxes.                         !
			// !     From 'fluxbelowinv' using reconstructed PBL height. Currently,!
			// !     the reconstructed PBLs are independently calculated for  each !
			// !     individual conservative scalar variables ( qt, thl, u, v ) in !
			// !     each 'fluxbelowinv',  instead of being uniquely calculated by !
			// !     using thvl. Turbulent flux at the surface is assumed to be 0. !
			// ! (2) " kinv <= k <= krel - 1 " : Non-buoyancy sorting fluxes       !
			// !     Assuming cumulus mass flux  and cumulus updraft thermodynamic !
			// !     properties (except u, v which are modified by the PGFc during !
			// !     upward motion) are conserved during a updraft motion from the !
			// !     PBL top interface to the release level. If these layers don't !
			// !     exist (e,g, when 'krel = kinv'), then  current routine do not !
			// !     perform this routine automatically. So I don't need to modify !
			// !     anything.                                                     ! 
			// ! (3) " krel <= k <= kbup - 1 " : Buoyancy sorting fluxes           !
			// !     From laterally entraining-detraining buoyancy sorting plumes. ! 
			// ! (4) " kbup <= k < kpen-1 " : Penetrative entrainment fluxes       !
			// !     From penetratively entraining plumes,                         !
			// !                                                                   !
			// ! In case of normal situation, turbulent interfaces  in each groups !
			// ! are mutually independent of each other. Thus double flux counting !
			// ! or ambiguous flux counting requiring the choice among the above 4 !
			// ! groups do not occur normally. However, in case that cumulus plume !
			// ! could not completely overcome the buoyancy barrier just above the !
			// ! PBL top interface and so 'kbup = krel' (.forcedCu=.true.) ( here, !
			// ! it can be either 'kpen = krel' as the initialization, or ' kpen > !
			// ! krel' if cumulus updraft just penetrated over the top of  release !
			// ! layer ). If this happens, we should be very careful in organizing !
			// ! the sequence of the 4 calculation routines above -  note that the !
			// ! routine located at the later has the higher priority.  Additional ! 
			// ! feature I must consider is that when 'kbup = kinv - 1' (this is a !
			// ! combined situation of 'kbup=krel-1' & 'krel = kinv' when I  chose !
			// ! 'kbup=krel-1' instead of current choice of 'kbup=krel'), a strong !
			// ! penetrative entrainment fluxes exists at the PBL top interface, & !
			// ! all of these fluxes are concentrated (deposited) within the layer ! 
			// ! just below PBL top interface (i.e., 'kinv-1' layer). On the other !
			// ! hand, in case of 'fluxbelowinv', only the compensating subsidence !
			// ! effect is concentrated in the 'kinv-1' layer and 'pure' turbulent !
			// ! heat and moisture fluxes ( 'pure' means the fluxes not associated !
			// ! with compensating subsidence) are linearly distributed throughout !
			// ! the whole PBL. Thus different choice of the above flux groups can !
			// ! produce very different results. Output variable should be written !
			// ! consistently to the choice of computation sequences.              !
			// ! When the case of 'kbup = krel(-1)' happens,another way to dealing !
			// ! with this case is to simply ' exit ' the whole cumulus convection !
			// ! calculation without performing any cumulus convection.     We can !
			// ! choose this approach by specifying a condition in the  'Filtering !
			// ! of unreasonable cumulus adjustment' just after 'iter_scaleh'. But !
			// ! this seems not to be a good choice (although this choice was used !
			// ! previous code ), since it might arbitrary damped-out  the shallow !
			// ! cumulus convection over the continent land, where shallow cumulus ! 
			// ! convection tends to be negatively buoyant.                        !
			// ! ----------------------------------------------------------------- !  
	 
			// ! --------------------------------------------------- !
			// ! 1. PBL fluxes :  0 <= k <= kinv - 1                 !
			// !    All the information necessary to reconstruct PBL ! 
			// !    height are passed to 'fluxbelowinv'.             !
			// ! --------------------------------------------------- !

			xsrc  = qtsrc;
       		xmean = qt0[kinv-1];
			xtop  = qt0[kinv] + ssqt0[kinv] * ( ps0[kinv]   - p0[kinv] );
			xbot  = qt0[kinv-2] + ssqt0[kinv-2] * ( ps0[kinv-1] - p0[kinv-2] );
			fluxbelowinv( cbmf, ps0, mkxtemp, kinv, dt, xsrc, xmean, xtop, xbot, xflx );
			for(m=0;m<=kinv-1;++m)
			{
				qtflx[m] = xflx[m];
			}

			xsrc  = thlsrc;
			xmean = thl0[kinv-1];
			xtop  = thl0[kinv] + ssthl0[kinv] * ( ps0[kinv]   - p0[kinv] );
			xbot  = thl0[kinv-2] + ssthl0[kinv-2] * ( ps0[kinv-1] - p0[kinv-2] );        
			fluxbelowinv( cbmf, ps0, mkxtemp, kinv, dt, xsrc, xmean, xtop, xbot, xflx );
			for(m=0;m<=kinv-1;++m)
			{
				slflx[m] = cp * exns0[m] * xflx[m];
			}
			
			xsrc  = usrc;
			xmean = u0[kinv-1];
			xtop  = u0[kinv] + ssu0[kinv] * ( ps0[kinv]   - p0[kinv] );
			xbot  = u0[kinv-2] + ssu0[kinv-2] * ( ps0[kinv-1] - p0[kinv-2] );
			fluxbelowinv( cbmf, ps0, mkxtemp, kinv, dt, xsrc, xmean, xtop, xbot, xflx );
			for(m=0;m<=kinv-1;++m)
			{
				uflx[m] = xflx[m];
			}

			xsrc  = vsrc;
			xmean = v0[kinv-1];
			xtop  = v0[kinv] + ssv0[kinv] * ( ps0[kinv]   - p0[kinv] );
			xbot  = v0[kinv-2] + ssv0[kinv-2] * ( ps0[kinv-1] - p0[kinv-2] );
			fluxbelowinv( cbmf, ps0, mkxtemp, kinv, dt, xsrc, xmean, xtop, xbot, xflx );
			for(m=0;m<=kinv-1;++m)
			{
				vflx[m] = xflx[m];
			}

			for(m=0;m<ncnst;++m)
			{
				xsrc  = trsrc[m];
				xmean = tr0[m*mkx+(kinv-1)];
				xtop  = tr0[m*mkx+kinv] + sstr0[m*mkx+kinv] * ( ps0[kinv]   - p0[kinv] );
				xbot  = tr0[m*mkx+kinv-2] + sstr0[m*mkx+(kinv-2)] * ( ps0[kinv-1] - p0[kinv-2] );        
				fluxbelowinv( cbmf, ps0, mkxtemp, kinv, dt, xsrc, xmean, xtop, xbot, xflx );
			// if(i == 7 && m==16) printf("kinv=%d,cbmf=%16.14f,dt=%16.14f,xsrc=%16.14f,xmean=%16.14f,xtop=%16.14f,xbot=%16.14f\n",kinv,
			// 									cbmf,dt,xsrc,xmean,xtop,xbot);
				for(int temp_k=0;temp_k<=kinv-1;++temp_k)
				{
		//	if(i == 7 && m==16) printf("xflx[temp_k]=%16.14f\n",xflx[temp_k]);
					trflx[m*(mkx+1)+temp_k] = xflx[temp_k];
				}
			}

			// ! -------------------------------------------------------------- !
			// ! 2. Non-buoyancy sorting fluxes : kinv <= k <= krel - 1         !
			// !    Note that when 'krel = kinv', below block is never executed !
			// !    as in a desirable, expected way ( but I must check  if this !
			// !    is the case ). The non-buoyancy sorting fluxes are computed !
			// !    only when 'krel > kinv'.                                    !
			// ! -------------------------------------------------------------- ! 

			uplus = 0.0;
			vplus = 0.0;
			for(k=kinv;k<=krel-1;++k)
			{
				kp1 = k + 1;
				qtflx[k] = cbmf * ( qtsrc  - (  qt0[kp1-1] +  ssqt0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) );          
          		slflx[k] = cbmf * ( thlsrc - ( thl0[kp1-1] + ssthl0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) ) * cp * exns0[k];
          		uplus    = uplus + PGFc * ssu0[k-1] * ( ps0[k] - ps0[k-1] );
         		vplus    = vplus + PGFc * ssv0[k-1] * ( ps0[k] - ps0[k-1] );
          		uflx[k]  = cbmf * ( usrc + uplus -  (  u0[kp1-1]  +   ssu0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) ); 
          		vflx[k]  = cbmf * ( vsrc + vplus -  (  v0[kp1-1]  +   ssv0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) );
				for(m=0;m<ncnst;++m)
				{
					trflx[m*(mkx+1)+k] = cbmf * ( trsrc[m]  - (  tr0[m*mkx+kp1-1] +  sstr0[m*mkx+(kp1-1)] * ( ps0[k] - p0[kp1-1] ) ) );
				}       
			}
			// ! ------------------------------------------------------------------------ !
			// ! 3. Buoyancy sorting fluxes : krel <= k <= kbup - 1                       !
			// !    In case that 'kbup = krel - 1 ' ( or even in case 'kbup = krel' ),    ! 
			// !    buoyancy sorting fluxes are not calculated, which is consistent,      !
			// !    desirable feature.                                                    !  
			// ! ------------------------------------------------------------------------ !

			for(k=krel;k<=kbup-1;++k)
			{
				kp1 = k + 1;
				slflx[k] = cp * exns0[k] * umf[k] * ( thlu[k] - ( thl0[kp1-1] + ssthl0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) );
				qtflx[k] = umf[k] * ( qtu[k] - ( qt0[kp1-1] + ssqt0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) );
				uflx[k]  = umf[k] * ( uu[k] - ( u0[kp1-1] + ssu0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) );
				vflx[k]  = umf[k] * ( vu[k] - ( v0[kp1-1] + ssv0[kp1-1] * ( ps0[k] - p0[kp1-1] ) ) );
				for(m=0;m<ncnst;++m)
				{
					trflx[m*(mkx+1)+k] = umf[k] * ( tru[m*(mkx+1)+k] - ( tr0[m*mkx+kp1-1] + sstr0[m*mkx+(kp1-1)] * ( ps0[k] - p0[kp1-1] ) ) );
				}
			}
			
			// ! ------------------------------------------------------------------------- !
			// ! 4. Penetrative entrainment fluxes : kbup <= k <= kpen - 1                 !
			// !    The only confliction that can happen is when 'kbup = kinv-1'. For this !
			// !    case, turbulent flux at kinv-1 is calculated  both from 'fluxbelowinv' !
			// !    and here as penetrative entrainment fluxes.  Since penetrative flux is !
			// !    calculated later, flux at 'kinv - 1 ' will be that of penetrative flux.!
			// !    However, turbulent flux calculated at 'kinv - 1' from penetrative entr.!
			// !    is less attractable,  since more reasonable turbulent flux at 'kinv-1' !
			// !    should be obtained from 'fluxbelowinv', by considering  re-constructed ! 
			// !    inversion base height. This conflicting problem can be solved if we can!
			// !    initialize 'kbup = krel', instead of kbup = krel - 1. This choice seems!
			// !    to be more reasonable since it is not conflicted with 'fluxbelowinv' in!
			// !    calculating fluxes at 'kinv - 1' ( for this case, flux at 'kinv-1' is  !
			// !    always from 'fluxbelowinv' ), and flux at 'krel-1' is calculated from  !
			// !    the non-buoyancy sorting flux without being competed with penetrative  !
			// !    entrainment fluxes. Even when we use normal cumulus flux instead of    !
			// !    penetrative entrainment fluxes at 'kbup <= k <= kpen-1' interfaces,    !
			// !    the initialization of kbup=krel perfectly works without any conceptual !
			// !    confliction. Thus it seems to be much better to choose 'kbup = krel'   !
			// !    initialization of 'kbup', which is current choice.                     !
			// !    Note that below formula uses conventional updraft cumulus fluxes for   !
			// !    shallow cumulus which did not overcome the first buoyancy barrier above!
			// !    PBL top while uses penetrative entrainment fluxes for the other cases  !
			// !    'kbup <= k <= kpen-1' interfaces. Depending on cases, however, I can   !
			// !    selelct different choice.                                              !
			// ! ------------------------------------------------------------------------------------------------------------------ !
			// !   if( forcedCu ) then                                                                                              !
			// !       slflx(k) = cp * exns0(k) * umf(k) * ( thlu(k) - ( thl0(kp1) + ssthl0(kp1) * ( ps0(k) - p0(kp1) ) ) )         !
			// !       qtflx(k) =                 umf(k) * (  qtu(k) - (  qt0(kp1) +  ssqt0(kp1) * ( ps0(k) - p0(kp1) ) ) )         !
			// !       uflx(k)  =                 umf(k) * (   uu(k) - (   u0(kp1) +   ssu0(kp1) * ( ps0(k) - p0(kp1) ) ) )         !
			// !       vflx(k)  =                 umf(k) * (   vu(k) - (   v0(kp1) +   ssv0(kp1) * ( ps0(k) - p0(kp1) ) ) )         !
			// !       do m = 1, ncnst                                                                                              !
			// !          trflx(k,m) = umf(k) * ( tru(k,m) - ( tr0(kp1,m) + sstr0(kp1,m) * ( ps0(k) - p0(kp1) ) ) )                 !
			// !       enddo                                                                                                        !
			// !   else                                                                                                             !
			// !       slflx(k) = cp * exns0(k) * emf(k) * ( thlu_emf(k) - ( thl0(k) + ssthl0(k) * ( ps0(k) - p0(k) ) ) )           !
			// !       qtflx(k) =                 emf(k) * (  qtu_emf(k) - (  qt0(k) +  ssqt0(k) * ( ps0(k) - p0(k) ) ) )           !
			// !       uflx(k)  =                 emf(k) * (   uu_emf(k) - (   u0(k) +   ssu0(k) * ( ps0(k) - p0(k) ) ) )           !
			// !       vflx(k)  =                 emf(k) * (   vu_emf(k) - (   v0(k) +   ssv0(k) * ( ps0(k) - p0(k) ) ) )           !
			// !       do m = 1, ncnst                                                                                              !
			// !          trflx(k,m) = emf(k) * ( tru_emf(k,m) - ( tr0(k,m) + sstr0(k,m) * ( ps0(k) - p0(k) ) ) )                   !
			// !       enddo                                                                                                        !
			// !   endif                                                                                                            !
			// !                                                                                                                    !
			// !   if( use_uppenent ) then ! Combined Updraft + Penetrative Entrainment Flux                                        !
			// !       slflx(k) = cp * exns0(k) * umf(k) * ( thlu(k)     - ( thl0(kp1) + ssthl0(kp1) * ( ps0(k) - p0(kp1) ) ) ) + & !
			// !                  cp * exns0(k) * emf(k) * ( thlu_emf(k) - (   thl0(k) +   ssthl0(k) * ( ps0(k) - p0(k) ) ) )       !
			// !       qtflx(k) =                 umf(k) * (  qtu(k)     - (  qt0(kp1) +  ssqt0(kp1) * ( ps0(k) - p0(kp1) ) ) ) + & !
			// !                                  emf(k) * (  qtu_emf(k) - (    qt0(k) +    ssqt0(k) * ( ps0(k) - p0(k) ) ) )       !                   
			// !       uflx(k)  =                 umf(k) * (   uu(k)     - (   u0(kp1) +   ssu0(kp1) * ( ps0(k) - p0(kp1) ) ) ) + & !
			// !                                  emf(k) * (   uu_emf(k) - (     u0(k) +     ssu0(k) * ( ps0(k) - p0(k) ) ) )       !                      
			// !       vflx(k)  =                 umf(k) * (   vu(k)     - (   v0(kp1) +   ssv0(kp1) * ( ps0(k) - p0(kp1) ) ) ) + & !
			// !                                  emf(k) * (   vu_emf(k) - (     v0(k) +     ssv0(k) * ( ps0(k) - p0(k) ) ) )       !                     
			// !       do m = 1, ncnst                                                                                              !
			// !          trflx(k,m) = umf(k) * ( tru(k,m) - ( tr0(kp1,m) + sstr0(kp1,m) * ( ps0(k) - p0(kp1) ) ) ) + &             ! 
			// !                       emf(k) * ( tru_emf(k,m) - ( tr0(k,m) + sstr0(k,m) * ( ps0(k) - p0(k) ) ) )                   ! 
			// !       enddo                                                                                                        !
			// ! ------------------------------------------------------------------------------------------------------------------ !
			// if((i == 7) )
			// {
			// 	printf("tr0_in1[7,0,16]=%e\n",tr0[16*mkx*iend+0*iend+7]);
			// }
			// if(i == 220)	printf("kbup=%d,kpen=%d\n",kbup,kpen);
			for(k=kbup;k<=kpen-1;++k)
			{
				kp1 = k + 1;
          		slflx[k] = cp * exns0[k] * emf[k] * ( thlu_emf[k] - ( thl0[k-1] + ssthl0[k-1] * ( ps0[k] - p0[k-1] ) ) );
          		qtflx[k] =                 emf[k] * (  qtu_emf[k] - (  qt0[k-1] +  ssqt0[k-1] * ( ps0[k] - p0[k-1] ) ) ); 
          		uflx[k]  =                 emf[k] * (   uu_emf[k] - (   u0[k-1] +   ssu0[k-1] * ( ps0[k] - p0[k-1] ) ) ); 
          		vflx[k]  =                 emf[k] * (   vu_emf[k] - (   v0[k-1] +   ssv0[k-1] * ( ps0[k] - p0[k-1] ) ) );
				for(m=0;m<ncnst;++m)
				{
					trflx[m*(mkx+1)+k] = emf[k] * ( tru_emf[m*(mkx+1)+k] - ( tr0[m*mkx+(k-1)] + sstr0[m*mkx+(k-1)] * ( ps0[k] - p0[k-1] ) ) );
				}
			}
//if(i == 7) printf("emf[1]=%16.14f,tru_emf[7,1,16]=%16.14f,tr0[7,0,16]=%16.14f,sstr0[7,0,16]=%16.14f,ps0[1]=%16.14f,p0[0]=%16.14f\n",emf[1],tru_emf[7+1*iend+16*(mkx+1)*iend],tr0[7+0*iend+16*(mkx)*iend],sstr0[7+0*iend+16*(mkx)*iend],ps0[1],p0[0]);
//if(i == 7) printf("trflx[7,1,16]=%16.14f\n",trflx[16*(mkx+1)*iend+1*iend+7]);
			// ! ------------------------------------------- !
			// ! Turn-off cumulus momentum flux as an option !
			// ! ------------------------------------------- !

			if( !use_momenflx )
			{
				for(k=0;k<mkx+1;++k)
				{
					uflx[k] = 0.0;
					vflx[k] = 0.0;
				}
			}  

			// ! -------------------------------------------------------- !
			// ! Condensate tendency by compensating subsidence/upwelling !
			// ! -------------------------------------------------------- !

			for(k=0;k<=mkx;++k)
			{
				uemf[k] = 0.0;
			}
			for(k=0;k<=kinv-2;++k)//! Assume linear updraft mass flux within the PBL.
			{
				uemf[k] = cbmf * ( ps0[0] - ps0[k] ) / ( ps0[0] - ps0[kinv-1] ); 
			}
			for(k=kinv-1;k<=krel-1;++k)
			{
				uemf[k] = cbmf;	
			}
			for(k=krel;k<=kbup-1;++k)
			{
				uemf[k] = umf[k];
			}
			for(k=kbup;k<=kpen-1;++k)
			{
				uemf[k] = emf[k]; //! Only use penetrative entrainment flux consistently.
			}

			for(j=0;j<mkx;++j)
			{
				comsub[j] = 0.0;
			}
			for(j=1;j<=kpen;++j)
			{
				comsub[j-1]  = 0.5 * ( uemf[j] + uemf[j-1] ); 
			}

			for(k=1;k<=kpen;++k)
			{
				if( comsub[k-1] >= 0.0 )
				{
					if( k == mkx )
					{
						thlten_sub = 0.0;
						qtten_sub  = 0.0;
						qlten_sub  = 0.0;
						qiten_sub  = 0.0;
						nlten_sub  = 0.0;
						niten_sub  = 0.0;
					}
					else
					{
						thlten_sub = g * comsub[k-1] * ( thl0[k] - thl0[k-1] ) / ( p0[k-1] - p0[k] );
						qtten_sub  = g * comsub[k-1] * (  qt0[k] -  qt0[k-1] ) / ( p0[k-1] - p0[k] );
						qlten_sub  = g * comsub[k-1] * (  ql0[k] -  ql0[k-1] ) / ( p0[k-1] - p0[k] );
						qiten_sub  = g * comsub[k-1] * (  qi0[k] -  qi0[k-1] ) / ( p0[k-1] - p0[k] );
						nlten_sub  = g * comsub[k-1] * (  tr0[(ixnumliq-1)*mkx+k] -  tr0[(ixnumliq-1)*mkx+(k-1)] ) / ( p0[k-1] - p0[k] );
						niten_sub  = g * comsub[k-1] * (  tr0[(ixnumice-1)*mkx+k] -  tr0[(ixnumice-1)*mkx+(k-1)] ) / ( p0[k-1] - p0[k] );
					}
				}
				else
				{
					if( k == 1 )
					{
						thlten_sub = 0.0;
						qtten_sub  = 0.0;
						qlten_sub  = 0.0;
						qiten_sub  = 0.0;
						nlten_sub  = 0.0;
						niten_sub  = 0.0;
					}
					else
					{
						thlten_sub = g * comsub[k-1] * ( thl0[k-1] - thl0[k-2] ) / ( p0[k-2] - p0[k-1] );
						qtten_sub  = g * comsub[k-1] * (  qt0[k-1] -  qt0[k-2] ) / ( p0[k-2] - p0[k-1] );
						qlten_sub  = g * comsub[k-1] * (  ql0[k-1] -  ql0[k-2] ) / ( p0[k-2] - p0[k-1] );
						qiten_sub  = g * comsub[k-1] * (  qi0[k-1] -  qi0[k-2] ) / ( p0[k-2] - p0[k-1] );
						nlten_sub  = g * comsub[k-1] * (  tr0[(ixnumliq-1)*mkx+(k-1)] -  tr0[(ixnumliq-1)*mkx+(k-2)] ) / ( p0[k-2] - p0[k-1] );
						niten_sub  = g * comsub[k-1] * (  tr0[(ixnumice-1)*mkx+(k-1)] -  tr0[(ixnumice-1)*mkx+(k-2)] ) / ( p0[k-2] - p0[k-1] );
					}
				}
				thl_prog = thl0[k-1] + thlten_sub * dt;
				qt_prog  = max( qt0[k-1] + qtten_sub * dt, 1.0e-12 );
				conden(p0[k-1],thl_prog,qt_prog,&thj,&qvj,&qlj,&qij,&qse,&id_check);
				if(id_check == 1)
				{
					id_exit = true;
					goto lable333;
				}
				//! qlten_sink(k) = ( qlj - ql0(k) ) / dt
				//! qiten_sink(k) = ( qij - qi0(k) ) / dt
				qlten_sink[k-1] = max( qlten_sub, - ql0[k-1] / dt );// ! For consistency with prognostic macrophysics scheme
				qiten_sink[k-1] = max( qiten_sub, - qi0[k-1] / dt );// ! For consistency with prognostic macrophysics scheme
				nlten_sink[k-1] = max( nlten_sub, - tr0[(ixnumliq-1)*mkx+(k-1)] / dt ); 
				niten_sink[k-1] = max( niten_sub, - tr0[(ixnumice-1)*mkx+(k-1)] / dt );
			}

			// ! --------------------------------------------- !
			// !                                               !
			// ! Calculate convective tendencies at each layer ! 
			// !                                               !
			// ! --------------------------------------------- !
			
			// ! ----------------- !
			// ! Momentum tendency !
			// ! ----------------- !
			
			for(k=1;k<=kpen;++k)
			{
				km1 = k - 1; 
				uten[k-1] = ( uflx[km1] - uflx[k] ) * g / dp0[k-1];
				vten[k-1] = ( vflx[km1] - vflx[k] ) * g / dp0[k-1]; 
				uf[k-1]   = u0[k-1] + uten[k-1] * dt;
				vf[k-1]   = v0[k-1] + vten[k-1] * dt;
				//! do m = 1, ncnst
				//!trten(k,m) = ( trflx(km1,m) - trflx(k,m) ) * g / dp0(k)
				//!  ! Limit trten(k,m) such that negative value is not developed.
				//!  ! This limitation does not conserve grid-mean tracers and future
				//!  ! refinement is required for tracer-conserving treatment.
				//!    trten(k,m) = max(trten(k,m),-tr0(k,m)/dt)              
			}

			// ! ----------------------------------------------------------------- !
			// ! Tendencies of thermodynamic variables.                            ! 
			// ! This part requires a careful treatment of bulk cloud microphysics.!
			// ! Relocations of 'precipitable condensates' either into the surface ! 
			// ! or into the tendency of 'krel' layer will be performed just after !
			// ! finishing the below 'do-loop'.                                    !        
			// ! ----------------------------------------------------------------- !

			rliq    = 0.0;
			rainflx = 0.0;
			snowflx = 0.0;

			for(k=1;k<=kpen;++k)
			{
				km1 = k - 1;

				// ! ------------------------------------------------------------------------------ !
				// ! Compute 'slten', 'qtten', 'qvten', 'qlten', 'qiten', and 'sten'                !
				// !                                                                                !
				// ! Key assumptions made in this 'cumulus scheme' are :                            !
				// ! 1. Cumulus updraft expels condensate into the environment at the top interface !
				// !    of each layer. Note that in addition to this expel process ('source' term), !
				// !    cumulus updraft can modify layer mean condensate through normal detrainment !
				// !    forcing or compensating subsidence.                                         !
				// ! 2. Expelled water can be either 'sustaining' or 'precipitating' condensate. By !
				// !    definition, 'suataining condensate' will remain in the layer where it was   !
				// !    formed, while 'precipitating condensate' will fall across the base of the   !
				// !    layer where it was formed.                                                  !
				// ! 3. All precipitating condensates are assumed to fall into the release layer or !
				// !    ground as soon as it was formed without being evaporated during the falling !
				// !    process down to the desinated layer ( either release layer of surface ).    !
				// ! ------------------------------------------------------------------------------ !
	  
				// ! ------------------------------------------------------------------------- !     
				// ! 'dwten(k)','diten(k)' : Production rate of condensate  within the layer k !
				// !      [ kg/kg/s ]        by the expels of condensate from cumulus updraft. !
				// ! It is important to note that in terms of moisture tendency equation, this !
				// ! is a 'source' term of enviromental 'qt'.  More importantly,  these source !
				// ! are already counted in the turbulent heat and moisture fluxes we computed !
				// ! until now, assuming all the expelled condensate remain in the layer where ! 
				// ! it was formed. Thus, in calculation of 'qtten' and 'slten' below, we MUST !
				// ! NOT add or subtract these terms explicitly in order not to double or miss !
				// ! count, unless some expelled condensates fall down out of the layer.  Note !
				// ! this falling-down process ( i.e., precipitation process ) and  associated !
				// ! 'qtten' and 'slten' and production of surface precipitation flux  will be !
				// ! treated later in 'zm_conv_evap' in 'convect_shallow_tend' subroutine.     ! 
				// ! In below, we are converting expelled cloud condensate into correct unit.  !
				// ! I found that below use of '0.5 * (umf(k-1) + umf(k))' causes conservation !
				// ! errors at some columns in global simulation. So, I returned to originals. !
				// ! This will cause no precipitation flux at 'kpen' layer since umf(kpen)=0.  !
				// ! ------------------------------------------------------------------------- !
 
				dwten[k-1] = dwten[k-1] * 0.5 * ( umf[k-1] + umf[k] ) * g / dp0[k-1]; //! [ kg/kg/s ]
				diten[k-1] = diten[k-1] * 0.5 * ( umf[k-1] + umf[k] ) * g / dp0[k-1]; //! [ kg/kg/s ]
				
				// ! dwten(k) = dwten(k) * umf(k) * g / dp0(k) ! [ kg/kg/s ]
				// ! diten(k) = diten(k) * umf(k) * g / dp0(k) ! [ kg/kg/s ]

				// ! --------------------------------------------------------------------------- !
				// ! 'qrten(k)','qsten(k)' : Production rate of rain and snow within the layer k !
				// !     [ kg/kg/s ]         by cumulus expels of condensates to the environment.!         
				// ! This will be falled-out of the layer where it was formed and will be dumped !
				// ! dumped into the release layer assuming that there is no evaporative cooling !
				// ! while precipitable condensate moves to the relaes level. This is reasonable ! 
				// ! assumtion if cumulus is purely vertical and so the path along which precita !
				// ! ble condensate falls is fully saturared. This 're-allocation' process of    !
				// ! precipitable condensate into the release layer is fully described in this   !
				// ! convection scheme. After that, the dumped water into the release layer will !
				// ! falling down across the base of release layer ( or LCL, if  exact treatment ! 
				// ! is required ) and will be allowed to be evaporated in layers below  release !
				// ! layer, and finally non-zero surface precipitation flux will be calculated.  !
				// ! This latter process will be separately treated 'zm_conv_evap' routine.      !
				// ! --------------------------------------------------------------------------- !
			//	printf("frc_rasn=%f dwten[k-1]=%f\n",frc_rasn,dwten[k-1]);
				qrten[k-1] = frc_rasn * dwten[k-1];
				qsten[k-1] = frc_rasn * diten[k-1];

				// ! ----------------------------------------------------------------------- !         
				// ! 'rainflx','snowflx' : Cumulative rain and snow flux integrated from the ! 
				// !     [ kg/m2/s ]       release leyer to the 'kpen' layer. Note that even !
				// ! though wtw(kpen) < 0 (and umf(kpen) = 0) at the top interface of 'kpen' !
				// ! layer, 'dwten(kpen)' and diten(kpen)  were calculated after calculating !
				// ! explicit cloud top height. Thus below calculation of precipitation flux !
				// ! is correct. Note that  precipitating condensates are formed only in the !
				// ! layers from 'krel' to 'kpen', including the two layers.                 !
				// ! ----------------------------------------------------------------------- !

				rainflx = rainflx + qrten[k-1] * dp0[k-1] / g;
				snowflx = snowflx + qsten[k-1] * dp0[k-1] / g;

				// ! ------------------------------------------------------------------------ !
				// ! 'slten(k)','qtten(k)'                                                    !
				// !  Note that 'slflx(k)' and 'qtflx(k)' we have calculated already included !
				// !  all the contributions of (1) expels of condensate (dwten(k), diten(k)), !
				// !  (2) mass detrainment ( delta * umf * ( qtu - qt ) ), & (3) compensating !
				// !  subsidence ( M * dqt / dz ). Thus 'slflx(k)' and 'qtflx(k)' we computed ! 
				// !  is a hybrid turbulent flux containing one part of 'source' term - expel !
				// !  of condensate. In order to calculate 'slten' and 'qtten', we should add !
				// !  additional 'source' term, if any. If the expelled condensate falls down !
				// !  across the base of the layer, it will be another sink (negative source) !
				// !  term.  Note also that we included frictional heating terms in the below !
				// !  calculation of 'slten'.                                                 !
				// ! ------------------------------------------------------------------------ !

				slten[k-1] = ( slflx[km1] - slflx[k] ) * g / dp0[k-1];
				if(k == 1)
				{
					slten[k-1] = slten[k-1] - g / 4.0 / dp0[k-1] * (                            
								 uflx[k]*(uf[k] - uf[k-1] + u0[k] - u0[k-1]) +     
								 vflx[k]*(vf[k] - vf[k-1] + v0[k] - v0[k-1]));
				}
				else if( (k >= 2) && (k <= kpen-1) )
				{
					slten[k-1] = slten[k-1] - g / 4.0 / dp0[k-1] * (                            
								 uflx[k]*(uf[k] - uf[k-1] + u0[k] - u0[k-1]) +     
								 uflx[k-1]*(uf[k-1] - uf[k-2] + u0[k-1] - u0[k-2]) +  
								 vflx[k]*(vf[k] - vf[k-1] + v0[k] - v0[k-1]) +     
								 vflx[k-1]*(vf[k-1] - vf[k-2] + v0[k-1] - v0[k-2]));
				}
				else if(k == kpen)
				{
					slten[k-1] = slten[k-1] - g / 4.0 / dp0[k-1] * (                            
								 uflx[k-1]*(uf[k-1] - uf[k-2] + u0[k-1] - u0[k-2]) +  
								 vflx[k-1]*(vf[k-1] - vf[k-2] + v0[k-1] - v0[k-2]));
				}
				qtten[k-1] = ( qtflx[km1] - qtflx[k] ) * g / dp0[k-1];

				// ! ---------------------------------------------------------------------------- !
				// ! Compute condensate tendency, including reserved condensate                   !
				// ! We assume that eventual detachment and detrainment occurs in kbup layer  due !
				// ! to downdraft buoyancy sorting. In the layer above the kbup, only penetrative !
				// ! entrainment exists. Penetrative entrained air is assumed not to contain any  !
				// ! condensate.                                                                  !
				// ! ---------------------------------------------------------------------------- !

				//! Compute in-cumulus condensate at the layer mid-point.

				if( (k < krel) || (k > kpen) )
				{
					qlu_mid = 0.0;
					qiu_mid = 0.0;
					qlj     = 0.0;
					qij     = 0.0;
				}
				else if( k == krel )
				{
					conden(prel,thlu[krel-1],qtu[krel-1],&thj,&qvj,&qlj,&qij,&qse,&id_check);
					if(id_check == 1)
					{
						exit_conden[i] = 1.0;
						id_exit = true;
						goto lable333;
					}
					qlubelow = qlj;       
					qiubelow = qij;  
					conden(ps0[k],thlu[k],qtu[k],&thj,&qvj,&qlj,&qij,&qse,&id_check);
					if(id_check == 1)
					{
						exit_conden[i] = 1.0;
						id_exit = true;
						goto lable333;
					}
					qlu_mid = 0.5 * ( qlubelow + qlj ) * ( prel - ps0[k] )/( ps0[k-1] - ps0[k] );
					qiu_mid = 0.5 * ( qiubelow + qij ) * ( prel - ps0[k] )/( ps0[k-1] - ps0[k] );
				}
				else if( k == kpen )
				{
					conden(ps0[k-1]+ppen,thlu_top,qtu_top,&thj,&qvj,&qlj,&qij,&qse,&id_check);
					if(id_check == 1)
					{
						exit_conden[i] = 1.0;
						id_exit = true;
						goto lable333;
					}
					qlu_mid = 0.5 * ( qlubelow + qlj ) * ( -ppen )        /( ps0[k-1] - ps0[k] );
					qiu_mid = 0.5 * ( qiubelow + qij ) * ( -ppen )        /( ps0[k-1] - ps0[k] );
					qlu_top = qlj;
					qiu_top = qij;
				}
				else
				{
					conden(ps0[k],thlu[k],qtu[k],&thj,&qvj,&qlj,&qij,&qse,&id_check);
					if(id_check == 1)
					{
						exit_conden[i] = 1.0;
						id_exit = true;
						goto lable333;
					}
					qlu_mid = 0.5 * ( qlubelow + qlj );
					qiu_mid = 0.5 * ( qiubelow + qij );
				}
				qlubelow = qlj;       
				qiubelow = qij;
				
				//! 1. Sustained Precipitation

				qc_l[k-1] = ( 1.0 - frc_rasn ) * dwten[k-1]; //! [ kg/kg/s ]
				qc_i[k-1] = ( 1.0 - frc_rasn ) * diten[k-1]; //! [ kg/kg/s ]

				//! 2. Detrained Condensate

				if(k <= kbup)
				{
					qc_l[k-1] = qc_l[k-1] + g * 0.5 * ( umf[k-1] + umf[k] ) * fdr[k-1] * qlu_mid; //! [ kg/kg/s ]
					qc_i[k-1] = qc_i[k-1] + g * 0.5 * ( umf[k-1] + umf[k] ) * fdr[k-1] * qiu_mid; //! [ kg/kg/s ]
					qc_lm   =         - g * 0.5 * ( umf[k-1] + umf[k] ) * fdr[k-1] * ql0[k-1];  
					qc_im   =         - g * 0.5 * ( umf[k-1] + umf[k] ) * fdr[k-1] * qi0[k-1];
				    //! Below 'nc_lm', 'nc_im' should be used only when frc_rasn = 1.
					nc_lm   =         - g * 0.5 * ( umf[k-1] + umf[k] ) * fdr[k-1] * tr0[(ixnumliq-1)*mkx+(k-1)];  
					nc_im   =         - g * 0.5 * ( umf[k-1] + umf[k] ) * fdr[k-1] * tr0[(ixnumice-1)*mkx+(k-1)];
				}
				else
				{
					qc_lm   = 0.0;
					qc_im   = 0.0;
					nc_lm   = 0.0;
					nc_im   = 0.0;
				}

				//! 3. Detached Updraft 

				if(k == kbup)
				{
					qc_l[k-1] = qc_l[k-1] + g * umf[k] * qlj     / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					qc_i[k-1] = qc_i[k-1] + g * umf[k] * qij     / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					qc_lm   = qc_lm   - g * umf[k] * ql0[k-1]  / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					qc_im   = qc_im   - g * umf[k] * qi0[k-1]  / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					nc_lm   = nc_lm   - g * umf[k] * tr0[(ixnumliq-1)*mkx+(k-1)] / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					nc_im   = nc_im   - g * umf[k] * tr0[(ixnumice-1)*mkx+(k-1)] / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
				}

				// ! 4. Cumulative Penetrative entrainment detrained in the 'kbup' layer
				// !    Explicitly compute the properties detrained penetrative entrained airs in k = kbup layer.

				if(k == kbup)
				{
					conden(p0[k-1],thlu_emf[k],qtu_emf[k],&thj,&qvj,&ql_emf_kbup,&qi_emf_kbup,&qse,&id_check);
					if(id_check == 1)
					{
						id_exit = true;
						goto lable333;
					}
					if(ql_emf_kbup > 0.0)
					{
						nl_emf_kbup = tru_emf[(ixnumliq-1)*(mkx+1)+k];
					}
					else
					{
						nl_emf_kbup = 0.0;
					}
					if(qi_emf_kbup > 0.0)
					{
						ni_emf_kbup = tru_emf[(ixnumice-1)*(mkx+1)+k];
					}
					else
					{
						ni_emf_kbup = 0.0;
					}
					qc_lm   = qc_lm   - g * emf[k] * ( ql_emf_kbup - ql0[k-1] ) / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					qc_im   = qc_im   - g * emf[k] * ( qi_emf_kbup - qi0[k-1] ) / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					nc_lm   = nc_lm   - g * emf[k] * ( nl_emf_kbup - tr0[(ixnumliq-1)*mkx+(k-1)] ) / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
					nc_im   = nc_im   - g * emf[k] * ( ni_emf_kbup - tr0[(ixnumice-1)*mkx+(k-1)] ) / ( ps0[k-1] - ps0[k] ); //! [ kg/kg/s ]
				}

				qlten_det   = qc_l[k-1] + qc_lm;
				qiten_det   = qc_i[k-1] + qc_im;

				// ! --------------------------------------------------------------------------------- !
				// ! 'qlten(k)','qiten(k)','qvten(k)','sten(k)'                                        !
				// ! Note that falling of precipitation will be treated later.                         !
				// ! The prevension of negative 'qv,ql,qi' will be treated later in positive_moisture. !
				// ! --------------------------------------------------------------------------------- ! 

				if(use_expconten)
				{
					if(use_unicondet)
					{
						qc_l[k-1] = 0.0;
						qc_i[k-1] = 0.0; 
						qlten[k-1] = frc_rasn * dwten[k-1] + qlten_sink[k-1] + qlten_det;
						qiten[k-1] = frc_rasn * diten[k-1] + qiten_sink[k-1] + qiten_det;
					}
					else
					{
						qlten[k-1] = qc_l[k-1] + frc_rasn * dwten[k-1] + ( max( 0.0, ql0[k-1] + ( qc_lm + qlten_sink[k-1] ) * dt ) - ql0[k-1] ) / dt;
						qiten[k-1] = qc_i[k-1] + frc_rasn * diten[k-1] + ( max( 0.0, qi0[k-1] + ( qc_im + qiten_sink[k-1] ) * dt ) - qi0[k-1] ) / dt;
						trten[(ixnumliq-1)*mkx+(k-1)] = max( nc_lm + nlten_sink[k-1], - tr0[(ixnumliq-1)*mkx+(k-1)] / dt );
						trten[(ixnumice-1)*mkx+(k-1)] = max( nc_im + niten_sink[k-1], - tr0[(ixnumice-1)*mkx+(k-1)] / dt );
					}
				}
				else
				{
					if(use_unicondet)
					{
						qc_l[k-1] = 0.0;
						qc_i[k-1] = 0.0; 
					}
					qlten[k-1] = dwten[k-1] + ( qtten[k-1] - dwten[k-1] - diten[k-1] ) * ( ql0[k-1] / qt0[k-1]);
					qiten[k-1] = diten[k-1] + ( qtten[k-1] - dwten[k-1] - diten[k-1] ) * ( qi0[k-1] / qt0[k-1]);
				}

				qvten[k-1] = qtten[k-1] - qlten[k-1] - qiten[k-1];
				sten[k-1]  = slten[k-1] + xlv * qlten[k-1] + xls * qiten[k-1];

				// ! -------------------------------------------------------------------------- !
				// ! 'rliq' : Verticall-integrated 'suspended cloud condensate'                 !
				// !  [m/s]   This is so called 'reserved liquid water'  in other subroutines   ! 
				// ! of CAM3, since the contribution of this term should not be included into   !
				// ! the tendency of each layer or surface flux (precip)  within this cumulus   !
				// ! scheme. The adding of this term to the layer tendency will be done inthe   !
				// ! 'stratiform_tend', just after performing sediment process there.           !
				// ! The main problem of these rather going-back-and-forth and stupid-seeming   ! 
				// ! approach is that the sediment process of suspendened condensate will not   !
				// ! be treated at all in the 'stratiform_tend'.                                !
				// ! Note that 'precip' [m/s] is vertically-integrated total 'rain+snow' formed !
				// ! from the cumulus updraft. Important : in the below, 1000 is rhoh2o ( water !
				// ! density ) [ kg/m^3 ] used for unit conversion from [ kg/m^2/s ] to [ m/s ] !
				// ! for use in stratiform.F90.                                                 !
				// ! -------------------------------------------------------------------------- ! 

				qc[k-1]  =  qc_l[k-1] +  qc_i[k-1];   
				rliq   =  rliq    + qc[k-1] * dp0[k-1] / g / 1000.0;    //! [ m/s ]

			}

			precip  =  rainflx + snowflx;                      //! [ kg/m2/s ]
			snow    =  snowflx;                                //! [ kg/m2/s ] 

			// ! ---------------------------------------------------------------- !
			// ! Now treats the 'evaporation' and 'melting' of rain ( qrten ) and ! 
			// ! snow ( qsten ) during falling process. Below algorithms are from !
			// ! 'zm_conv_evap' but with some modification, which allows separate !
			// ! treatment of 'rain' and 'snow' condensates. Note that I included !
			// ! the evaporation dynamics into the convection scheme for complete !
			// ! development of cumulus scheme especially in association with the ! 
			// ! implicit CIN closure. In compatible with this internal treatment !
			// ! of evaporation, I should modify 'convect_shallow',  in such that !
			// ! 'zm_conv_evap' is not performed when I choose UW PBL-Cu schemes. !                                          
			// ! ---------------------------------------------------------------- !

			evpint_rain    = 0.0; 
			evpint_snow    = 0.0;
			for(j=0;j<mkx+1;++j)
			{
				flxrain[j] = 0.0;
				flxsnow[j] = 0.0;
			}
			for(j=0;j<mkx;++j)
			{
				ntraprd[j] = 0.0;
				ntsnprd[j] = 0.0;
			}

			//lifeilifei
			for(k=mkx-1;k>=0;k--)//! 'k' is a layer index : 'mkx'('1') is the top ('bottom') layer
			{

				// ! ----------------------------------------------------------------------------- !
				// ! flxsntm [kg/m2/s] : Downward snow flux at the top of each layer after melting.! 
				// ! snowmlt [kg/kg/s] : Snow melting tendency.                                    !
				// ! Below allows melting of snow when it goes down into the warm layer below.     !
				// ! ----------------------------------------------------------------------------- !

				if(t0[k] > 273.16)
				{
					snowmlt = max( 0.0, flxsnow[k+1] * g / dp0[k] );
				}
				else
				{
					snowmlt = 0.0;
				}

				// ! ----------------------------------------------------------------- !
				// ! Evaporation rate of 'rain' and 'snow' in the layer k, [ kg/kg/s ] !
				// ! where 'rain' and 'snow' are coming down from the upper layers.    !
				// ! I used the same evaporative efficiency both for 'rain' and 'snow'.!
				// ! Note that evaporation is not allowed in the layers 'k >= krel' by !
				// ! assuming that inside of cumulus cloud, across which precipitation !
				// ! is falling down, is fully saturated.                              !
				// ! The asumptions in association with the 'evplimit_rain(snow)' are  !
				// !   1. Do not allow evaporation to supersate the layer              !
				// !   2. Do not evaporate more than the flux falling into the layer   !
				// !   3. Total evaporation cannot exceed the input total surface flux !
				// ! ----------------------------------------------------------------- !

				qsat_arglf[0] = t0[k];
				pelf[0] = p0[k];
				status = qsat(qsat_arglf,pelf,es,qs,gam, 1);          
				subsat = max( ( 1.0 - qv0[k]/qs[0] ), 0.0 );
				if(noevap_krelkpen)
				{
					if( k >= krel ) subsat = 0.0;
				}

				evprain  = kevp * subsat * sqrt(flxrain[k+1]+snowmlt*dp0[k]/g); 
				evpsnow  = kevp * subsat * sqrt(max(flxsnow[k+1]-snowmlt*dp0[k]/g,0.0));

				evplimit = max( 0.0, ( qw0_in[k*iend+i] - qv0[k] ) / dt );
				
				evplimit_rain = min( evplimit,      ( flxrain[k+1] + snowmlt * dp0[k] / g ) * g / dp0[k] );
				evplimit_rain = min( evplimit_rain, ( rainflx - evpint_rain ) * g / dp0[k] );
				evprain = max(0.0,min( evplimit_rain, evprain ));

				evplimit_snow = min( evplimit,   max( flxsnow[k+1] - snowmlt * dp0[k] / g , 0.0 ) * g / dp0[k] );
				evplimit_snow = min( evplimit_snow, ( snowflx - evpint_snow ) * g / dp0[k] );
				evpsnow = max(0.0,min( evplimit_snow, evpsnow ));

				if( ( evprain + evpsnow ) > evplimit )
				{
					tmp1 = evprain * evplimit / ( evprain + evpsnow );
					tmp2 = evpsnow * evplimit / ( evprain + evpsnow );
					evprain = tmp1;
					evpsnow = tmp2;
				}

				evapc[k] = evprain + evpsnow;

				// ! ------------------------------------------------------------- !
				// ! Vertically-integrated evaporative fluxes of 'rain' and 'snow' !
				// ! ------------------------------------------------------------- !

				evpint_rain = evpint_rain + evprain * dp0[k] / g;
				evpint_snow = evpint_snow + evpsnow * dp0[k] / g;

				// ! -------------------------------------------------------------- !
				// ! Net 'rain' and 'snow' production rate in the layer [ kg/kg/s ] !
				// ! -------------------------------------------------------------- !

				ntraprd[k] = qrten[k] - evprain + snowmlt;
				ntsnprd[k] = qsten[k] - evpsnow - snowmlt;

				// ! -------------------------------------------------------------------------------- !
				// ! Downward fluxes of 'rain' and 'snow' fluxes at the base of the layer [ kg/m2/s ] !
				// ! Note that layer index increases with height.                                     !
				// ! -------------------------------------------------------------------------------- !

				flxrain[k] = flxrain[k+1] + ntraprd[k] * dp0[k] / g;
				flxsnow[k] = flxsnow[k+1] + ntsnprd[k] * dp0[k] / g;
				flxrain[k] = max( flxrain[k], 0.0 );
				if( flxrain[k] == 0.0 ) ntraprd[k] = -flxrain[k+1] * g / dp0[k];
				flxsnow[k] = max( flxsnow[k], 0.0 );         
				if( flxsnow[k] == 0.0 ) ntsnprd[k] = -flxsnow[k+1] * g / dp0[k];

				// ! ---------------------------------- !
				// ! Calculate thermodynamic tendencies !
				// ! --------------------------------------------------------------------------- !
				// ! Note that equivalently, we can write tendency formula of 'sten' and 'slten' !
				// ! by 'sten(k)  = sten(k) - xlv*evprain  - xls*evpsnow - (xls-xlv)*snowmlt' &  !
				// !    'slten(k) = sten(k) - xlv*qlten(k) - xls*qiten(k)'.                      !
				// ! The above formula is equivalent to the below formula. However below formula !
				// ! is preferred since we have already imposed explicit constraint on 'ntraprd' !
				// ! and 'ntsnprd' in case that flxrain(k-1) < 0 & flxsnow(k-1) < 0._r8          !
				// ! Note : In future, I can elborate the limiting of 'qlten','qvten','qiten'    !
				// !        such that that energy and moisture conservation error is completely  !
				// !        suppressed.                                                          !
				// ! Re-storation to the positive condensate will be performed later below       !
				// ! --------------------------------------------------------------------------- !

				qlten[k] = qlten[k] - qrten[k];
				qiten[k] = qiten[k] - qsten[k];
				qvten[k] = qvten[k] + evprain  + evpsnow;
				qtten[k] = qlten[k] + qiten[k] + qvten[k];
				if( ( (qv0[k] + qvten[k]*dt ) < qmin[0]) || 
				    ( (ql0[k] + qlten[k]*dt ) < qmin[1]) || 
				    ( (qi0[k] + qiten[k]*dt ) < qmin[2]) )
				{
///					limit_negcon[i] = 1.0;
				}
				sten[k]  = sten[k] - xlv*evprain  - xls*evpsnow - (xls-xlv)*snowmlt;
				slten[k] = sten[k] - xlv*qlten[k] - xls*qiten[k];

				// !  slten(k) = slten(k) + xlv * ntraprd(k) + xls * ntsnprd(k)         
				// !  sten(k)  = slten(k) + xlv * qlten(k)   + xls * qiten(k)
			}

			// ! ------------------------------------------------------------- !
			// ! Calculate final surface flux of precipitation, rain, and snow !
			// ! Convert unit to [m/s] for use in 'check_energy_chng'.         !  
			// ! ------------------------------------------------------------- !

			precip  = ( flxrain[0] + flxsnow[0] ) / 1000.0;
			snow    =   flxsnow[0] / 1000.0;			
			// ! --------------------------------------------------------------------------- !
			// ! Until now, all the calculations are done completely in this shallow cumulus !
			// ! scheme. If you want to use this cumulus scheme other than CAM3, then do not !
			// ! perform below block. However, for compatible use with the other subroutines !
			// ! in CAM3, I should subtract the effect of 'qc(k)' ('rliq') from the tendency !
			// ! equation in each layer, since this effect will be separately added later in !
			// ! in 'stratiform_tend' just after performing sediment process there. In order !
			// ! to be consistent with 'stratiform_tend', just subtract qc(k)  from tendency !
			// ! equation of each layer, but do not add it to the 'precip'. Apprently,  this !
			// ! will violate energy and moisture conservations.    However, when performing !
			// ! conservation check in 'tphysbc.F90' just after 'convect_shallow_tend',   we !
			// ! will add 'qc(k)' ( rliq ) to the surface flux term just for the purpose  of !
			// ! passing the energy-moisture conservation check. Explicit adding-back of 'qc'!
			// ! to the individual layer tendency equation will be done in 'stratiform_tend' !
			// ! after performing sediment process there. Simply speaking, in 'tphysbc' just !
			// ! after 'convect_shallow_tend', we will dump 'rliq' into surface as a  'rain' !
			// ! in order to satisfy energy and moisture conservation, and  in the following !
			// ! 'stratiform_tend', we will restore it back to 'qlten(k)' ( 'ice' will go to !  
			// ! 'water' there) from surface precipitation. This is a funny but conceptually !
			// ! entertaining procedure. One concern I have for this complex process is that !
			// ! output-writed stratiform precipitation amount will be underestimated due to !
			// ! arbitrary subtracting of 'rliq' in stratiform_tend, where                   !
			// ! ' prec_str = prec_sed + prec_pcw - rliq' and 'rliq' is not real but fake.   ! 
			// ! However, as shown in 'srfxfer.F90', large scale precipitation amount (PRECL)!
			// ! that is writed-output is corrected written since in 'srfxfer.F90',  PRECL = !
			// ! 'prec_sed + prec_pcw', without including 'rliq'. So current code is correct.!
			// ! Note also in 'srfxfer.F90', convective precipitation amount is 'PRECC =     ! 
			// ! prec_zmc(i) + prec_cmf(i)' which is also correct.                           !
			// ! --------------------------------------------------------------------------- !

			for(k=1;k<=kpen;++k)
			{
				qtten[k-1] = qtten[k-1] - qc[k-1];
				qlten[k-1] = qlten[k-1] - qc_l[k-1];
				qiten[k-1] = qiten[k-1] - qc_i[k-1];
				slten[k-1] = slten[k-1] + ( xlv * qc_l[k-1] + xls * qc_i[k-1] );
				// ! ---------------------------------------------------------------------- !
				// ! Since all reserved condensates will be treated  as liquid water in the !
				// ! 'check_energy_chng' & 'stratiform_tend' without an explicit conversion !
				// ! algorithm, I should consider explicitly the energy conversions between !
				// ! 'ice' and 'liquid' - i.e., I should convert 'ice' to 'liquid'  and the !
				// ! necessary energy for this conversion should be subtracted from 'sten'. ! 
				// ! Without this conversion here, energy conservation error come out. Note !
				// ! that there should be no change of 'qvten(k)'.                          !
				// ! ---------------------------------------------------------------------- !
				sten[k-1]  = sten[k-1]  - ( xls - xlv ) * qc_i[k-1];
			}
			//for(k=1;k<=kpen;++k)
			//{
			//	qlten[k-1] = qlten[k-1]/1000000000;
			//}

			// ! --------------------------------------------------------------- !
			// ! Prevent the onset-of negative condensate at the next time step  !
			// ! Potentially, this block can be moved just in front of the above !
			// ! block.                                                          ! 
			// ! --------------------------------------------------------------- !

			// ! Modification : I should check whether this 'positive_moisture_single' routine is
			// !                consistent with the one used in UW PBL and cloud macrophysics schemes.
			// ! Modification : Below may overestimate resulting 'ql, qi' if we use the new 'qc_l', 'qc_i'
			// !                in combination with the original computation of qlten, qiten. However,
			// !                if we use new 'qlten,qiten', there is no problem.

			for(j=0;j<mkx;++j)
			{
//		if(i==22) printf("j=%d,qvten=%e\n",j,qvten[j]);
				qv0_star[j] = qv0[j] + qvten[j] * dt;
				ql0_star[j] = ql0[j] + qlten[j] * dt;
				qi0_star[j] = qi0[j] + qiten[j] * dt;
				s0_star[j]  =  s0[j] +  sten[j] * dt;
			}

			positive_moisture_single(xlv, xls, mkxtemp, dt, qmin[0], qmin[1], qmin[2], dp0, qv0_star, ql0_star, qi0_star, s0_star, qvten, qlten, qiten, sten );
			for(j=0;j<mkx;++j)
			{
				qtten[j]    = qvten[j] + qlten[j] + qiten[j];
				slten[j]    = sten[j]  - xlv * qlten[j] - xls * qiten[j];
			}

			// ! --------------------- !
			// ! Tendencies of tracers !
			// ! --------------------- !
			
			for(m=3;m<ncnst;++m)
			{
				if( ((m+1) != ixnumliq) && ((m+1) != ixnumice) )
				{
					trmin = qmin[m];
//条件编译
#ifdef MODAL_AERO
					for(mm=1;mm<=ntot_amode;++mm)
					{
						if( (m+1) == numptr_amode[mm-1] )
						{
							trmin = 1.0e-5;
							goto lable55;
						}
					}
lable55:
				//	 aaa=1;//continue;
#endif
					for(j=0;j<mkx+1;++j)
					{
						trflx_d[j] = 0.0;
						trflx_u[j] = 0.0;
					}
					for(k=1;k<mkx;++k)
					{
						if(false)//if(__constituents_MOD_cnst_get_type_byind(m+1) == "wet") //! cnst_type(m) .eq. 'wet' 
						{
							pdelx = dp0[k-1];
						}
						else
						{
							pdelx = dpdry0[k-1];
						}
						km1 = k - 1;
						dum = ( tr0[m*mkx+(k-1)] - trmin ) *  pdelx / g / dt + trflx[m*(mkx+1)+km1] - trflx[m*(mkx+1)+k] + trflx_d[km1];
             			trflx_d[k] = min( 0.0, dum );
// if((i == 7)&&(k == 1)&&(m == 16) )
// {
// 	printf("-------------------------------------------------------------------\n");
// 	printf("dp0(1) =%e, dpdry0(1) =%e\n",dp0[0],dpdry0[0]);
// 	printf("tr0(7,1,16) =%15.13e\n",tr0[m*mkx*iend+(k-1)*iend+i]);
// 	printf("pdelx=%e, dum=%e\n",pdelx,dum);
// 	printf("-------------------------------------------------------------------\n");
// }
						// !======================== zhh debug 2012-02-09 =======================     
						// !              if (i==8 .and. k==1 .and. m==17) then
						// !                 print*, 'dp0(1) =', dp0(1), ' dpdry0(1) =', dpdry0(1)
						// !                 print*, 'pdelx =', pdelx, ' dum =', dum
						// !                 print*, '-------------------------------------------------------------------'
						// !              end if
						// !======================== zhh debug 2012-02-09 ======================= 
					}
					for(k=mkx;k>=2;k--)
					{
						if(false)//if(__constituents_MOD_cnst_get_type_byind(m+1) == "wet") //!cnst_type(m) .eq. 'wet' 
						{
							pdelx = dp0[k-1];
						}
						else
						{
							pdelx = dpdry0[k-1];
						}
						km1 = k - 1;
						dum = ( tr0[m*mkx+(k-1)] - trmin ) * pdelx / g / dt + trflx[m*(mkx+1)+km1] - trflx[m*(mkx+1)+k] + 
                                                           trflx_d[km1] - trflx_d[k] - trflx_u[k]; 
             			trflx_u[km1] = max( 0.0, -dum ); 
					}
					for(k=1;k<=mkx;++k)
					{
						if(false)//if(__constituents_MOD_cnst_get_type_byind(m+1) == "wet") //! cnst_type(m) .eq. 'wet'
						{
							pdelx = dp0[k-1];
						}
						else
						{
							pdelx = dpdry0[k-1];
						}
						km1 = k - 1;
						// ! Check : I should re-check whether '_u', '_d' are correctly ordered in 
						// !         the below tendency computation.
						trten[m*mkx+(k-1)] = ( trflx[m*(mkx+1)+km1] - trflx[m*(mkx+1)+k] +  
									 					   trflx_d[km1] - trflx_d[k] + 
									   					   trflx_u[km1] - trflx_u[k] ) * g / pdelx;
						
// if((i == 7)&&(k == 1)&&(m == 16) )
// {
// 	printf("trflx[16,0,7]=%e, trflx[16,1,7]=%e\n",trflx[16*(mkx+1)*iend+0*iend+7],trflx[16*(mkx+1)*iend+1*iend+7]);
// 	printf("trflx_d[0]=%e, trflx_d[1]=%e\n",trflx_d[0],trflx_d[1]);
// 	printf("trflx_u[0]=%e, trflx_u[1]=%e\n",trflx_u[0],trflx_u[1]);
// 	printf("pdelx=%e\n",pdelx);
// 	printf("-------------------------------------------------------------------\n");
// }
						// !======================== zhh debug 2012-02-09 =======================     
						// !!              if (i==8 .and. k==1 .and. m==17) then
						// !!                 print*, 'trflx(0,17) =', trflx(0,17), ' trflx(1,17) =', trflx(1,17)
						// !!                 print*, 'trflx_d(0) =', trflx_d(0), ' trflx_d(1) =', trflx_d(1)
						// !!                 print*, 'trflx_u(0) =', trflx_u(0), ' trflx_u(1) =', trflx_u(1)
						// !!                 print*, 'pdelx =', pdelx
						// !!                 print*, '-------------------------------------------------------------------'
						// !!              end if
						// !======================== zhh debug 2012-02-09 =======================   
					}
				}
			}
			// !======================== zhh debug 2012-02-09 =======================     
			// !       if (i==8) then
			// !          print*, '3rd: trten(1,17) =', trten(1,17)
			// !          print*, '-------------------------------------------------------------------'
			// !       end if
			// !======================== zhh debug 2012-02-09 ======================= 

			// ! ---------------------------------------------------------------- !
			// ! Cumpute default diagnostic outputs                               !
			// ! Note that since 'qtu(krel-1:kpen-1)' & 'thlu(krel-1:kpen-1)' has !
			// ! been adjusted after detraining cloud condensate into environment ! 
			// ! during cumulus updraft motion,  below calculations will  exactly !
			// ! reproduce in-cloud properties as shown in the output analysis.   !
			// ! ---------------------------------------------------------------- ! 
	//printf("lifeilifeilifei2\n");
			conden(prel,thlu[krel-1],qtu[krel-1],&thj,&qvj,&qlj,&qij,&qse,&id_check);
			if(id_check == 1)
			{
				exit_conden[i] = 1.0;
				id_exit = true;
				goto lable333;
			}
			qcubelow = qlj + qij;
			qlubelow = qlj;       
			qiubelow = qij;       
			rcwp     = 0.0;
			rlwp     = 0.0;
			riwp     = 0.0;

			// ! --------------------------------------------------------------------- !
			// ! In the below calculations, I explicitly considered cloud base ( LCL ) !
			// ! and cloud top height ( ps0(kpen-1) + ppen )                           !
			// ! ----------------------------------------------------------------------! 
			for(k=krel;k<=kpen;++k) //! This is a layer index
			{
				// ! ------------------------------------------------------------------ ! 
				// ! Calculate cumulus condensate at the upper interface of each layer. !
				// ! Note 'ppen < 0' and at 'k=kpen' layer, I used 'thlu_top'&'qtu_top' !
				// ! which explicitly considered zero or non-zero 'fer(kpen)'.          !
				// ! ------------------------------------------------------------------ ! 
				if( k == kpen )
				{
					conden(ps0[k-1]+ppen,thlu_top,qtu_top,&thj,&qvj,&qlj,&qij,&qse,&id_check);
				}
				else
				{
					conden(ps0[k],thlu[k],qtu[k],&thj,&qvj,&qlj,&qij,&qse,&id_check);
				}
				if(id_check == 1)
				{
					exit_conden[i] = 1.0;
					id_exit = true;
					goto lable333;
				}
				// ! ---------------------------------------------------------------- !
				// ! Calculate in-cloud mean LWC ( qlu(k) ), IWC ( qiu(k) ),  & layer !
				// ! mean cumulus fraction ( cufrc(k) ),  vertically-integrated layer !
				// ! mean LWP and IWP. Expel some of in-cloud condensate at the upper !
				// ! interface if it is largr than criqc. Note cumulus cloud fraction !
				// ! is assumed to be twice of core updraft fractional area. Thus LWP !
				// ! and IWP will be twice of actual value coming from our scheme.    !
				// ! ---------------------------------------------------------------- !
				qcu[k-1]   = 0.5 * ( qcubelow + qlj + qij );
				qlu[k-1]   = 0.5 * ( qlubelow + qlj );
				qiu[k-1]   = 0.5 * ( qiubelow + qij );
				cufrc[k-1] = ( ufrc[k-1] + ufrc[k] );
				if(k == krel)
				{
					cufrc[k-1] = ( ufrclcl + ufrc[k] )*( prel - ps0[k] )/( ps0[k-1] - ps0[k] );
				}
				else if(k == kpen)
				{
					cufrc[k-1] = ( ufrc[k-1] + 0.0 )*( -ppen )        /( ps0[k-1] - ps0[k] );
					if( (qlj + qij) > criqc )
					{
						qcu[k-1] = 0.5 * ( qcubelow + criqc );
						qlu[k-1] = 0.5 * ( qlubelow + criqc * qlj / ( qlj + qij ) );
						qiu[k-1] = 0.5 * ( qiubelow + criqc * qij / ( qlj + qij ) );
					}
				}
				rcwp = rcwp + ( qlu[k-1] + qiu[k-1] ) * ( ps0[k-1] - ps0[k] ) / g * cufrc[k-1];
				rlwp = rlwp +   qlu[k-1]            * ( ps0[k-1] - ps0[k] ) / g * cufrc[k-1];
				riwp = riwp +   qiu[k-1]            * ( ps0[k-1] - ps0[k] ) / g * cufrc[k-1];
				qcubelow = qlj + qij;
				qlubelow = qlj;
				qiubelow = qij;
			}
			// ! ------------------------------------ !      
			// ! Cloud top and base interface indices !
			// ! ------------------------------------ !
			cnt = (double)kpen;    //real( kpen, r8 )
			cnb = (double)(krel - 1);    //real( krel - 1, r8 )

			// ! ------------------------------------------------------------------------- !
			// ! End of formal calculation. Below blocks are for implicit CIN calculations ! 
			// ! with re-initialization and save variables at iter_cin = 1._r8             !
			// ! ------------------------------------------------------------------------- !

			// ! --------------------------------------------------------------- !
			// ! Adjust the original input profiles for implicit CIN calculation !
			// ! --------------------------------------------------------------- !
			
			if( iter != iter_cin )
			{
				// ! ------------------------------------------------------------------- !
				// ! Save the output from "iter_cin = 1"                                 !
				// ! These output will be writed-out if "iter_cin = 1" was not performed !
				// ! for some reasons.                                                   !
				// ! ------------------------------------------------------------------- !

				for(j=0;j<mkx;++j)
				{
		//		if(i==22)	printf("j=%d,qv0[j]=%e,qvten[j]=%e,dt=%e\n",j,qv0[j],qvten[j],dt);
					qv0_s[j]           = qv0[j] + qvten[j] * dt;
					ql0_s[j]           = ql0[j] + qlten[j] * dt;
					qi0_s[j]           = qi0[j] + qiten[j] * dt;
					s0_s[j]            = s0[j]  +  sten[j] * dt;
					u0_s[j]            = u0[j]  +  uten[j] * dt;
					v0_s[j]            = v0[j]  +  vten[j] * dt;
				}
				for(j=0;j<mkx;++j)
				{
					qt0_s[j]           = qv0_s[j] + ql0_s[j] + qi0_s[j];
					t0_s[j]            = t0[j]  +  sten[j] * dt / cp;
				}
				for(k=0;k<ncnst;++k)
				{
					for(j=0;j<mkx;++j)
					{
//						tr0_s[k*mkx*iend+j*iend+i] = tr0[k*mkx+j] + trten[k*mkx*iend+j*iend+i] * dt;
					}
				}

				for(j=0;j<mkx+1;++j)
				{
					umf_s[j]          = umf[j];

					slflx_s[j]        = slflx[j];  
					qtflx_s[j]        = qtflx[j];

///					ufrc_s[j]         = ufrc[j]; 
	  
///					uflx_s[j]         = uflx[j];  
///					vflx_s[j]         = vflx[j]; 

					wu_s[j]           = wu[j];
///					qtu_s[j*iend+i]          = qtu[j];
///					thlu_s[j*iend+i]         = thlu[j];
///					thvu_s[j*iend+i]         = thvu[j];
///					uu_s[j*iend+i]           = uu[j];
///					vu_s[j*iend+i]           = vu[j];
///					qtu_emf_s[j*iend+i]      = qtu_emf[j];
///					thlu_emf_s[j*iend+i]     = thlu_emf[j];
///					uu_emf_s[j*iend+i]       = uu_emf[j];
///					vu_emf_s[j*iend+i]       = vu_emf[j];
///					uemf_s[j*iend+i]         = uemf[j];

///					flxrain_s[j*iend+i]      = flxrain[j];
///					flxsnow_s[j*iend+i]      = flxsnow[j];
				}
				for(j=0;j<mkx;++j)
				{
					qvten_s[j]         = qvten[j];
					qlten_s[j]         = qlten[j]; 
					qiten_s[j]         = qiten[j];
					sten_s[j]          = sten[j];
					uten_s[j]          = uten[j];  
					vten_s[j]          = vten[j];
					qrten_s[j]         = qrten[j];
					qsten_s[j]         = qsten[j]; 

					evapc_s[j]         = evapc[j];

					cufrc_s[j]         = cufrc[j];

					qcu_s[j]           = qcu[j];
					qlu_s[j]           = qlu[j];  
					qiu_s[j]           = qiu[j];  
///					fer_s[j*iend+i]           = fer[j];  
///					fdr_s[j*iend+i]           = fdr[j]; 

					qc_s[j]            = qc[j];

///					qtten_s[j*iend+i]         = qtten[j];
///					slten_s[j*iend+i]         = slten[j];

///					dwten_s[j*iend+i]         = dwten[j];
///					diten_s[j*iend+i]         = diten[j];
	
///					ntraprd_s[j*iend+i]       = ntraprd[j];
///					ntsnprd_s[j*iend+i]       = ntsnprd[j];
		  
///					excessu_arr_s[j*iend+i]   = excessu_arr[j];
///					excess0_arr_s[j*iend+i]   = excess0_arr[j];
///					xc_arr_s[j*iend+i]        = xc_arr[j];
///					aquad_arr_s[j*iend+i]     = aquad_arr[j];
///					bquad_arr_s[j*iend+i]     = bquad_arr[j];
///					cquad_arr_s[j*iend+i]     = cquad_arr[j];
///					bogbot_arr_s[j*iend+i]    = bogbot_arr[j];
///					bogtop_arr_s[j*iend+i]    = bogtop_arr[j];	
				}

				precip_s              = precip;
				snow_s                = snow;
				
				cush_s                = cush;
				  
				cin_s                 = cin;
				cinlcl_s              = cinlcl;
				cbmf_s                = cbmf;
				rliq_s                = rliq;
				
				cnt_s                 = cnt;
				cnb_s                 = cnb;
	 
				ufrcinvbase_s         = ufrcinvbase;
				ufrclcl_s             = ufrclcl; 
				winvbase_s            = winvbase;
				wlcl_s                = wlcl;
				plcl_s                = plcl;
				pinv_s                = ps0[kinv-1];
				plfc_s                = plfc;        
				pbup_s                = ps0[kbup];
				ppen_s                = ps0[kpen-1] + ppen;        
				qtsrc_s               = qtsrc;
				thlsrc_s              = thlsrc;
				thvlsrc_s             = thvlsrc;
				emfkbup_s             = emf[kbup];
				cbmflimit_s           = cbmflimit;
				tkeavg_s              = tkeavg;
				zinv_s                = zs0[kinv-1];
				rcwp_s                = rcwp;
				rlwp_s                = rlwp;
				riwp_s                = riwp;
	  
				for(k=0;k<ncnst;++k)
				{
					for(j=0;j<mkx;++j)
					{
						trten_s[k*mkx+j]    = trten[k*mkx+j];
					}
				}
				for(k=0;k<ncnst;++k)
				{
					for(j=0;j<mkx+1;++j)
					{
//						trflx_s[k*(mkx+1)*iend+j*iend+i]   = trflx[k*(mkx+1)*iend+j*iend+i];
//						tru_s[k*(mkx+1)*iend+j*iend+i]     = tru[k*(mkx+1)*iend+j*iend+i];
//						tru_emf_s[k*(mkx+1)*iend+j*iend+i] = tru_emf[k*(mkx+1)*iend+j*iend+i];
					}
				}

				// ! ----------------------------------------------------------------------------- ! 
				// ! Recalculate environmental variables for new cin calculation at "iter_cin = 2" ! 
				// ! using the updated state variables. Perform only for variables necessary  for  !
				// ! the new cin calculation.                                                      !
				// ! ----------------------------------------------------------------------------- !
// for(j=0;j<mkx;++j)
// 	if(i==22) printf("j=%d,qv0[j]=%e,ql0[j]=%e,qi0[j]=%e\n",j,qv0[j],ql0[j],qi0[j]);
				for(j=0;j<mkx;++j)
				{
					qv0[j]   = qv0_s[j];
					ql0[j]   = ql0_s[j];
					qi0[j]   = qi0_s[j];
					s0[j]    = s0_s[j];
					t0[j]    = t0_s[j];
				}
				for(j=0;j<mkx;++j)
				{
					qt0[j]   = (qv0[j] + ql0[j] + qi0[j]);
					thl0[j]  = (t0[j] - xlv*ql0[j]/cp - xls*qi0[j]/cp)/exn0[j];
				}
				for(j=0;j<mkx;++j)
				{
					thvl0[j] = (1.0 + zvir*qt0[j])*thl0[j];
				}
// for(m=0;m<mkx;++m)
// {
// 	if(i==22) printf("m=%d,qt0[m]=%f\n",m,qt0[m]);
// }
				slope(ssthl0,mkxtemp,thl0,p0); //! Dimension of ssthl0(:mkx) is implicit
				slope(ssqt0,mkxtemp,qt0 ,p0);
				slope(ssu0,mkxtemp,u0  ,p0);
				slope(ssv0,mkxtemp,v0  ,p0);
			//	double temp2_sstr0[mkx];//temp2_tr0[mkx];
				for(k=0;k<ncnst;++k)
				{
					for(j=0;j<mkx;++j)
						temp_tr0[j] = tr0[k*mkx+j];
					slope(temp_sstr0,mkxtemp,temp_tr0,p0);
					for(j=0;j<mkx;++j)
						sstr0[k*mkx+j] = temp_sstr0[j];
				}

				for(k=0;k<mkx;++k)
				{
					thl0bot = thl0[k] + ssthl0[k] * ( ps0[k] - p0[k] );
					qt0bot  = qt0[k]  + ssqt0[k]  * ( ps0[k] - p0[k] );
//if(i==22) printf("k=%d,qt0[k]=%e,ssqt0[k]=%e,p0[k]=%e\n",k,qt0[k],ssqt0[k],p0[k]);
					conden(ps0[k],thl0bot,qt0bot,&thj,&qvj,&qlj,&qij,&qse,&id_check);
					if(id_check == 1)
					{
						exit_conden[i] = 1.0;
						id_exit = true;
						goto lable333;
					}
					thv0bot[k]  = thj * ( 1.0 + zvir*qvj - qlj - qij );
					thvl0bot[k] = thl0bot * ( 1.0 + zvir*qt0bot );

					thl0top = thl0[k] + ssthl0[k] * ( ps0[k+1] - p0[k] );
					qt0top  =  qt0[k] + ssqt0[k]  * ( ps0[k+1] - p0[k] );
//if(i==22) printf("k=%d,ps0[k+1]=%e,thl0top=%e,qt0top=%e\n",k,ps0[k+1],thl0top,qt0top);
					conden(ps0[k+1],thl0top,qt0top,&thj,&qvj,&qlj,&qij,&qse,&id_check);
					if(id_check == 1)
					{
						exit_conden[i] = 1.0;
						id_exit = true;
						goto lable333;
					}
					thv0top[k]  = thj * ( 1.0 + zvir*qvj - qlj - qij );
					thvl0top[k] = thl0top * ( 1.0 + zvir*qt0top );
//if(i==22)	printf("k=%d,thj=%e,zvir=%e,qvj=%e,qlj=%e,qij=%e\n",k,thj,zvir,qvj,qlj,qij);
				}	

			} //! End of 'if(iter .ne. iter_cin)' if sentence. 
		
		} //! End of implicit CIN loop (cin_iter) 
		
		// ! ----------------------- !
		// ! Update Output Variables !
		// ! ----------------------- !

		for(j=0;j<mkx+1;++j)
		{
			umf_out[j*iend+i]             = umf[j];
//!wxc zmh
			wu_out[j*iend+i]              = wu[j];

			slflx_out[j*iend+i]           = slflx[j];
			qtflx_out[j*iend+i]           = qtflx[j];
			//!the indices are not reversed, these variables go into compute_mcshallow_inv, this is why they are called "flxprc1" and "flxsnow1".
			flxprc1_out[j*iend+i]         = flxrain[j] + flxsnow[j];
			flxsnow1_out[j*iend+i]        = flxsnow[j];

		}
		for(j=0;j<mkx;++j)
		{
			qvten_out[j*iend+i]            = qvten[j];
			qlten_out[j*iend+i]            = qlten[j];
			qiten_out[j*iend+i]            = qiten[j];
			sten_out[j*iend+i]             = sten[j];
			uten_out[j*iend+i]             = uten[j];
			vten_out[j*iend+i]             = vten[j];
			qrten_out[j*iend+i]            = qrten[j];
			qsten_out[j*iend+i]            = qsten[j];

			evapc_out[j*iend+i]            = evapc[j];
			cufrc_out[j*iend+i]            = cufrc[j];
			qcu_out[j*iend+i]              = qcu[j];
			qlu_out[j*iend+i]              = qlu[j];
			qiu_out[j*iend+i]              = qiu[j];

			qc_out[j*iend+i]               = qc[j];
		}

		precip_out[i]                = precip;
		snow_out[i]                  = snow;

		cush_inout[i]                = cush;
		cbmf_out[i]                  = cbmf;
		rliq_out[i]                  = rliq;
		
		cnt_out[i]                   = cnt;
		cnb_out[i]                   = cnb;
   
		for(k=0;k<ncnst;++k)
		{
			for(j=0;j<mkx;++j)
			{
				trten_out[k*mkx*iend+j*iend+i] = trten[k*mkx+j];
			}
		}

		// !======================== zhh debug 2012-02-09 =======================     
		// !    if (i==8) then
		// !       print*, '4th: trten_out(8,1,17) =', trten_out(8,1,17)
		// !       print*, '-------------------------------------------------------------------'
		// !    end if
		// !======================== zhh debug 2012-02-09 =======================   

		// ! ------------------------------------------------- !
		// ! Below are specific diagnostic output for detailed !
		// ! analysis of cumulus scheme                        !
		// ! ------------------------------------------------- !

		for(j=mkx-1;j>=0;j--)
		{
            int jtemp=mkx-1-j;

//			fer_out[j*iend+i]          = fer[jtemp];  
//			fdr_out[j*iend+i]          = fdr[jtemp]; 

//			qtten_out[j*iend+i]        = qtten[jtemp];
//			slten_out[j*iend+i]        = slten[jtemp];

///			dwten_out[j*iend+i]        = dwten[jtemp];
///			diten_out[j*iend+i]        = diten[jtemp];

///			ntraprd_out[j*iend+i]      = ntraprd[jtemp];
///			ntsnprd_out[j*iend+i]      = ntsnprd[jtemp];
	   
///			excessu_arr_out[j*iend+i]  = excessu_arr[jtemp];
///			excess0_arr_out[j*iend+i]  = excess0_arr[jtemp];
///			xc_arr_out[j*iend+i]       = xc_arr[jtemp];
///			aquad_arr_out[j*iend+i]    = aquad_arr[jtemp];
///			bquad_arr_out[j*iend+i]    = bquad_arr[jtemp];
///			cquad_arr_out[j*iend+i]    = cquad_arr[jtemp];
///			bogbot_arr_out[j*iend+i]   = bogbot_arr[jtemp];
///			bogtop_arr_out[j*iend+i]   = bogtop_arr[jtemp];
		}

		for(j=mkx;j>=0;j--)
		{
			int jtemp = mkx-j;

//			ufrc_out[j*iend+i]         = ufrc[jtemp];
//			uflx_out[j*iend+i]         = uflx[jtemp];  
//			vflx_out[j*iend+i]         = vflx[jtemp];

///			wu_out[j*iend+i]           = wu[jtemp];
///			qtu_out[j*iend+i]          = qtu[jtemp];
///			thlu_out[j*iend+i]         = thlu[jtemp];
///			thvu_out[j*iend+i]         = thvu[jtemp];
///			uu_out[j*iend+i]           = uu[jtemp];
///			vu_out[j*iend+i]           = vu[jtemp];
///			qtu_emf_out[j*iend+i]      = qtu_emf[jtemp];
///			thlu_emf_out[j*iend+i]     = thlu_emf[jtemp];
///			uu_emf_out[j*iend+i]       = uu_emf[jtemp];
///			vu_emf_out[j*iend+i]       = vu_emf[jtemp];
///			uemf_out[j*iend+i]         = uemf[jtemp];

///			flxrain_out[j*iend+i]      = flxrain[jtemp];
///			flxsnow_out[j*iend+i]      = flxsnow[jtemp];
		}
 
//		cinh_out[i]                  = cin;
///		cinlclh_out[i]               = cinlcl;
		
///		ufrcinvbase_out[i]           = ufrcinvbase;
///		ufrclcl_out[i]               = ufrclcl; 
///		winvbase_out[i]              = winvbase;
///		wlcl_out[i]                  = wlcl;
///		plcl_out[i]                  = plcl;
///		pinv_out[i]                  = ps0[kinv-1];
		// plfc_out[i]                  = plfc;    
		// pbup_out[i]                  = ps0[kbup];        
		// ppen_out[i]                  = ps0[kpen-1] + ppen;            
		// qtsrc_out[i]                 = qtsrc;
		// thlsrc_out[i]                = thlsrc;
		// thvlsrc_out[i]               = thvlsrc;
		// emfkbup_out[i]               = emf[kbup];
		// cbmflimit_out[i]             = cbmflimit;
		// tkeavg_out[i]                = tkeavg;
		// zinv_out[i]                  = zs0[kinv-1];
		// rcwp_out[i]                  = rcwp;
		// rlwp_out[i]                  = rlwp;
		// riwp_out[i]                  = riwp;

		for(k=0;k<ncnst;k++)
		{
			for(j=mkx;j>=0;j--)
			{
				int jtemp = mkx-j;

//				trflx_out[k*(mkx+1)*iend+j*iend+i] = trflx[k*(mkx+1)*iend+jtemp*iend+i];  
//				tru_out[k*(mkx+1)*iend+j*iend+i]     = tru[k*(mkx+1)*iend+jtemp*iend+i];
//				tru_emf_out[k*(mkx+1)*iend+j*iend+i] = tru_emf[k*(mkx+1)*iend+jtemp*iend+i];
			}
		}

//	__syncthreads();

 lable333:	

		if(id_exit) //! Exit without cumulus convection
		{
		//	printf("lifei\n");
			exit_UWCu[i] = 1.0;

			// ! --------------------------------------------------------------------- !
			// ! Initialize output variables when cumulus convection was not performed.!
			// ! --------------------------------------------------------------------- !

			for(j=0;j<mkx+1;++j)
			{
				umf_out[j*iend+i]             = 0.0; 
//!wxc zmh
				wu_out[j*iend+i]              = 0.0;				

				slflx_out[j*iend+i]           = 0.0;
				qtflx_out[j*iend+i]           = 0.0;
			}
			for(j=0;j<mkx;++j)
			{
				qvten_out[j*iend+i]            = 0.0;
				qlten_out[j*iend+i]            = 0.0;
				qiten_out[j*iend+i]            = 0.0;
				sten_out[j*iend+i]             = 0.0;
				uten_out[j*iend+i]             = 0.0;
				vten_out[j*iend+i]             = 0.0;
				qrten_out[j*iend+i]            = 0.0;
				qsten_out[j*iend+i]            = 0.0;

				evapc_out[j*iend+i]            = 0.0;
				cufrc_out[j*iend+i]            = 0.0;
				qcu_out[j*iend+i]              = 0.0;
				qlu_out[j*iend+i]              = 0.0;
				qiu_out[j*iend+i]              = 0.0;

				qc_out[j*iend+i]               = 0.0;
			}

			for(j=mkx-1;j>=0;j--)
			{
//				fer_out[j*iend+i]          = 0.0;  
//				fdr_out[j*iend+i]          = 0.0; 

//				qtten_out[j*iend+i]        = 0.0;
//				slten_out[j*iend+i]        = 0.0;

///				dwten_out[j*iend+i]        = 0.0;    
///				diten_out[j*iend+i]        = 0.0;

///				ntraprd_out[j*iend+i]      = 0.0;    
///				ntsnprd_out[j*iend+i]      = 0.0;

///				excessu_arr_out[j*iend+i]  = 0.0;    
///				excess0_arr_out[j*iend+i]  = 0.0;    
///				xc_arr_out[j*iend+i]       = 0.0;    
///				aquad_arr_out[j*iend+i]    = 0.0;    
///				bquad_arr_out[j*iend+i]    = 0.0;    
///				cquad_arr_out[j*iend+i]    = 0.0;    
///				bogbot_arr_out[j*iend+i]   = 0.0;    
///				bogtop_arr_out[j*iend+i]   = 0.0; 
			}
			for(j=mkx;j>=0;j--)
			{
//				ufrc_out[j*iend+i]         = 0.0;
//				uflx_out[j*iend+i]         = 0.0;  
//				vflx_out[j*iend+i]         = 0.0; 

///				wu_out[j*iend+i]           = 0.0;    
///				qtu_out[j*iend+i]          = 0.0;        
///				thlu_out[j*iend+i]         = 0.0;         
///				thvu_out[j*iend+i]         = 0.0;         
///				uu_out[j*iend+i]           = 0.0;        
///				vu_out[j*iend+i]           = 0.0;        
///				qtu_emf_out[j*iend+i]      = 0.0;         
///				thlu_emf_out[j*iend+i]     = 0.0;         
///				uu_emf_out[j*iend+i]       = 0.0;          
///				vu_emf_out[j*iend+i]       = 0.0;    
///				uemf_out[j*iend+i]         = 0.0; 

///				flxrain_out[j*iend+i]      = 0.0;     
///				flxsnow_out[j*iend+i]      = 0.0; 
			}


			precip_out[i]                = 0.0;
			snow_out[i]                  = 0.0;

			cush_inout[i]                = -1.0;
			cbmf_out[i]                  = 0.0;   
			rliq_out[i]                  = 0.0;
			
			cnt_out[i]                   = 1.0;
			cnb_out[i]                   = (double)mkx; //real(mkx, r8)
	   
 
//			cinh_out[i]                  = -1.0; 
///			cinlclh_out[i]               = -1.0; 

 
	   
///			ufrcinvbase_out[i]           = 0.0; 
///			ufrclcl_out[i]               = 0.0; 
///			winvbase_out[i]              = 0.0;    
///			wlcl_out[i]                  = 0.0;    
///			plcl_out[i]                  = 0.0;    
///			pinv_out[i]                  = 0.0;     
			// plfc_out[i]                  = 0.0;     
			// pbup_out[i]                  = 0.0;    
			// ppen_out[i]                  = 0.0;    
			// qtsrc_out[i]                 = 0.0;    
			// thlsrc_out[i]                = 0.0;    
			// thvlsrc_out[i]               = 0.0;    
			// emfkbup_out[i]               = 0.0;
			// cbmflimit_out[i]             = 0.0;    
			// tkeavg_out[i]                = 0.0;    
			// zinv_out[i]                  = 0.0;    
			// rcwp_out[i]                  = 0.0;    
			// rlwp_out[i]                  = 0.0;    
			// riwp_out[i]                  = 0.0;    
	   
			for(k=0;k<ncnst;++k)
			{
				for(j=0;j<mkx;++j)
				{
					trten_out[k*mkx*iend+j*iend+i]       = 0.0;
				}
			}
			for(k=0;k<ncnst;++k)
			{
				for(j=mkx;j>=0;j--)
				{
//					trflx_out[k*(mkx+1)*iend+j*iend+i]   = 0.0;  
//					tru_out[k*(mkx+1)*iend+j*iend+i]     = 0.0;
//					tru_emf_out[k*(mkx+1)*iend+j*iend+i] = 0.0;
				}
			}

		}
	//	__syncthreads();

	} //! end of big i loop for each column.

	// ! ---------------------------------------- !
	// ! Writing main diagnostic output variables !
	// ! ---------------------------------------- !

	// !======================== zhh debug 2012-02-09 =======================     
	// !       print*, '---------- At the end of sub. compute_uwshcu ---------------'
	// !       print*, '5th: trten_out(8,1,17) =', trten_out(8,1,17)
	// !======================== zhh debug 2012-02-09 =======================   
	return;

  // }

}

// ! ------------------------------ !
// !                                ! 
// ! Beginning of subroutine blocks !
// !                                !
// ! ------------------------------ !

__device__ double exnf(double pressure)
{
	double eexnf;
	eexnf = pow((pressure/p00),rovcp);
	return eexnf;
}

__device__ double eerfc(double x)
{
	double result;
	int jint = 1;

	int i;
	double y,ysq,xnum,xden,del;

	// !------------------------------------------------------------------
	// !  Mathematical constants
	// !------------------------------------------------------------------

	double zero = 0.0e0;
	double four = 4.0e0;
	double one  = 1.0e0;
	double half = 0.5e0;
	double two  = 2.0e0;
	double sqrpi= 5.6418958354775628695e-1;
	double thresh = 0.46875e0;
	double sixten = 16.0e0;

// 	!------------------------------------------------------------------
// 	!  Machine-dependent constants: IEEE single precision values
// 	!------------------------------------------------------------------
// 	!S      real, parameter :: XINF   =  3.40E+38
// 	!S      real, parameter :: XNEG   = -9.382E0
// 	!S      real, parameter :: XSMALL =  5.96E-8 
// 	!S      real, parameter :: XBIG   =  9.194E0
// 	!S      real, parameter :: XHUGE  =  2.90E3
// 	!S      real, parameter :: XMAX   =  4.79E37

//    !------------------------------------------------------------------
//    !  Machine-dependent constants: IEEE double precision values
//    !------------------------------------------------------------------
	double xinf   =   1.79E308;
	double xneg   = -26.628e0;
	double xsmall =   1.11e-16;
	double xbig   =  26.543e0;
	double xhuge  =   6.71e7;
	double xmax   =   2.53e307;

	// !------------------------------------------------------------------
	// !  Coefficients for approximation to  erf  in first interval
	// !------------------------------------------------------------------
	double A[5]={ 3.16112374387056560E00, 1.13864154151050156E02, 
				  3.77485237685302021E02, 3.20937758913846947E03, 
				  1.85777706184603153E-1};
	double B[4]={ 2.36012909523441209E01, 2.44024637934444173E02, 
				  1.28261652607737228E03, 2.84423683343917062E03};
	
    // !------------------------------------------------------------------
    // !  Coefficients for approximation to  erfc  in second interval
	// !------------------------------------------------------------------
	double C[9] = { 5.64188496988670089E-1, 8.88314979438837594E00, 
					6.61191906371416295E01, 2.98635138197400131E02, 
					8.81952221241769090E02, 1.71204761263407058E03, 
					2.05107837782607147E03, 1.23033935479799725E03, 
					2.15311535474403846E-8};
	double D[8] = { 1.57449261107098347E01, 1.17693950891312499E02, 
					5.37181101862009858E02, 1.62138957456669019E03, 
					3.29079923573345963E03, 4.36261909014324716E03,
					3.43936767414372164E03, 1.23033935480374942E03};
	
	// !------------------------------------------------------------------
   	// !  Coefficients for approximation to  erfc  in third interval
    // !------------------------------------------------------------------
	double P[6] = { 3.05326634961232344E-1, 3.60344899949804439E-1, 
					1.25781726111229246E-1, 1.60837851487422766E-2, 
					6.58749161529837803E-4, 1.63153871373020978E-2 };
	double Q[5] = { 2.56852019228982242E00, 1.87295284992346047E00, 
					5.27905102951428412E-1, 6.05183413124413191E-2,
					2.33520497626869185E-3 };
	// !------------------------------------------------------------------
	y = fabs(x);
	if(y <= thresh)
	{
		// !------------------------------------------------------------------
		// !  Evaluate  erf  for  |X| <= 0.46875
		// !------------------------------------------------------------------
		ysq = zero;
		if(y > xsmall)
			ysq = y*y;
		xnum = A[4]*ysq;
		xden = ysq;
		for(i=1;i<=3;++i)
		{
			xnum = (xnum+A[i-1])*ysq;
			xden = (xden+B[i-1])*ysq;
		}
		result = x*(xnum + A[3]) / (xden + B[3]);
		if(jint != 0) result = one - result;
		if(jint == 2) result = exp(ysq) * result;
		goto llable80;
	}
	else if(y <= four)
	{
		// !------------------------------------------------------------------
		// !  Evaluate  erfc  for 0.46875 <= |X| <= 4.0
		// !------------------------------------------------------------------
		xnum = C[8]*y;
		xden = y;
		for(i=1;i<=7;++i)
		{
			xnum = (xnum+C[i-1])*y;
			xden = (xden+D[i-1])*y;
		}
		result = (xnum + C[7]) / (xden + D[7]);
		if(jint != 2)
		{
			ysq = (int)(y*sixten)/sixten;
			del = (y - ysq)*(y + ysq);
			result = exp(-ysq*ysq) * exp(-del) * result;
		}
	}
	else
	{
		// !------------------------------------------------------------------
		// !  Evaluate  erfc  for |X| > 4.0
		// !------------------------------------------------------------------
		result = zero;
		if(y >= xbig)
		{
			if((jint != 2) || (y >= xmax)) goto llable30;
			if(y >= xhuge)
			{
				result = sqrpi / y;
				goto llable30;
			}
		}
		ysq = one / (y*y);
		xnum= P[5]*ysq;
		xden= ysq;
		for(i=1;i<=4;++i)
		{
			xnum = (xnum + P[i-1]) * ysq;
			xden = (xden + Q[i-1]) * ysq;
		}
		result = ysq * (xnum + P[4]) / (xden + Q[4]);
		result = (sqrpi - result) / y;
		if(jint != 2)
		{
			ysq = (int)(y*sixten) / sixten;
			del = (y - ysq)*(y + ysq);
			result = exp(-ysq*ysq) * exp(-del) * result;
		}
	}
llable30: 
	// !------------------------------------------------------------------
	// !  Fix up for negative argument, erf, etc.
	// !------------------------------------------------------------------
	if(jint == 0)
	{
		result = (half - result) + half;
		if(x < zero) result = -result;
	}
	else if(jint == 1)
	{
		if(x < zero) result = two - result;
	}
	else
	{
		if(x < zero)
		{
			if(x < xneg)
			{
				result = xinf;
			}
			else
			{
				ysq = (int)(x * sixten) / sixten;
				del = (x - ysq)*(x + ysq);
				y = exp(ysq*ysq) * exp(del);
				result = (y+y) - result;
			}
		}
	}
llable80:

	return result;
}



__device__ void slope(double *sslop, int mkxtemp, double *field, double *p0)
{
	double below,above;
	int k;
	below = ( field[1] - field[0] ) / ( p0[1] - p0[0] );
	for(k=1;k<mkx;++k)
	{
		above = ( field[k] - field[k-1] ) / ( p0[k] - p0[k-1] );
		if(above > 0.0)
			sslop[k-1] = max(0.0,min(above,below));
		else
			sslop[k-1] = min(0.0,max(above,below));
		below = above;
	}
	sslop[mkx-1] = sslop[mkx-2];
	return;
}



__device__ double estblf(double td)
{
	double e;      // ! intermediate variable for es look-up
	double ai;
	double estblf_temp;
	int i;

//	double wv_saturation_mp_tmin__ = (double)173.1599999999999965894;
//	double wv_saturation_mp_tmax__ = (double)375.1600000000000250111;

	e = max(min(td,wv_saturation_mp_tmax__),wv_saturation_mp_tmin__);   //! partial pressure
	i = (int)(e-wv_saturation_mp_tmin__)+1;
	ai = (int)(e-wv_saturation_mp_tmin__);
	estblf_temp = (wv_saturation_mp_tmin__+ai-e+1.0)*
			  	   wv_saturation_mp_estbl__[i-1]-(wv_saturation_mp_tmin__+ai-e)* 
			  	   wv_saturation_mp_estbl__[i];
	return estblf_temp;
}

__device__ int qsat(double *t,double *p,double *es,double *qs,double *gam,int len)
{
	bool lflg;
	int i;
 //!
   double omeps;    //! 1. - 0.622
   double trinv;     //! reciprocal of ttrice (transition range)
   double tc;        //! temperature (in degrees C)
   double weight;    //! weight for es transition from water to ice
   double hltalt;    //! appropriately modified hlat for T derivatives
 //!
   double hlatsb;    //! hlat weighted in transition region
   double hlatvp;    //! hlat modified for t changes above freezing
   double tterm;     //! account for d(es)/dT in transition region
   double desdt;     //! d(es)/dT
   double pcf[5];

//    !
//    !-----------------------------------------------------------------------
//    !

//    double wv_saturation_mp_tmin__ = (double)173.1599999999999965894;
//    double wv_saturation_mp_tmax__ = (double)375.1600000000000250111;
//    double wv_saturation_mp_ttrice__ = (double)20.0000000000000000000;
//    double wv_saturation_mp_epsqs__ = (double)0.6219705862045155076;
//    double wv_saturation_mp_rgasv__ = (double)461.5046398201599231470;
//    double wv_saturation_mp_hlatf__ = (double)333700.0000000000000000000;
//    double wv_saturation_mp_hlatv__ = (double)2501000.0000000000000000000;
//    double wv_saturation_mp_cp__ = (double)1004.6399999999999863576;
//    double wv_saturation_mp_tmelt__ = (double)273.1499999999999772626;
//    bool wv_saturation_mp_icephs__ = true;


pcf[0] =  (double)5.04469588506e-01;
pcf[1] = (double)-5.47288442819e+00;
pcf[2] = (double)-3.67471858735e-01;
pcf[3] = (double)-8.95963532403e-03;
pcf[4] = (double)-7.78053686625e-05;

   omeps = 1.0 - wv_saturation_mp_epsqs__;
   for(i=1;i<=len;++i)
   {
	es[i-1] = estblf(t[i-1]);
	// !
	// ! Saturation specific humidity
	// !
	qs[i-1] = wv_saturation_mp_epsqs__*es[i-1]/(p[i-1] - omeps*es[i-1]);
	// !
	// ! The following check is to avoid the generation of negative
	// ! values that can occur in the upper stratosphere and mesosphere
	// !
	qs[i-1] = min(1.0,qs[i-1]);
	//!
	if(qs[i-1] < 0.0)
	{
		qs[i-1] = 1.0;
		es[i-1] = p[i-1];	
	}
   }
//    !
//    ! "generalized" analytic expression for t derivative of es
//    ! accurate to within 1 percent for 173.16 < t < 373.16
//    !
	trinv = 0.0;
	if ((!wv_saturation_mp_icephs__) || (wv_saturation_mp_ttrice__ == 0.0)) goto llable10;
		trinv = 1.0/wv_saturation_mp_ttrice__;
	for(i=1;i<=len;++i)
	{
		// !
		// ! Weighting of hlat accounts for transition from water to ice
		// ! polynomial expression approximates difference between es over
		// ! water and es over ice from 0 to -ttrice (C) (min of ttrice is
		// ! -40): required for accurate estimate of es derivative in transition
		// ! range from ice to water also accounting for change of hlatv with t
		// ! above freezing where const slope is given by -2369 j/(kg c) = cpv - cw
		// !
		tc     = t[i-1] - wv_saturation_mp_tmelt__;
		lflg   = ((tc >= -wv_saturation_mp_ttrice__) && (tc < 0.0));
		weight = min(-tc*trinv,1.0);
		hlatsb = wv_saturation_mp_hlatv__ + weight*wv_saturation_mp_hlatf__;
		hlatvp = wv_saturation_mp_hlatv__ - 2369.0*tc;
		if(t[i-1] < wv_saturation_mp_tmelt__)
		{
			hltalt = hlatsb;
		}
		else
		{
			hltalt = hlatvp;
		}
		if(lflg)
		{
			tterm = pcf[0] + tc*(pcf[1] + tc*(pcf[2] + tc*(pcf[3] + tc*pcf[4])));
		}
		else
		{
			tterm = 0.0;
		}
		desdt  = hltalt*es[i-1]/(wv_saturation_mp_rgasv__*t[i-1]*t[i-1]) + tterm*trinv;
     	gam[i-1] = hltalt*qs[i-1]*p[i-1]*desdt/(wv_saturation_mp_cp__*es[i-1]*(p[i-1] - omeps*es[i-1]));
		 if(qs[i-1] == 1.0) gam[i-1] = 0.0;
	}
	return 1;

// !
// ! No icephs or water to ice transition
// !
llable10: 
		  for(i=1;i<=len;++i)
		  {
		  	// !
			// ! Account for change of hlatv with t above freezing where
			// ! constant slope is given by -2369 j/(kg c) = cpv - cw
			// !
			hlatvp = wv_saturation_mp_hlatv__ - 2369.0*(t[i-1]-wv_saturation_mp_tmelt__);
			if (wv_saturation_mp_icephs__) 
         		hlatsb = wv_saturation_mp_hlatv__ + wv_saturation_mp_hlatf__;
      		else
         		hlatsb = wv_saturation_mp_hlatv__;
      		if (t[i-1] < wv_saturation_mp_tmelt__)
         		hltalt = hlatsb;
      		else
         		hltalt = hlatvp;
			desdt  = hltalt*es[i-1]/(wv_saturation_mp_rgasv__*t[i-1]*t[i-1]);
			gam[i-1] = hltalt*qs[i-1]*p[i-1]*desdt/(wv_saturation_mp_cp__*es[i-1]*(p[i-1] - omeps*es[i-1]));
			if (qs[i-1] == 1.0) gam[i-1] = 0.0;
		  }
		  //!
   		  return 1;
		  //!

}

// __device__ int qsat(double *t,double *p,double *es,double *qs,double *gam,int len)
// {
// 	int i;
// 	double omeps;
// 	omeps = 1.0-0.622;
// 	for(i=0;i<len;++i)
// 	{
// 		es[i] = t[i];
// 		qs[i] = 0.622*es[i]/(p[i]-omeps*es[i]);
// 		qs[i] = min(1.0,qs[i]);
// 		if(qs[i] < 0.0)
// 		{
// 			qs[i] = 1.0;
// 			es[i] = p[i];
// 		}
// 	}
// 	for(i=0;i<len;++i)
// 	{
// 		gam[i] = qs[i]*p[i]/(es[i]*(p[i] - omeps*es[i]));
// 		if(qs[i] == 1.0)
// 			gam[i] = 0.0;
// 	}
// 	//__wv_saturation_MOD_vqsatd(t ,p ,es ,qs ,gam , len);
// 	return 1;
// }

__device__ void conden(double p,double thl,double qt,double *th,double *qv,double *ql,
			double *qi,double *rvls,int *id_check)
{
	double tc,temps[1],t,temp_p[1];
	double leff,nu,qc;
	int iteration;
	double es[1]; //! Saturation vapor pressure
	double qs[1]; //! Saturation spec. humidity
	double gam[1];//! (L/cp)*dqs/dT
	int status;   //! Return status of qsat call
	tc = thl*exnf(p);
	//! Modification : In order to be compatible with the dlf treatment in stratiform.F90,
  	//!                we may use ( 268.15, 238.15 ) with 30K ramping instead of 20 K,
  	//!                in computing ice fraction below. 
  	//!                Note that 'cldwat_fice' uses ( 243.15, 263.15 ) with 20K ramping for stratus.
	nu   = max(min((268.0 - tc)/20.0,1.0),0.0); // ! Fraction of ice in the condensate. 
	leff = (1.0 - nu)*xlv + nu*xls;             // ! This is an estimate that hopefully speeds convergence
	//! --------------------------------------------------------------------------- !
    //! Below "temps" and "rvls" are just initial guesses for iteration loop below. !
    //! Note that the output "temps" from the below iteration loop is "temperature" !
    //! NOT "liquid temperature".                                                   !
    //! --------------------------------------------------------------------------- !
	temps[0]  = tc;
	////////////////
	temp_p[0] = p; 
    status = qsat(temps,temp_p,es,qs,gam,1);
    *rvls   = qs[0];

	if(qs[0] >= qt)
	{
		*id_check = 0;
		*qv = qt;
		qc = 0.0;
		*ql = 0.0;
		*qi = 0.0;
		*th = tc/exnf(p);
	}
	else
	{
		for(iteration=0;iteration<10;++iteration)
		{
			temps[0]  = temps[0] + ( (tc-temps[0])*cp/leff + qt - (*rvls) )/( cp/leff + ep2*leff*(*rvls)/r/temps[0]/temps[0] );
            status = qsat(temps,temp_p,es,qs,gam,1);
            *rvls   = qs[0];
		}
		qc = max(qt - qs[0],0.0);
        *qv = qt - qc;
        *ql = qc*(1.0 - nu);
        *qi = nu*qc;
        *th = temps[0]/exnf(p);
		if( fabs((temps[0]-(leff/cp)*qc)-tc) >= 1.0 )
            *id_check = 1;
        else
            *id_check = 0;
	}

	return;
}

__device__ void getbuoy(double pbot,double thv0bot,double ptop,double thv0top,double thvubot,double thvutop,double *plfc,double *cin)
{
	double frc;

	if( (thvubot > thv0bot) && (thvutop > thv0top) )
	{
		*plfc = pbot;
		return;
	}
	else if( (thvubot <= thv0bot) && (thvutop <= thv0top) )
	{
		*cin  = *cin - ( (thvubot/thv0bot - 1.0) + (thvutop/thv0top - 1.0)) * (pbot - ptop) /        
		( pbot/(r*thv0bot*exnf(pbot)) + ptop/(r*thv0top*exnf(ptop)) );
	}
	else if( (thvubot > thv0bot) && (thvutop <= thv0top) )
	{
		frc  = ( thvutop/thv0top - 1.0 ) / ( (thvutop/thv0top - 1.0) - (thvubot/thv0bot - 1.0) );
		*cin  = *cin - ( thvutop/thv0top - 1.0 ) * ( (ptop + frc*(pbot - ptop)) - ptop ) /         
					 ( pbot/(r*thv0bot*exnf(pbot)) + ptop/(r*thv0top*exnf(ptop)) );
	}
	else
	{
		frc  = ( thvubot/thv0bot - 1.0 ) / ( (thvubot/thv0bot - 1.0) - (thvutop/thv0top - 1.0) );
		*plfc = pbot - frc * ( pbot - ptop );
		*cin  = (*cin) - ( thvubot/thv0bot - 1.0)*(pbot - (*plfc))/
					 ( pbot/(r*thv0bot*exnf(pbot)) + ptop/(r*thv0top * exnf(ptop)));
	}

	return;
}

__device__ double single_cin(double pbot,double thv0bot,double ptop,double thv0top,double thvubot,double thvutop)
{
	// ! ------------------------------------------------------- !
	// ! Function to calculate a single layer CIN by summing all ! 
	// ! positive and negative CIN.                              !
	// ! ------------------------------------------------------- !
	double temp_single_cin;
	
    temp_single_cin = ( (1.0 - thvubot/thv0bot) + (1.0 - thvutop/thv0top)) * ( pbot - ptop ) / 
                 ( pbot/(r*thv0bot*exnf(pbot)) + ptop/(r*thv0top*exnf(ptop)) );
	return temp_single_cin;
}

__device__ void roots(double a,double b,double c,double *r1,double *r2,int *status)
{
	double q;

	*status = 0;

	if(a == 0.0)
	{
		if(b == 0.0)
			*status =1;
		else
			*r1 = -c/b;
		*r2 = *r1;
	}
	else
	{
		if(b == 0.0)
		{
			if(a*c > 0.0)
				*status = 2;
			else
				*r1 = sqrt(-c/a);
			*r2 = -(*r1);
		}
		else
		{
			if((pow(b,2)-4.0*a*c) < 0.0)
				*status = 3;
			else
			{
				q = -0.5*(b + fabs(1.0)*(b/(fabs(b)))*sqrt(pow(b,2) - 4.0*a*c));//sign(1.0,b)*sqrt(pow(b,2) - 4.0*a*c));
				*r1 = q/a;
				*r2 = c/q;
			}

		}
	}
		return;
}

__device__ double qsinvert(double qt,double thl,double psfc)
{
	// ! ----------------------------------------------------------------- !
	// ! Function calculating saturation pressure ps (or pLCL) from qt and !
	// ! thl ( liquid potential temperature,  NOT liquid virtual potential ! 
	// ! temperature) by inverting Bolton formula. I should check later if !
	// ! current use of 'leff' instead of 'xlv' here is reasonable or not. !
	// ! ----------------------------------------------------------------- !
	double temp_qsinvert;
	double ps, Pis, Ts, err, dlnqsdT, dTdPis, dPisdps, dlnqsdps, derrdps, dps, Ti, rhi, TLCL, PiLCL, psmin, dpsmax,temp_psfc[1],temp_Ti[1],temp_Ts[1],temp_ps[1];
	int i;
	double es[1],qs[1],gam[1];
	int status;
	double leff,nu;

	psmin  = 100.0*100.0; //! Default saturation pressure [Pa] if iteration does not converge
    dpsmax = 1.0;           //! Tolerance [Pa] for convergence of iteration
	
	// ! ------------------------------------ !
    // ! Calculate best initial guess of pLCL !
    // ! ------------------------------------ !

	Ti      =  thl*pow((psfc/p00),rovcp);
	temp_psfc[0]=  psfc;
	temp_Ti[0] = Ti; 
    status   =  qsat(temp_Ti,temp_psfc,es,qs,gam,1);
    rhi     =  qt/qs[0];
	if(rhi <= 0.01)
	{
		//write(iulog,*) 'Source air is too dry and pLCL is set to psmin in uwshcu.F90' 
		temp_qsinvert = psmin;
		return temp_qsinvert;
	} 
	TLCL     =  55.0 + 1.0/(1.0/(Ti-55.0)-log(rhi)/2840.0); //! Bolton's formula. MWR.1980.Eq.(22)
    PiLCL    =  TLCL/thl;
    ps       =  p00*pow((PiLCL),(1.0/rovcp));

	for(i=0;i<10;++i)
	{
		Pis = pow((ps/p00),rovcp);
		Ts  = thl*Pis;
		temp_Ts[0] = Ts;
		temp_ps[0] = ps;
		status   =  qsat(temp_Ts,temp_ps,es,qs,gam,1);
       	err      =  qt - qs[0];
       	nu       =  max(min((268.0 - Ts)/20.0,1.0),0.0);        
       	leff     =  (1.0 - nu)*xlv + nu*xls;                   
       	dlnqsdT  =  gam[0]*(cp/leff)/qs[0];
       	dTdPis   =  thl;
       	dPisdps  =  rovcp*Pis/ps; 
       	dlnqsdps = -1.0/(ps - (1.0 - ep2)*es[0]);
       	derrdps  = -qs[0]*(dlnqsdT * dTdPis * dPisdps + dlnqsdps);
       	dps      = -err/derrdps;
       	ps       =  ps + dps;
		if(ps < 0.0)
		{
			//write(iulog,*) 'pLCL iteration is negative and set to psmin in uwshcu.F90', qt, thl, psfc 
			temp_qsinvert = psmin;
			return temp_qsinvert;
		}
		if(fabs(dps) <= dpsmax)
		{
			temp_qsinvert = ps;
			return temp_qsinvert;
		}
	}
	//write(iulog,*) 'pLCL does not converge and is set to psmin in uwshcu.F90', qt, thl, psfc 
	temp_qsinvert = psmin;
	return temp_qsinvert;
}

__device__ double compute_alpha(double del_CIN,double ke)
{
	// ! ------------------------------------------------ !
	// ! Subroutine to compute proportionality factor for !
	// ! implicit CIN calculation.                        !   
	// ! ------------------------------------------------ !
	double x0,x1;
	int iteration;
	double temp_compute_alpha;

	x0 = 0.0;
	for(iteration=0;iteration<10;++iteration)
	{
		x1 = x0 - (exp(-x0*ke*del_CIN) - x0)/(-ke*del_CIN*exp(-x0*ke*del_CIN) - 1.0);
       	x0 = x1;
	}
	temp_compute_alpha = x0;

	return temp_compute_alpha;
}

__device__ double compute_mumin2(double mulcl,double rmaxfrac,double mulow)
{
	// ! --------------------------------------------------------- !
	// ! Subroutine to compute critical 'mu' (normalized CIN) such ! 
	// ! that updraft fraction at the LCL is equal to 'rmaxfrac'.  !
	// ! --------------------------------------------------------- ! 
	double x0, x1, ex, ef, exf, f, fs;
	int iteration;
	double temp_compute_mumin2;

	x0 = mulow;
	for(iteration = 0;iteration < 10;++iteration)
	{
		ex = exp((-1)*pow(x0,2));
		ef = erfc(x0);  //!!!erfc(x0)
		// ! if(x0.ge.3._r8) then
		// !    compute_mumin2 = 3._r8 
		// !    goto 20
		// ! endif 
		exf = ex/ef;
		f  = 0.5*pow(exf,2) - 0.5*pow((ex/2.0/rmaxfrac),2) - pow((mulcl*2.5066/2.0),2);
		fs = (2.0*pow(exf,2))*(exf/sqrt(3.141592)-x0) + (0.5*x0*pow(ex,2))/(pow(rmaxfrac,2));
		x1 = x0 - f/fs;     
		x0 = x1;
	}
	temp_compute_mumin2 = x0;
	return temp_compute_mumin2;
}

__device__ double compute_ppen(double wtwb,double D,double bogbot,double bogtop,double rho0j,double dpen)
{
	// ! ----------------------------------------------------------- !
	// ! Subroutine to compute critical 'ppen[Pa]<0' ( pressure dis. !
	// ! from 'ps0(kpen-1)' to the cumulus top where cumulus updraft !
	// ! vertical velocity is exactly zero ) by considering exact    !
	// ! non-zero fer(kpen).                                         !  
	// ! ----------------------------------------------------------- !  
	double x0, x1, f, fs, SB, s00;
	int iteration;
	double temp_compute_ppen;

	//! Buoyancy slope
	SB = ( bogtop - bogbot ) / dpen;
  	//! Sign of slope, 'f' at x = 0
  	//! If 's00>0', 'w' increases with height.
	s00 = bogbot / rho0j - D * wtwb;

	if((D*dpen) < 1.0e-8)
	{
		if(s00 >= 0.0)
			x0 = dpen;
		else
			x0 = max(0.0,min(dpen,-0.5*wtwb/s00));
	}
	else
	{
		if(s00 >= 0.0)
			x0 = dpen;
		else
			x0 = 0.0;
		for(iteration=1;iteration<=5;++iteration)
		{
			f  = exp(-2.0*D*x0)*(wtwb-(bogbot-SB/(2.0*D))/(D*rho0j)) + 
				(SB*x0+bogbot-SB/(2.0*D))/(D*rho0j);
			fs = -2.0*D*exp(-2.0*D*x0)*(wtwb-(bogbot-SB/(2.0*D))/(D*rho0j)) + 
				(SB)/(D*rho0j);
			if(fs >= 0.0)
			{
				fs = max(fs, 1.0e-10);
			}
			else
			{
				fs = min(fs,-1.0e-10);
			}
			x1 = x0 - f/fs;     
			x0 = x1;
		}
	}
	
	temp_compute_ppen = -max(0.0,min(dpen,x0));
	return temp_compute_ppen;
}

__device__ void fluxbelowinv(double cbmf,double *ps0,int mkxtemp,int kinv,double dt,double xsrc,double xmean,double xtopin,double xbotin,double *xflx)
{
	// ! ------------------------------------------------------------------------- !
	// ! Subroutine to calculate turbulent fluxes at and below 'kinv-1' interfaces.!
	// ! Check in the main program such that input 'cbmf' should not be zero.      !  
	// ! If the reconstructed inversion height does not go down below the 'kinv-1' !
	// ! interface, then turbulent flux at 'kinv-1' interface  is simply a product !
	// ! of 'cmbf' and 'qtsrc-xbot' where 'xbot' is the value at the top interface !
	// ! of 'kinv-1' layer. This flux is linearly interpolated down to the surface !
	// ! assuming turbulent fluxes at surface are zero. If reconstructed inversion !
	// ! height goes down below the 'kinv-1' interface, subsidence warming &drying !
	// ! measured by 'xtop-xbot', where  'xtop' is the value at the base interface !
	// ! of 'kinv+1' layer, is added ONLY to the 'kinv-1' layer, using appropriate !
	// ! mass weighting ( rpinv and rcbmf, or rr = rpinv / rcbmf ) between current !
	// ! and next provisional time step. Also impose a limiter to enforce outliers !
	// ! of thermodynamic variables in 'kinv' layer  to come back to normal values !
	// ! at the next step.                                                         !
	// ! ------------------------------------------------------------------------- ! 
	int k;
	double rcbmf, rpeff, dp, rr, pinv_eff, xtop, xbot, pinv, xtop_ori, xbot_ori;
	for(k=0;k<mkx+1;++k)
	{
		xflx[k] = 0.0;
	}
	dp = ps0[kinv-1] - ps0[kinv];

	if(fabs(xbotin-xtopin) <= 1.0e-13)
	{
		xbot = xbotin - 1.0e-13;
        xtop = xtopin + 1.0e-13;
	}
	else
	{
		xbot = xbotin;
        xtop = xtopin;
	}
	// ! -------------------------------------- !
    // ! Compute reconstructed inversion height !
    // ! -------------------------------------- !
	xtop_ori = xtop;
    xbot_ori = xbot;
    rcbmf = ( cbmf * g * dt ) / dp;                  //! Can be larger than 1 : 'OK'      
    rpeff = ( xmean - xtop ) / ( xbot - xtop ); 
    rpeff = min( max(0.0,rpeff), 1.0 );          //! As of this, 0<= rpeff <= 1   
	if((rpeff == 0.0) || (rpeff == 1.0))
	{
		xbot = xmean;
        xtop = xmean;
	}
	// ! Below two commented-out lines are the old code replacing the above 'if' block.   
    // ! if(rpeff.eq.1) xbot = xmean
    // ! if(rpeff.eq.0) xtop = xmean 
	rr       = rpeff / rcbmf;
    pinv     = ps0[kinv-1] - rpeff * dp;             //! "pinv" before detraining mass
    pinv_eff = ps0[kinv-1] + ( rcbmf - rpeff ) * dp; //! Effective "pinv" after detraining mass
	// ! ----------------------------------------------------------------------- !
    // ! Compute turbulent fluxes.                                               !
    // ! Below two cases exactly converges at 'kinv-1' interface when rr = 1._r8 !
    // ! ----------------------------------------------------------------------- !
	for(k=0;k<=kinv-1;++k)
	{
		xflx[k] = cbmf * ( xsrc - xbot ) * ( ps0[0] - ps0[k] ) / ( ps0[0] - pinv );
	}
	if(rr <= 1.0)
	{
		xflx[kinv-1] =  xflx[kinv-1] - ( 1.0 - rr ) * cbmf * ( xtop_ori - xbot_ori );
	}

	return;
}

__device__ void positive_moisture_single(double xlv,double xls,int mkxtemp,double dt,double qvmin,double qlmin,double qimin,double *dp,double *qv, 
							  double *ql,double *qi,double *s,double *qvten,double *qlten,double *qiten,double *sten )
{
	// ! ------------------------------------------------------------------------------- !
	// ! If any 'ql < qlmin, qi < qimin, qv < qvmin' are developed in any layer,         !
	// ! force them to be larger than minimum value by (1) condensating water vapor      !
	// ! into liquid or ice, and (2) by transporting water vapor from the very lower     !
	// ! layer. '2._r8' is multiplied to the minimum values for safety.                  !
	// ! Update final state variables and tendencies associated with this correction.    !
	// ! If any condensation happens, update (s,t) too.                                  !
	// ! Note that (qv,ql,qi,s) are final state variables after applying corresponding   !
	// ! input tendencies and corrective tendencies                                      !
	// ! ------------------------------------------------------------------------------- !
	int k;
	double dql, dqi, dqv, sum, aa, dum;

	for(k=mkx-1;k>=0;k--)	//! From the top to the 1st (lowest) layer from the surface
	{
		dql = max(0.0,1.0*qlmin-ql[k]);
		dqi = max(0.0,1.0*qimin-qi[k]);
		qlten[k] = qlten[k] +  dql/dt;
		qiten[k] = qiten[k] +  dqi/dt;
		qvten[k] = qvten[k] - (dql+dqi)/dt;
		sten[k]  = sten[k]  + xlv * (dql/dt) + xls * (dqi/dt);
		ql[k]    = ql[k] +  dql;
		qi[k]    = qi[k] +  dqi;
		qv[k]    = qv[k] -  dql - dqi;
		s[k]     = s[k]  +  xlv * dql + xls * dqi;
		dqv      = max(0.0,1.0*qvmin-qv[k]);
		qvten[k] = qvten[k] + dqv/dt;
		qv[k]    = qv[k]   + dqv;
		if( (k+1) != 1 )
		{
			qv[k-1]    = qv[k-1]    - dqv*dp[k]/dp[k-1];
			qvten[k-1] = qvten[k-1] - dqv*dp[k]/dp[k-1]/dt;
		} 
		qv[k] = max(qv[k],qvmin);
		ql[k] = max(ql[k],qlmin);
		qi[k] = max(qi[k],qimin);
	}
	// ! Extra moisture used to satisfy 'qv(i,1)=qvmin' is proportionally 
    // ! extracted from all the layers that has 'qv > 2*qvmin'. This fully
    // ! preserves column moisture.
	if(dqv > 1.0e-20)
	{
		sum = 0.0;
		for(k=0;k<mkx;++k)
		{
			if(qv[k] > 2.0*qvmin) sum = sum + qv[k]*dp[k];
		}
		aa = dqv*dp[0]/max(1.0e-20,sum);
		if(aa < 0.5)
		{
			for(k=0;k<mkx;++k)
			{
				if(qv[k] > 2.0*qvmin)
				{
					dum      = aa*qv[k];
                   	qv[k]    = qv[k] - dum;
                   	qvten[k] = qvten[k] - dum/dt;
				}
			}
		}
		else
		{
			printf("Full positive_moisture is impossible in uwshcu");
		}
	}
	return;
}

 double estblf__(double td)
{
	double e;      // ! intermediate variable for es look-up
	double ai;
	double estblf_temp;
	int i;

//	double wv_saturation_mp_tmin__ = 173.1599999999999965894;
//	double wv_saturation_mp_tmax__ = 375.1600000000000250111;

	e = max(min(td,wv_saturation_mp_tmax_),wv_saturation_mp_tmin_);   //! partial pressure
	i = (int)(e-wv_saturation_mp_tmin_)+1;
	ai = (int)(e-wv_saturation_mp_tmin_);
	estblf_temp = (wv_saturation_mp_tmin_+ai-e+1.0)*
			  	   wv_saturation_mp_estbl_[i-1]-(wv_saturation_mp_tmin_+ai-e)* 
			  	   wv_saturation_mp_estbl_[i];
	return estblf_temp;
}

 void findsp__(int lchnk_find,int iend_find,double *q,double *t,double *p,double *tsp,double *qsp)
{
/////!!!!!!!!!!!!----pcols=1024,pver=51时----!!!!!!!!!!!!!!!!!!!!!!//////////////
	// !----------------------------------------------------------------------- 
	// ! 
	// ! Purpose: 
	// !     find the wet bulb temperature for a given t and q
	// !     in a longitude height section
	// !     wet bulb temp is the temperature and spec humidity that is 
	// !     just saturated and has the same enthalpy
	// !     if q > qs(t) then tsp > t and qsp = qs(tsp) < q
	// !     if q < qs(t) then tsp < t and qsp = qs(tsp) > q
	// !
	// ! Method: 
	// ! a Newton method is used
	// ! first guess uses an algorithm provided by John Petch from the UKMO
	// ! we exclude points where the physical situation is unrealistic
	// ! e.g. where the temperature is outside the range of validity for the
	// !      saturation vapor pressure, or where the water vapor pressure
	// !      exceeds the ambient pressure, or the saturation specific humidity is 
	// !      unrealistic
	// ! 
	// ! Author: P. Rasch
	// ! 
	// !-----------------------------------------------------------------------

//	double wv_saturation_mp_pcf_[5];

// 	double wv_saturation_mp_tmin_ = 173.1599999999999965894;
// //	double wv_saturation_mp_tmax__ = 375.1600000000000250111;
// 	double wv_saturation_mp_ttrice_ = 20.0000000000000000000;
// 	double wv_saturation_mp_epsqs_ = 0.6219705862045155076;
// 	double wv_saturation_mp_rgasv_ = 461.5046398201599231470;
// 	double wv_saturation_mp_hlatf_ = 333700.0000000000000000000;
// 	double wv_saturation_mp_hlatv_ = 2501000.0000000000000000000;
// 	double wv_saturation_mp_cp_ = 1004.6399999999999863576;
//	double wv_saturation_mp_tmelt__ = 273.1499999999999772626;
//	bool wv_saturation_mp_icephs__ = false;

// pcf[0] =  5.04469588506e-01;
// pcf[1] = -5.47288442819e+00;
// pcf[2] = -3.67471858735e-01;
// pcf[3] = -8.95963532403e-03;
// pcf[4] = -7.78053686625e-05;

// !
// ! local variables
// !
   int i;//                 ! work variable
   int k;//                 ! work variable
   bool lflg;//              ! work variable
   int iter;//              ! work variable
   int l;//                 ! work variable
   bool error_found;

   double omeps;//                ! 1 minus epsilon
   double trinv;//                ! work variable
   double es;//                   ! sat. vapor pressure
   double desdt;//                ! change in sat vap pressure wrt temperature
//!     real(r8) desdp                ! change in sat vap pressure wrt pressure
   double dqsdt;//                ! change in sat spec. hum. wrt temperature
   double dgdt;//                 ! work variable
   double g;//                    ! work variable
   double weight[mix];//        ! work variable
   double hlatsb;//               ! (sublimation)
   double hlatvp;//               ! (vaporization)
   double hltalt[mix*mkx];//   ! lat. heat. of vap.
   double tterm;//                ! work var.
   double qs;//                   ! spec. hum. of water vapor
   double tc;//                   ! crit temp of transition to ice
   double tt0;
   
   //! work variables
   double t1, q1, dt, dq;
   double dtm, dqm;
   double qvd, a1, tmp;
   double rair;
   double r1b, c1, c2, c3;
   double denom;
   double dttol;
   double dqtol;
   int doit[mix]; 
   double enin[mix], enout[mix];
   double tlim[mix];

   omeps = 1.0 - wv_saturation_mp_epsqs_;
   trinv = 1.0/wv_saturation_mp_ttrice_;
   a1 = 7.5*log(10.0);
   rair =  287.04;
   c3 = rair*a1/wv_saturation_mp_cp_;
   dtm = 0.0;//    ! needed for iter=0 blowup with f90 -ei
   dqm = 0.0;//    ! needed for iter=0 blowup with f90 -ei
   dttol = 1.0e-4;// ! the relative temp error tolerance required to quit the iteration
   dqtol = 1.0e-4;// ! the relative moisture error tolerance required to quit the iteration
   tt0 = 273.15;//  ! Freezing temperature 
// !  tmin = 173.16 ! the coldest temperature we can deal with
// !
// ! max number of times to iterate the calculation
	iter = 8;
//!
	for(k=1;k<=mkx;++k)
	{
		// !
		// ! first guess on the wet bulb temperature
		// !
		for(i=1;i<=iend_find;++i) //mix=ncol时
		{
//#ifdef DEBUG
//          if ( (lchnk == lchnklook(nlook) ) && (i == icollook(nlook) ) ) 
// 		 {
// 			// write(iulog,*) ' '
//             // write(iulog,*) ' level, t, q, p', k, t(i,k), q(i,k), p(i,k)
// 			printf("iulog\n");
// 		 }     
// #endif
//! limit the temperature range to that relevant to the sat vap pres tables
//#if ( ! defined WACCM_MOZART )
         tlim[i-1] = min(max(t[(k-1)*iend_find+i-1],173.0),373.0);
// #else
//           tlim[i-1] = min(max(t[(k-1)*iend_find+i-1],128.0),373.0);
// #endif
		es = estblf__(tlim[i-1]);
		denom = p[(k-1)*iend_find+i-1] - omeps*es;
		qs = wv_saturation_mp_epsqs_*es/denom;
		doit[i-1] = 0;
		enout[i-1] = 1.0;
//! make sure a meaningful calculation is possible
		if ((p[(k-1)*iend_find+i-1] > 5.0*es) && ((qs > 0.0) && (qs < 0.5)))
		{
			// !
			// ! Saturation specific humidity
			// !
			qs = min(wv_saturation_mp_epsqs_*es/denom,1.0);
			// !
			// ! "generalized" analytic expression for t derivative of es
			// !  accurate to within 1 percent for 173.16 < t < 373.16
			// !
			// ! Weighting of hlat accounts for transition from water to ice
			// ! polynomial expression approximates difference between es over
			// ! water and es over ice from 0 to -ttrice (C) (min of ttrice is
			// ! -40): required for accurate estimate of es derivative in transition
			// ! range from ice to water also accounting for change of hlatv with t
			// ! above freezing where const slope is given by -2369 j/(kg c) = cpv - cw
			// !
			tc     = tlim[i-1] - tt0;
			lflg   = ((tc >= -wv_saturation_mp_ttrice_) && (tc < 0.0));
			weight[i-1] = min(-tc*trinv,1.0);
			hlatsb = wv_saturation_mp_hlatv_ + weight[i-1]*wv_saturation_mp_hlatf_;
			hlatvp = wv_saturation_mp_hlatv_ - 2369.0*tc;
			if (tlim[i-1] < tt0) 
				hltalt[(k-1)*iend_find+i-1] = hlatsb;
		 	else
				hltalt[(k-1)*iend_find+i-1] = hlatvp;
		 	
		 	enin[i-1] = wv_saturation_mp_cp_*tlim[i-1] + hltalt[(k-1)*iend_find+i-1]*q[(k-1)*iend_find+i-1];

			//! make a guess at the wet bulb temp using a UKMO algorithm (from J. Petch)
			tmp =  q[(k-1)*iend_find+i-1] - qs;
			c1 = hltalt[(k-1)*iend_find+i-1]*c3;
			c2 = pow((tlim[i-1] + 36.0),2); 
			r1b    = c2/(c2 + c1*qs);
			qvd   = r1b*tmp;
			tsp[(k-1)*iend_find+i-1] = tlim[i-1] + ((hltalt[(k-1)*iend_find+i-1]/wv_saturation_mp_cp_)*qvd);
//#ifdef DEBUG
//              if ( (lchnk == lchnklook(nlook) ) && (i == icollook(nlook) ) ) 
// 			 {
//                 // write(iulog,*) ' relative humidity ', q(i,k)/qs
//                 // write(iulog,*) ' first guess ', tsp(i,k)
// 				printf("iulog\n");
// 			 }
//#endif
			es = estblf__(tsp[(k-1)*iend_find+i-1]);
			qsp[(k-1)*iend_find+i-1] = min(wv_saturation_mp_epsqs_*es/(p[(k-1)*iend_find+i-1] - omeps*es),1.0);
		}
		else
		{
			doit[i-1] = 1;
			tsp[(k-1)*iend_find+i-1] = tlim[i-1];
			qsp[(k-1)*iend_find+i-1] = q[(k-1)*iend_find+i-1];
			enin[i-1] = 1.0;
		}
		}
// !
// ! now iterate on first guess
// !	
		for(l=1;l<=iter;++l)
		{
			dtm = 0;
			dqm = 0;
			for(i=1;i<=iend_find;++i)//mix==ncol时
			{
				if (doit[i-1] == 0) 
				{
              		es = estblf__(tsp[(k-1)*iend_find+i-1]);
					// !
					// ! Saturation specific humidity
					// !
					qs = min(wv_saturation_mp_epsqs_*es/(p[(k-1)*iend_find+i-1] - omeps*es),1.0);
					// !
					// ! "generalized" analytic expression for t derivative of es
					// ! accurate to within 1 percent for 173.16 < t < 373.16
					// !
					// ! Weighting of hlat accounts for transition from water to ice
					// ! polynomial expression approximates difference between es over
					// ! water and es over ice from 0 to -ttrice (C) (min of ttrice is
					// ! -40): required for accurate estimate of es derivative in transition
					// ! range from ice to water also accounting for change of hlatv with t
					// ! above freezing where const slope is given by -2369 j/(kg c) = cpv - cw
					// !
					tc     = tsp[(k-1)*iend_find+i-1] - tt0;
					lflg   = ((tc >= -wv_saturation_mp_ttrice_) && (tc < 0.0));
					weight[i-1] = min(-tc*trinv,1.0);
					hlatsb = wv_saturation_mp_hlatv_ + weight[i-1]*wv_saturation_mp_hlatf_;
					hlatvp = wv_saturation_mp_hlatv_ - 2369.0*tc;
					if (tsp[(k-1)*iend_find+i-1] < tt0) 
						hltalt[(k-1)*iend_find+i-1] = hlatsb;
				 	else
						hltalt[(k-1)*iend_find+i-1] = hlatvp;
				 	if(lflg)
					{
						tterm = wv_saturation_mp_pcf_[0] + tc*(wv_saturation_mp_pcf_[1] + tc*(wv_saturation_mp_pcf_[2]+tc*(wv_saturation_mp_pcf_[3] + tc*wv_saturation_mp_pcf_[4])));
					}
					else
						tterm = 0.0;
					desdt = hltalt[(k-1)*iend_find+i-1]*es/(wv_saturation_mp_rgasv_*tsp[(k-1)*iend_find+i-1]*tsp[(k-1)*iend_find+i-1]) + tterm*trinv;
					dqsdt = (wv_saturation_mp_epsqs_ + omeps*qs)/(p[(k-1)*iend_find+i-1] - omeps*es)*desdt;
					//!              g = cp*(tlim(i)-tsp(i,k)) + hltalt(i,k)*q(i,k)- hltalt(i,k)*qsp(i,k)
               		g = enin[i-1] - (wv_saturation_mp_cp_*tsp[(k-1)*iend_find+i-1] + hltalt[(k-1)*iend_find+i-1]*qsp[(k-1)*iend_find+i-1]);
					dgdt = -(wv_saturation_mp_cp_ + hltalt[(k-1)*iend_find+i-1]*dqsdt);
					t1 = tsp[(k-1)*iend_find+i-1] - g/dgdt;
					dt = fabs(t1 - tsp[(k-1)*iend_find+i-1])/t1;
					tsp[(k-1)*iend_find+i-1] = max(t1,wv_saturation_mp_tmin_);
					es = estblf__(tsp[(k-1)*iend_find+i-1]);
					q1 = min(wv_saturation_mp_epsqs_*es/(p[(k-1)*iend_find+i-1] - omeps*es),1.0);
					dq = fabs(q1 - qsp[(k-1)*iend_find+i-1])/max(q1,1.0e-12);
					qsp[(k-1)*iend_find+i-1] = q1;
//#ifdef DEBUG
// 					if ( (lchnk == lchnklook(nlook) ) && (i == icollook(nlook) ) ) 
// 					{
// 					   //write(iulog,*) ' rel chg lev, iter, t, q ', k, l, dt, dq, g
// 					   printf("iulog\n");
// 					}			
//#endif
					dtm = max(dtm,dt);
					dqm = max(dqm,dq);
					//! if converged at this point, exclude it from more iterations
					if((dt < dttol) && (dq < dqtol))
					{
						doit[i-1] = 2;
					}
					enout[i-1] = wv_saturation_mp_cp_*tsp[(k-1)*iend_find+i-1] + hltalt[(k-1)*iend_find+i-1]*qsp[(k-1)*iend_find+i-1];
					//! bail out if we are too near the end of temp range
//#if ( ! defined WACCM_MOZART )  
								   if (tsp[(k-1)*iend_find+i-1] < 174.16) 
// #else
// 								   if (tsp[(k-1)*iend_find+i-1] < 130.16)
// #endif
					{
						doit[i-1] = 4;
					}

				}

			}// ! do i = 1,ncol

			if((dtm < dttol) && (dqm < dqtol))
			{
				goto lablelable10;
			}
		}
lablelable10: 

		error_found = false;
		if((dtm > dttol) || (dqm > dqtol))
		{
			for(i=1;i<=iend_find;++i)//mix==ncol时
			{
				if (doit[i-1] == 0) error_found = true;
			}
			if(error_found)
			{
				for(i=1;i<=iend_find;++i)//mix==ncol时
				{
					if (doit[i-1] == 0) 
					{
                  		// write(iulog,*) ' findsp not converging at point i, k ', i, k
                  		// write(iulog,*) ' t, q, p, enin ', t(i,k), q(i,k), p(i,k), enin(i)
                  		// write(iulog,*) ' tsp, qsp, enout ', tsp(i,k), qsp(i,k), enout(i)
                  		// call endrun ('FINDSP')
						printf("FINDSP iulog error\n");
						return;
					}
				}
			}
		}
		for(i=1;i<=iend_find;++i)//mix==ncol时
		{
			if ((doit[i-1] == 2) && fabs((enin[i-1]-enout[i-1])/(enin[i-1]+enout[i-1])) > 1.0e-4)
            	error_found = true;
		}
	
	if(error_found)
	{
		for(i=1;i<=iend_find;++i)//mix==ncol时
		{
			if((doit[i-1] == 2) && (fabs((enin[i-1]-enout[i-1])/(enin[i-1]+enout[i-1])) > 1.0e-4))
			{
				// write(iulog,*) ' the enthalpy is not conserved for point ', &
				// 				i, k, enin(i), enout(i)
			 	// write(iulog,*) ' t, q, p, enin ', t(i,k), q(i,k), p(i,k), enin(i)
				// write(iulog,*) ' tsp, qsp, enout ', tsp(i,k), qsp(i,k), enout(i)
			 	// call endrun ('FINDSP')
				 printf("FINDSP iulog error\n");
				 return;
			}
		}
	}
	}   //! level loop (k=1,pver)

	return;
}

// ! ------------------------ !
// !                          ! 
// ! End of subroutine blocks !
// !                          !
// ! ------------------------ !



extern "C" void bxcompute_uwshcu_inv_(int *pcols_c ,int *pver_c  ,int *ncol_c  ,int *pcnst_c   ,double *ztodt_c  ,       
	double *ps0_inv  ,double *zs0_inv    ,double *p0_inv        , double *z0_inv    ,double *dp0_inv  ,  
	double *u0_inv   ,double *v0_inv     ,double *qv0_inv       ,double *ql0_inv   ,double *qi0_inv  ,  
	double *t0_inv   ,double *s0_inv     ,double *tr0_inv       ,                         
	double *tke_inv  ,double *cldfrct_inv,double *concldfrct_inv,double *pblh      ,double *cush     ,   
	double *umf_inv  ,double *slflx_inv  ,double *qtflx_inv     ,                          
	double *flxprc1_inv,double *flxsnow1_inv,     				 
	double *qvten_inv,double *qlten_inv  ,double *qiten_inv     ,                         
	double *sten_inv ,double *uten_inv   ,double *vten_inv      ,double *trten_inv ,               
	double *qrten_inv,double *qsten_inv  ,double *precip        ,double *snow      ,double *evapc_inv,  
	double *cufrc_inv,double *qcu_inv    ,double *qlu_inv       ,double *qiu_inv   ,                
	double *cbmf     ,double *qc_inv     ,double *rliq          ,                         
	double *cnt_inv  ,double *cnb_inv    ,int *lchnk_c     ,double *dpdry0_inv ,double *wu_inv)
{
	int pcols=pcols_c[0];
	int pver=pver_c[0];
	int ncol=ncol_c[0];
	int pcnst=pcnst_c[0];
	double ztodt=ztodt_c[0];
	int lchnk=lchnk_c[0];
	int pvers = pver+1;
int ret1,ret2,ret3,ret4,ret5,ret6,ret10;

ret2= GPTLinitialize();
// for(int i=0;i<25;++i)
// {
// 	for(int j=0;j<3;++j)
// 	{
// 		printf("%c",constituents_mp_cnst_type_[i][j]);
// 	}
// 	printf("\n");
// }
ret2 = GPTLstart("Malloc");
	//主机端使用数组
	///////input//////////
	double *ps0_c;
	hipHostMalloc((double**) &ps0_c,pvers*pcols*sizeof(double));
	double *zs0_c;
	hipHostMalloc((double**) &zs0_c,pvers*pcols*sizeof(double));
	double *p0_c;
	hipHostMalloc((double**) &p0_c,pver*pcols*sizeof(double));
	double *z0_c;
	hipHostMalloc((double**) &z0_c,pver*pcols*sizeof(double));
	double *dp0_c;
	hipHostMalloc((double**) &dp0_c,pver*pcols*sizeof(double));
	double *dpdry0_c;
	hipHostMalloc((double**) &dpdry0_c,pver*pcols*sizeof(double));
	double *u0_c;
	hipHostMalloc((double**) &u0_c,pver*pcols*sizeof(double));
	double *v0_c;
	hipHostMalloc((double**) &v0_c,pver*pcols*sizeof(double));
	double *qv0_c;
	hipHostMalloc((double**) &qv0_c,pver*pcols*sizeof(double));
	double *ql0_c;
	hipHostMalloc((double**) &ql0_c,pver*pcols*sizeof(double));
	double *qi0_c;
	hipHostMalloc((double**) &qi0_c,pver*pcols*sizeof(double));
	double *t0_c;
	hipHostMalloc((double**) &t0_c,pver*pcols*sizeof(double));
	double *s0_c;
	hipHostMalloc((double**) &s0_c,pver*pcols*sizeof(double));
	double *tr0_c;
	hipHostMalloc((double**) &tr0_c,pcnst*pver*pcols*sizeof(double));
	double *tke_c;
	hipHostMalloc((double**) &tke_c,pvers*pcols*sizeof(double));
	double *cldfrct_c;
	hipHostMalloc((double**) &cldfrct_c,pver*pcols*sizeof(double));
	double *concldfrct_c;
	hipHostMalloc((double**) &concldfrct_c,pver*pcols*sizeof(double));
	////double pblh_c[pcols];
	////hipHostMalloc((void**) &pblh_c,pcols*sizeof(double));
	
	///////////////////////in-out////////////////
	////double cush_c[pcols]; 
	////hipHostMalloc((void**) &cush_c,pcols*sizeof(double));

	// //////////////////////output/////////////////
	double *umf_c;
	hipHostMalloc((double**) &umf_c,pvers*pcols*sizeof(double));
	double *qvten_c;
	hipHostMalloc((double**) &qvten_c,pver*pcols*sizeof(double));
	double *qlten_c;
	hipHostMalloc((double**) &qlten_c,pver*pcols*sizeof(double));
	double *qiten_c;
	hipHostMalloc((double**) &qiten_c,pver*pcols*sizeof(double));
	double *sten_c;
	hipHostMalloc((double**) &sten_c,pver*pcols*sizeof(double));
	double *uten_c;
	hipHostMalloc((double**) &uten_c,pver*pcols*sizeof(double));
	double *vten_c;
	hipHostMalloc((double**) &vten_c,pver*pcols*sizeof(double));
	double *trten_c;
	hipHostMalloc((double**) &trten_c,pcnst*pver*pcols*sizeof(double));
	double *qrten_c;
	hipHostMalloc((double**) &qrten_c,pver*pcols*sizeof(double));
	double *qsten_c;
	hipHostMalloc((double**) &qsten_c,pver*pcols*sizeof(double));
//	double *precip_c[pcols];
//	//hipHostMalloc((void**) &precip_c,pcols*sizeof(double));
//	double *snow_c[pcols];
//	//hipHostMalloc((void**) &snow_c,pcols*sizeof(double));
	double *evapc_c;
	hipHostMalloc((double**) &evapc_c,pver*pcols*sizeof(double));
	// double *rliq_c[pcols];
	// //hipHostMalloc((void**) &rliq_c,pcols*sizeof(double));slflx_c
	double *slflx_c;
	hipHostMalloc((double**) &slflx_c,pvers*pcols*sizeof(double));
	double *qtflx_c;
	hipHostMalloc((double**) &qtflx_c,pvers*pcols*sizeof(double));
	double *flxprc1_c;
	hipHostMalloc((double**) &flxprc1_c,pvers*pcols*sizeof(double));
	double *flxsnow1_c;
	hipHostMalloc((double**) &flxsnow1_c,pvers*pcols*sizeof(double));

	double *cufrc_c;
	hipHostMalloc((double**) &cufrc_c,pver*pcols*sizeof(double));
	double *qcu_c;
	hipHostMalloc((double**) &qcu_c,pver*pcols*sizeof(double));
	double *qlu_c;
	hipHostMalloc((double**) &qlu_c,pver*pcols*sizeof(double));
	double *qiu_c;
	hipHostMalloc((double**) &qiu_c,pver*pcols*sizeof(double));
	double *qc_c;
	hipHostMalloc((double**) &qc_c,pver*pcols*sizeof(double));
	// double *cbmf_c[pcols];
	// //hipHostMalloc((void**) &cbmf_c,pcols*sizeof(double));
	double *cnt_c;
	hipHostMalloc((double**) &cnt_c,pcols*sizeof(double));
	double *cnb_c;
	hipHostMalloc((double**) &cnb_c,pcols*sizeof(double));

	// double *wv_saturation_mp_estbl_c;
	// hipHostMalloc((double**) &wv_saturation_mp_estbl_c,250*sizeof(double));
	// double *wv_saturation_mp_pcf_c;
	// hipHostMalloc((double**) &wv_saturation_mp_pcf_c,6*sizeof(double));
	
	double *wu_c;
	hipHostMalloc((double**) &wu_c,pvers*pcols*sizeof(double));

	double *tw0_c;
	hipHostMalloc((double**) &tw0_c,pver*pcols*sizeof(double));
	double *qw0_c;
	hipHostMalloc((double**) &qw0_c,pver*pcols*sizeof(double));
///////output/////////



//设备端使用数组
/////input//////////
double *ps0_d;
double *zs0_d;
double *p0_d;
double *z0_d;
double *dp0_d;
double *dpdry0_d;
double *u0_d;
double *v0_d;
double *qv0_d;
double *ql0_d;
double *qi0_d;
double *t0_d;
double *s0_d;
double *tr0_d;
double *tke_d;
double *cldfrct_d;
double *concldfrct_d;
double *pblh_d;

double *cush_d; 

//////////////////////output/////////////////
double *umf_d;
double *qvten_d;
double *qlten_d;
double *qiten_d;
double *sten_d;
double *uten_d;
double *vten_d;
double *trten_d;
double *qrten_d;
double *qsten_d;
double *precip_d;
double *snow_d;
double *evapc_d;
double *rliq_d;
double *slflx_d;
double *qtflx_d;
double *flxprc1_d;
double *flxsnow1_d;
double *cufrc_d;
double *qcu_d;
double *qlu_d;
double *qiu_d;
double *qc_d;
double *cbmf_d;
double *cnt_d;
double *cnb_d;

double *qmin;
int *numptr_amode;

double *tw0_in;
double *qw0_in;

double *wu_d;

double *wv_saturation_mp_estbl_d;
//double *wv_saturation_mp_pcf_d;

hipMalloc((void**) &tw0_in,pver*pcols*sizeof(double));

hipMalloc((void**) &qw0_in,pver*pcols*sizeof(double));

// int *lchnk_d,*ncol_d;


/////input//////////
hipMalloc((void**) &ps0_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &zs0_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &p0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &z0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &dp0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &dpdry0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &u0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &v0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qv0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &ql0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qi0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &t0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &s0_d,pver*pcols*sizeof(double));

hipMalloc((void**) &tr0_d,pcnst*pver*pcols*sizeof(double));

hipMalloc((void**) &tke_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &cldfrct_d,pver*pcols*sizeof(double));

hipMalloc((void**) &concldfrct_d,pver*pcols*sizeof(double));

hipMalloc((void**) &pblh_d,pcols*sizeof(double));

// 	///////////////////////in-out////////////////

hipMalloc((void**) &cush_d,pcols*sizeof(double));



//////////////////////output/////////////////

hipMalloc((void**) &umf_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &qvten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qlten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qiten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &sten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &uten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &vten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &trten_d,pcnst*pver*pcols*sizeof(double));

hipMalloc((void**) &qrten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qsten_d,pver*pcols*sizeof(double));

hipMalloc((void**) &precip_d,pcols*sizeof(double));

hipMalloc((void**) &snow_d,pcols*sizeof(double));

hipMalloc((void**) &evapc_d,pver*pcols*sizeof(double));

hipMalloc((void**) &rliq_d,pcols*sizeof(double));

hipMalloc((void**) &slflx_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &qtflx_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &flxprc1_d,pvers*pcols*sizeof(double));

hipMalloc((void**) &flxsnow1_d,pvers*pcols*sizeof(double));


hipMalloc((void**) &cufrc_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qcu_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qlu_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qiu_d,pver*pcols*sizeof(double));

hipMalloc((void**) &qc_d,pver*pcols*sizeof(double));

hipMalloc((void**) &cbmf_d,pcols*sizeof(double));

hipMalloc((void**) &cnt_d,pcols*sizeof(double));

hipMalloc((void**) &cnb_d,pcols*sizeof(double));

hipMalloc((void**) &qmin,pcnst*sizeof(double));

hipMalloc((void**) &numptr_amode,3*sizeof(int));


hipMalloc((void**) &wv_saturation_mp_estbl_d,250*sizeof(double));

hipMalloc((void**) &wu_d,pvers*pcols*sizeof(double));
// hipMalloc((void**) &wv_saturation_mp_pcf_d,6*sizeof(double));
///////output/////////
ret2 = GPTLstop("Malloc");
	int k,k_inv,m,i;
 	int mixtemp=pcols;
 	int mkxtemp=pver;
	int iend=ncol;
 	int ncnsttemp=pcnst;
 //	double dt= ztodt;
 ret3= GPTLinitialize();
 ret3 = GPTLstart("CPU computing1");
	for(k=0; k<mkx; ++k)
		for(i=0;i<iend;++i)
		{
			k_inv = mkx - 1 - k;
			p0_c[k*iend + i] = p0_inv[k_inv*iend + i];
			u0_c[k*iend + i] = u0_inv[k_inv*iend + i];
			v0_c[k*iend + i] = v0_inv[k_inv*iend + i];
			z0_c[k*iend + i] = z0_inv[k_inv*iend + i];
			dp0_c[k*iend + i] = dp0_inv[k_inv*iend + i];
			dpdry0_c[k*iend + i] = dpdry0_inv[k_inv*iend + i];
			qv0_c[k*iend + i] = qv0_inv[k_inv*iend + i];
			ql0_c[k*iend + i] = ql0_inv[k_inv*iend + i];
			qi0_c[k*iend + i] = qi0_inv[k_inv*iend + i];
			t0_c[k*iend + i] = t0_inv[k_inv*iend + i];
			s0_c[k*iend + i] = s0_inv[k_inv*iend + i];
			cldfrct_c[k*iend + i] = cldfrct_inv[k_inv*iend + i];
			concldfrct_c[k*iend + i] = concldfrct_inv[k_inv*iend + i];
		}
	for(m=0; m<pcnst; ++m)
		for(k=0; k<pver; ++k)
			for(i=0;i<iend;++i)
			{
				k_inv = pver - 1 - k;
				tr0_c[m*pver*iend + k*iend + i] = tr0_inv[m*pver*iend+k_inv*iend+i];
			}
				
	for(k=0; k<mkx+1; ++k)
		for(i=0;i<iend;++i)
		{
			k_inv = mkx - k;
			ps0_c[k*iend + i] = ps0_inv[k_inv*iend + i];
			zs0_c[k*iend + i] = zs0_inv[k_inv*iend + i];
			tke_c[k*iend + i] = tke_inv[k_inv*iend + i];
		}


	//		printf("tr0_c[7,9,1] =%e\n",tr0_c[1*mkx*iend+9*iend+7]);

	// for( k=mkx;k>=0;--k)	
	// 	for(i=0;i<iend;++i)	
	// 		printf("zs0_c=%e\n",zs0_c[k*iend+i]);
//		{
		//if(umf_c[k*ncol+i]>=1.0e-12)		
// 		printf("p0_c=%e\n",p0_c[66]);
// 		printf("u0_c=%e\n",u0_c[66]);
// 		printf("v0_c=%e\n",v0_c[66]);
// 		printf("z0_c=%e\n",z0_c[66]);
// 		printf("dp0_c=%e\n",dp0_c[66]);
			
// //		}

// //		for(i=0;i<iend;++i)	
// //		{
// 	//if(umf_c[k*ncol+i]>=1.0e-12)		
// 		printf("dpdry0_c=%e\n",dpdry0_c[66]);
// 		printf("qv0_c=%e\n",qv0_c[66]);
// 		printf("cldfrct_c=%e\n",cldfrct_c[66]);
// 		printf("concldfrct_c=%e\n",concldfrct_c[66]);
// 		printf("tr0_c=%e\n",tr0_c[66]);
// 		printf("ps0_c=%e\n",ps0_c[66]);			
// //		}
// //	for( k=mkx-1;k>=0;--k)	
// //		for(i=0;i<iend;++i)	
// //		{
// 	//if(umf_c[k*ncol+i]>=1.0e-12)		
// 		printf("zs0_c=%e\n",zs0_c[66]);
// 		printf("tke_c=%e\n",tke_c[66]);
		// printf("sten_c=%e\n",sten_c[66]);
		// printf("evapc_c=%e\n",evapc_c[66]);
		// printf("qlu_c=%e\n",qlu_c[66]);


		 for(k=0;k<mkx;++k)
		  for(i=0;i<iend;++i)
	      {
			tw0_c[k*iend + i] = 0.0;
			qw0_c[k*iend+ i ] = 1.0;
		  }
	// *lchnk_d = lchnk;
	// *ncol_d  = ncol;

ret3 = GPTLstop("CPU computing1");

ret4 = GPTLinitialize();
ret4 = GPTLstart("CPU computing2");
findsp__( lchnk, ncol, qv0_c, t0_c, p0_c, tw0_c, qw0_c );
hipDeviceSynchronize();
ret4 = GPTLstop("CPU computing2");
	//printf("lifeilifei\n");
	//	hipMemcpy

	// printf("wv_saturation_mp_estbl_[5]=%20.19f\n",wv_saturation_mp_estbl_[5]);
	// printf("wv_saturation_mp_tmin_=%20.19f\n",wv_saturation_mp_tmin_);
	// printf("wv_saturation_mp_tmax_=%20.19f\n",wv_saturation_mp_tmax_);
	// printf("wv_saturation_mp_ttrice_=%20.19f\n",wv_saturation_mp_ttrice_);
	// printf("wv_saturation_mp_epsqs_=%20.19f\n",wv_saturation_mp_epsqs_);
	// printf("wv_saturation_mp_rgasv_=%20.19f\n",wv_saturation_mp_rgasv_);
	// printf("wv_saturation_mp_hlatf_=%20.19f\n",wv_saturation_mp_hlatf_);
	// printf("wv_saturation_mp_hlatv_=%20.19f\n",wv_saturation_mp_hlatv_);
	// printf("wv_saturation_mp_cp_=%20.19f\n",wv_saturation_mp_cp_);
	// printf("wv_saturation_mp_tmelt_=%20.19f\n",wv_saturation_mp_tmelt_);
	// printf("wv_saturation_mp_icephs_=%d\n",wv_saturation_mp_icephs_);
	// int tempp;
	// for(tempp = 0;tempp < 6;++tempp)
	// 	wv_saturation_mp_pcf_c[tempp] = wv_saturation_mp_pcf_[tempp];
	// for(tempp = 0;tempp < 250;++tempp)
	// 	wv_saturation_mp_estbl_c[tempp] = wv_saturation_mp_estbl_[tempp];

ret10 = GPTLinitialize();
ret10 = GPTLstart("cudaMemcpy3");
//	hipMemcpyToSymbol(HIP_SYMBOL(wv_saturation_mp_estbl__), wv_saturation_mp_estbl_, 250*sizeof(double));
	hipMemcpy(wv_saturation_mp_estbl_d, wv_saturation_mp_estbl_, 250*sizeof(double),hipMemcpyHostToDevice);
hipDeviceSynchronize();

//hipMemcpyToSymbol(HIP_SYMBOL(wv_saturation_mp_pcf__), wv_saturation_mp_pcf_, 6*sizeof(double));

ret10 = GPTLstop("cudaMemcpy3");
	//hipMemcpy(wv_saturation_mp_estbl__, wv_saturation_mp_estbl_, 250*sizeof(double),hipMemcpyHostToDevice);
	//hipMemcpy(wv_saturation_mp_tmin__, wv_saturation_mp_tmin_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_tmax__, wv_saturation_mp_tmax_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_ttrice__, wv_saturation_mp_ttrice_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_epsqs__, wv_saturation_mp_epsqs_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_rgasv__, wv_saturation_mp_rgasv_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_hlatf__, wv_saturation_mp_hlatf_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_hlatv__, wv_saturation_mp_hlatv_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_cp__, wv_saturation_mp_cp_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_tmelt__, wv_saturation_mp_tmelt_, 1*sizeof(double),hipMemcpyHostToDevice);
	// hipMemcpy(wv_saturation_mp_icephs__, wv_saturation_mp_icephs_, 1*sizeof(bool),hipMemcpyHostToDevice);
ret5 = GPTLinitialize();
ret5 = GPTLstart("cudaMemcpy1");
	hipMemcpy(p0_d, p0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(u0_d, u0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(v0_d, v0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(z0_d, z0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(dp0_d, dp0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(dpdry0_d, dpdry0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(qv0_d, qv0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(ql0_d, ql0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(qi0_d, qi0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(t0_d, t0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(s0_d, s0_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(cldfrct_d, cldfrct_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(concldfrct_d, concldfrct_c, pver*pcols*sizeof(double),hipMemcpyHostToDevice);

	hipMemcpy(tr0_d, tr0_c, pcnst*pver*pcols*sizeof(double),hipMemcpyHostToDevice);

	hipMemcpy(ps0_d, ps0_c, pvers*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(zs0_d, zs0_c, pvers*pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(tke_d, tke_c, pvers*pcols*sizeof(double),hipMemcpyHostToDevice);

	hipMemcpy(pblh_d, pblh, pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(cush_d, cush, pcols*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(qmin, constituents_mp_qmin_, pcnst*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(numptr_amode, modal_aero_data_mp_numptr_amode_, 3*sizeof(int),hipMemcpyHostToDevice);

//	hipMemcpy(tw0_in, tw0_c, pcols*pver*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(qw0_in, qw0_c, pcols*pver*sizeof(double),hipMemcpyHostToDevice);
hipDeviceSynchronize();
ret5 = GPTLstop("cudaMemcpy1");
	hipError_t err1 = hipGetLastError();
	if (err1 != hipSuccess) {
		printf("CUDA Error1: %s\n", hipGetErrorString(err1));
	}
ret1= GPTLinitialize();
ret1 = GPTLstart("compute_uwshcu111");
		hipLaunchKernelGGL(compute_uwshcu, dim3(ceil(ncol/32)), dim3(32), 0, 0, 
			uwshcu_mp_xlv_,
			uwshcu_mp_xlf_,
			uwshcu_mp_xls_,
		    uwshcu_mp_cp_,
			uwshcu_mp_zvir_,
			uwshcu_mp_r_,
			uwshcu_mp_g_,
			uwshcu_mp_ep2_,
			uwshcu_mp_p00_,
			uwshcu_mp_rovcp_,
			uwshcu_mp_rpen_,
		    wv_saturation_mp_tmin_,
			wv_saturation_mp_tmax_,
			wv_saturation_mp_ttrice_,
	//		wv_saturation_mp_pcf_d,
			wv_saturation_mp_epsqs_,
			wv_saturation_mp_rgasv_,
			wv_saturation_mp_hlatf_,
			wv_saturation_mp_hlatv_,
			wv_saturation_mp_cp_,
			wv_saturation_mp_tmelt_,
			wv_saturation_mp_icephs_,
			wv_saturation_mp_estbl_d,

			mixtemp  , mkxtemp    , iend      , ncnsttemp , ztodt ,
			ps0_d  , zs0_d    , p0_d        , z0_d    , dp0_d  , 
			u0_d   , v0_d     , qv0_d       , ql0_d   , qi0_d  ,  
			t0_d   , s0_d     , tr0_d       ,                 
			tke_d  , cldfrct_d, concldfrct_d, pblh_d  , cush_d ,  
			umf_d  , slflx_d  , qtflx_d     ,                  
			flxprc1_d  , flxsnow1_d  ,		            
			qvten_d, qlten_d  , qiten_d     ,                 
			sten_d , uten_d   , vten_d      , trten_d ,        
			qrten_d, qsten_d  , precip_d    , snow_d  , evapc_d, 
			cufrc_d, qcu_d    , qlu_d       , qiu_d   ,        
			cbmf_d , qc_d     , rliq_d      ,                
			cnt_d  , cnb_d    , lchnk , dpdry0_d , wu_d, qmin , numptr_amode, qw0_in);
	hipError_t err3 = hipGetLastError();
	if (err3 != hipSuccess) {
		printf("CUDA Error3: %s\n", hipGetErrorString(err3));
	}
	hipDeviceSynchronize();
ret1 = GPTLstop("compute_uwshcu111");

ret6 = GPTLinitialize();
ret6 = GPTLstart("cudaMemcpy2");
	//	hipMemcpy
	//hipMemcpy(pblh, pblh_d, pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(cush, cush_d, pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(snow, snow_d, pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(precip, precip_d, pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(cbmf, cbmf_d, pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(rliq, rliq_d, pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(cnt_c, cnt_d, pcols*sizeof(double),hipMemcpyDeviceToHost);

	hipMemcpy(umf_c, umf_d, pvers*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(slflx_c, slflx_d, pvers*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qtflx_c, qtflx_d, pvers*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(flxprc1_c, flxprc1_d, pvers*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(flxsnow1_c, flxsnow1_d, pvers*pcols*sizeof(double),hipMemcpyDeviceToHost);

	hipMemcpy(qrten_c, qrten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qvten_c, qvten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qlten_c, qlten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qiten_c, qiten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(sten_c, sten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(uten_c, uten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(vten_c, vten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qsten_c, qsten_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(evapc_c, evapc_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(cufrc_c, cufrc_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qcu_c, qcu_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qlu_c, qlu_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qiu_c, qiu_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(qc_c, qc_d, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(cnb_c, cnb_d, pcols*sizeof(double),hipMemcpyDeviceToHost);

	hipMemcpy(trten_c, trten_d, pcnst*pver*pcols*sizeof(double),hipMemcpyDeviceToHost);

	hipMemcpy(wu_c, wu_d, pvers*pcols*sizeof(double),hipMemcpyDeviceToHost);
ret6 = GPTLstop("cudaMemcpy2");	
	//hipMemcpy(qw0_c, qw0_in, pver*pcols*sizeof(double),hipMemcpyDeviceToHost);
	//printf("qw0_c[0]=\n",qw0_c[0]);
	//printf("lifeilifeilifei\n");
	// for( k=mkx;k>=0;--k)	
	// 	for(i=0;i<iend;++i)	
	// 	{
	// 	if(umf_c[k*ncol+i]>=1.0e-12)		
	// 		printf("umf_c=%33.32e\n",umf_c[k*ncol+i]); 
	// 	}
// 			printf("slflx_c=%e\n",slflx_c[66]);
// 			printf("qtflx_c=%e\n",qtflx_c[66]);
// 			printf("flxprc1_c=%e\n",flxprc1_c[66]);
// 			printf("flxsnow1_c=%e\n",flxsnow1_c[66]);
				
// //		}

// //		for(i=0;i<iend;++i)	
// //		{
// 		//if(umf_c[k*ncol+i]>=1.0e-12)		
// 			printf("cush=%e\n",cush[66]);
// 			printf("snow=%e\n",snow[66]);
// 			printf("precip=%e\n",precip[66]);
// 			printf("cbmf=%e\n",cbmf[66]);
// 			printf("rliq=%e\n",rliq[66]);
// 			printf("cnt_c=%e\n",cnt_c[66]);			
// //		}
// //	for( k=mkx-1;k>=0;--k)	
// //		for(i=0;i<iend;++i)	
// //		{
// 		//if(umf_c[k*ncol+i]>=1.0e-12)		
// 			printf("qrten_c=%e\n",qrten_c[66]);
// 			printf("qvten_c=%e\n",qvten_c[66]);
// 			printf("sten_c=%e\n",sten_c[66]);
// 			printf("evapc_c=%e\n",evapc_c[66]);
// 			printf("qlu_c=%e\n",qlu_c[66]);
				
//		}
		//! Reverse cloud top/base interface indices
		
		for(i=0;i<iend;++i)
		{
			cnt_inv[i] = mkx + 1 - cnt_c[i];
			cnb_inv[i] = mkx + 1 - cnb_c[i];
		}
		for(k=0;k<mkx+1;++k)
		{
			for(i=0;i<iend;++i)
			{
				k_inv                  = mkx  - k;
       			umf_inv[k_inv*iend+i]   = umf_c[k*iend+i];   

				wu_inv[k_inv*iend+i]    = wu_c[k*iend+i];

       			slflx_inv[k_inv*iend+i] = slflx_c[k*iend+i];     
       			qtflx_inv[k_inv*iend+i] = qtflx_c[k*iend+i];
       			flxprc1_inv[k_inv*iend+i] = flxprc1_c[k*iend+i];    //! reversed for output to cam
       			flxsnow1_inv[k_inv*iend+i] = flxsnow1_c[k*iend+i];  //! ""
			}
		}
		for(k=0;k<mkx;++k)
		{
			for(i=0;i<iend;++i)
			{
				k_inv                          = mkx - 1 - k;
				qvten_inv[k_inv*iend+i]        = qvten_c[k*iend+i];   
				qlten_inv[k_inv*iend+i]        = qlten_c[k*iend+i];   
				qiten_inv[k_inv*iend+i]        = qiten_c[k*iend+i];   
				sten_inv[k_inv*iend+i]         = sten_c[k*iend+i];    
				uten_inv[k_inv*iend+i]         = uten_c[k*iend+i];    
				vten_inv[k_inv*iend+i]         = vten_c[k*iend+i];    
				qrten_inv[k_inv*iend+i]        = qrten_c[k*iend+i];   
				qsten_inv[k_inv*iend+i]        = qsten_c[k*iend+i];   
				evapc_inv[k_inv*iend+i]        = evapc_c[k*iend+i];
				cufrc_inv[k_inv*iend+i]        = cufrc_c[k*iend+i];   
				qcu_inv[k_inv*iend+i]          = qcu_c[k*iend+i];     
				qlu_inv[k_inv*iend+i]          = qlu_c[k*iend+i];     
				qiu_inv[k_inv*iend+i]          = qiu_c[k*iend+i];     
				qc_inv[k_inv*iend+i]           = qc_c[k*iend+i]; 
			}
		}
		for(m=0;m<ncnst;++m)
		{
			for(k=0;k<mkx;++k)
			{
				for(i=0;i<iend;++i)
				{
					k_inv = mkx - 1 - k;
					trten_inv[m*mkx*iend+k_inv*iend+i]   = trten_c[m*mkx*iend+k*iend+i];
				}
			}
		}
		// for(m=0;m<ncnst;++m)
		// {
		// 	for(k=0;k<mkx;++k)
		// 	{
		// 		if(trten_inv[m*mkx*iend+k_inv*iend+221]!=0.0)
		// 		printf("trten_inv=%e\n",trten_inv[m*mkx*iend+k_inv*iend+221]);
		// 	}
		// }
		// for(k=0;k<mkx;++k)
		// 	if(umf_inv[k*iend+221]!=0.0)
		// 	printf("umf_inv=%e\n",umf_inv[k*iend+221]);

		// !======================== zhh debug 2012-02-09 =======================     
		// !!    print*, '------------ At the end of sub. compute_uwshcu_inv ---------------'
		// !!    print*, 'trten(8,1,17) =', trten(8,1,17)
		// !!    print*, '-------------------------------------------------------------------'
		// !======================== zhh debug 2012-02-09 ======================= 
// printf("cnt_inv[1]=%e\n",cnt_inv[1]);
// printf("cnb_inv[1]=%e\n",cnb_inv[1]);
// printf("umf_inv[33*iend+1]=%e\n",umf_inv[33*iend+1]);
// printf("slflx_inv[33*iend+1]=%e\n",slflx_inv[33*iend+1]);
// printf("qtflx_inv[33*iend+1]=%e\n",qtflx_inv[33*iend+1]);
// printf("flxprc1_inv[33*iend+1]=%e\n",flxprc1_inv[33*iend+1]);
// printf("flxsnow1_inv[33*iend+1]=%e\n",flxsnow1_inv[33*iend+1]);

// printf("qvten_inv[33*iend+1]=%e\n",qvten_inv[33*iend+1]);
// printf("qlten_inv[33*iend+1]=%e\n",qlten_inv[33*iend+1]);
// printf("qiten_inv[33*iend+1]=%e\n",qiten_inv[33*iend+1]);

// printf("sten_inv[33*iend+1]=%e\n",sten_inv[33*iend+1]);
// printf("uten_inv[33*iend+1]=%e\n",uten_inv[33*iend+1]);
// printf("vten_inv[33*iend+1]=%e\n",vten_inv[33*iend+1]);

// printf("qrten_inv[33*iend+1]=%e\n",qrten_inv[33*iend+1]);
// printf("qsten_inv[33*iend+1]=%e\n",qsten_inv[33*iend+1]);
// printf("evapc_inv[33*iend+1]=%e\n",evapc_inv[33*iend+1]);
// printf("cufrc_inv[33*iend+1]=%e\n",cufrc_inv[33*iend+1]);
// printf("qcu_inv[33*iend+1]=%e\n",qcu_inv[33*iend+1]);
// printf("qlu_inv[33*iend+1]=%e\n",qlu_inv[33*iend+1]);
// printf("qiu_inv[33*iend+1]=%e\n",qiu_inv[33*iend+1]);
// printf("qc_inv[33*iend+1]=%e\n",qc_inv[33*iend+1]);

// printf("trten_inv[20*iend*mkx+33*iend+1]=%e\n",trten_inv[20*iend*mkx+33*iend+1]);

hipHostFree(ps0_c);
hipHostFree(zs0_c);

hipHostFree(p0_c);

hipHostFree(z0_c);

hipHostFree(dp0_c);

hipHostFree(dpdry0_c);

hipHostFree(u0_c);

hipHostFree(v0_c);

hipHostFree(qv0_c);

hipHostFree(ql0_c);

hipHostFree(qi0_c);

hipHostFree(t0_c);

hipHostFree(s0_c);

hipHostFree(tr0_c);

hipHostFree(tke_c);

hipHostFree(cldfrct_c);

hipHostFree(concldfrct_c);

////double pblh_c[pcols];
////hipHostMalloc((void**) &pblh_c,pcols*sizeof(double));

///////////////////////in-out////////////////
////double cush_c[pcols]; 
////hipHostMalloc((void**) &cush_c,pcols*sizeof(double));

// //////////////////////output/////////////////
hipHostFree(umf_c);

hipHostFree(qvten_c);

hipHostFree(qlten_c);

hipHostFree(qiten_c);

hipHostFree(sten_c);

hipHostFree(uten_c);

hipHostFree(vten_c);

hipHostFree(trten_c);

hipHostFree(qrten_c);

hipHostFree(qsten_c);

//	double *precip_c[pcols];
//	//hipHostMalloc((void**) &precip_c,pcols*sizeof(double));
//	double *snow_c[pcols];
//	//hipHostMalloc((void**) &snow_c,pcols*sizeof(double));
hipHostFree(evapc_c);

// double *rliq_c[pcols];
// //hipHostMalloc((void**) &rliq_c,pcols*sizeof(double));slflx_c
hipHostFree(slflx_c);

hipHostFree(qtflx_c);

hipHostFree(flxprc1_c);

hipHostFree(flxsnow1_c);


hipHostFree(cufrc_c);

hipHostFree(qcu_c);

hipHostFree(qlu_c);

hipHostFree(qiu_c);

hipHostFree(qc_c);

// double *cbmf_c[pcols];
// //hipHostMalloc((void**) &cbmf_c,pcols*sizeof(double));
hipHostFree(cnt_c);

hipHostFree(cnb_c);

// hipHostFree(wv_saturation_mp_estbl_c);

// hipHostFree(wv_saturation_mp_pcf_c);
hipHostFree(qw0_c);

hipHostFree(tw0_c);

hipHostFree(wu_c);


hipFree(ps0_d);	//ps0 = NULL;
hipFree(zs0_d);	//zs0 = NULL;
hipFree(p0_d);	//p0 = NULL;
hipFree(z0_d);	//z0 = NULL;
hipFree(dp0_d);	//dp0 = NULL;
hipFree(dpdry0_d);	//dpdry0 = NULL;
hipFree(u0_d);	//u0 = NULL;
hipFree(v0_d);	//v0 = NULL;
hipFree(qv0_d);	//qv0 = NULL;
hipFree(ql0_d);	//ql0 = NULL;
hipFree(qi0_d);	//qi0 = NULL;
hipFree(t0_d);	//t0 = NULL;
hipFree(s0_d);	//s0 = NULL;
hipFree(tr0_d);	//tr0 = NULL;
hipFree(tke_d);	//tke = NULL;
hipFree(cldfrct_d);	//cldfrct = NULL;
hipFree(concldfrct_d);	//concldfrct = NULL;
hipFree(pblh_d);	//pblh = NULL;
hipFree(cush_d);	//cush = NULL;
hipFree(umf_d);	//umf = NULL;
hipFree(qvten_d);	//qvten = NULL;
hipFree(qlten_d);	//qlten = NULL;
hipFree(qiten_d);	//qiten = NULL;
hipFree(sten_d);	//sten = NULL;
hipFree(uten_d);	//uten = NULL;
hipFree(vten_d);	//vten = NULL;
hipFree(trten_d);	//trten = NULL;
hipFree(qrten_d);	//qrten = NULL;
hipFree(qsten_d);	//qsten = NULL;
hipFree(precip_d);	//precip = NULL;
hipFree(snow_d);	//snow = NULL;
hipFree(evapc_d);	//evapc = NULL;
hipFree(rliq_d);	//rliq = NULL;
hipFree(slflx_d);	//slflx = NULL;
hipFree(qtflx_d);	//qtflx = NULL;
hipFree(flxprc1_d);	//flxprc1 = NULL;
hipFree(flxsnow1_d);	//flxsnow1 = NULL;
hipFree(cufrc_d);	//cufrc = NULL;
hipFree(qcu_d);	//qcu = NULL;
hipFree(qlu_d);	//qlu = NULL;
hipFree(qiu_d);	//qiu = NULL;
hipFree(qc_d);	//qc = NULL;
hipFree(cbmf_d);	//cbmf = NULL;
hipFree(cnt_d);	//cnt = NULL;
hipFree(cnb_d);	//cnb = NULL;
hipFree(qmin);
hipFree(numptr_amode);
hipFree(tw0_in);
hipFree(qw0_in);

 hipFree(wv_saturation_mp_estbl_d);
// hipFree(wv_saturation_mp_pcf_d);
hipFree(wu_d);
		return;
}
