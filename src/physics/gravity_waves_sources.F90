! zmh added the following line for SCAM
#if (! defined BFB_CAM_SCAM_IOP )  

module gravity_waves_sources


  use shr_kind_mod, only: r8 => shr_kind_r8
  use pmgrid, only: plev, plevp, beglatxy, endlatxy, beglonxy, endlonxy

  implicit none
  save
  private
  public :: gws_src_fnct

  ! ghosting added by Francis Vitt -- 7 July 2008
  ! zmh added vorticity, horizontal shear and tq gradient calculations, put in top 5 layers

  contains


!=================================================================

    subroutine gws_src_fnct (u3,v3,pt, q3, pe, grid, frontgf, frontga)

      use physconst,     only: zvir, cappa, aearth => rearth
      use ppgrid,        only: pcols
      use hycoef,        only: psr,ps0
      use dynamics_vars, only: T_FVDYCORE_GRID

      implicit none

! Input/Output arguments
      real(r8), intent(in) :: u3(beglonxy:endlonxy,plev,beglatxy:endlatxy)          ! zonal velocity
      real(r8), intent(in) :: v3(beglonxy:endlonxy,plev,beglatxy:endlatxy)          ! meridional velocity
!zmh corrected original code
      !zmh real(r8), intent(in) :: pt(beglonxy:endlonxy,beglatxy:endlatxy,plev)          ! virtual temperature
      !zmh real(r8), intent(in) :: q3(beglonxy:endlonxy,beglatxy:endlatxy,plev)          ! water constituent
      real(r8), intent(in) :: pt(beglonxy:endlonxy,plev,beglatxy:endlatxy)          ! virtual temperature
      real(r8), intent(in) :: q3(beglonxy:endlonxy,plev,beglatxy:endlatxy)          ! water constituent

      real(r8), intent(in) :: pe(beglonxy:endlonxy,plevp,beglatxy:endlatxy)         ! interface pressure
      type (T_FVDYCORE_GRID), intent(in) :: grid    ! grid for XY decomp

      real(r8), intent(out) :: frontgf(beglonxy:endlonxy,plev,beglatxy:endlatxy)    ! Frontogenesis function
      real(r8), intent(out) :: frontga(beglonxy:endlonxy,plev,beglatxy:endlatxy)    ! Frontogenesis angle

! Locals
      real(r8) :: ptemp(beglonxy:endlonxy,plev,beglatxy:endlatxy)       ! temperature
      real(r8) :: pm(beglonxy:endlonxy   ,plev,beglatxy:endlatxy)       ! mid-point pressure
      real(r8) :: pexf                                ! Exner function
      real(r8) :: dummy

      real(r8) :: pty(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! temperature meridional gradient
      real(r8) :: ptx(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! temperature zonal gradient
!zmh
      real(r8) :: qy(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! water vapor meridional gradient
      real(r8) :: qx(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! water vapor zonal gradient

      real(r8) :: uy(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! U-wind meridional gradient
      real(r8) :: ux(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! U-wind zonal gradient

      real(r8) :: vy(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! V-wind meridional gradient
      real(r8) :: vx(beglonxy:endlonxy,plev,beglatxy:endlatxy)         ! V-wind zonal gradient

      real(r8) :: ptg(beglonxy-1:endlonxy+1,plev,beglatxy-1:endlatxy+1)  ! temperature ghosted
!zmh
      real(r8) ::  qg(beglonxy-1:endlonxy+1,plev,beglatxy-1:endlatxy+1)  ! water vapor ghosted

      real(r8) ::  ug(beglonxy-1:endlonxy+1,plev,beglatxy-1:endlatxy+1)  ! U-wind ghosted
      real(r8) ::  vg(beglonxy-1:endlonxy+1,plev,beglatxy-1:endlatxy+1)  ! V-wind ghosted

      real(r8) :: tglat                               ! tangent-latitude
      integer  :: i,j,k
      integer  :: im, ip, k2  !zmh added k2

!-----------------------------------------------------------------------------------------

      pty(:,:,:) = 0._r8
      ptx(:,:,:) = 0._r8
! zmh
      qy(:,:,:) = 0._r8
      qx(:,:,:) = 0._r8

      uy(:,:,:) = 0._r8
      ux(:,:,:) = 0._r8
      vy(:,:,:) = 0._r8
      vx(:,:,:) = 0._r8
      frontgf(:,:,:) = 0._r8
      frontga(:,:,:) = 0._r8

      !$omp parallel do private (i,j,k,pexf)
      do j = beglatxy, endlatxy
         do k = 1, plev
            do i = beglonxy, endlonxy

               ! Calculate pressure and Exner function
               pm(i,k,j) = 0.5_r8 * ( pe(i,k,j) + pe(i,k+1,j) )
               pexf      = (ps0 / pm(i,k,j))**cappa

               ! Convert virtual temperature to temperature and calculate potential temperature
!zmh
!               ptemp(i,k,j) = pt(i,j,k) / (1._r8 + zvir*q3(i,j,k)) * pexf
               ptemp(i,k,j) = pt(i,k,j) / (1._r8 + zvir*q3(i,k,j)) * pexf

            end do
         end do
      end do

!      write(*,*) 'pt:',pt(beglonxy,plev,:)
!      write(*,*) 'u3:',u3(beglonxy,plev,:)
!      write(*,*) 'q3:',q3(beglonxy,plev,:)

      call ghost_array( ptemp, ptg, grid )
      call ghost_array( u3, ug, grid )
      call ghost_array( v3, vg, grid )
!zmh
      call ghost_array( q3, qg, grid )

!      write(*,*) 'ptg:',ptg(beglonxy,plev,:)
!      write(*,*) 'ug:',ug(beglonxy,plev,:)
!      write(*,*) 'qg:',qg(beglonxy,plev,:)

      !$omp parallel do private (i,j,k)
      do k=1, plev
         do i=beglonxy, endlonxy
            do j=beglatxy, endlatxy

               ! Pot. Temperature
               pty(i,k,j) = ( ptg(i,k,j+1) - ptg(i,k,j-1) ) / (2._r8 * grid%dp)
               pty(i,k,j) = pty(i,k,j) / aearth
!zmh
               ! humidity to avoid numerical stripes in y
               qy(i,k,j) = ( qg(i,k,j+1) - qg(i,k,j-1) ) / (2._r8 * grid%dp)
               qy(i,k,j) = qy(i,k,j) / aearth
               ! just take the difference 
               !qy(i,k,j) = max( abs(qg(i,k,j+1) - qg(i,k,j) ),abs(qg(i,k,j) - qg(i,k,j-1) ) )

               ! U-wind
               uy(i,k,j) = ( ug(i,k,j+1) - ug(i,k,j-1) ) / (2._r8 * grid%dp)
               uy(i,k,j) = uy(i,k,j) / aearth

               ! V-wind
               vy(i,k,j) = ( vg(i,k,j+1) - vg(i,k,j-1) ) / (2._r8 * grid%dp)
               vy(i,k,j) = vy(i,k,j) / aearth

            end do
         end do
      end do

      !++rrg use 1.e-3 floor on cosine terms in the denominator of frontgf

      !$omp parallel do private (i,j,k,im,ip)
      do k=1, plev
         do j=beglatxy, endlatxy
            do i=beglonxy, endlonxy

               im = i-1
               ip = i+1

               ! Pot. Temperature
               ptx(i,k,j) = ( ptg(ip,k,j) - ptg(im,k,j) ) / (2._r8 * grid%dl)
               ptx(i,k,j) = ptx(i,k,j) / (aearth * (grid%cosp(j)+1.e-3_r8))
!zmh
               ! humidity
               qx(i,k,j) = ( qg(ip,k,j) - qg(im,k,j) ) / (2._r8 *grid%dl)
               qx(i,k,j) = qx(i,k,j) / (aearth * (grid%cosp(j)+1.e-3_r8))
               !qx(i,k,j) = max( abs(qg(ip,k,j) - qg(i,k,j)),abs(qg(i,k,j) - qg(im,k,j))) 

               ! U-wind
               ux(i,k,j) = ( ug(ip,k,j) - ug(im,k,j) ) / (2._r8 *grid%dl)
               ux(i,k,j) = ux(i,k,j) / (aearth * (grid%cosp(j)+1.e-3_r8))

               ! V-wind
               vx(i,k,j) = ( vg(ip,k,j) - vg(im,k,j) ) / (2._r8 *grid%dl)
               vx(i,k,j) = vx(i,k,j) / (aearth * (grid%cosp(j)+1.e-3_r8))

            end do
         end do
      end do

      !$omp parallel do private (i,j,k, tglat)
      do j=beglatxy, endlatxy

         tglat = grid%sinp(j) / (grid%cosp(j)+1.e-3_r8)

         do k=1, plev
            do i=beglonxy, endlonxy

               frontgf(i,k,j) =                                                                          &
                    - ptx(i,k,j)**2._r8 * (ux(i,k,j) - v3(i,k,j) * tglat / aearth)                          &
                    - pty(i,k,j)**2._r8 * vy(i,k,j)                                                         &
                    - ptx(i,k,j) * pty(i,k,j) * ( vx(i,k,j) + uy(i,k,j) + u3(i,k,j) * tglat / aearth )

            end do
         end do

      end do

      !--rrg use 1.e-3 floor on cosine terms in the denominator of frontgf

      !$omp parallel do private (i,j,k)
      do j=beglatxy, endlatxy
         do k=1, plev
            do i=beglonxy, endlonxy
               frontga(i,k,j) = atan2 ( pty(i,k,j) , ptx(i,k,j) + 1.e-10_r8 )
            end do
         end do
      end do


!zmh 2018-09-20
      !$omp parallel do private (i,j,k,k2)
      do j=beglatxy, endlatxy
         do k=1,5 
            k2 = plev-k+1 
            do i=beglonxy, endlonxy
               frontga(i,k,j) = vx(i,k2,j) - uy(i,k2,j)
            end do
         end do
      end do

! stored pt gradient of the lowerest layer into frontgf
      !$omp parallel do private (i,j) !,k2)
      k2 = plev
      do j=beglatxy, endlatxy
            do i=beglonxy, endlonxy
               frontgf(i,1,j) = sqrt( ptx(i,k2,j)*ptx(i,k2,j) + pty(i,k2,j)*pty(i,k2,j) )  ! 
               frontgf(i,2,j) = sqrt( qx(i,k2,j)*qx(i,k2,j) + qy(i,k2,j)*qy(i,k2,j) )
               !frontgf(i,2,j) = max( qx(i,k2,j), qy(i,k2,j) )
               frontgf(i,3,j) =  qx(i,k2,j)
               frontgf(i,4,j) =  qy(i,k2,j) 
               frontgf(i,5,j) =  ptx(i,k2,j) 
               frontgf(i,6,j) =  pty(i,k2,j) 
            end do
      end do

      return

    end subroutine gws_src_fnct

    subroutine ghost_array( x, xg, grid )

      ! subroutine added by Francis Vitt -- 7 July 2008

#if defined( SPMD )
      use mod_comm,      only: mp_send3d, mp_recv3d
#endif
      use dynamics_vars, only: T_FVDYCORE_GRID

      implicit none

      ! Input/Output arguments
      type (T_FVDYCORE_GRID), intent(in) :: grid    ! grid for XY decomp
      real(r8), intent(in)  :: x(beglonxy:endlonxy,plev,beglatxy:endlatxy)          ! zonal velocity
      real(r8), intent(out) :: xg(beglonxy-1:endlonxy+1,plev,beglatxy-1:endlatxy+1)          ! zonal velocity

      ! local variables
      real(r8) :: north(beglonxy:endlonxy,plev)
      real(r8) :: south(beglonxy:endlonxy,plev)
      real(r8) :: east(plev,beglatxy:endlatxy)
      real(r8) :: west(plev,beglatxy:endlatxy)
      integer  :: im, jm, km, ifirstxy, ilastxy, jfirstxy, jlastxy, iam, myidxy_y, nprxy_x, nprxy_y
      integer  :: itot, dest, src, j, k

      im = grid%im
      jm = grid%jm
      km = grid%km

      ifirstxy = grid%ifirstxy
      ilastxy  = grid%ilastxy
      jfirstxy = grid%jfirstxy
      jlastxy  = grid%jlastxy

      iam      = grid%iam
      myidxy_y = grid%myidxy_y
      nprxy_x  = grid%nprxy_x
      nprxy_y  = grid%nprxy_y
      itot = ilastxy-ifirstxy+1

      xg(ifirstxy:ilastxy,:,jfirstxy:jlastxy) = x(ifirstxy:ilastxy,:,jfirstxy:jlastxy)

#if defined( SPMD )

  ! north
      call mp_send3d( grid%commxy, iam-nprxy_x, iam+nprxy_x, im, km, jm,      &
                      ifirstxy, ilastxy, 1, km, jfirstxy, jlastxy,           &
                      ifirstxy, ilastxy, 1, km, jfirstxy, jfirstxy, x )
      call mp_recv3d( grid%commxy, iam+nprxy_x, im, jm, km,                   &
                      ifirstxy, ilastxy, 1, km, jlastxy+1, jlastxy+1,        &
                      ifirstxy, ilastxy, 1, km, jlastxy+1, jlastxy+1, north )

  ! south
      call mp_send3d( grid%commxy, iam+nprxy_x, iam-nprxy_x, im, km, jm,      &
                      ifirstxy, ilastxy, 1, km, jfirstxy, jlastxy,           &
                      ifirstxy, ilastxy, 1, km, jlastxy,  jlastxy, x )
      call mp_recv3d( grid%commxy, iam-nprxy_x, im, jm, km,                   &
                      ifirstxy, ilastxy, 1, km, jfirstxy-1, jfirstxy-1,        &
                      ifirstxy, ilastxy, 1, km, jfirstxy-1, jfirstxy-1, south )

#endif

      if (itot .ne. im) then
#if defined( SPMD )

   ! east

         dest = myidxy_y*nprxy_x + MOD(iam+nprxy_x-1,nprxy_x)
         src  = myidxy_y*nprxy_x + MOD(iam+1,nprxy_x)
         call mp_send3d( grid%commxy, dest, src, im, km, jm,                  &
                         ifirstxy, ilastxy, 1, km, jfirstxy, jlastxy,        &
                         ifirstxy, ifirstxy, 1, km, jfirstxy, jlastxy, x )
         call mp_recv3d( grid%commxy, src, im, km, jm,                        &
                         ilastxy+1, ilastxy+1, 1, km, jfirstxy, jlastxy,     &
                         ilastxy+1, ilastxy+1, 1, km, jfirstxy, jlastxy, east )

   ! west

         dest = myidxy_y*nprxy_x + MOD(iam+1,nprxy_x)
         src  = myidxy_y*nprxy_x + MOD(iam+nprxy_x-1,nprxy_x)
         call mp_send3d( grid%commxy, dest, src, im, km, jm,                  &
                         ifirstxy, ilastxy, 1, km, jfirstxy, jlastxy,          &
                         ilastxy,  ilastxy, 1, km, jfirstxy, jlastxy, x )
         call mp_recv3d( grid%commxy, src, im, km, jm,                        &
                         ifirstxy-1, ifirstxy-1, 1, km, jfirstxy, jlastxy, &
                         ifirstxy-1, ifirstxy-1, 1, km, jfirstxy, jlastxy, west )
#endif

      else
!$omp parallel do private(j, k)
         do k = 1,km
            do j=jfirstxy,jlastxy
               east(k,j) = x(1, k,j)
               west(k,j) = x(im,k,j)
            enddo
         enddo
      endif

      if ( jfirstxy == 1 ) then
         xg(ifirstxy:ilastxy,:,jfirstxy-1) = xg(ifirstxy:ilastxy,:,jfirstxy)
      else
         xg(ifirstxy:ilastxy,:,jfirstxy-1) = south
      endif

      if ( jlastxy == jm ) then
         xg(ifirstxy:ilastxy,:,jlastxy+1) = xg(ifirstxy:ilastxy,:,jlastxy)
      else
         xg(ifirstxy:ilastxy,:,jlastxy+1) = north
      endif

      xg(ifirstxy-1,:,jfirstxy:jlastxy) = west
      xg( ilastxy+1,:,jfirstxy:jlastxy) = east

    end subroutine ghost_array
!=================================================================

  end module gravity_waves_sources
#endif
