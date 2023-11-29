module mod_foutput

use spmd_utils,     only: iam

implicit none
save

public  foutput_1d, foutput_2d


contains
!=============================================================================
subroutine foutput_2d(str, indata, nx, ny)
    use shr_kind_mod,       only: r8 => shr_kind_r8
    use time_manager,       only: get_nstep
implicit none
    character(len=*) :: str
    integer :: nx, ny
    real(r8) :: indata(nx, ny)

    character(len=255) :: tstr, cpustr
    character(len=255) :: fname 
    integer :: nstep, k
    
    nstep = get_nstep()
    write (tstr, '(I5.5)' ) nstep
    write (cpustr, '(I4.4)') iam
    fname = './foutput/foutput_'//trim(cpustr)//'_step_'//trim(tstr)//'.txt'
    open(unit=10, file=trim(fname), position="append")
    write(10, *) str, minval(indata), maxval(indata)
    close(10)
end subroutine foutput_2d
!=============================================================================
subroutine foutput_1d(str, indata, insize)
    use shr_kind_mod,       only: r8 => shr_kind_r8
    use time_manager,       only: get_nstep
implicit none
    character(len=*) :: str
    integer :: insize
    real(r8) :: indata(insize)

    character(len=255) :: tstr, cpustr
    character(len=255) :: fname 
    integer :: nstep, k
    
    nstep = get_nstep()
    write (tstr, '(I5.5)' ) nstep
    write (cpustr, '(I4.4)') iam
    fname = './foutput/foutput_'//trim(cpustr)//'_step_'//trim(tstr)//'.txt'
    open(unit=10, file=trim(fname), position="append")
    write(10, *) str, minval(indata(1:insize)), maxval(indata(1:insize))
    close(10)
end subroutine foutput_1d


end module mod_foutput
