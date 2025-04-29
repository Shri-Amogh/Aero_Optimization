program solver
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use mesh_operations
    implicit none

    integer :: i, n_vertices, n_faces
    real(dp), allocatable :: vertices(:,:), faces(:,:)
    real(dp) :: U_inf(3), rho
    real(dp), allocatable :: normals(:,:), areas(:)
    real(dp) :: total_force(3), drag, lift
    real(dp) :: cp

    U_inf = (/ 1.0_dp, 0.0_dp, 0.0_dp /)
    rho = 1.0_dp

    call read_mesh('data/vertices.dat', 'data/faces.dat', vertices, faces)

    n_vertices = size(vertices, 1)
    n_faces = size(faces, 1)

    allocate(normals(n_faces, 3))
    allocate(areas(n_faces))


    call compute_normals_areas(vertices, faces, normals, areas)


    total_force = 0.0_dp


    do i = 1, n_faces
        if (areas(i) > 1.0e-12_dp) then
            cp = -2.0_dp * dot_product(normals(i,:), U_inf)
            total_force = total_force + cp * areas(i) * normals(i,:)
        end if
    end do


    drag = -total_force(1) 
    lift = total_force(2) 


    if (.not.(isfinite(drag))) drag = 0.0_dp
    if (.not.(isfinite(lift))) lift = 0.0_dp


    open(unit=10, file='data/forces.dat', status='replace', action='write')
    write(10,'(2F12.6)') drag, lift
    close(10)


    
    deallocate(vertices, faces, normals, areas)

contains
    logical function isfinite(x)
        real(dp), intent(in) :: x
        isfinite = (x == x) .and. (abs(x) < huge(1.0_dp))
    end function isfinite

end program solver
