module mesh_operations
    use, intrinsic :: iso_fortran_env, only: dp => real64
    implicit none
contains

    subroutine read_mesh(vertices_file, faces_file, vertices, faces)
        character(len=*), intent(in) :: vertices_file, faces_file
        real(dp), allocatable, intent(out) :: vertices(:,:), faces(:,:)
        integer :: n_vertices, n_faces, i
        character(len=256) :: line

        ! Read vertices
        open(unit=11, file=vertices_file, status='old', action='read')
        n_vertices = 0
        do
            read(11,'(A)', end=100) line
            n_vertices = n_vertices + 1
        end do
100     continue
        rewind(11)
        allocate(vertices(n_vertices,3))
        do i = 1, n_vertices
            read(11,*) vertices(i,1), vertices(i,2), vertices(i,3)
        end do
        close(11)

        ! Read faces
        open(unit=12, file=faces_file, status='old', action='read')
        n_faces = 0
        do
            read(12,'(A)', end=200) line
            n_faces = n_faces + 1
        end do
200     continue
        rewind(12)
        allocate(faces(n_faces,3))
        do i = 1, n_faces
            read(12,*) faces(i,1), faces(i,2), faces(i,3)
        end do
        close(12)

    end subroutine read_mesh

    subroutine compute_normals_areas(vertices, faces, normals, areas)
        real(dp), intent(in) :: vertices(:,:), faces(:,:)
        real(dp), intent(out) :: normals(:,:), areas(:)
        integer :: i
        real(dp) :: v1(3), v2(3), v3(3)
        real(dp) :: vec1(3), vec2(3)
        real(dp) :: cross(3), norm_cross

        do i = 1, size(faces,1)
            v1 = vertices(int(faces(i,1)), :)
            v2 = vertices(int(faces(i,2)), :)
            v3 = vertices(int(faces(i,3)), :)

            vec1 = v2 - v1
            vec2 = v3 - v1
            cross = cross_product(vec1, vec2)
            norm_cross = norm2(cross)

            if (norm_cross > 0.0_dp) then
                normals(i,:) = cross / norm_cross
            else
                normals(i,:) = (/ 0.0_dp, 0.0_dp, 0.0_dp /)
            end if

            areas(i) = 0.5_dp * norm_cross
        end do

    end subroutine compute_normals_areas

    function cross_product(a, b) result(c)
        real(dp), intent(in) :: a(3), b(3)
        real(dp) :: c(3)

        c(1) = a(2)*b(3) - a(3)*b(2)
        c(2) = a(3)*b(1) - a(1)*b(3)
        c(3) = a(1)*b(2) - a(2)*b(1)
    end function cross_product

    function norm2(vec) result(val)
        real(dp), intent(in) :: vec(3)
        real(dp) :: val

        val = sqrt(sum(vec**2))
    end function norm2

end module mesh_operations
