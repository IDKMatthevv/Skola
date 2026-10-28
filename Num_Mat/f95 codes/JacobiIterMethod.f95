program JacobiIter
 implicit none
 integer,parameter :: n=4
 integer :: i 
 real(8),parameter :: eps=1.0d-6
 real(8) :: a(n,n),b(n),x(n)
!
! Data "natvrdo" do programu :-)
!
 a=RESHAPE((/ 4.0d0,-1.0d0, 0.0d0, 2.0d0,    &
             -1.0d0,10.0d0, 2.0d0,-1.0d0,    &
	      0.0d0, 2.0d0, 7.0d0,-1.0d0,    &
	      2.0d0,-1.0d0,-1.0d0, 5.0d0/),  &
 	    (/4,4/))
 b=(/7.0d0,-10.0d0,4.0d0,7.0d0/)
!
! Pocatecni iterace = 0
!
 x=(/0.0d0,0.0d0,0.0d0,0.0d0/)   
!
 call Jacobi(n,eps,a,b,x)
!
 print*," x:"
! 
 do i=1,n
  print*,x(i)
 enddo
! 
 stop " JacobiIter: Konec vypoctu!"
end program JacobiIter
!
subroutine Jacobi(n,eps,a,b,x)
 implicit none
 integer,parameter :: kmax=1000
 integer :: n,i,j,k 
 real(8) :: a(n,n),b(n),x(n),x0(n),  &
            eps,dsum,SN
!
 x0=x
!
 do k=1,kmax
  SN=0.0d0  ! Akumulator pro sloupcovou normu
  do i=1,n
   dsum=b(i)
   do j=1,n
    if(i == j) cycle
    dsum=dsum-a(i,j)*x0(j)
   enddo
   x(i)=dsum/a(i,i)
   SN=SN+dabs(x(i)-x0(i))  ! Sloupcova norma
  enddo
  print*," iterace =",k,"sloupcova norma =",SN
  if(SN < eps) exit
  x0=x
 enddo
 if(k > kmax) stop " Jacobi: Prilis mnoho iteraci!"
end subroutine Jacobi
