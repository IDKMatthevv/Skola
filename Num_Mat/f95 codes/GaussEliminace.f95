program GaussElim
 implicit none
! 
 integer,parameter :: n=3
 integer :: i 
 real(8) :: a(n,n),b(n),x(n)
!
! Data "natvrdo" do programu :-)
!
 a=RESHAPE((/ 1.0d0, 4.0d0, 9.0d0,  &
              2.0d0, 5.0d0, 6.0d0,  &
	          3.0d0, 6.0d0, 5.0d0/),  &
	       (/3,3/))
 b=(/14.0d0,32.0d0,36.0d0/)
!
 call Gauss(n,a,b,x)
!
 print*," x:"
 do i=1,n
  print*,x(i)
 enddo
! 
 stop " GaussElim: Konec vypoctu!"
end program GaussElim
!
subroutine Gauss(n,a,b,x)
 implicit none
 integer :: n,k,i,j 
 real(8) :: a(n,n),b(n),x(n),temp,tsum
!
! Primy chod
!
 do k=1,n-1
  do i=k+1,n
   temp=a(i,k)/a(k,k)
!   
   do j=k+1,n
    a(i,j)=a(i,j)-temp*a(k,j)
   enddo
!   
   b(i)=b(i)-temp*b(k)
  enddo
 enddo
!
! Zpetny chod
!
 x(n)=b(n)/a(n,n)
! 
 do i=n-1,1,-1
  tsum=0.0d0
!  
  do j=i+1,n
   tsum=tsum+a(i,j)*x(j)
  enddo
!  
  x(i)=(b(i)-tsum)/a(i,i)
 enddo
end subroutine Gauss
