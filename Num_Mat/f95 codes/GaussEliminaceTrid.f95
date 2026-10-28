program GaussElimTriD
 implicit none
! 
 integer,parameter :: n=5
 integer :: i 
 real(8) :: p(n),d(n),s(n),b(n)
!
! Data "natvrdo" do programu :-)
!
 p=(/0.0d0,-0.1125d0,-0.1125d0,-0.1125d0,-0.1125d0/)
 d=(/1.225d0,1.225d0,1.225d0,1.225d0,1.225d0/)
 s=(/-0.1125d0,-0.1125d0,-0.1125d0,-0.1125d0,0.0d0/)
 b=(/1.307452d0,0.9725676d0,0.8852738d0,0.8472736d0,0.7876443d0/) 
!
 call GaussTriD(n,d,p,s,b)
!
 print*," x:"
 do i=1,n
  print*,b(i)
 enddo
! 
 stop " GaussElimTriD: Konec vypoctu!" 
end program GaussElimTriD
!
subroutine GaussTriD(n,d,p,s,b)
 implicit none
 integer :: i,n 
 real(8) :: d(*),p(*),s(*),b(*),temp
!
! Primy chod
!
 do i=2,n
  temp=p(i)/d(i-1)
  d(i)=d(i)-temp*s(i-1)
  b(i)=b(i)-temp*b(i-1)
 enddo
!
! Zpetny chod
!
 b(n)=b(n)/d(n)
! 
 do i=n-1,1,-1
  b(i)=(b(i)-s(i)*b(i+1))/d(i)
 enddo
end subroutine GaussTriD
