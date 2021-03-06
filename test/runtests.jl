using StrongArrays
using StrongArrays: IncompatibleTypes
using StaticArrays
using Test

@StrongInt Foo
@StrongInt Bar
VFoo = StrongVector{Foo}
VBar = StrongVector{Bar}
MFooBar = StrongMatrix{Foo,Bar}
@testset "StrongInt" begin#««1
@test_throws ErrorException Foo(1)+1
@test Foo(1)+one(Foo) == Foo(2)
@test_throws IncompatibleTypes Foo(1)==1
@test 1Foo isa Foo
end
@testset "StrongArray" begin#««1
v0=[0,13,21]
v1=VFoo(v0)
v2=VBar(v0)
@test_throws TypeError v1[1]
v1[Foo(1)] = 8
@test v1[Foo(1)] == 8
@test v1[Foo[1,2]] isa VFoo
@test typeof(similar(v1)) == typeof(v1)
v1[Foo[1,2,3]] = [0,0,0]
@test iszero(v1)
@test_throws TypeError v2[Foo(1)]

m0=[1 2;3 4]
m1=MFooBar(m0)
@test m1[Foo(1),:] isa VBar
@test m1[Foo(1),Bar(2)] == 2
@test_throws TypeError m1[1,1]
# @test_throws TypeError m0[Foo(1),Bar(2)]

end

@testset "Broadcasting" begin#««1
m1=MFooBar([1 2;3 4])
@test iszero(m1-m1)
@test !iszero(2m1)
end
@testset "`Strong` shortcut" begin#««1
@test Strong{Foo}Vector == StrongArray{1,Tuple{Foo}}
@test Strong{Foo}Vector{Int} == StrongArray{1,Tuple{Foo},Int,Vector{Int}}
@test Strong{Foo}SVector{2,Int} isa DataType
@test Strong{Foo}SVector{2} isa UnionAll
@test (S2i=Strong{Foo}SVector{2,Int}; S2i([1,2]).array isa SVector{2,Int})
@test (S2=Strong{Foo}SVector{2}; S2([1,2]).array isa SVector{2,Int})
@test Strong{Foo}Vector([1,2,3]) isa StrongVector{Foo,Int}
@test Strong{Foo}([1,2,3]) isa StrongVector{Foo,Int}
@test Strong{Foo}[1,2,3] isa StrongVector{Foo,Int}
@test Strong{Foo,Bar}[1 2;3 4] isa StrongMatrix{Foo,Bar,Int}
@test Strong{Foo,Bar}([1 2;3 4]) isa StrongMatrix{Foo,Bar,Int}
end
