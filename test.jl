include("StrongArrays.jl"); using .StrongArrays; S=StrongArrays


V=Base.OneTo
f(::V{T},::V{T}) where{T} = "val is $T"
f(::V,::V) = "any val"

W=Val
g(::W{T},::W{T}) where{T} = "val is $T"
g(::W,::W) = "any val"
@StrongInt Foo
@StrongInt Bar

VFoo{T} = StrongVector{Foo,T,Vector{T}}
MFooBar{T} = StrongMatrix{Foo,Bar,T,Matrix{T}}
MFooInt{T} = StrongMatrix{Foo,Int,T,Matrix{T}}

v0=[i^2 for i in 1:5]
m0 =[1 2;3 4]
v = VFoo(v0)
v2=VFoo(Foo[1,2])
m1 = MFooBar([1 2;3 4])
m2 = MFooInt([1 3; 4 6])
c0=CartesianIndices(axes(m0))
c1=CartesianIndices(axes(m1))
p0=pairs(m0);
p1=pairs(m1);

a0=axes(m0)
a1=axes(m1)
i0=a0[1]
i1=a1[1]

B=Broadcast

bc1=B.broadcasted(+,m1,m1)
bc0=B.broadcasted(+,m0,m0)
bci1=B.instantiate(bc1)
bci0=B.instantiate(bc0)

m1+m1
2m1
