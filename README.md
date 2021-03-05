# StrongArrays

This package introduces arrays with strongly typed indices,

Using these “funny integers” as indices prevents accidentally using the
wrong variable for indexing an array. Since everything uses very thin
wrappers of the standard array types, this should have zero performance
cost.


## Example
```julia
using StrongArrays

# first define an index type:
Foo = StrongInt{:Foo}
# or via this macro, which also adds a `Base.show` method:
@StrongInt Bar

Foo(1)
Foo(1) + Foo(2) # basic operations are supported
Foo(1) + 1 # but this is an error

# Basic arithmetic is supported, but only additive additions, since these
# are the only ones which make sense for indices. (More formally,
# `StrongInt` types behave as free modules over the ring of `Int`s.)

Foo(1) + one(Foo(1)) # incrementation is done this way


# conversion is possible:
Foo(1) == Foo(Bar(1))
Int(Foo(2)) == 2


# then define an array type:
VFoo = StrongVector{Foo}
VBar = StrongVector{Bar}

# and use this to build arrays:
v1 = VFoo([8,13,21])
v1[1] # TypeError: in coordinate 1, expected Foo, got Int64
v1[Foo(1)] # 8

v2 = VBar{Foo}([3,2,1])
v2[Bar(1)] # returns Foo(3)
v1[v2[Bar(3)]] # returns 8


MFooBar = StrongMatrix{Foo,Bar}
MFooInt = StrongMatrix{Foo,Int}

m1 = MFooBar([1 2;3 4])
m2 = MFooInt([1 2;3 4])

m1[:,Bar(2)] # is a VFoo

# conversion is possible:
Vector(v1)
Matrix(m1)

```
