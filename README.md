# StrongArrays

This package introduces arrays with strongly typed indices,

Using these “funny integers” as indices prevents accidentally using the
wrong variable for indexing an array. Since everything uses very thin
wrappers of the standard array types, this should have zero performance
cost.

This package also contains a number of shorthands
(using the large extensibility of Julia's syntax)
so that typechecking your array indices is not too much of a hassle.


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
1Foo == Foo(1) # two notations are available for the same meaning

# Basic arithmetic is supported, but only additive additions, since these
# are the only ones which make sense for indices. (More formally,
# `StrongInt` types behave as free modules over the ring of `Int`s.)

Foo(1) + one(Foo(1)) # incrementation is done this way
Foo(1) + one # or this way


# conversion is possible, but needs to be explicited:
Foo(1) == Foo(Bar(1))
Int(Foo(2)) == 2
# or a::Foo = 1 (only on a local variable for now)


# then define an array type:
FooVec = StrongVector{Foo}
BarVec = Strong{Bar}Vector # this also works

# and use this to build arrays:
v1 = FooVec([8,13,21])
v1[1] # TypeError: in coordinate 1, expected Foo, got Int64
v1[Foo(1)] # 8

v2 = BarVec{Foo}([3,2,1])
v2[Bar(1)] # returns Foo(3)
v1[v2[Bar(3)]] # returns 8


FooBarMat = StrongMatrix{Foo,Bar}
FooIntMat = Strong{Foo,Int}Matrix{Int}

m1 = FooBarMat([1 2;3 4])
m2 = FooIntMat([1 2;3 4])
m3 = Strong{Foo,Int}[1 2;3 4]

m1[Foo(1),Bar(1)]
m1[:,Bar(2)] # is a FooVec

# conversion is possible:
Vector(v1)
Matrix(m1)

```
## Strongly-typed integers

Each strongly-typed `Int` alias is declared using the `@StrongInt` macro:
```julia
@StrongInt MyIndex
a = MyIndex(3)
```
These types have the properties of a module over `Int`: namely,
they can be added, subtracted, and compared together,
but not multiplied (which would not make sense for array indices);
they can be multiplied by integers, reduced modulo integers,
and divided by integers.

Since `a+1` is not supported, incrementation is done by `a+one(a)`;
the shorthand `a+one` is also provided.
Likewise, shorthands `a > zero` and `a > one` (etc.) are also provided.

As another shorthand, a `MyIndex` literal may be obtained by postfixing
the literal by the type, in this way: `a == 3MyIndex`.
(This saves **a whole, complete character** compared to typing
`MyIndex(3)`!!eleven!)

## Declaring strongly-indexed arrays

The following types are provided:

 - `StrongVector{I,T}` is a vector with contents of type `T`,
 indexed by values with strong type `I`; `I` may be any `Integer` type,
 including `Int`.
 - `StrongMatrix{I,J,T}` is a matrix with contents of type `T`,
 indexed by rows with strong type `I` and columns with strong type `J`.
 - `StrongArray` is the general supertype.

## Array operations

Non-scalar indexation (i.e. by an array or a colon `:`) is supported.
Slice extraction returns `StrongArray` with matching index types for the
remaining dimensions. View returns a `StrongArray` wrapping a view.

Broadcast operations are supported;
therefore, for any `StrongArray` with numeric contents,
`A + A` or `2A` should work.
However, linear algebra is not (yet) implemented:
`A*A` (assuming compatible index types) does not work,
nor do scalar products, etc.

Arrays with dimension ``n ≥ 2``
generally have incompatible index types along their dimensions;
thus, the default indexation for these `StrongArray` objects
is cartesian. (This is different from the base `Array` case).

Iterating over pairs return `StrongCartesianIndices` objects,
which behave much like `Base.CartesianIndices`.
(They actually even derive from the same abstract type,
`Base.AbstractCartesianIndices`;
however, not too many methods in `Base` actually use that abstract type).

## The `Strong` constructor

As a shortcut, the `Strong` constructor allows access to more or less all
constructions:
 - `Strong{I}Vector` is equivalent to `StrongVector{I}`;
 - `Strong{I}Vector{T}` is equivalent to `StrongVector{I,T}`;
 this notation helps separate index and data types, and also gives access
 to the following:
 - `Strong{I}SVector{2,T}` wraps a `SVector`
 (from [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl))
 in a strongly-indexed type;

The same notation also works for individual arrays:
 - `Strong{I}[1,2,3]` is a strongly-indexed vector;
 - `Strong{I,J}[1 2;3 4]` is a strongly-indexed matrix.

## `wrap` and `unwrap`

These two functions (not exported by the module)
allow easy conversion between standard arrays and strongly-indexed
arrays.

 - `wrap((T1,T2,...), array)` wraps the array (no copy is done) as a
	 `StrongArray`.
 - `unwrap(strongarray)` accesses the inner array of a `StrongArray`
	 (again, no copy is done).
