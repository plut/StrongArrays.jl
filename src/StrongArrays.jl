module StrongArrays

using Base: tail, OneTo, AbstractCartesianIndex, @propagate_inbounds
using Base.Broadcast: Broadcasted

# Tools««1
# Tuple unrolling««2
@inline function tuple_take(::Val{N}, itr, state...) where{N}
	(value, newstate) = iterate(itr, state...)
	return (value, tuple_take(Val(N-1), itr, newstate)...)
end
@inline tuple_take(::Val{0}, itr, state...) = ()
	
# Strong integers««1
# Types««2
abstract type StrongInteger <: Integer end
"""
    StrongInt{S}

Strongly-typed alias of integers.

The symbol `S` is used for domain separation.

Use the `@StrongInt` macro to generate such a type.
"""
struct StrongInt{S} <: StrongInteger
	value::Int
end
Int(x::StrongInt) = x.value
name(::Type{StrongInt{S}}) where{S} = S
name(x::StrongInt) = name(typeof(x))
# disambiguation with a Core constructor from "boot.jl":
(::Type{StrongInt{S}})(a::StrongInt{S}) where{S} = StrongInt{S}(Int(a))
# (1MyIndex) is an alternate form of constructor
Base.:*(a::Int, b::Type{<:StrongInteger}) = b(a)

# @StrongInt macro««2
"""
    @StrongInt MyIndexType

Generates a new strongly-typed `Int` alias,
along with the `Base.show` method that goes with it.
"""
macro StrongInt(name)
	quote
		$(esc(name)) = StrongInt{$(QuoteNode(name))}
		Base.show(io::IO, ::Type{$(esc(name))}) =
			print(io, $(string(name)))
	end
end

# Arithmetic««2
# (MyIndex(1) + one) is supported
for op in (:+, :-) @eval begin
	Base.$op(a::J, b::J) where{J<:StrongInteger} = J($op(Int(a), Int(b)))
	Base.$op(a::StrongInteger, ::typeof(one)) = typeof(a)($op(Int(a), 1))
end end
Base.:-(a::J) where{J<:StrongInteger} = J(-Int(a))
for op in (:<, :<=) @eval begin
	Base.$op(a::J, b::J) where{J<:StrongInteger} = $op(Int(a), Int(b))
	Base.$op(a::StrongInteger, ::typeof(zero)) = $op(Int(a), 0)
	Base.$op(::typeof(zero), a::StrongInteger) = $op(0, Int(a))
	Base.$op(a::StrongInteger, ::typeof(one)) = $op(Int(a), 1)
	Base.$op(::typeof(one), a::StrongInteger) = $op(1, Int(a))
end end
Base.:*(a::Int, b::StrongInteger) = (typeof(a))(a*Int(b))
for op in (:div, :rem, :fld, :mod, :fld1, :mod1) @eval begin
	Base.$op(a::StrongInteger, b::Integer) = (typeof(a))($op(Int(a), b))
end end
for op in (:divrem, :fldmod, :fldmod1) @eval begin
	Base.$op(a::StrongInteger, b::Integer) = (typeof(a)).($op(Int(a), b))
end end

# Detecting indexation with incompatible type««2
struct IncompatibleTypes <: Exception
	t1::Type
	t2::Type
end
function Base.showerror(io::IO, e::IncompatibleTypes)
	print(io, "Incompatible types: ", e.t1, " and ", e.t2)
end
Base.promote(a::Int, b::StrongInteger)= throw(IncompatibleTypes(Int, typeof(b)))
Base.promote(b::StrongInteger, a::Int) = throw(IncompatibleTypes(Int, typeof(b)))
Base.promote(a::StrongInteger, b::StrongInteger) =
	 throw(IncompatibleTypes(typeof(a), typeof(b)))
Base.promote(a::T, b::T) where{T<:StrongInteger} = (a,b)

# Base.show(io::IO, x::StrongInteger) = print(io, name(x), "(", Int(x), ")")
Base.string(x::StrongInt) = string(name(x))*"("*string(Int(x))*")"

# Ranges««1
StrongOneTo{T} = OneTo{StrongInt{T}}
@inline Base.eltype(::Type{OneTo{T}}) where{T} = T
@inline Base.length(r::StrongOneTo) = Int(r.stop)
@inline Base.length(r::AbstractUnitRange{<:StrongInteger}) =
	Int(Base.unsafe_length(r))
AbstractUnitRange{Int}(r::StrongOneTo) = OneTo(Int(r.stop))

# Index««2
struct StrongCartesianIndex{N,J} <: AbstractCartesianIndex{N}
	I::J
	@inline StrongCartesianIndex{N,J}(a...) where{N,J} = new{N,J}(a)
end
@inline coordtypes(::Type{StrongCartesianIndex{N,J}}) where{N,J} =
	(collect(J.parameters)...,)
@inline StrongCartesianIndex(a...) =
	StrongCartesianIndex{length(a),typeof(a)}(a...)
@inline Base.Tuple(idx::StrongCartesianIndex) = idx.I

# forced conversion to and from CartesianIndex
@inline Base.CartesianIndex(idx::StrongCartesianIndex) =
	CartesianIndex(Int.(Tuple(idx)))
@inline (T::Type{StrongCartesianIndex{N,J}})(
	idx::AbstractCartesianIndex{N}) where{N,J}=
	T([x(y) for (x,y) in zip(coordtypes(T), Tuple(idx))]...)

function Base.show(io::IO, c::StrongCartesianIndex{N,J}) where{N,J}
	print(io, "StrongCartesianIndex(", join(string.(Tuple(c)), ","), ")")
end
@inline Base.getindex(idx::Base.AbstractCartesianIndex, i::Integer) =
	getindex(Tuple(idx), i)
@inline Base.length(idx::Base.AbstractCartesianIndex) =
	length(Tuple(idx))

# Indices««2
struct StrongCartesianIndices{N,I,R<:NTuple{N,AbstractUnitRange}} <:
# 		AbstractArray{Int,N}
	AbstractArray{StrongCartesianIndex{N,I},N}
	indices::R
	StrongCartesianIndices{N,I}(indices) where{N,I} =
		new{N,I,typeof(indices)}(indices)
end

# the Union allows us not to overwrite the default constructor in
# `multidimensional.jl`
function Base.CartesianIndices(
	indices::NTuple{N,Union{AbstractUnitRange{Int},StrongOneTo}}) where{N}
	I = Tuple{eltype.(indices)...}
	return StrongCartesianIndices{N,I}(indices)
end

# size and getindex««2
@inline Base.size(c::StrongCartesianIndices) = length.(c.indices)
@inline function Base.getindex(c::StrongCartesianIndices{N},
		i::Vararg{Int,N}) where{N}
	I = coordtypes(eltype(c))
	return StrongCartesianIndex{N,Tuple{I...}}(
		[T(j) for (T,j) in zip(I, i)]...,)
end

# iterate««2
Base.first(it::StrongCartesianIndices) =
	StrongCartesianIndex(map(first, it.indices)...)
Base.last(it::StrongCartesianIndices) =
	StrongCartesianIndex(map(last, it.indices)...)
# mirrored from `multidimensional.jl`, replacing 1 by one(...)
@inline function Base.iterate(it::StrongCartesianIndices)
	f = first(it)
	any(map(>, Tuple(f), Tuple(last(it)))) && return nothing
	return (f, f)
end
@inline function Base.iterate(it::StrongCartesianIndices, state)
	valid, I = __inc(Tuple(state), Tuple(first(it)), Tuple(last(it)))
	valid || return nothing
	return StrongCartesianIndex(I...), StrongCartesianIndex(I...)
end
@inline function __inc(state, start, stop)
	f = first(state)
	(f < first(stop)) && 
		return (true, (f+one(f), tail(state)...))
	valid, I = __inc(tail(state), tail(start), tail(stop))
	return (valid, (first(start), I...))
end
@inline function __inc(state::Tuple{<:Integer}, start::Tuple{<:Integer},
	stop::Tuple{<:Integer})
	f = first(state)
	return (f < first(stop), (f + one(f),))
end
@inline __inc(::Tuple{}, ::Tuple{}, ::Tuple{}) = (false, ())

# Arrays««1
# Type ««2
# I is a tuple of index types
"""
    StrongArray{N,I,T,A}

Wrapper for an array of type `A`, exposed as an `AbstractArray{T,N}`,
with index types for each coordinates given by the Tuple of types `I`.

`A` should be an `AbstractArray{T,N}` and `T` should be a `Tuple`
of `N` concrete `Integer` types.

"""
struct StrongArray{N,I,T,A} <: AbstractArray{T,N}
	array::A
	StrongArray{N,I,T,A}(a) where{N,I,T,A} = new{N,I,T,A}(copy(a))
	StrongArray{N,I,T,A}(::Val{:nocopy}, a) where{N,I,T,A} =
		new{N,I,T,A}(a)
end
StrongArray{N,I,T}(a::AbstractArray{T,N}) where{N,I,T} =
		StrongArray{N,I,T,typeof(a)}(a)
StrongArray{N,I,T}(a) where{N,I,T} =
	StrongArray{N,I,T}(AbstractArray{T,N}(a))
# (::Type{StrongArray{X,N,I,A} where{X}})(a::AbstractArray) where{N,I,A} =
# 	StrongArray{eltype(a),N,I,A}(a)
StrongArray{N,I}(a::AbstractArray) where{N,I} =
	StrongArray{N,I,eltype(a)}(a)
# this makes (Strong{Int}SVector{3})([1,2,3]) work as intended:
(X::Type{StrongArray{N,I,T,<:A} where{T}})(a) where{N,I,A} =
	StrongArray{N,I}(X.body.var.ub(a))

(A::Type{<:StrongArray{N,I,T}})(::UndefInitializer, dims::Tuple) where{N,I,T} =
	A(Array{T}(undef, dims))
(A::Type{<:StrongArray{N,I,T}})(::UndefInitializer, dims::Int...) where{N,I,T} =
	A(undef, dims)

@inline unwrap(a::StrongArray) = a.array
Array{T,N}(a::StrongArray{N}) where{T,N} = Array{T,N}(unwrap(a))

@inline Base.ndims(::Type{<:StrongArray{N}}) where{N} = N
@inline Base.ndims(a::StrongArray) = ndims(typeof(a))
@inline indextype(::Type{<:StrongArray{N,I}}) where{N,I} = I
@inline indextype(a::StrongArray) = indextype(typeof(a))
@inline Base.eltype(::Type{<:StrongArray{N,I,T}}) where{N,I,T} = T
@inline Base.eltype(a::StrongArray) = eltype(typeof(a))
@inline indextypes(a::StrongArray) = (indextype(a).parameters...,)

"""
    StrongArrays.wrap((T1,T2,...), array)

Wraps a *N*-dimensional array as a `StrongArray`
indexed by the *N* given types `T1`, `T2`, etc.
"""
wrap(::Tuple{}, a) = a
wrap(types::Tuple{Any,Vararg{Any}}, a::AbstractArray) =
	StrongArray{length(types),Tuple{types...},eltype(a),typeof(a)}(Val(:nocopy), a)


# Displaying ««2
Base.size(a::StrongArray) = size(unwrap(a))
@inline print_typename(io::IO, a::StrongArray, x...) =
	print(io, "StrongArray{", indextype(a), "}", x...)
function Base.show(io::IO, T::MIME"text/plain", a::StrongArray)
	print_typename(io, a, " wrapping ")
	show(io, T, unwrap(a))
end

Base.show(io::IO, a::StrongArray) =
	print(io, "StrongArray{", ndims(a), ",", indextype(a), "}(",
		unwrap(a), ")")

Base.copy(a::StrongArray) = (typeof(a))(Val(:nocopy), copy(unwrap(a)))
# Indexation««2

Base.axes(a::StrongArray) = (map(((t,n),)->OneTo(t(n)),
	zip(indextypes(a), size(a)))...,)
# to_index: returns a pair (real index, keep this dimension)
@inline _to_index(T::Type{<:Integer}, i, n, k) =
	throw(TypeError(:StrongArray, "coordinate $k", T, i))
@inline _to_index(::Type{T}, i::T, n, k) where{T<:Integer} =
	(Int(i), false)
@inline _to_index(::Type{T}, i::AbstractVector{T}, n, k) where{T<:Integer} =
	(Int.(i), true)
@inline _to_index(::Type{<:Integer}, ::Colon, n, k) = ((:), true)
@inline function _to_indices(a::StrongArray{N}, idx) where{N}
	newidx = ()
	newtypes = ()
	# could be unrolled as a recursive call instead?
	# let's place our hopes on constant propagation for now
	for i in 1:N
		(j, keep) = _to_index(indextypes(a)[i], idx[i], size(a)[i], i)
		newidx = (newidx..., j)
		keep && (newtypes = (newtypes..., indextypes(a)[i]))
	end
	return (newidx, newtypes)
end
# Base.getindex(a::StrongArray, idx::CartesianIndex) = getindex(a.array, idx)
@inline @propagate_inbounds Base.getindex(a::StrongArray,
		idx::StrongCartesianIndex{N,I}) where{T,N,I} =
	getindex(unwrap(a), Int.(Tuple(idx))...)
@inline @propagate_inbounds Base.setindex!(a::StrongArray, v,
		idx::StrongCartesianIndex{N,I}) where{T,N,I} =
	setindex!(unwrap(a), v, Int.(Tuple(idx))...)
@inline @propagate_inbounds function Base.getindex(a::StrongArray, idx...)
	(newidx, newtypes) = _to_indices(a, idx)
	r = getindex(unwrap(a), newidx...)
	return wrap(newtypes, r)
end

@inline @propagate_inbounds function Base.setindex!(a::StrongArray,
	v, idx...)
	# FIXME: check type of v
	(newidx, newtypes) = _to_indices(a, idx)
	if newtypes == ()
		setindex!(unwrap(a), v, newidx...)
	else
		setindex!(unwrap(a), Array(v), newidx...)
	end
	return v
end

@inline @propagate_inbounds function Base.view(a::StrongArray, idx...)
	(newidx, newtypes) = _to_indices(a, idx)
	v = view(unwrap(a), newidx...)
	return wrap(newtypes, v)
end

# Arrays which are not `Strong` are allowed to be indexed by any type of
# integer, including StrongInt. (If you want an array indexed *only*
# by `Int`, you may use `StrongArray{Int}`!).
# To disallow indexation by `StrongInt`, uncomment the
# following method definition:
# Base.to_index(a::AbstractArray, i::StrongInt) =
# 	throw(TypeError(Base.typename(typeof(a)).name, "coordinate", Int, i))
# disable getindex and setindex
 
# Similar and broadcast««2
Base.similar(a::StrongArray, ::Type{T}, dims::Dims{N}) where{T,N} =
	StrongArray{ndims(a),indextype(a),T}(similar(unwrap(a), T, dims))
# this is a (compatible) domain extension of the definition of
# `Broadcast.axistype` (hence slight piracy).
Broadcast.axistype(a::T,b::T) where{T<:OneTo} = a
Broadcast.axistype(a::OneTo, b::OneTo) = throw(DmensionMismatch(
	"Index types $(typeof(a.stop)) and $(typeof(b.stop)) are incompatible"))

struct StrongArrayStyle{N,I} <: Broadcast.AbstractArrayStyle{N} end

(T::Type{<:StrongArrayStyle{N}})(::Val{N}) where{N} = T()
Base.BroadcastStyle(T::Type{<:StrongArray}) =
	StrongArrayStyle{ndims(T),indextype(T)}()
Broadcast.similar(bc::Broadcast.Broadcasted{StrongArrayStyle{N,I}},
		::Type{T}, dims) where{N,I,T} =
	similar(StrongArray{N,I,T}, dims)

@inline function copyto!(dest::StrongArray, bc::Broadcast.Broadcasted{Nothing})
	c = copyto!(dest.array, bc)
	return dest
end

@inline function Base.getindex(bc::Broadcast.Broadcasted,
		I::AbstractCartesianIndex) 
	@boundscheck checkbounds(bc, I)
	@inbounds Broadcast._broadcast_getindex(bc, I)
end
@inline Base.checkbounds(bc::Base.Broadcast.Broadcasted,
	I::AbstractCartesianIndex) =
	Base.checkbounds_indices(Bool, Base.AbstractUnitRange{Int}.(axes(bc)),
		(CartesianIndex(I),)) || Base.throw_boundserror(bc, (I,))
@inline Broadcast.newindex(idx::StrongCartesianIndex, keep, Idefault) =
	typeof(idx)(Broadcast.newindex(CartesianIndex(idx), keep, Int.(Idefault)))


# Strong{...} constructor««2
const _STRONG_MAX=4
_nvar(n) = tuple((Symbol(:X,i) for i in 1:n)...,)
# _strong_nvar(n) = Expr(:curly, :Strong, _nvar(n)...)
@eval abstract type Strong{$(_nvar(_STRONG_MAX)...)} end

for i in 1:_STRONG_MAX
	v = _nvar(i)
	S = :(Strong{$(v...)})
	@eval begin
	# Strong{Int}Vector
Base.:*(::Type{$S}, ::Type{Array{T,$i} where T}) where {$(v...)} =
		StrongArray{$i, Tuple{$(v...)}}
	# Strong{Int}SVector{2,Int}
Base.:*(::Type{$S}, A::Type{<:AbstractArray{T,$i}}) where{$(v...),T} =
		StrongArray{$i, Tuple{$(v...)}, T, A}
	# Strong{Int}SVector{2}
Base.:*(::Type{$S}, A::Type{<:AbstractArray{T,$i} where T}) where{$(v...)} =
		StrongArray{$i, Tuple{$(v...)}, T, <:A} where{T}
	# Strong{Int}Vector([1,2,3])
Base.:*(::Type{$S}, a::AbstractArray{T,$i} where{T}) where{$(v...)} =
		wrap(($(v...),), a)
	# Strong{Int}([1,2,3])
$S(a::AbstractArray{T,$i} where{T}) where{$(v...)} =
		wrap(($(v...),), a)
	end
end

"""
    Strong{I}

General-purpose prefix allowing transformation to a strongly-indexed array type;
in particular:

 - `Strong{I}Vector`: alias for `StrongVector{I,J}`.
 - `Strong{I}Vector{T}`: alias for `StrongVector{I,T}`
   (helps separating index type from content type).
 - `Strong{I}SVector{3}` (or any other `AbstractArray` type):
   wraps this type in a strongly-indexed array.
 - `Strong{I}([1,2,3])`: wraps (without copy) the right-hand side vector
   as a `StrongVector`.
 - `Strong{I}[1,2,3]`: same as above.
 - `Strong{I,J}Matrix`: alias for `StrongMatrix{I,J}`.
 - `Strong{I,J}([1 2;3 4])`: wraps (no copy) as a `StrongMatrix`.
 - `Strong{I,J}[1 2;3 4]`: same as above.
"""
Strong

# StrongVector and StrongMatrix ««2
"""
    StrongVector{T}

One-dimensional vector with strongly-typed index of type `T`.
`StrongVector{T,X}` has index type `T` and data type `X`.
"""
StrongVector{T} = StrongArray{1,Tuple{T}}
@inline print_typename(io::IO, a::StrongVector, x...) =
	print(io, "StrongVector{", indextypes(a)..., "}", x...)
Base.LinearIndices(v::StrongVector) = OneTo(indextypes(v)[1](length(v)))
# Strong{Int}[1,2,3]
Base.getindex(::Type{Strong{I}}, a...) where{I} =
	wrap((I,), [a...])

"""
    StrongMatrix{T1,T2}

Two-dimensional matrix with strongly-typed indices of type `T1` and `T2`.
`StrongMatrix{T1,T2,X}` has index types `T1`,`T2` and data type `X`.
"""
StrongMatrix{T1,T2} = StrongArray{2,Tuple{T1,T2}}
@inline print_typename(io::IO, a::StrongMatrix, x...) =
	print(io, "StrongMatrix{", indextypes(a)[1], ",", indextypes(a)[2], "}", x...)
# Strong{Int,Int}[1 2;3 4]
Base.typed_hvcat(::Type{Strong{I,J}}, rows::Tuple{Vararg{Int}},
	a::Number...) where{I,J}= wrap((I,J), hvcat(rows, a...))
Base.typed_hvcat(::Type{Strong{I,J}}, rows::Tuple{Vararg{Int}},
	a...) where{I,J}= wrap((I,J), hvcat(rows, a...))

# TODO: linear algebra ««1

# exports««1
export StrongInt, @StrongInt
export StrongArray, StrongVector, StrongMatrix, Strong
end # module
#»»1
