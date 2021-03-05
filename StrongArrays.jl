# todo: pairs
# wrap pairs(a.array)

module StrongArrays

using Base: tail, OneTo, AbstractCartesianIndex, @propagate_inbounds
using Base.Broadcast: Broadcasted

# Strong integers««1
struct StrongInt{S} <: Integer
	value::Int
end
Int(x::StrongInt) = x.value
name(::Type{StrongInt{S}}) where{S} = S
name(x::StrongInt) = name(typeof(x))
(::Type{StrongInt{S}})(a::StrongInt{S}) where{S} = StrongInt{S}(Int(a))
Base.:<(a::J, b::J) where{J<:StrongInt} = Int(a) < Int(b)
Base.:+(a::J, b::J) where{J<:StrongInt} = J(Int(a) + Int(b))
Base.:-(a::J, b::J) where{J<:StrongInt} = J(Int(a) - Int(b))
Base.:*(a::Int, b::J) where{J<:StrongInt} = J(a*Int(b))
Base.:div(a::J, b::Integer) where{J<:StrongInt} = J(div(Int(a),b))
Base.:rem(a::J, b::Integer) where{J<:StrongInt} = J(rem(Int(a),b))
Base.:fld(a::J, b::Integer) where{J<:StrongInt} = J(fld(Int(a),b))
Base.:mod(a::J, b::Integer) where{J<:StrongInt} = J(mod(Int(a),b))
# oneunit(J::Type{<:StrongInt}) = J(1)
# zero(J::Type{<:StrongInt}) = J(0)

macro StrongInt(name)
	quote
		$(esc(name)) = StrongInt{$(QuoteNode(name))}
		Base.show(io::IO, ::Type{$(esc(name))}) =
			print(io, $(string(name)))
	end
end

struct IncompatibleTypes <: Exception
	t1::Type
	t2::Type
end
function Base.showerror(io::IO, e::IncompatibleTypes)
	print(io, "Incompatible types: ", e.t1, " and ", e.t2)
end
Base.promote(a::Int, b::StrongInt) = throw(IncompatibleTypes(Int, typeof(b)))
Base.promote(b::StrongInt, a::Int) = throw(IncompatibleTypes(Int, typeof(b)))
Base.promote(a::StrongInt, b::StrongInt) =
	 throw(IncompatibleTypes(typeof(a), typeof(b)))
Base.promote(a::T, b::T) where{T<:StrongInt} = (a,b)

Base.show(io::IO, x::StrongInt) = print(io, name(x), "(", Int(x), ")")
Base.string(x::StrongInt) = string(name(x))*"("*string(Int(x))*")"

# Ranges««1
StrongOneTo{T} = OneTo{StrongInt{T}}
@inline Base.eltype(::Type{OneTo{T}}) where{T} = T
@inline Base.length(r::StrongOneTo) = Int(r.stop)
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
struct StrongArray{T,N,I<:Tuple{Vararg{Integer}},A<:AbstractArray{T,N}} <: AbstractArray{T,N}
	array::A
	StrongArray{T,N,I,A}(a) where{T,N,I,A} = new{T,N,I,A}(a)
	StrongArray{T,N,I}(a::AbstractArray{T,N}) where{T,N,I} =
		new{T,N,I,typeof(a)}(a)
end
(A::Type{<:StrongArray{T,N,I}})(::UndefInitializer, dims::Tuple) where{T,N,I} =
	A(Array{T}(undef, dims))
(A::Type{<:StrongArray{T,N,I}})(::UndefInitializer, dims::Int...) where{T,N,I} =
	A(undef, dims)
StrongArray{T,N,I}(a) where{T,N,I} =
	StrongArray{T,N,I}(AbstractArray{T,N}(a))
# (::Type{StrongArray{X,N,I,A} where{X}})(a::AbstractArray) where{N,I,A} =
# 	StrongArray{eltype(a),N,I,A}(a)
(::Type{<:StrongArray{X,N,I} where{X}})(a::AbstractArray) where{N,I} =
	StrongArray{eltype(a),N,I}(a)

@inline Base.eltype(::Type{<:StrongArray{T}}) where{T} = T
@inline Base.eltype(a::StrongArray) = eltype(typeof(a))
@inline Base.ndims(::Type{<:StrongArray{T,N}}) where{T,N} = N
@inline Base.ndims(a::StrongArray) = ndims(typeof(a))
@inline indextype(::Type{<:StrongArray{X,Y,I} where{X,Y}}) where{I} = I
@inline indextype(a::StrongArray) = indextype(a)

# Indexation ««2
Base.size(a::StrongArray) = size(a.array)
# indextype(::Type{<:StrongArray{T,N,I}}) where{T,N,I} = I
indextype(a::StrongArray) = indextype(typeof(a))
indextypes(a::StrongArray) = (indextype(a).parameters...,)
function Base.show(io::IO, T::MIME"text/plain", a::StrongArray)
	print(io, "StrongArray on ", indextype(a))
	show(io, T, a.array)
end

Base.show(io::IO, a::StrongArray) =
	print(io, "StrongArray on ", indextype(a), a.array)

# Iteration««2

Base.axes(a::StrongArray) = (map(((t,n),)->OneTo(t(n)),
	zip(indextypes(a), size(a)))...,)
@inline _to_index(T::Type{<:Integer}, i, n, k) =
	throw(TypeError(:StrongArray, "coordinate $k", T, i))
@inline _to_index(::Type{T}, i::T, n, k) where{T<:Integer} = Int(i)
@inline _to_index(::Type{T}, i::AbstractVector{T}, n, k) where{T<:Integer} =
	Integer.(i)
@inline _to_index(::Type{<:Integer}, ::Colon, n, k) = (:)

# Base.getindex(a::StrongArray, idx::CartesianIndex) = getindex(a.array, idx)
@propagate_inbounds Base.getindex(a::StrongArray,
		idx::StrongCartesianIndex{N,I}) where{T,N,I} =
	getindex(a.array, Int.(Tuple(idx))...)
@propagate_inbounds Base.setindex!(a::StrongArray, v,
		idx::StrongCartesianIndex{N,I}) where{T,N,I} =
	setindex!(a.array, v, Int.(Tuple(idx))...)
Base.getindex(a::StrongArray, idx...) = getindex(a.array,
	map(x->_to_index(x...), zip(indextypes(a), idx, size(a), 1:ndims(a)))...)

Base.setindex!(a::StrongArray, v, idx::Integer...) = setindex!(a.array, v,
	map(x->_to_index(x...), zip(indextypes(a), idx, size(a), 1:ndims(a)))...)
 
# Similar and broadcast««2
Base.similar(a::StrongArray, ::Type{T}, dims::Dims{N}) where{T,N} =
	StrongArray{T,ndims(a),indextype(a)}(similar(a.array, T, dims))
# this is a (compatible) domain extension of the definition of
# `Broadcast.axistype` (hence slight piracy).
Broadcast.axistype(a::T,b::T) where{T<:OneTo} = a
Broadcast.axistype(a::OneTo, b::OneTo) = throw(DimensionMismatch(
	"Index types $(typeof(a.stop)) and $(typeof(b.stop)) are incompatible"))

struct StrongArrayStyle{N,I} <: Broadcast.AbstractArrayStyle{N} end

(T::Type{<:StrongArrayStyle{N}})(::Val{N}) where{N} = T()
Base.BroadcastStyle(T::Type{<:StrongArray}) =
	StrongArrayStyle{ndims(T),indextype(T)}()
Broadcast.similar(bc::Broadcast.Broadcasted{StrongArrayStyle{N,I}},
		::Type{T}, dims) where{N,I,T} =
	similar(StrongArray{T,N,I}, dims)

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

# StrongVector and StrongMatrix ««2
StrongVector{S,T,A} = StrongArray{T,1,Tuple{S},A}
StrongMatrix{S1,S2,T,A} = StrongArray{T,2,Tuple{S1,S2},A}

Base.LinearIndices(v::StrongVector) = OneTo(indextypes(v)[1](length(v)))

# TODO: linear algebra ««1

# exports««1
export StrongInt, @StrongInt
export StrongArray, StrongVector, StrongMatrix
end # module
#»»1

nothing
