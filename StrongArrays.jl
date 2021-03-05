# todo: pairs
# wrap pairs(a.array)

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

# Base.show(io::IO, x::StrongInt) = print(io, name(x), "(", Int(x), ")")
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
struct StrongArray{N,I<:Tuple{Vararg{Integer}},T,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
	array::A
	StrongArray{N,I,T,A}(a) where{N,I,T,A} = new{N,I,T,A}(a)
end
StrongArray{N,I,T}(a::AbstractArray{T,N}) where{N,I,T} =
		StrongArray{N,I,T,typeof(a)}(a)
StrongArray{N,I,T}(a) where{N,I,T} =
	StrongArray{N,I,T}(AbstractArray{T,N}(a))
# (::Type{StrongArray{X,N,I,A} where{X}})(a::AbstractArray) where{N,I,A} =
# 	StrongArray{eltype(a),N,I,A}(a)
StrongArray{N,I}(a::AbstractArray) where{N,I} =
	StrongArray{N,I,eltype(a)}(a)

(A::Type{<:StrongArray{N,I,T}})(::UndefInitializer, dims::Tuple) where{N,I,T} =
	A(Array{T}(undef, dims))
(A::Type{<:StrongArray{N,I,T}})(::UndefInitializer, dims::Int...) where{N,I,T} =
	A(undef, dims)

Array{T,N}(a::StrongArray{N}) where{T,N} = Array{T,N}(a.array)

@inline Base.ndims(::Type{<:StrongArray{N}}) where{N} = N
@inline Base.ndims(a::StrongArray) = ndims(typeof(a))
@inline indextype(::Type{<:StrongArray{N,I}}) where{N,I} = I
@inline indextype(a::StrongArray) = indextype(a)
@inline Base.eltype(::Type{<:StrongArray{N,I,T}}) where{N,I,T} = T
@inline Base.eltype(a::StrongArray) = eltype(typeof(a))

# Indexation ««2
Base.size(a::StrongArray) = size(a.array)
indextype(a::StrongArray) = indextype(typeof(a))
indextypes(a::StrongArray) = (indextype(a).parameters...,)
function Base.show(io::IO, T::MIME"text/plain", a::StrongArray)
	print(io, "StrongArray{", indextype(a), "} wrapping ")
	show(io, T, a.array)
end

Base.show(io::IO, a::StrongArray) =
	print(io, "StrongArray{", ndims(a), ",", indextype(a), "}(",
		a.array, ")")

# Iteration««2

Base.axes(a::StrongArray) = (map(((t,n),)->OneTo(t(n)),
	zip(indextypes(a), size(a)))...,)
# to_index: returns a pair (real index, keep this dimension)
@inline _to_index(T::Type{<:Integer}, i, n, k) =
	throw(TypeError(:StrongArray, "coordinate $k", T, i))
@inline _to_index(::Type{T}, i::T, n, k) where{T<:Integer} =
	(Int(i), false)
@inline _to_index(::Type{T}, i::AbstractVector{T}, n, k) where{T<:Integer} =
	(Integer.(i), true)
@inline _to_index(::Type{<:Integer}, ::Colon, n, k) = ((:), true)
@inline function _to_indices(a::StrongArray{N}, idx) where{N}
	newidx = ()
	newtypes = ()
	# could be unrolled as a recursive call instead?
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
	getindex(a.array, Int.(Tuple(idx))...)
@inline @propagate_inbounds Base.setindex!(a::StrongArray, v,
		idx::StrongCartesianIndex{N,I}) where{T,N,I} =
	setindex!(a.array, v, Int.(Tuple(idx))...)
@inline @propagate_inbounds function Base.getindex(a::StrongArray, idx...)
	(newidx, newtypes) = _to_indices(a, idx)
	r = getindex(a.array, newidx...)
	(newtypes == ()) && return r
	return StrongArray{length(newtypes),Tuple{newtypes...}}(r)
end

Base.setindex!(a::StrongArray, v, idx::Integer...) =
	setindex!(a.array, v, _to_indices(a, idx)...)
# disable getindex and setindex
 
# Similar and broadcast««2
Base.similar(a::StrongArray, ::Type{T}, dims::Dims{N}) where{T,N} =
	StrongArray{ndims(a),indextype(a),T}(similar(a.array, T, dims))
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

# StrongVector and StrongMatrix ««2
StrongVector{S} = StrongArray{1,Tuple{S}}
StrongMatrix{S1,S2} = StrongArray{2,Tuple{S1,S2}}

Base.LinearIndices(v::StrongVector) = OneTo(indextypes(v)[1](length(v)))

# TODO: linear algebra ««1

# exports««1
export StrongInt, @StrongInt
export StrongArray, StrongVector, StrongMatrix
end # module
#»»1
