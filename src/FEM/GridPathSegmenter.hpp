#include <array>
#include <cmath>
#include <type_traits>

namespace ippl {

template<int I, int End, class F>
KOKKOS_INLINE_FUNCTION
constexpr void static_for(F&& f) {
  if constexpr (I < End) {
    f(std::integral_constant<int, I>{});
    static_for<I+1, End>(std::forward<F>(f));
  }
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void sort2(T& a, T& b) { if (a > b) { T t=a; a=b; b=t; } }

template<typename T>
KOKKOS_INLINE_FUNCTION
void sort3(T& a, T& b, T& c) { sort2(a,b); sort2(b,c); sort2(a,b); }

template<unsigned Dim, typename T>
KOKKOS_INLINE_FUNCTION
std::array<T,Dim> lerp_point(const std::array<T,Dim>& A,
                             const std::array<T,Dim>& B, T t) {
  std::array<T,Dim> out{}; 
  for (unsigned a=0; a<Dim; ++a) out[a] = A[a] + (B[a]-A[a]) * t;
  return out;
}

// ---------- per-axis cut times (for DefaultCellCrossingRule) ----------
template<unsigned Dim, typename T>
struct CutTimes { std::array<T,Dim> t; };

template<unsigned Dim, typename T>
KOKKOS_INLINE_FUNCTION
CutTimes<Dim,T> compute_axis_cuts_default(
  const std::array<T,Dim>& A,
  const std::array<T,Dim>& B,
  const std::array<T,Dim>& origin,
  const std::array<T,Dim>& h)
{
  CutTimes<Dim,T> cuts;
  for (unsigned a=0; a<Dim; ++a) cuts.t[a] = (T)2; // sentinel (>1)

  const T eps1 = (T)1e-12, one = (T)1;

  auto axis_cut = [&](auto Ax) {
    constexpr int a = Ax;
    const T d   = B[a] - A[a];
    const T eps = eps1 * h[a];
    if (std::fabs(d) <= eps) return; // no motion → no crossing

    const T nA  = (A[a] - origin[a]) / h[a]; // in cell units
    // nearest plane index in direction toward B; bias off-plane a hair
    const T k   = (d > 0) ? std::ceil(nA - eps1) : std::floor(nA + eps1);
    const T pa  = origin[a] + k * h[a];
    const T t   = (pa - A[a]) / d;

    if (t > eps1 && t < one - eps1) cuts.t[a] = t;
  };

  static_for<0,Dim>(axis_cut);
  return cuts;
}

// ---------- endpoints builder: [0, t_sorted..., 1] (keeps duplicates) ----------
template<unsigned Dim, typename T>
KOKKOS_INLINE_FUNCTION
void make_endpoints_fixed(const CutTimes<Dim,T>& cuts,
                          std::array<T,Dim+2>& Tcuts /*out*/)
{
  T t0 = cuts.t[0];
  T t1 = (Dim>=2) ? cuts.t[1] : (T)2;
  T t2 = (Dim==3) ? cuts.t[2] : (T)2;

  if constexpr (Dim==2) {
    sort2(t0, t1);
  } else { // Dim==3
    sort3(t0, t1, t2);
  }

  const T one = (T)1;
  auto clamp_or_one = [&](T v)->T { return (v >= (T)1.5) ? one : (v < one ? v : one); };

  Tcuts[0] = (T)0;
  Tcuts[1] = clamp_or_one(t0);
  if constexpr (Dim>=2) Tcuts[2] = clamp_or_one(t1);
  if constexpr (Dim==3) Tcuts[3] = clamp_or_one(t2);
  Tcuts[Dim+1] = one;
}

// ---------------------------------------------------------------------------------
// DefaultCellCrossingRule specialization 
// ---------------------------------------------------------------------------------

template<unsigned Dim, typename T, typename Rule>
KOKKOS_INLINE_FUNCTION
std::array<Segment<Dim,T>, Dim+1>
GridPathSegmenter<Dim,T,Rule>::split(
    const std::array<T,Dim>& A,
    const std::array<T,Dim>& B,
    const std::array<T,Dim>& origin,
    const std::array<T,Dim>& h)
{
  const auto cuts = compute_axis_cuts_default<Dim,T>(A,B,origin,h);

  std::array<T,Dim+2> Tcuts{};
  make_endpoints_fixed<Dim,T>(cuts, Tcuts);

  std::array<Segment<Dim,T>, Dim+1> segs{};
  for (unsigned i = 0; i < Dim + 1; ++i) {
    const T ta = Tcuts[i];
    const T tb = Tcuts[i+1];
    segs[i].p0 = lerp_point<Dim,T>(A,B,ta);
    segs[i].p1 = lerp_point<Dim,T>(A,B,tb);
  }
  return segs;
}

} // namespace ippl 
