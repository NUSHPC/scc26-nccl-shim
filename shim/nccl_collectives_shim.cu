// LD_PRELOAD shim: copy-engine schedules for NCCL AllReduce, AllGather,
// ReduceScatter and Broadcast when one process drives exactly two GPUs
// (devices 0 and 1) through one two-rank communicator pair. Built for the
// SC26 SCC H100 NVL pair (NV12), CUDA 13.2, NCCL 2.28.3. Every call the shim
// does not handle goes to the real libnccl.so through dlsym(RTLD_NEXT).
//
// Layers, each depending only on the ones above it:
//   1. Tuning      frozen constants from the job-1574 repeat-10 run.
//   2. Gate        pure eligibility decisions, per call and per buffered pair.
//   3. Plan        a pure function from (route, sizes, ranks) to an ordered
//                  list of record / wait / copy / reduce ops with byte offsets.
//   4. Executor    binds plan ids to CUDA streams, events and buffers and
//                  submits each op, checking every CUDA return code.
//   5. Interposer  NCCL group buffering, pairing and fallback.
// Layers 1-3 compile without CUDA when NCCL_SHIM_PLAN_ONLY is defined;
// This keeps the pure scheduling layers available for independent inspection.
//
// V3 screening candidate. Default policy is byte-for-byte plan equivalent to v2.2.
// Arm policy data lives in Tuning; no measured GPU result belongs to v3.
// The first eligible pair and graph captures use NCCL. The pinned ABI,
// communicator/group/registration rules and the existing schedules are retained.
// Device allocations must pass range/ownership/non-pool checks. VMM additionally
// requires READWRITE on both GPUs and legacy-IPC capability 0; ordinary
// allocations require legacy-IPC capability 1, zero access flags or INVALID_VALUE,
// and bidirectional peer capability. Ambiguous or failed
// driver queries fail closed. Stable mappings/lifetimes and no hidden physical
// aliases are application obligations. No implementation proof is claimed.
// NCCL_SHIM_SUMMARY=1 enables exit-only submission counters; default off.
//
// Error policy: a CUDA failure before a pair's first copy/kernel attempt
// falls back to NCCL. A failure at or after that attempt returns
// ncclUnhandledCudaError (never an NCCL replay over half-written buffers),
// attempts to drain both devices, and disables custom paths for the process.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace shim {

constexpr size_t kMiB = size_t(1) << 20;
constexpr int kMaxLanes = 4;

// =====================================================================
// 1. Tuning
// =====================================================================
// V2 dispatch policy; scheduling constants retained from the parent.
struct Tuning {
  bool rs_enabled = true;  // one switch after a fresh reviewed GPU A/B
  size_t inputhybrid_min = 64 * kMiB;
  size_t ar_hole_begin = size_t(1) << 30;
  size_t ar_hole_end = size_t(2) << 30;
  size_t rs_min = 64 * kMiB;  // ReduceScatter input threshold
  size_t bcast_min = 32 * kMiB;  // Broadcast threshold
  size_t chunk = 128 * kMiB;  // direct peer-input piece
  size_t tail_chunk = 24 * kMiB;  // final peer-input pieces
  size_t host_chunk = 24 * kMiB;  // unchanged allocation and host-share quantum
  size_t host_piece = 24 * kMiB;  // selected in Plan; never used to size buffers
  // BEGIN V3 POLICY DATA (the only arm-dependent source region)
  bool ar_fill_hole = true;
  size_t ar_small_end = size_t(1) << 30;
  size_t ar_small_piece = 6 * kMiB;
  size_t ar_gap_split = 1536 * kMiB;
  size_t ar_gap_lo_divisor = 9;
  size_t ar_gap_hi_divisor = 7;
  size_t ar_gap_piece = 24 * kMiB;
  size_t ar_two_piece = 0;
  size_t rs_small_end = size_t(1) << 30;
  size_t rs_small_piece = 6 * kMiB;
  size_t bcast_scale_end = 256 * kMiB;
  size_t bcast_host_scale_min = 24 * kMiB;
  size_t bcast_host_min_chunks = 4;
  size_t ag_min = 32 * kMiB;
  // END V3 POLICY DATA
  int lanes = 4;  // reusable host lanes
  size_t ar_input_host_divisor = 7;  // AllReduce input host share
  size_t ar_output_host_divisor = 7;  // AllReduce output host share
  size_t ag_host_divisor = 7;  // AllGather host share
  size_t rs_host_divisor = 7;  // ReduceScatter host share
  size_t bcast_host_divisor = 6;  // Broadcast host share
  size_t ag_host_scale_min = 16 * kMiB;  // AllGather host-piece scaling threshold
  size_t ag_host_min_chunks = 4;  // AllGather minimum host pieces
  int threads = 256;  // reduction threads per block
  int blocks_per_sm = 2;  // reduction blocks per SM
  size_t max_message_bytes = size_t(8) << 30;  // enforced logical message cap
  size_t prewarm_bytes = size_t(8) << 30;  // maximum prewarmed pair
};
constexpr Tuning kTuning{};
static_assert(kTuning.lanes >= 1 && kTuning.lanes <= kMaxLanes, "lane count");
static_assert(kTuning.host_chunk % 16 == 0 && kTuning.chunk % 16 == 0 &&
                  kTuning.tail_chunk % 16 == 0,
              "pieces must keep the 16-byte reduction vectors aligned");

constexpr bool valid_host_piece(size_t n) {
  return n > 0 && n <= kTuning.host_chunk && n % 16 == 0;
}
static_assert(valid_host_piece(kTuning.ar_small_piece) &&
              valid_host_piece(kTuning.ar_gap_piece) &&
              (!kTuning.ar_two_piece || valid_host_piece(kTuning.ar_two_piece)) &&
              valid_host_piece(kTuning.rs_small_piece), "v3 host piece bounds/alignment");
static_assert(kTuning.ar_gap_lo_divisor > 0 && kTuning.ar_gap_hi_divisor > 0,
              "v3 share divisors must be nonzero");

// =====================================================================
// 2. Gate
// =====================================================================
enum class Kind : uint8_t { AllReduce, AllGather, ReduceScatter, Broadcast };
enum class DType : uint8_t { Unsupported, I8, U8, F16, BF16, I32, U32, F32, I64, U64, F64 };
enum class RedOp : uint8_t { Sum, Other };

inline size_t elem_size(DType t) {
  switch (t) {
    case DType::I8: case DType::U8: return 1;
    case DType::F16: case DType::BF16: return 2;
    case DType::I32: case DType::U32: case DType::F32: return 4;
    case DType::I64: case DType::U64: case DType::F64: return 8;
    default: return 0;
  }
}
inline bool reducible(DType t) {
  return t == DType::F32 || t == DType::F16 || t == DType::BF16;
}
inline const char* kind_name(Kind k) {
  switch (k) {
    case Kind::AllGather: return "allgather";
    case Kind::ReduceScatter: return "reducescatter";
    case Kind::Broadcast: return "broadcast";
    default: return "allreduce";
  }
}

enum class Route : uint8_t { Fallback, ArPeerRead, ArInputHybrid, AllGather, ReduceScatter, Broadcast };

enum class Reason : uint8_t {
  None, NotGrouped, UnsupportedType, UnsupportedOp, NullBuffer, SizeOverflow,
  BelowThreshold, Misaligned, BadSlot, PartialOverlap, BadRank, GroupShape,
  NotPair, Mismatch, Capturing, FirstPass, SetupFailed, SubmitFailedEarly,
  Poisoned, AboveLimit, AddressOverflow, PolicyFallback, MetadataQuery,
  RuntimeUnsupported, Count
};

// One intercepted call. `count` follows NCCL: AllGather send count,
// ReduceScatter receive count, element count otherwise.
struct CallDesc {
  Kind kind = Kind::AllReduce;
  DType dtype = DType::Unsupported;
  RedOp op = RedOp::Sum;
  size_t count = 0;
  uintptr_t send = 0, recv = 0;
  int device = -1, rank = -1, root = -1;
};

// Per-rank payload: the AllReduce/Broadcast buffer, AllGather send slot or
// ReduceScatter receive slot.
inline size_t call_bytes(const CallDesc& c) { return c.count * elem_size(c.dtype); }
inline size_t send_extent(const CallDesc& c) {
  return c.kind == Kind::ReduceScatter ? 2 * call_bytes(c) : call_bytes(c);
}
inline size_t recv_extent(const CallDesc& c) {
  return c.kind == Kind::AllGather ? 2 * call_bytes(c) : call_bytes(c);
}
// The in-place layouts NCCL defines. Any other send/recv overlap falls back.
inline bool call_in_place(const CallDesc& c) {
  const size_t n = call_bytes(c);
  switch (c.kind) {
    case Kind::AllGather: return c.send == c.recv + static_cast<size_t>(c.rank) * n;
    case Kind::ReduceScatter: return c.recv == c.send + static_cast<size_t>(c.rank) * n;
    default: return c.send == c.recv;
  }
}
inline bool ranges_overlap(uintptr_t a, size_t an, uintptr_t b, size_t bn) {
  return a < b + bn && b < a + an;
}

// Pure message checks need no communicator, stream or pointer queries.
// Keep this shared with gate_call so early rejection cannot weaken its gate.
inline Reason gate_message(const CallDesc& c, const Tuning& t) {
  const size_t elem = elem_size(c.dtype);
  if (elem == 0) return Reason::UnsupportedType;
  const bool reduces = c.kind == Kind::AllReduce || c.kind == Kind::ReduceScatter;
  if (reduces && !reducible(c.dtype)) return Reason::UnsupportedType;
  if (reduces && c.op != RedOp::Sum) return Reason::UnsupportedOp;
  if (c.count > (SIZE_MAX / 4) / elem) return Reason::SizeOverflow;
  const size_t n = call_bytes(c);
  const size_t logical = (c.kind == Kind::AllGather || c.kind == Kind::ReduceScatter) ? 2 * n : n;
  if (logical > t.max_message_bytes) return Reason::AboveLimit;
  switch (c.kind) {
    case Kind::AllReduce:
      if (n < t.inputhybrid_min) return Reason::BelowThreshold;
      if (!t.ar_fill_hole && n >= t.ar_hole_begin && n < t.ar_hole_end) return Reason::PolicyFallback;
      break;
    case Kind::AllGather:
      if (2 * n < t.ag_min) return Reason::BelowThreshold;
      break;
    case Kind::ReduceScatter:
      if (!t.rs_enabled) return Reason::PolicyFallback;
      if (2 * n < t.rs_min) return Reason::BelowThreshold;
      if (n % 16 != 0) return Reason::BadSlot;  // slot offsets feed 16-byte vectors
      break;
    case Kind::Broadcast:
      if (n < t.bcast_min) return Reason::BelowThreshold;
      break;
  }
  return Reason::None;
}

// Full per-call eligibility. `device` and `rank` come from the communicator.
inline Reason gate_call(const CallDesc& c, const Tuning& t) {
  const Reason message = gate_message(c, t);
  if (message != Reason::None) return message;
  if (!c.send || !c.recv) return Reason::NullBuffer;
  if ((c.device != 0 && c.device != 1) || (c.rank != 0 && c.rank != 1))
    return Reason::BadRank;
  // Check represented ranges before in-place/overlap additions. The ignored
  // non-root Broadcast send pointer needs no readable extent.
  if (c.recv > UINTPTR_MAX - recv_extent(c) ||
      (!(c.kind == Kind::Broadcast && c.rank != c.root) &&
       c.send > UINTPTR_MAX - send_extent(c))) return Reason::AddressOverflow;
  const bool reduces = c.kind == Kind::AllReduce || c.kind == Kind::ReduceScatter;
  // The reduction kernels issue 128-bit loads and stores.
  if (reduces && (c.send % 16 != 0 || c.recv % 16 != 0)) return Reason::Misaligned;
  // A Broadcast non-root's send buffer is never read.
  if (c.kind == Kind::Broadcast && c.rank != c.root) return Reason::None;
  if (!call_in_place(c) &&
      ranges_overlap(c.send, send_extent(c), c.recv, recv_extent(c)))
    return Reason::PartialOverlap;
  return Reason::None;
}

// Process state the pair decision depends on, gathered by the interposer.
struct PairState {
  bool known_pair = false;       // the two communicators were created together
  bool capturing = false;        // either stream is capturing (or unknown)
  bool first_pass_done = false;  // an eligible pair already ran on NCCL
  bool poisoned = false;         // a post-enqueue CUDA error disabled the shim
};
struct PairDecision {
  Route route;
  Reason reason;
};

inline PairDecision gate_pair(const CallDesc& a, const CallDesc& b,
                              const PairState& s, const Tuning& t) {
  const auto fallback = [](Reason r) { return PairDecision{Route::Fallback, r}; };
  if (s.poisoned) return fallback(Reason::Poisoned);
  const bool devices = (a.device == 0 && b.device == 1) || (a.device == 1 && b.device == 0);
  if (!devices || !s.known_pair) return fallback(Reason::NotPair);
  const bool ranks = (a.rank == 0 && b.rank == 1) || (a.rank == 1 && b.rank == 0);
  if (a.kind != b.kind || a.count != b.count || a.dtype != b.dtype || a.op != b.op || !ranks)
    return fallback(Reason::Mismatch);
  if (a.kind == Kind::Broadcast && (a.root != b.root || (a.root != 0 && a.root != 1)))
    return fallback(Reason::Mismatch);
  const Reason ra = gate_call(a, t);
  if (ra != Reason::None) return fallback(ra);
  const Reason rb = gate_call(b, t);
  if (rb != Reason::None) return fallback(rb);
  if (s.capturing) return fallback(Reason::Capturing);
  if (!s.first_pass_done) return fallback(Reason::FirstPass);
  switch (a.kind) {
    case Kind::AllGather: return {Route::AllGather, Reason::None};
    case Kind::ReduceScatter: return {Route::ReduceScatter, Reason::None};
    case Kind::Broadcast: return {Route::Broadcast, Reason::None};
    default:
      return {Route::ArInputHybrid, Reason::None};
  }
}

// A gated pair as the plan sees it: no pointers, everything indexed by device.
struct PairDesc {
  DType dtype = DType::Unsupported;
  size_t bytes = 0;  // call_bytes
  int rank[2] = {0, 1};
  bool in_place[2] = {false, false};
  int root_dev = -1;  // Broadcast only
};
inline PairDesc describe_pair(const CallDesc& a, const CallDesc& b) {
  const CallDesc* by_dev[2] = {a.device == 0 ? &a : &b, a.device == 0 ? &b : &a};
  PairDesc p;
  p.dtype = a.dtype;
  p.bytes = call_bytes(a);
  for (int d = 0; d < 2; ++d) {
    p.rank[d] = by_dev[d]->rank;
    p.in_place[d] = call_in_place(*by_dev[d]);
  }
  if (a.kind == Kind::Broadcast) p.root_dev = by_dev[0]->rank == a.root ? 0 : 1;
  return p;
}

// =====================================================================
// 3. Plan
// =====================================================================
// Streams. Caller is the application's stream; the rest belong to the shim.
// D2H/H2D lanes are per direction: `dev` is the source GPU (owner) of the
// bytes, the D2H stream lives on it and the H2D stream on the other GPU.
enum class Strm : uint8_t { Caller, Copy, HostReduce, Side, D2H, H2D, Count };
struct StreamId {
  Strm kind;
  uint8_t dev;
  uint8_t lane;
};
inline bool operator==(StreamId x, StreamId y) {
  return x.kind == y.kind && x.dev == y.dev && x.lane == y.lane;
}
inline int stream_device(StreamId s) { return s.kind == Strm::H2D ? 1 - s.dev : s.dev; }

// Events. Copied is indexed per NVLink input piece; D2H/H2D-done per lane
// (`dev` is again the owner). Every event is recorded on its own device.
enum class Ev : uint8_t { InputReady, Reduced, HostReduced, SideDone, Join, Copied, D2HDone, H2DDone, Count };
struct EventId {
  Ev kind;
  uint8_t dev;
  uint8_t lane;
  uint32_t index;
};
inline int event_device(EventId e) { return e.kind == Ev::H2DDone ? 1 - e.dev : e.dev; }

// Buffers. Send/Recv are the caller's; Temp is a per-GPU scratch buffer;
// Bounce is the pinned-host buffer of one lane (`dev` = owner).
enum class Buf : uint8_t { Send, Recv, Temp, Bounce };
struct Loc {
  Buf buf;
  uint8_t dev;
  uint8_t lane;
  size_t off;
};
inline int loc_device(Loc l) { return l.buf == Buf::Bounce ? -1 : l.dev; }
inline Loc advance(Loc l, size_t by) {
  l.off += by;
  return l;
}

// Record: event := current position of stream.     Wait: stream waits event.
// Copy: dst[0,bytes) := a[0,bytes).                 Reduce: dst := a + b.
enum class OpKind : uint8_t { Record, Wait, Copy, Reduce };
struct PlanOp {
  OpKind kind;
  StreamId stream;
  EventId event;
  Loc dst, a, b;
  size_t bytes;
};

// Resources a plan touches; the executor allocates them before submitting.
struct Needs {
  size_t temp_bytes[2] = {0, 0};
  uint32_t copied_events[2] = {0, 0};
  size_t bounce_bytes = 0;
  void merge(const Needs& o) {
    for (int d = 0; d < 2; ++d) {
      temp_bytes[d] = std::max(temp_bytes[d], o.temp_bytes[d]);
      copied_events[d] = std::max(copied_events[d], o.copied_events[d]);
    }
    bounce_bytes = std::max(bounce_bytes, o.bounce_bytes);
  }
};

struct Plan {
  Route route = Route::Fallback;
  DType dtype = DType::Unsupported;
  std::vector<PlanOp> ops;
  Needs needs;
};

inline StreamId caller(int d) { return {Strm::Caller, uint8_t(d), 0}; }
inline StreamId copy_stream(int d) { return {Strm::Copy, uint8_t(d), 0}; }
inline StreamId host_reduce_stream(int d) { return {Strm::HostReduce, uint8_t(d), 0}; }
inline StreamId side_stream(int d) { return {Strm::Side, uint8_t(d), 0}; }
inline StreamId d2h_stream(int owner, int lane) { return {Strm::D2H, uint8_t(owner), uint8_t(lane)}; }
inline StreamId h2d_stream(int owner, int lane) { return {Strm::H2D, uint8_t(owner), uint8_t(lane)}; }
inline EventId input_ready(int d) { return {Ev::InputReady, uint8_t(d), 0, 0}; }
inline EventId reduced(int d) { return {Ev::Reduced, uint8_t(d), 0, 0}; }
inline EventId host_reduced(int d) { return {Ev::HostReduced, uint8_t(d), 0, 0}; }
inline EventId side_done(int d) { return {Ev::SideDone, uint8_t(d), 0, 0}; }
inline EventId join_done(int d) { return {Ev::Join, uint8_t(d), 0, 0}; }
inline EventId copied(int d, size_t i) { return {Ev::Copied, uint8_t(d), 0, uint32_t(i)}; }
inline EventId d2h_done(int owner, int lane) { return {Ev::D2HDone, uint8_t(owner), uint8_t(lane), 0}; }
inline EventId h2d_done(int owner, int lane) { return {Ev::H2DDone, uint8_t(owner), uint8_t(lane), 0}; }
inline Loc send_at(int d, size_t off) { return {Buf::Send, uint8_t(d), 0, off}; }
inline Loc recv_at(int d, size_t off) { return {Buf::Recv, uint8_t(d), 0, off}; }
inline Loc temp_at(int d, size_t off) { return {Buf::Temp, uint8_t(d), 0, off}; }
inline Loc bounce_at(int owner, int lane) { return {Buf::Bounce, uint8_t(owner), uint8_t(lane), 0}; }

class PlanBuilder {
 public:
  PlanBuilder(Plan& plan, Route route, DType dtype) : p_(plan) {
    p_.route = route;
    p_.dtype = dtype;
    p_.ops.clear();
    p_.needs = Needs{};
  }
  void record(EventId e, StreamId s) {
    if (e.kind == Ev::Copied)
      p_.needs.copied_events[e.dev] = std::max(p_.needs.copied_events[e.dev], e.index + 1);
    p_.ops.push_back({OpKind::Record, s, e, {}, {}, {}, 0});
  }
  void wait(StreamId s, EventId e) { p_.ops.push_back({OpKind::Wait, s, e, {}, {}, {}, 0}); }
  void copy(StreamId s, Loc dst, Loc src, size_t n) {
    data(OpKind::Copy, s, dst, src, {}, n);
  }
  void reduce(StreamId s, Loc dst, Loc a, Loc b, size_t n) {
    data(OpKind::Reduce, s, dst, a, b, n);
  }

 private:
  // Both data ops skip empty ranges and account for every scratch/host operand.
  void data(OpKind k, StreamId s, Loc dst, Loc a, Loc b, size_t n) {
    if (n == 0) return;
    for (Loc l : {dst, a, b}) {
      if (l.buf == Buf::Temp) p_.needs.temp_bytes[l.dev] = std::max(p_.needs.temp_bytes[l.dev], l.off + n);
      if (l.buf == Buf::Bounce) p_.needs.bounce_bytes = std::max(p_.needs.bounce_bytes, l.off + n);
    }
    p_.ops.push_back({k, s, {}, dst, a, b, n});
  }
  Plan& p_;
};

// Every pair first waits for the previous custom pair's join on both
// devices. Scratch, bounce buffers, internal streams and events are shared
// process-wide, so this keeps two pairs from overlapping on the GPU even when
// they use unrelated caller streams. Each caller stream then marks its input
// ready; every peer access below waits on that mark.
inline void plan_prologue(PlanBuilder& b) {
  for (int d = 0; d < 2; ++d) {
    b.wait(caller(d), join_done(0));
    b.wait(caller(d), join_done(1));
  }
  for (int d = 0; d < 2; ++d) b.record(input_ready(d), caller(d));
}

// Each caller stream waits until the other device's schedule is done, so
// neither rank returns while its peer still reads its send buffer or writes
// its receive buffer.
inline void plan_join(PlanBuilder& b) {
  for (int d = 0; d < 2; ++d) b.record(join_done(d), caller(d));
  for (int d = 0; d < 2; ++d) b.wait(caller(d), join_done(1 - d));
}

// Stage `n` bytes from `src` (on GPU `owner`) to `dst` (on the other GPU)
// through the owner's pinned-host lanes in `chunk` pieces, round-robin. A
// lane's D2H waits for its previous H2D (bounce reuse) and for `ready`.
// `landed(rel, cb, lane)` may consume each piece after its H2D. Returns the
// piece count.
template <typename Landed>
inline size_t plan_host_stage(PlanBuilder& b, const Tuning& t, int owner, Loc dst, Loc src,
                              size_t n, size_t chunk, const EventId* ready, int nready,
                              Landed&& landed) {
  size_t rel = 0, index = 0;
  while (rel < n) {
    const int lane = static_cast<int>(index++ % static_cast<size_t>(t.lanes));
    const size_t cb = std::min(chunk, n - rel);
    b.wait(d2h_stream(owner, lane), h2d_done(owner, lane));
    for (int i = 0; i < nready; ++i) b.wait(d2h_stream(owner, lane), ready[i]);
    b.copy(d2h_stream(owner, lane), bounce_at(owner, lane), advance(src, rel), cb);
    b.record(d2h_done(owner, lane), d2h_stream(owner, lane));
    b.wait(h2d_stream(owner, lane), d2h_done(owner, lane));
    b.copy(h2d_stream(owner, lane), advance(dst, rel), bounce_at(owner, lane), cb);
    b.record(h2d_done(owner, lane), h2d_stream(owner, lane));
    landed(rel, cb, lane);
    rel += cb;
  }
  return index;
}
inline void no_consumer(size_t, size_t, int) {}

// `s` waits for the last H2D of every lane `pieces` used. Lanes are in-order
// streams, so the last H2D covers the earlier ones on the same lane.
inline void plan_wait_lanes(PlanBuilder& b, const Tuning& t, StreamId s, int owner, size_t pieces) {
  for (int lane = 0; lane < t.lanes && static_cast<size_t>(lane) < pieces; ++lane)
    b.wait(s, h2d_done(owner, lane));
}

// NVLink input pieces: `chunk` bytes, with the final <= chunk bytes cut into
// `tail_chunk` pieces so the last reduction drains sooner.
template <typename Piece>
inline void for_each_input_piece(size_t n, const Tuning& t, Piece&& piece) {
  size_t rel = 0, i = 0;
  while (rel < n) {
    const size_t remaining = n - rel;
    size_t cb = std::min(t.chunk, remaining);
    if (t.tail_chunk > 0 && t.tail_chunk < t.chunk && remaining <= t.chunk)
      cb = std::min(cb, t.tail_chunk);
    piece(i++, rel, cb);
    rel += cb;
  }
}

// AllReduce halves: device 0 owns [0, split), device 1 [split, bytes).
inline size_t ar_split(size_t bytes) { return (bytes / 2) & ~size_t(15); }

// Pinned-host share of one device's inbound input half. The suffix is
// reduced with 16-byte vectors, so it must start 16-byte aligned: an odd half
// gives its < 16-byte remainder to the suffix (job 1445 misaligned fault).
inline size_t ar_input_host_bytes(size_t part, const Tuning& t) {
  size_t host = ((part / t.ar_input_host_divisor) / t.host_chunk) * t.host_chunk;
  if (host > 0) host = part - ((part - host) & ~size_t(15));
  return host;
}
inline size_t ar_output_host_bytes(size_t part, const Tuning& t) {
  return ((part / t.ar_output_host_divisor) / t.host_chunk) * t.host_chunk;
}

// Collective host share: about part/divisor in whole pieces. With
// min_chunks > 0, a share of at least scale_min bytes shrinks its piece (whole
// MiB, at least 1 MiB) so it splits into min_chunks pieces instead of
// rounding to zero. `chunk` returns the piece size to stage with.
inline size_t coll_host_bytes(size_t part, size_t divisor, size_t scale_min,
                              size_t min_chunks, size_t& chunk) {
  if (divisor == 0 || chunk == 0) return 0;
  const size_t target = part / divisor;
  if (min_chunks > 0 && target >= scale_min)
    chunk = std::min(chunk, std::max(kMiB, (target / min_chunks) & ~(kMiB - 1)));
  return (target / chunk) * chunk;
}

// Reduce local + staged peer input into dst. Each Loc already includes the
// collective's slot/half offset; scratch offsets stay relative to this slice.
inline void plan_staged_reduce(PlanBuilder& b, const Tuning& t, Loc dst, Loc local, Loc peer,
                               size_t bytes, size_t host, size_t chunk) {
  const int dev = dst.dev, owner = peer.dev;
  const size_t direct = bytes - host;
  b.wait(copy_stream(dev), input_ready(dev));
  b.wait(copy_stream(dev), input_ready(owner));
  b.wait(caller(dev), input_ready(owner));
  for_each_input_piece(direct, t, [&](size_t i, size_t rel, size_t cb) {
    b.copy(copy_stream(dev), temp_at(dev, rel), advance(peer, rel), cb);
    b.record(copied(dev, i), copy_stream(dev));
    b.wait(caller(dev), copied(dev, i));
    b.reduce(caller(dev), advance(dst, rel), advance(local, rel), temp_at(dev, rel), cb);
  });
  if (host > 0) {
    const EventId ready[2] = {input_ready(owner), input_ready(dev)};
    plan_host_stage(b, t, owner, temp_at(dev, direct), advance(peer, direct), host,
                    chunk, ready, 2, [&](size_t rel, size_t cb, int lane) {
                      const size_t pos = direct + rel;
                      b.wait(host_reduce_stream(dev), h2d_done(owner, lane));
                      b.reduce(host_reduce_stream(dev), advance(dst, pos),
                               advance(local, pos), temp_at(dev, pos), cb);
                    });
    b.record(host_reduced(dev), host_reduce_stream(dev));
    b.wait(caller(dev), host_reduced(dev));
  }
}

// AllReduce, 32-192 MiB. Each GPU reduces its half while reading the peer's
// input directly, then pulls the peer's reduced half.
inline void plan_ar_peerread(PlanBuilder& b, const PairDesc& p, const Tuning&) {
  const size_t split = ar_split(p.bytes);
  const size_t part[2] = {split, p.bytes - split}, off[2] = {0, split};
  plan_prologue(b);
  for (int dev = 0; dev < 2; ++dev) {
    b.wait(caller(dev), input_ready(1 - dev));
    b.reduce(caller(dev), recv_at(dev, off[dev]), send_at(dev, off[dev]),
             send_at(1 - dev, off[dev]), part[dev]);
    b.record(reduced(dev), caller(dev));
  }
  for (int dst = 0; dst < 2; ++dst) {
    const int owner = 1 - dst;
    b.wait(caller(dst), reduced(owner));
    b.copy(caller(dst), recv_at(dst, off[owner]), recv_at(owner, off[owner]), part[owner]);
  }
  plan_join(b);
}

// AllReduce, >= 192 MiB. Input phase: the peer's copy of this GPU's half
// arrives in NVLink chunks on the copy stream (reduced on the caller stream as
// each lands) and as a pinned-host suffix (reduced on the host-reduce stream).
// Output phase, after both halves are reduced: NVLink prefix plus pinned-host
// suffix of each reduced half, in both directions.
inline void plan_ar_inputhybrid(PlanBuilder& b, const PairDesc& p, const Tuning& t) {
  const size_t split = ar_split(p.bytes);
  const size_t part[2] = {split, p.bytes - split}, off[2] = {0, split};
  plan_prologue(b);
  for (int dev = 0; dev < 2; ++dev) {
    const size_t host = ar_input_host_bytes(part[dev], t);
    plan_staged_reduce(b, t, recv_at(dev, off[dev]), send_at(dev, off[dev]),
                       send_at(1 - dev, off[dev]), part[dev], host, t.host_piece);
    b.record(reduced(dev), caller(dev));
  }
  size_t host[2], direct[2];
  for (int owner = 0; owner < 2; ++owner) {
    host[owner] = ar_output_host_bytes(part[owner], t);
    direct[owner] = part[owner] - host[owner];
  }
  for (int dst = 0; dst < 2; ++dst) {
    const int owner = 1 - dst;
    b.wait(caller(dst), reduced(owner));
    if (direct[owner]) {
      b.copy(caller(dst), recv_at(dst, off[owner]), recv_at(owner, off[owner]), direct[owner]);
    }
  }
  size_t pieces[2];
  for (int owner = 0; owner < 2; ++owner) {
    const EventId ready[1] = {reduced(owner)};
    const size_t base = off[owner] + direct[owner];
    pieces[owner] = plan_host_stage(b, t, owner, recv_at(1 - owner, base), recv_at(owner, base),
                                    host[owner], t.host_piece, ready, 1, no_consumer);
  }
  for (int dst = 0; dst < 2; ++dst) plan_wait_lanes(b, t, caller(dst), 1 - dst, pieces[1 - dst]);
  plan_join(b);
}

// AllGather: rank r's input lands in slot r of both outputs. Each GPU pulls
// the peer's slot (NVLink prefix, pinned-host suffix) while its own slot is
// copied locally on the side stream (skipped when in place).
inline void plan_allgather(PlanBuilder& b, const PairDesc& p, const Tuning& t) {
  const size_t n = p.bytes;
  size_t chunk = t.host_chunk;
  const size_t host = coll_host_bytes(n, t.ag_host_divisor, t.ag_host_scale_min,
                                      t.ag_host_min_chunks, chunk);
  const size_t direct = n - host;
  plan_prologue(b);
  for (int dev = 0; dev < 2; ++dev) {
    b.wait(caller(dev), input_ready(1 - dev));
    if (!p.in_place[dev]) {
      b.wait(side_stream(dev), input_ready(dev));
      b.copy(side_stream(dev), recv_at(dev, size_t(p.rank[dev]) * n), send_at(dev, 0), n);
      b.record(side_done(dev), side_stream(dev));
    }
  }
  for (int dst = 0; dst < 2; ++dst) {
    const int owner = 1 - dst;
    if (!direct) continue;
    b.copy(caller(dst), recv_at(dst, size_t(p.rank[owner]) * n), send_at(owner, 0), direct);
  }
  // Join the self copy only after the NVLink copy is queued; waiting first
  // serialized the two (jobs 1440/1441).
  for (int dev = 0; dev < 2; ++dev)
    if (!p.in_place[dev]) b.wait(caller(dev), side_done(dev));
  for (int dst = 0; dst < 2; ++dst) {
    const int owner = 1 - dst;
    if (!host) continue;
    const EventId ready[2] = {input_ready(owner), input_ready(dst)};
    const size_t pieces = plan_host_stage(
        b, t, owner, recv_at(dst, size_t(p.rank[owner]) * n + direct),
        send_at(owner, direct), host, chunk, ready, 2, no_consumer);
    plan_wait_lanes(b, t, caller(dst), owner, pieces);
  }
  plan_join(b);
}

// ReduceScatter: rank r's output is the sum of slot r of both inputs. The
// AllReduce input phase with the reduced slot written to the receive buffer.
inline void plan_reducescatter(PlanBuilder& b, const PairDesc& p, const Tuning& t) {
  const size_t n = p.bytes;
  size_t chunk = t.host_chunk;
  const size_t host = coll_host_bytes(n, t.rs_host_divisor, 0, 0, chunk);
  chunk = t.host_piece;  // stage the unchanged 24 MiB-rounded share
  plan_prologue(b);
  for (int dev = 0; dev < 2; ++dev) {
    const size_t slot = size_t(p.rank[dev]) * n;
    plan_staged_reduce(b, t, recv_at(dev, 0), send_at(dev, slot), send_at(1 - dev, slot),
                       n, host, chunk);
  }
  plan_join(b);
}

// Broadcast: the root's buffer reaches the other GPU over NVLink (prefix) and
// the pinned-host lanes (suffix); the root copies its own buffer when out of
// place.
inline void plan_broadcast(PlanBuilder& b, const PairDesc& p, const Tuning& t) {
  const int rd = p.root_dev, ld = 1 - p.root_dev;
  const size_t n = p.bytes;
  size_t chunk = t.host_chunk;
  const size_t host = coll_host_bytes(n, t.bcast_host_divisor,
      n < t.bcast_scale_end ? t.bcast_host_scale_min : 0,
      n < t.bcast_scale_end ? t.bcast_host_min_chunks : 0, chunk);
  const size_t direct = n - host;
  plan_prologue(b);
  if (!p.in_place[rd]) b.copy(caller(rd), recv_at(rd, 0), send_at(rd, 0), n);
  b.wait(caller(ld), input_ready(rd));
  b.copy(caller(ld), recv_at(ld, 0), send_at(rd, 0), direct);
  if (host) {
    const EventId ready[2] = {input_ready(rd), input_ready(ld)};
    const size_t pieces = plan_host_stage(b, t, rd, recv_at(ld, direct), send_at(rd, direct),
                                          host, chunk, ready, 2, no_consumer);
    plan_wait_lanes(b, t, caller(ld), rd, pieces);
  }
  plan_join(b);
}

// Select once from pair-wide facts. No rank, device, root, placement or runtime
// observation participates. host_chunk stays the allocation/share quantum.
inline Tuning plan_tuning(Route r, const PairDesc& p, const Tuning& base) {
  Tuning t = base;
  t.host_piece = base.host_chunk;
  if (r == Route::ArInputHybrid) {
    if (p.bytes < t.ar_small_end) t.host_piece = t.ar_small_piece;
    if (t.ar_fill_hole && p.bytes >= t.ar_hole_begin && p.bytes < t.ar_hole_end) {
      const size_t divisor = p.bytes < t.ar_gap_split ? t.ar_gap_lo_divisor : t.ar_gap_hi_divisor;
      t.ar_input_host_divisor = t.ar_output_host_divisor = divisor;
      t.host_piece = t.ar_gap_piece;
    }
    if (p.bytes >= (size_t(2) << 30) && p.bytes < (size_t(4) << 30) && t.ar_two_piece)
      t.host_piece = t.ar_two_piece;
  }
  if (r == Route::ReduceScatter && p.bytes < t.rs_small_end) t.host_piece = t.rs_small_piece;
  return t;
}

inline void build_plan(Route r, const PairDesc& p, const Tuning& base, Plan& out) {
  const Tuning t = plan_tuning(r, p, base);
  PlanBuilder b(out, r, p.dtype);
  switch (r) {
    case Route::ArPeerRead: plan_ar_peerread(b, p, t); break;
    case Route::ArInputHybrid: plan_ar_inputhybrid(b, p, t); break;
    case Route::AllGather: plan_allgather(b, p, t); break;
    case Route::ReduceScatter: plan_reducescatter(b, p, t); break;
    case Route::Broadcast: plan_broadcast(b, p, t); break;
    default: break;
  }
}

// Resources for the largest pair prewarm covers: AllReduce inputhybrid and
// ReduceScatter at prewarm_bytes, plus every route's lanes.
inline Needs prewarm_needs(const Tuning& t) {
  Needs needs;
  Plan plan;
  PairDesc p;
  p.dtype = DType::F32;
  p.bytes = t.prewarm_bytes;
  build_plan(Route::ArInputHybrid, p, t, plan);
  needs.merge(plan.needs);
  p.bytes = t.prewarm_bytes / 2;
  build_plan(Route::ReduceScatter, p, t, plan);
  needs.merge(plan.needs);
  needs.bounce_bytes = std::max(needs.bounce_bytes, t.host_chunk);
  // Smaller pairs can need more input pieces than the largest one: the final
  // <= chunk bytes split into up to ceil(chunk / tail_chunk) pieces.
  const size_t half = t.prewarm_bytes / 2;
  const size_t tail_pieces = t.tail_chunk > 0 && t.tail_chunk < t.chunk
                                 ? (t.chunk + t.tail_chunk - 1) / t.tail_chunk
                                 : 1;
  const uint32_t pieces = static_cast<uint32_t>((half + t.chunk - 1) / t.chunk - 1 + tail_pieces);
  for (int d = 0; d < 2; ++d) {
    needs.copied_events[d] = std::max(needs.copied_events[d], pieces);
    // ar_split rounds down to 16 bytes, so device 1's half reaches half + 15.
    needs.temp_bytes[d] = std::max(needs.temp_bytes[d], half + 16);
  }
  return needs;
}

}  // namespace shim

#if !defined(NCCL_SHIM_PLAN_ONLY)
// =====================================================================
// 4. Executor
// =====================================================================
#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <atomic>
#include <cctype>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

using shim::kMaxLanes;

// Arithmetic stays type-specific: float additions for fp32, native CUDA
// intrinsics for fp16/bf16 (eight packed elements per 16-byte vector).
struct Half8 { __half2 a0, a1, a2, a3; };
struct Bf168 { __nv_bfloat162 a0, a1, a2, a3; };
__device__ inline float4 reduce_add_vector(float4 x, float4 y) {
  x.x += y.x; x.y += y.y; x.z += y.z; x.w += y.w;
  return x;
}
template <typename Packed>
__device__ inline Packed reduce_add_vector(Packed x, Packed y) {
  Packed r;
  r.a0 = __hadd2(x.a0, y.a0); r.a1 = __hadd2(x.a1, y.a1);
  r.a2 = __hadd2(x.a2, y.a2); r.a3 = __hadd2(x.a3, y.a3);
  return r;
}
__device__ inline float reduce_add_scalar(float x, float y) { return x + y; }
template <typename Scalar>
__device__ inline Scalar reduce_add_scalar(Scalar x, Scalar y) { return __hadd(x, y); }

// dst may alias a: each thread reads its inputs before writing its output.
template <typename Vector>
__global__ void shim_reduce_write_vector(Vector* dst, const Vector* a, const Vector* b, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = reduce_add_vector(a[i], b[i]);
}
template <typename Scalar>
__global__ void shim_reduce_write_tail(Scalar* dst, const Scalar* a, const Scalar* b, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = reduce_add_scalar(a[i], b[i]);
}

// cudaLaunchKernel returns this launch's own error, unlike <<<>>> followed by
// cudaGetLastError, which would also report (and clear) an unrelated earlier
// error from the application.
template <typename T>
static cudaError_t launch_kernel(void (*kernel)(T*, const T*, const T*, size_t),
                                 unsigned blocks, unsigned threads, cudaStream_t stream,
                                 T* dst, const T* a, const T* b, size_t count) {
  void* argv[] = {&dst, &a, &b, &count};
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(blocks), dim3(threads),
                          argv, 0, stream);
}

// dst = a + b over n bytes: a vector kernel for whole 16-byte units, then a
// scalar kernel for the 0-7 trailing elements (nccl-tests' 8-byte size has no
// whole vector at all).
template <typename Vector, typename Scalar>
static cudaError_t launch_reduce_typed(char* dst, const char* a, const char* b, size_t n,
                                      unsigned blocks, unsigned threads, cudaStream_t s) {
  static_assert(sizeof(Vector) == 16, "reduction vector size");
  const size_t n_vec = n / 16, tail_off = n_vec * 16;
  const size_t tail_n = (n - tail_off) / sizeof(Scalar);
  cudaError_t e = cudaSuccess;
  if (n_vec > 0) {
    e = launch_kernel(shim_reduce_write_vector<Vector>, blocks, threads, s,
                      reinterpret_cast<Vector*>(dst), reinterpret_cast<const Vector*>(a),
                      reinterpret_cast<const Vector*>(b), n_vec);
    if (e != cudaSuccess) return e;
  }
  if (tail_n > 0)
    e = launch_kernel(shim_reduce_write_tail<Scalar>, 1, static_cast<unsigned>(tail_n), s,
                      reinterpret_cast<Scalar*>(dst + tail_off),
                      reinterpret_cast<const Scalar*>(a + tail_off),
                      reinterpret_cast<const Scalar*>(b + tail_off), tail_n);
  return e;
}

static cudaError_t launch_reduce(char* dst, const char* a, const char* b, size_t n,
                                 shim::DType dt, unsigned blocks, unsigned threads,
                                 cudaStream_t s) {
  switch (dt) {
    case shim::DType::F32:
      return launch_reduce_typed<float4, float>(dst, a, b, n, blocks, threads, s);
    case shim::DType::F16:
      return launch_reduce_typed<Half8, __half>(dst, a, b, n, blocks, threads, s);
    default:
      return launch_reduce_typed<Bf168, __nv_bfloat16>(dst, a, b, n, blocks, threads, s);
  }
}

// ---------------------------------------------------------------------
// Process-wide resources, guarded by g_exec_mu. Creation is idempotent: a
// failed step leaves earlier handles valid and the next pair retries the
// rest, and a handle is stored only after its creation succeeded.
// ---------------------------------------------------------------------
struct Resources {
  bool base_ready = false;
  bool peer_enabled[2] = {false, false};
  int sm_count[2] = {0, 0};
  // Plan IDs index these tables; unused kind/lane slots stay null.
  cudaStream_t streams[static_cast<unsigned>(shim::Strm::Count)][2][kMaxLanes] = {};
  cudaEvent_t events[static_cast<unsigned>(shim::Ev::Count)][2][kMaxLanes] = {};
  std::vector<cudaEvent_t> copied[2];
  void* temp[2] = {};
  size_t temp_bytes[2] = {0, 0};
  void* bounce[2][kMaxLanes] = {};
  size_t bounce_bytes[2][kMaxLanes] = {};
  // Nonzero mapping size distinguishes registered mmap from cudaHostAlloc.
  // An unregistered mapping is retained with capacity zero if munmap fails.
  size_t bounce_mapped[2][kMaxLanes] = {};
  bool bounce_registered[2][kMaxLanes] = {};
  cudaStream_t& stream(shim::StreamId s) {
    return streams[static_cast<unsigned>(s.kind)][s.dev][s.lane];
  }
  cudaEvent_t& event(shim::EventId e) {
    if (e.kind == shim::Ev::Copied) return copied[e.dev][e.index];
    return events[static_cast<unsigned>(e.kind)][e.dev][e.lane];
  }
};
static Resources g_res;
static std::mutex g_exec_mu;
static shim::Plan g_plan;
static bool g_poisoned = false;
static bool g_did_first_pass = false;

static bool make_stream(cudaStream_t& s) {
  if (s) return true;
  cudaStream_t created = nullptr;
  if (cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking) != cudaSuccess) return false;
  s = created;
  return true;
}
static bool make_event(cudaEvent_t& e) {
  if (e) return true;
  cudaEvent_t created = nullptr;
  if (cudaEventCreateWithFlags(&created, cudaEventDisableTiming) != cudaSuccess) return false;
  e = created;
  return true;
}
static bool sync_device(int dev) {
  return cudaSetDevice(dev) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

// Peer access is process state. Enabling it is illegal during stream capture,
// so successful setup is remembered.
static bool enable_peer_access() {
  for (int dev = 0; dev < 2; ++dev) {
    if (g_res.peer_enabled[dev]) continue;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, dev, 1 - dev) != cudaSuccess || !can) return false;
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    const cudaError_t e = cudaDeviceEnablePeerAccess(1 - dev, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
    } else if (e != cudaSuccess) {
      fprintf(stderr, "[nccl-shim] cudaDeviceEnablePeerAccess %d->%d failed: %s\n", dev,
              1 - dev, cudaGetErrorString(e));
      return false;
    }
    g_res.peer_enabled[dev] = true;
  }
  return true;
}

static bool ensure_base() {
  if (g_res.base_ready) return true;
  if (!enable_peer_access()) return false;
  const int lanes = shim::kTuning.lanes;
  for (int dev = 0; dev < 2; ++dev) {
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
      return false;
    g_res.sm_count[dev] = sms;
    for (auto kind : {shim::Strm::Copy, shim::Strm::HostReduce, shim::Strm::Side})
      if (!make_stream(g_res.stream({kind, uint8_t(dev), 0}))) return false;
    for (auto kind : {shim::Ev::InputReady, shim::Ev::Reduced, shim::Ev::HostReduced,
                      shim::Ev::SideDone, shim::Ev::Join})
      if (!make_event(g_res.event({kind, uint8_t(dev), 0, 0}))) return false;
  }
  for (int owner = 0; owner < 2; ++owner)
    for (int lane = 0; lane < lanes; ++lane) {
      if (cudaSetDevice(owner) != cudaSuccess || !make_stream(g_res.stream(shim::d2h_stream(owner, lane))) ||
          !make_event(g_res.event(shim::d2h_done(owner, lane))))
        return false;
      if (cudaSetDevice(1 - owner) != cudaSuccess || !make_stream(g_res.stream(shim::h2d_stream(owner, lane))) ||
          !make_event(g_res.event(shim::h2d_done(owner, lane))))
        return false;
    }
  g_res.base_ready = true;
  return true;
}

// CUDA headers differ in whether this enumerator exists. Dependent enum
// lookup selects the PCI/sysfs path at compile time when it is absent.
template <typename Attr>
static auto cuda_numa_node(int dev, int) -> decltype(Attr::cudaDevAttrNumaId, int()) {
  int node = -1;
  if (cudaDeviceGetAttribute(&node, Attr::cudaDevAttrNumaId, dev) == cudaSuccess) return node;
  cudaGetLastError();
  return -1;
}
template <typename Attr>
static int cuda_numa_node(int, long) { return -1; }

static int staging_numa_node(int dev) {
  const int node = cuda_numa_node<cudaDeviceAttr>(dev, 0);
  if (node >= 0) return node;
  char pci[32] = {};
  if (cudaDeviceGetPCIBusId(pci, sizeof(pci), dev) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  for (char* c = pci; *c; ++c) {
    const unsigned char v = static_cast<unsigned char>(*c);
    if (!std::isxdigit(v) && *c != ':' && *c != '.') return -1;
    *c = static_cast<char>(std::tolower(v));
  }
  char path[96];
  std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", pci);
  FILE* f = ::fopen(path, "r");
  if (!f) return -1;
  int placed = -1;
  const bool parsed = ::fscanf(f, "%d", &placed) == 1;
  const bool closed = ::fclose(f) == 0;
  return parsed && closed ? placed : -1;
}

// Query at most 16 evenly spaced pages, including both ends. A negative
// per-page status, mixed placement or syscall failure is unknown, never local.
static int staging_placement(void* p, size_t n, size_t page) {
  if (!n || !page) return -1;
  const size_t pages = 1 + (n - 1) / page;
  const size_t count = std::min(pages, size_t(16));
  void* addresses[16];
  int status[16];
  for (size_t i = 0; i < count; ++i) {
    const size_t index = count == 1 ? 0 : (pages - 1) * i / (count - 1);
    addresses[i] = static_cast<char*>(p) + index * page;
    status[i] = -1;
  }
  if (syscall(SYS_move_pages, 0, count, addresses, nullptr, status, 0) != 0) return -1;
  for (size_t i = 0; i < count; ++i)
    if (status[i] < 0 || status[i] != status[0]) return -1;
  return status[0];
}

static void staging_summary(int owner, int lane, int target, int placed, bool bound);

static bool free_bounce(int owner, int lane) {
  void*& p = g_res.bounce[owner][lane];
  size_t& capacity = g_res.bounce_bytes[owner][lane];
  size_t& mapped = g_res.bounce_mapped[owner][lane];
  bool& registered = g_res.bounce_registered[owner][lane];
  if (!p) return true;
  if (mapped) {
    if (registered) {
      if (cudaHostUnregister(p) != cudaSuccess) { cudaGetLastError(); return false; }
      registered = false;
      capacity = 0;  // A failed munmap must never publish usable capacity.
    }
    if (munmap(p, mapped) != 0) return false;
  } else if (cudaFreeHost(p) != cudaSuccess) {
    return false;
  }
  p = nullptr;
  capacity = mapped = 0;
  return true;
}

static bool allocate_bounce(int owner, int lane, size_t n) {
  const int target = staging_numa_node(owner);
  const long page_size = sysconf(_SC_PAGESIZE);
  const size_t page = page_size > 0 ? static_cast<size_t>(page_size) : 0;
  // Bound the mask size, and reject page rounding overflow before mmap.
  if (target >= 0 && target < 4096 && page && n <= SIZE_MAX - (page - 1)) {
    const size_t rounded = ((n + page - 1) / page) * page;
    void* p = mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
      g_res.bounce[owner][lane] = p;
      g_res.bounce_mapped[owner][lane] = rounded;
      constexpr size_t bits = sizeof(unsigned long) * 8;
      // Linux's raw nodemask ABI consumes maxnode-1 bits. Keep one
      // trailing zero bit, including when target is a mask-word boundary.
      std::vector<unsigned long> mask((static_cast<size_t>(target) + 1) / bits + 1, 0);
      mask[static_cast<size_t>(target) / bits] |= 1UL << (static_cast<size_t>(target) % bits);
      if (syscall(SYS_mbind, p, rounded, MPOL_BIND, mask.data(),
                  static_cast<unsigned long>(target + 2), MPOL_MF_STRICT | MPOL_MF_MOVE) == 0) {
        // Volatile writes force physical allocation before the placement query.
        for (size_t off = 0; off < rounded; off += page)
          static_cast<volatile unsigned char*>(p)[off] = 0;
        if (staging_placement(p, rounded, page) == target) {
          if (cudaHostRegister(p, rounded, cudaHostRegisterPortable) == cudaSuccess) {
            g_res.bounce_registered[owner][lane] = true;
            g_res.bounce_bytes[owner][lane] = n;
            staging_summary(owner, lane, target, target, true);
            return true;
          }
          cudaGetLastError();
        }
      }
      // Cleanup must complete before fallback allocation. If unmap fails,
      // retain this unusable mapping for the next setup retry, never leak it.
      if (!free_bounce(owner, lane)) return false;
    }
  }
  void* p = nullptr;
  if (cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  g_res.bounce[owner][lane] = p;
  g_res.bounce_bytes[owner][lane] = n;
  staging_summary(owner, lane, target, staging_placement(p, n, page), false);
  return true;
}

// Allocate what `needs` asks for. Growing a buffer first drains the devices
// that may still use the old one.
static bool ensure_capacity(const shim::Needs& needs) {
  if (!ensure_base()) return false;
  for (int dev = 0; dev < 2; ++dev) {
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    auto& events = g_res.copied[dev];
    while (events.size() < needs.copied_events[dev]) {
      cudaEvent_t e = nullptr;
      if (!make_event(e)) return false;
      events.push_back(e);
    }
    if (needs.temp_bytes[dev] > g_res.temp_bytes[dev]) {
      if (g_res.temp[dev]) {
        if (!sync_device(dev) || cudaFree(g_res.temp[dev]) != cudaSuccess) return false;
        g_res.temp[dev] = nullptr;
        g_res.temp_bytes[dev] = 0;
      }
      void* p = nullptr;
      const cudaError_t e = cudaMalloc(&p, needs.temp_bytes[dev]);
      if (e != cudaSuccess) {
        fprintf(stderr, "[nccl-shim] cudaMalloc(%zu) on device %d failed: %s\n",
                needs.temp_bytes[dev], dev, cudaGetErrorString(e));
        cudaGetLastError();
        return false;
      }
      g_res.temp[dev] = p;
      g_res.temp_bytes[dev] = needs.temp_bytes[dev];
    }
  }
  for (int owner = 0; owner < 2; ++owner)
    for (int lane = 0; lane < shim::kTuning.lanes; ++lane) {
      if (needs.bounce_bytes <= g_res.bounce_bytes[owner][lane]) continue;
      if (g_res.bounce[owner][lane]) {
        if (!sync_device(0) || !sync_device(1) || !free_bounce(owner, lane)) return false;
      }
      if (!allocate_bounce(owner, lane, needs.bounce_bytes)) return false;
    }
  return true;
}

struct Call {
  // Count/root live in desc; retain native dtype/op for faithful NCCL replay.
  shim::CallDesc desc;
  const void* sendbuff;
  void* recvbuff;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  ncclComm_t comm;
  cudaStream_t stream;
};

static cudaStream_t bind_stream(shim::StreamId s, Call* const by_dev[2]) {
  return s.kind == shim::Strm::Caller ? by_dev[s.dev]->stream : g_res.stream(s);
}
static char* bind_loc(shim::Loc l, Call* const by_dev[2]) {
  switch (l.buf) {
    case shim::Buf::Send: return const_cast<char*>(static_cast<const char*>(by_dev[l.dev]->sendbuff)) + l.off;
    case shim::Buf::Recv: return static_cast<char*>(by_dev[l.dev]->recvbuff) + l.off;
    case shim::Buf::Temp: return static_cast<char*>(g_res.temp[l.dev]) + l.off;
    default: return static_cast<char*>(g_res.bounce[l.dev][l.lane]) + l.off;
  }
}

static cudaError_t submit(const shim::PlanOp& op, const shim::Plan& plan, Call* const by_dev[2]) {
  const cudaStream_t s = bind_stream(op.stream, by_dev);
  switch (op.kind) {
    case shim::OpKind::Record: return cudaEventRecord(g_res.event(op.event), s);
    case shim::OpKind::Wait: return cudaStreamWaitEvent(s, g_res.event(op.event), 0);
    case shim::OpKind::Copy: {
      char* dst = bind_loc(op.dst, by_dev);
      const char* src = bind_loc(op.a, by_dev);
      const int ddev = shim::loc_device(op.dst), sdev = shim::loc_device(op.a);
      if (ddev < 0) return cudaMemcpyAsync(dst, src, op.bytes, cudaMemcpyDeviceToHost, s);
      if (sdev < 0) return cudaMemcpyAsync(dst, src, op.bytes, cudaMemcpyHostToDevice, s);
      if (ddev == sdev) return cudaMemcpyAsync(dst, src, op.bytes, cudaMemcpyDeviceToDevice, s);
      return cudaMemcpyPeerAsync(dst, ddev, src, sdev, op.bytes, s);
    }
    case shim::OpKind::Reduce: {
      const int dev = shim::stream_device(op.stream);
      const unsigned blocks = static_cast<unsigned>(g_res.sm_count[dev] * shim::kTuning.blocks_per_sm);
      return launch_reduce(bind_loc(op.dst, by_dev), bind_loc(op.a, by_dev), bind_loc(op.b, by_dev),
                           op.bytes, plan.dtype, blocks, static_cast<unsigned>(shim::kTuning.threads), s);
    }
    default: return cudaErrorNotSupported;
  }
}

enum class ExecStatus { Ok, NotStarted, Failed };

// Submit every op, checking each CUDA call. NotStarted means no copy or
// kernel was attempted: only event records/waits reached the streams, which
// order but never touch data, so NCCL may still run the pair.
static ExecStatus execute_plan(const shim::Plan& plan, Call* const by_dev[2], cudaError_t* error) {
  bool started = false;
  int current = -1;
  for (size_t i = 0; i < plan.ops.size(); ++i) {
    const shim::PlanOp& op = plan.ops[i];
    const int dev = shim::stream_device(op.stream);
    cudaError_t e = cudaSuccess;
    if (dev != current) {
      e = cudaSetDevice(dev);
      if (e == cudaSuccess) current = dev;
    }
    if (e == cudaSuccess) {
      started |= op.kind == shim::OpKind::Copy || op.kind == shim::OpKind::Reduce;
      e = submit(op, plan, by_dev);
    }
    if (e != cudaSuccess) {
      *error = e;
      return started ? ExecStatus::Failed : ExecStatus::NotStarted;
    }
  }
  return ExecStatus::Ok;
}

// =====================================================================
// 5. Interposer
// =====================================================================
// Use NCCL's declared function types so signatures have one source of truth.
static decltype(&ncclGetVersion) real_ncclGetVersion = nullptr;
static decltype(&ncclCommGetAsyncError) real_ncclCommGetAsyncError = nullptr;
static bool g_supported_nccl = false;
static decltype(&ncclAllReduce) real_ncclAllReduce = nullptr;
static decltype(&ncclGroupStart) real_ncclGroupStart = nullptr;
static decltype(&ncclGroupEnd) real_ncclGroupEnd = nullptr;
static decltype(&ncclCommCuDevice) real_ncclCommCuDevice = nullptr;
static decltype(&ncclCommInitAll) real_ncclCommInitAll = nullptr;
static decltype(&ncclCommInitRank) real_ncclCommInitRank = nullptr;
static decltype(&ncclCommInitRankConfig) real_ncclCommInitRankConfig = nullptr;
static decltype(&ncclCommDestroy) real_ncclCommDestroy = nullptr;
static decltype(&ncclCommUserRank) real_ncclCommUserRank = nullptr;
static decltype(&ncclAllGather) real_ncclAllGather = nullptr;
static decltype(&ncclReduceScatter) real_ncclReduceScatter = nullptr;
static decltype(&ncclBroadcast) real_ncclBroadcast = nullptr;

// The build injects the hash of the complete source file. Embedding a literal
// inside that same file would not hash to itself. Missing identity blocks builds.
#ifndef NCCL_SHIM_SOURCE_SHA256
#error "Define NCCL_SHIM_SOURCE_SHA256 to the SHA-256 of this source file"
#endif
static_assert(sizeof(NCCL_SHIM_SOURCE_SHA256) == 65, "source SHA-256 length");
struct SubmissionSummary {
  const bool enabled = [] {
    const char* value = std::getenv("NCCL_SHIM_SUMMARY");
    return value && std::strcmp(value, "1") == 0;
  }();
  std::atomic<unsigned long long> pairs[6]{}, bytes[6]{};
  std::atomic<unsigned long long> fallbacks[static_cast<size_t>(shim::Reason::Count)]{};
  ~SubmissionSummary() {
    if (!enabled) return;
    const char* routes[] = {"fallback", "ar_peerread", "ar_inputhybrid", "allgather",
                            "reducescatter", "broadcast"};
    const char* reasons[] = {"None", "NotGrouped", "UnsupportedType", "UnsupportedOp",
      "NullBuffer", "SizeOverflow", "BelowThreshold", "Misaligned", "BadSlot",
      "PartialOverlap", "BadRank", "GroupShape", "NotPair", "Mismatch", "Capturing",
      "FirstPass", "SetupFailed", "SubmitFailedEarly", "Poisoned", "AboveLimit",
      "AddressOverflow", "PolicyFallback", "MetadataQuery", "RuntimeUnsupported"};
    static_assert(sizeof(reasons) / sizeof(*reasons) == static_cast<size_t>(shim::Reason::Count),
                  "reason names");
    fprintf(stderr, "[nccl-shim-summary] source_sha256=%s\n", NCCL_SHIM_SOURCE_SHA256);
    for (size_t i = 1; i < 6; ++i)
      fprintf(stderr, "[nccl-shim-summary] route=%s custom_pairs=%llu logical_bytes=%llu\n",
              routes[i], pairs[i].load(std::memory_order_relaxed), bytes[i].load(std::memory_order_relaxed));
    for (size_t i = 1; i < static_cast<size_t>(shim::Reason::Count); ++i)
      fprintf(stderr, "[nccl-shim-summary] fallback_reason=%s calls=%llu\n",
              reasons[i], fallbacks[i].load(std::memory_order_relaxed));
  }
};
static SubmissionSummary g_summary;
static void staging_summary(int owner, int lane, int target, int placed, bool bound) {
  if (g_summary.enabled)
    fprintf(stderr, "[nccl-shim-summary] staging_numa target=%d placed=%d method=%s owner=%d lane=%d\n",
            target, placed, bound ? "mbind" : "fallback", owner, lane);
}
static void count_fallback(shim::Reason reason) {
  if (g_summary.enabled)
    g_summary.fallbacks[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}
static void count_custom(shim::Route route, const shim::CallDesc& call) {
  if (!g_summary.enabled) return;
  const size_t logical = shim::call_bytes(call) *
      ((call.kind == shim::Kind::AllGather || call.kind == shim::Kind::ReduceScatter) ? 2 : 1);
  g_summary.pairs[static_cast<size_t>(route)].fetch_add(1, std::memory_order_relaxed);
  g_summary.bytes[static_cast<size_t>(route)].fetch_add(logical, std::memory_order_relaxed);
}


template <typename Fn>
static void resolve(const char* sym, Fn& fn) {
  fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, sym));
  if (!fn) {
    fprintf(stderr, "[nccl-shim] dlsym failed for %s: %s\n", sym, dlerror());
    abort();
  }
}

static void resolve_all() {
  static std::once_flag once;
  std::call_once(once, [] {
    resolve("ncclGetVersion", real_ncclGetVersion);
    resolve("ncclCommGetAsyncError", real_ncclCommGetAsyncError);
    int version = 0;
    g_supported_nccl = real_ncclGetVersion(&version) == ncclSuccess && version == 22803;
    resolve("ncclAllReduce", real_ncclAllReduce);
    resolve("ncclGroupStart", real_ncclGroupStart);
    resolve("ncclGroupEnd", real_ncclGroupEnd);
    resolve("ncclCommCuDevice", real_ncclCommCuDevice);
    resolve("ncclCommInitAll", real_ncclCommInitAll);
    resolve("ncclCommInitRank", real_ncclCommInitRank);
    resolve("ncclCommInitRankConfig", real_ncclCommInitRankConfig);
    resolve("ncclCommDestroy", real_ncclCommDestroy);
    resolve("ncclCommUserRank", real_ncclCommUserRank);
    resolve("ncclAllGather", real_ncclAllGather);
    resolve("ncclReduceScatter", real_ncclReduceScatter);
    resolve("ncclBroadcast", real_ncclBroadcast);
  });
}

static shim::DType to_dtype(ncclDataType_t t) {
  switch (t) {
    case ncclInt8: return shim::DType::I8;
    case ncclUint8: return shim::DType::U8;
    case ncclFloat16: return shim::DType::F16;
    case ncclBfloat16: return shim::DType::BF16;
    case ncclInt32: return shim::DType::I32;
    case ncclUint32: return shim::DType::U32;
    case ncclFloat32: return shim::DType::F32;
    case ncclInt64: return shim::DType::I64;
    case ncclUint64: return shim::DType::U64;
    case ncclFloat64: return shim::DType::F64;
    default: return shim::DType::Unsupported;
  }
}

static Call make_call(shim::Kind kind, const void* sendbuff, void* recvbuff, size_t count,
                      ncclDataType_t dt, ncclRedOp_t op, int root, ncclComm_t comm,
                      cudaStream_t stream) {
  Call c{{}, sendbuff, recvbuff, dt, op, comm, stream};
  c.desc.kind = kind;
  c.desc.dtype = to_dtype(dt);
  c.desc.op = op == ncclSum ? shim::RedOp::Sum : shim::RedOp::Other;
  c.desc.count = count;
  c.desc.send = reinterpret_cast<uintptr_t>(sendbuff);
  c.desc.recv = reinterpret_cast<uintptr_t>(recvbuff);
  c.desc.root = root;
  return c;
}

static ncclResult_t real_call(const Call& c, shim::Reason reason = shim::Reason::GroupShape) {
  count_fallback(reason);
  switch (c.desc.kind) {
    case shim::Kind::AllGather:
      return real_ncclAllGather(c.sendbuff, c.recvbuff, c.desc.count, c.datatype, c.comm, c.stream);
    case shim::Kind::ReduceScatter:
      return real_ncclReduceScatter(c.sendbuff, c.recvbuff, c.desc.count, c.datatype, c.op, c.comm,
                                    c.stream);
    case shim::Kind::Broadcast:
      return real_ncclBroadcast(c.sendbuff, c.recvbuff, c.desc.count, c.datatype, c.desc.root, c.comm,
                                c.stream);
    default:
      return real_ncclAllReduce(c.sendbuff, c.recvbuff, c.desc.count, c.datatype, c.op, c.comm,
                                c.stream);
  }
}

// Thread-local group state, matching NCCL's rule that a group is issued from
// one thread.
struct PendingInit {
  ncclComm_t* output;
  ncclUniqueId id;
  int rank;
};
struct GroupState {
  int depth = 0;
  bool passthrough = false;
  shim::Reason passthrough_reason = shim::Reason::GroupShape;
  bool init_pair_only = true;
  std::vector<Call> pending;
  std::vector<PendingInit> initializations;
};
static thread_local GroupState g_group;
static thread_local unsigned g_init_all_depth = 0;

// Buffered calls must reach NCCL in their original order.
static ncclResult_t replay_pending(shim::Reason reason = shim::Reason::GroupShape) {
  ncclResult_t first_error = ncclSuccess;
  for (const auto& c : g_group.pending) {
    const ncclResult_t r = real_call(c, reason);
    if (first_error == ncclSuccess && r != ncclSuccess) first_error = r;
  }
  g_group.pending.clear();
  return first_error;
}

// Buffer one eligible call of an outer group or hand it to NCCL. Only two
// calls of one kind can pair; anything else replays what was buffered, in
// order, and keeps the rest of the outer group on NCCL.
static ncclResult_t shim_enqueue(Call c) {
  if (g_group.depth > 0) g_group.init_pair_only = false;
  if (g_group.depth == 0) return real_call(c, shim::Reason::NotGrouped);
  if (g_group.passthrough) return real_call(c, g_group.passthrough_reason);
  const bool same_kind = g_group.pending.empty() || g_group.pending[0].desc.kind == c.desc.kind;
  shim::Reason reason = shim::Reason::GroupShape;
  if (g_group.pending.size() < 2 && same_kind) {
    reason = shim::gate_message(c.desc, shim::kTuning);
    if (reason == shim::Reason::None) {
      reason = shim::Reason::MetadataQuery;
      if (real_ncclCommCuDevice(c.comm, &c.desc.device) == ncclSuccess &&
          real_ncclCommUserRank(c.comm, &c.desc.rank) == ncclSuccess)
        reason = shim::gate_call(c.desc, shim::kTuning);
    }
    if (reason == shim::Reason::None) {
      g_group.pending.push_back(c);
      return ncclSuccess;  // deferred, as NCCL defers grouped calls
    }
  }
  // A pure-check rejection still flushes the buffered prefix in order, then
  // makes this outer group sticky passthrough. Preserve first-error propagation.
  const ncclResult_t replay_result = replay_pending(reason);
  g_group.passthrough = true;
  g_group.passthrough_reason = reason;
  return replay_result != ncclSuccess ? replay_result : real_call(c, reason);
}

// real_ncclGroupStart and real_ncclGroupEnd are each called exactly once per
// interposed call, whatever happens inside: NCCL's own ncclCommInitAll uses
// these public symbols too, and an unmatched pair corrupts its group depth.
// An empty real group, when the shim handles the pair itself, is legal.
extern "C" ncclResult_t ncclGroupStart() {
  resolve_all();
  ncclResult_t replay_result = ncclSuccess;
  if (g_group.depth > 0) {  // nested: the outer group stays on NCCL
    g_group.init_pair_only = false;
    replay_result = replay_pending();
    g_group.passthrough = true;
    g_group.passthrough_reason = shim::Reason::GroupShape;
  } else {
    g_group.pending.clear();
    g_group.initializations.clear();
    g_group.init_pair_only = true;
    g_group.passthrough = false;
    g_group.passthrough_reason = shim::Reason::GroupShape;
  }
  const ncclResult_t r = real_ncclGroupStart();
  if (r == ncclSuccess) ++g_group.depth;
  return replay_result != ncclSuccess ? replay_result : r;
}

// Only two communicators created together (one ncclCommInitAll over devices
// 0 and 1, or ncclCommInitRank calls sharing a unique ID) may pair; equal
// counts and devices alone cannot tell independent communicators apart.
static std::mutex g_pair_mu;
static std::unordered_map<ncclComm_t, ncclComm_t> g_known_peers;
// All lifecycle transitions remove both peers before native processing.
// Only a later successful supported InitAll may establish a new eligible pair.
static void forget_pair(ncclComm_t comm) {
  std::lock_guard<std::mutex> lk(g_pair_mu);
  const auto it = g_known_peers.find(comm);
  if (it != g_known_peers.end()) {
    const ncclComm_t peer = it->second;
    g_known_peers.erase(it);
    g_known_peers.erase(peer);
  }
}
static bool same_pair(ncclComm_t c0, ncclComm_t c1) {
  std::lock_guard<std::mutex> lk(g_pair_mu);
  const auto it = g_known_peers.find(c0);
  return g_supported_nccl && it != g_known_peers.end() && it->second == c1;
}

// Every unsupported operation flushes the prefix before it is forwarded.
// The sticky flag also handles unsupported operations before either pair call.
static ncclResult_t force_passthrough(bool initialization = false) {
  if (g_group.depth == 0) return ncclSuccess;
  if (!initialization) g_group.init_pair_only = false;
  g_group.passthrough = true;
  g_group.passthrough_reason = shim::Reason::GroupShape;
  return replay_pending();
}

// The pinned library defaults to blocking. Reject environment overrides other
// than an explicit 1 and reject configurations whose blocking status is unknown.
static bool blocking_configuration(const ncclConfig_t* config) {
  const char* environment = std::getenv("NCCL_COMM_BLOCKING");
  if (environment && std::strcmp(environment, "1") != 0) return false;
  if (!config) return true;
  return config->size >= offsetof(ncclConfig_t, blocking) + sizeof(config->blocking) &&
         config->magic == 0xcafebeef && config->version >= 21400 && config->version <= 22803 &&
         (config->blocking == 1 || config->blocking == NCCL_CONFIG_UNDEF_INT);
}

static void note_initialization(ncclResult_t result, ncclComm_t* output, int nranks,
                                ncclUniqueId id, int rank, bool blocking) {
  if (g_init_all_depth || g_group.depth == 0) return;
  if (!g_supported_nccl || result != ncclSuccess || !blocking || !output ||
      nranks != 2 || (rank != 0 && rank != 1)) {
    g_group.init_pair_only = false;
    return;
  }
  // Grouped init may defer writing *output until the outer native GroupEnd.
  g_group.initializations.push_back({output, id, rank});
}

static void register_initializations() {
  const auto& pending = g_group.initializations;
  if (!g_supported_nccl || !g_group.init_pair_only || pending.size() != 2 ||
      pending[0].rank == pending[1].rank || pending[0].output == pending[1].output ||
      std::memcmp(pending[0].id.internal, pending[1].id.internal, sizeof(ncclUniqueId)) != 0)
    return;
  const ncclComm_t comm[2] = {*pending[0].output, *pending[1].output};
  if (!comm[0] || !comm[1] || comm[0] == comm[1]) return;
  int device[2] = {-1, -1};
  for (int i = 0; i < 2; ++i) {
    int rank = -1;
    ncclResult_t asynchronous = ncclSuccess;
    if (real_ncclCommCuDevice(comm[i], &device[i]) != ncclSuccess ||
        real_ncclCommUserRank(comm[i], &rank) != ncclSuccess || rank != pending[i].rank ||
        real_ncclCommGetAsyncError(comm[i], &asynchronous) != ncclSuccess || asynchronous != ncclSuccess)
      return;
  }
  if (!((device[0] == 0 && device[1] == 1) || (device[0] == 1 && device[1] == 0))) return;
  std::lock_guard<std::mutex> lk(g_pair_mu);
  g_known_peers[comm[0]] = comm[1];
  g_known_peers[comm[1]] = comm[0];
}

// Allocate every resource the largest pair needs before the application can
// begin graph capture: communicator creation always precedes the first
// collective. One-rank-per-process (ncclCommInitRank) jobs never pair, so they
// allocate nothing.
static void prewarm_once() {
  static bool done = false;
  std::lock_guard<std::mutex> lk(g_exec_mu);
  if (done) return;
  done = true;
  const shim::Needs needs = shim::prewarm_needs(shim::kTuning);
  const bool ok = ensure_capacity(needs);
  // Drain allocation and event creation before NCCL or capture starts.
  const bool drained = sync_device(0) && sync_device(1);
  if (!ok || !drained) cudaGetLastError();
}

extern "C" ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  resolve_all();
  const bool outer_init = g_group.depth == 0;
  const ncclResult_t prefix = force_passthrough();
  if (prefix != ncclSuccess) return prefix;
  // Do not read a devlist entry for an invalid/non-pair initialization.
  const int d0 = ndev == 2 ? (devlist ? devlist[0] : 0) : -1;
  const int d1 = ndev == 2 ? (devlist ? devlist[1] : 1) : -1;
  const bool pair = outer_init && g_supported_nccl && blocking_configuration(nullptr) && ndev == 2 &&
                    ((d0 == 0 && d1 == 1) || (d0 == 1 && d1 == 0));
  bool device_restored = true;
  if (pair) {
    int caller_device = -1;
    if (cudaGetDevice(&caller_device) == cudaSuccess) {
      prewarm_once();
      device_restored = cudaSetDevice(caller_device) == cudaSuccess;
    }
  }
  ++g_init_all_depth;
  const ncclResult_t r = real_ncclCommInitAll(comms, ndev, devlist);
  --g_init_all_depth;
  if (r == ncclSuccess && pair && device_restored && comms && comms[0] && comms[1] &&
      comms[0] != comms[1]) {
    std::lock_guard<std::mutex> lk(g_pair_mu);
    g_known_peers[comms[0]] = comms[1];
    g_known_peers[comms[1]] = comms[0];
  }
  if (r == ncclSuccess && !device_restored) return ncclUnhandledCudaError;
  return r;
}

extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  resolve_all();
  const ncclResult_t prefix = force_passthrough(true);
  if (prefix != ncclSuccess) return prefix;
  const ncclResult_t result = real_ncclCommInitRank(comm, nranks, commId, rank);
  note_initialization(result, comm, nranks, commId, rank, blocking_configuration(nullptr));
  return result;
}

extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                                 int rank, ncclConfig_t* config) {
  resolve_all();
  const ncclResult_t prefix = force_passthrough(true);
  if (prefix != ncclSuccess) return prefix;
  const ncclResult_t result = real_ncclCommInitRankConfig(comm, nranks, commId, rank, config);
  note_initialization(result, comm, nranks, commId, rank, blocking_configuration(config));
  return result;
}

extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  resolve_all();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real_ncclCommDestroy(comm);
}

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) {
  resolve_all();
  return shim_enqueue(make_call(shim::Kind::AllReduce, sendbuff, recvbuff, count, datatype, op, -1,
                                comm, stream));
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                       ncclDataType_t datatype, ncclComm_t comm,
                                       cudaStream_t stream) {
  resolve_all();
  return shim_enqueue(make_call(shim::Kind::AllGather, sendbuff, recvbuff, sendcount, datatype,
                                ncclSum, -1, comm, stream));
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                           ncclDataType_t datatype, ncclRedOp_t op,
                                           ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  return shim_enqueue(make_call(shim::Kind::ReduceScatter, sendbuff, recvbuff, recvcount, datatype,
                                op, -1, comm, stream));
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, int root, ncclComm_t comm,
                                       cudaStream_t stream) {
  resolve_all();
  return shim_enqueue(make_call(shim::Kind::Broadcast, sendbuff, recvbuff, count, datatype, ncclSum,
                                root, comm, stream));
}

// Legacy in-place broadcast, interposed so a buffered pair cannot be
// reordered around it; it replays as ncclBroadcast(buff, buff, ...).
extern "C" ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
                                   ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

// The custom schedules use cross-device event dependencies, which per-stream
// graph captures (nccl-tests -G) cannot represent; captured pairs use NCCL.
// An unknown capture state counts as capturing.
static bool pair_capturing(const Call& a, const Call& b) {
  for (const Call* c : {&a, &b}) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaSetDevice(c->desc.device) != cudaSuccess ||
        cudaStreamIsCapturing(c->stream, &status) != cudaSuccess) {
      cudaGetLastError();
      return true;
    }
    if (status != cudaStreamCaptureStatusNone) return true;
  }
  return false;
}

// Classify by queried properties, not allocator names. VMM requires successful
// READWRITE queries and legacy-IPC capability 0 at both endpoints. Ordinary
// memory requires legacy-IPC capability 1 and each access query must succeed
// with flags 0 or return INVALID_VALUE (observed CUDA 13.2 cudaMalloc uses 0).
// Mixed classes, partial permissions, other errors and unavailable attributes
// are unsupported. Both paths require one covering non-pool range and access
// checks at both endpoints for both devices; ordinary requires peer capability
// in both directions (ensure_base still enables it before any submission).
// No cached pointer eligibility: mapping stability/physical aliasing remain the
// caller's obligations. A failed query cannot lead to a data operation.
static bool supported_buffer(const void* ptr, size_t bytes, int dev) {
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice || attributes.device != dev) {
    cudaGetLastError();
    return false;
  }
  using RangeFn = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);
  using AttrFn = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
  using AccessFn = CUresult (*)(unsigned long long*, const CUmemLocation*, CUdeviceptr);
  static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  static auto range = driver ? reinterpret_cast<RangeFn>(dlsym(driver, "cuMemGetAddressRange_v2")) : nullptr;
  static auto attribute = driver ? reinterpret_cast<AttrFn>(dlsym(driver, "cuPointerGetAttribute")) : nullptr;
  static auto access = driver ? reinterpret_cast<AccessFn>(dlsym(driver, "cuMemGetAccess")) : nullptr;
  if (!range || !attribute || !access || bytes == 0) return false;
  const auto address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
  if (bytes - 1 > UINTPTR_MAX - address) return false;
  const CUdeviceptr last = address + bytes - 1;
  CUdeviceptr base = 0, last_base = 0;
  size_t extent = 0, last_extent = 0;
  if (range(&base, &extent, address) != CUDA_SUCCESS || address < base ||
      address - base >= extent || bytes > extent - (address - base) ||
      range(&last_base, &last_extent, last) != CUDA_SUCCESS ||
      last_base != base || last_extent != extent) return false;
  bool vmm = true, ordinary = true;
  for (CUdeviceptr endpoint : {address, last}) {
    CUmemoryPool pool = nullptr;
    if (attribute(&pool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, endpoint) != CUDA_SUCCESS || pool)
      return false;
    int legacy_ipc = 0;
    if (attribute(&legacy_ipc, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, endpoint) != CUDA_SUCCESS)
      return false;
    if (legacy_ipc != 0) vmm = false;
    if (legacy_ipc != 1) ordinary = false;
    for (int device = 0; device < 2; ++device) {
      CUmemLocation location{};
      location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      location.id = device;
      unsigned long long flags = 0;
      const CUresult result = access(&flags, &location, endpoint);
      if (result == CUDA_SUCCESS) {
        if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) vmm = false;
        if (flags != 0) ordinary = false;
      } else if (result == CUDA_ERROR_INVALID_VALUE) {
        vmm = false;
      } else {
        return false;
      }
    }
  }
  if (vmm) return true;
  if (!ordinary) return false;  // inconsistent endpoint/device classification
  for (int device = 0; device < 2; ++device) {
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, device, 1 - device) != cudaSuccess || !can) {
      cudaGetLastError();
      return false;
    }
  }
  return true;
}

static bool supported_runtime_pair(const Call& a, const Call& b) {
  for (const Call* c : {&a, &b}) {
    const int dev = c->desc.device;
    ncclResult_t asynchronous = ncclSuccess;
    if (real_ncclCommGetAsyncError(c->comm, &asynchronous) != ncclSuccess ||
        asynchronous != ncclSuccess) {
      forget_pair(c->comm);
      return false;
    }
    int stream_dev = -1;
    if (cudaSetDevice(dev) != cudaSuccess ||
        cudaStreamGetDevice(c->stream, &stream_dev) != cudaSuccess || stream_dev != dev) {
      cudaGetLastError();
      return false;
    }
    if (!supported_buffer(c->recvbuff, shim::recv_extent(c->desc), dev)) return false;
    if (!(c->desc.kind == shim::Kind::Broadcast && c->desc.rank != c->desc.root) &&
        !supported_buffer(c->sendbuff, shim::send_extent(c->desc), dev)) return false;
  }
  return true;
}

enum class PairOutcome { Handled, Fallback, Error };

// Gate, plan and execute one buffered pair under the process-wide mutex.
static PairOutcome run_pair(Call& a, Call& b, shim::Reason& reason) {
  std::lock_guard<std::mutex> lk(g_exec_mu);
  shim::PairState state;
  state.poisoned = g_poisoned;
  state.known_pair = same_pair(a.comm, b.comm);
  state.first_pass_done = g_did_first_pass;
  const bool devices = (a.desc.device == 0 && b.desc.device == 1) ||
                       (a.desc.device == 1 && b.desc.device == 0);
  state.capturing = devices && state.known_pair && pair_capturing(a, b);
  const shim::PairDecision d = shim::gate_pair(a.desc, b.desc, state, shim::kTuning);
  reason = d.reason;
  if (d.reason == shim::Reason::FirstPass) g_did_first_pass = true;
  if (d.route != shim::Route::Fallback) {
    reason = shim::Reason::RuntimeUnsupported;
    if (!supported_runtime_pair(a, b)) return PairOutcome::Fallback;
    Call* by_dev[2] = {a.desc.device == 0 ? &a : &b, a.desc.device == 0 ? &b : &a};
    shim::build_plan(d.route, shim::describe_pair(a.desc, b.desc), shim::kTuning, g_plan);
    cudaError_t error = cudaSuccess;
    const bool capacity = ensure_capacity(g_plan.needs);
    const ExecStatus status = capacity ? execute_plan(g_plan, by_dev, &error) : ExecStatus::NotStarted;
    reason = capacity ? shim::Reason::SubmitFailedEarly : shim::Reason::SetupFailed;
    if (status == ExecStatus::Failed) {
      g_poisoned = true;
      fprintf(stderr,
              "[nccl-shim] %s: CUDA error after custom work was enqueued (%s); returning "
              "ncclUnhandledCudaError, custom paths disabled for this process\n",
              shim::kind_name(a.desc.kind), cudaGetErrorString(error));
      // Let the submitted work stop touching the caller's buffers before the
      // error reaches the application.
      sync_device(0);
      sync_device(1);
      return PairOutcome::Error;
    }
    if (status == ExecStatus::Ok) {
      count_custom(d.route, a.desc);
      return PairOutcome::Handled;
    }
    cudaGetLastError();  // consumed here; NCCL must not see it
  }
  return PairOutcome::Fallback;
}

static ncclResult_t group_end_impl() {
  if (g_group.depth > 1) {
    --g_group.depth;
    return real_ncclGroupEnd();
  }
  if (g_group.depth == 0) return real_ncclGroupEnd();
  g_group.depth = 0;
  PairOutcome outcome = PairOutcome::Fallback;
  shim::Reason reason = shim::Reason::GroupShape;
  if (!g_group.passthrough && g_group.pending.size() == 2)
    outcome = run_pair(g_group.pending[0], g_group.pending[1], reason);
  g_group.passthrough = false;
  // Never replay over partially written buffers; otherwise replay inside
  // the real group ncclGroupStart already opened. Preserve the first error.
  ncclResult_t result = ncclSuccess;
  if (outcome == PairOutcome::Error) result = ncclUnhandledCudaError;
  else if (outcome == PairOutcome::Fallback) result = replay_pending(reason);
  g_group.pending.clear();
  const ncclResult_t end_result = real_ncclGroupEnd();
  return result != ncclSuccess ? result : end_result;
}

extern "C" ncclResult_t ncclGroupEnd() {
  resolve_all();
  int caller_device = -1;
  const cudaError_t get_error = cudaGetDevice(&caller_device);
  const bool outer = g_group.depth == 1;
  const ncclResult_t result = group_end_impl();
  const cudaError_t restore_error =
      get_error == cudaSuccess ? cudaSetDevice(caller_device) : get_error;
  if (outer) {
    if (result == ncclSuccess && restore_error == cudaSuccess) register_initializations();
    g_group.initializations.clear();
  }
  if (restore_error != cudaSuccess) {
    return result == ncclSuccess ? ncclUnhandledCudaError : result;
  }
  return result;
}


// Unsupported operations are visible to the group state machine.
extern "C" ncclResult_t ncclReduce(const void* send, void* recv, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  static auto real = [] { decltype(&::ncclReduce) fn = nullptr; resolve("ncclReduce", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(send, recv, count, type, op, root, comm, stream);
}

extern "C" ncclResult_t ncclSend(const void* send, size_t count, ncclDataType_t type, int peer, ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  static auto real = [] { decltype(&::ncclSend) fn = nullptr; resolve("ncclSend", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(send, count, type, peer, comm, stream);
}

extern "C" ncclResult_t ncclRecv(void* recv, size_t count, ncclDataType_t type, int peer, ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  static auto real = [] { decltype(&::ncclRecv) fn = nullptr; resolve("ncclRecv", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(recv, count, type, peer, comm, stream);
}

extern "C" ncclResult_t ncclAlltoAll(const void* send, void* recv, size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  static auto real = [] { decltype(&::ncclAlltoAll) fn = nullptr; resolve("ncclAlltoAll", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(send, recv, count, type, comm, stream);
}

extern "C" ncclResult_t ncclGather(const void* send, void* recv, size_t count, ncclDataType_t type, int root, ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  static auto real = [] { decltype(&::ncclGather) fn = nullptr; resolve("ncclGather", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(send, recv, count, type, root, comm, stream);
}

extern "C" ncclResult_t ncclScatter(const void* send, void* recv, size_t count, ncclDataType_t type, int root, ncclComm_t comm, cudaStream_t stream) {
  resolve_all();
  static auto real = [] { decltype(&::ncclScatter) fn = nullptr; resolve("ncclScatter", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(send, recv, count, type, root, comm, stream);
}

extern "C" ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommFinalize) fn = nullptr; resolve("ncclCommFinalize", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm);
}

extern "C" ncclResult_t ncclCommAbort(ncclComm_t comm) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommAbort) fn = nullptr; resolve("ncclCommAbort", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm);
}

extern "C" ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* output, ncclConfig_t* config) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommSplit) fn = nullptr; resolve("ncclCommSplit", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm, color, key, output, config);
}

extern "C" ncclResult_t ncclCommShrink(ncclComm_t comm, int* excluded, int count, ncclComm_t* output, ncclConfig_t* config, int flags) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommShrink) fn = nullptr; resolve("ncclCommShrink", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm, excluded, count, output, config, flags);
}

extern "C" ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buffer, size_t bytes, void** handle) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommRegister) fn = nullptr; resolve("ncclCommRegister", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm, buffer, bytes, handle);
}

extern "C" ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommDeregister) fn = nullptr; resolve("ncclCommDeregister", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm, handle);
}

extern "C" ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buffer, size_t bytes, ncclWindow_t* window, int flags) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommWindowRegister) fn = nullptr; resolve("ncclCommWindowRegister", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm, buffer, bytes, window, flags);
}

extern "C" ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t window) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommWindowDeregister) fn = nullptr; resolve("ncclCommWindowDeregister", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  forget_pair(comm);
  return prefix != ncclSuccess ? prefix : real(comm, window);
}

extern "C" ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t type, ncclScalarResidence_t residence, ncclComm_t comm) {
  resolve_all();
  static auto real = [] { decltype(&::ncclRedOpCreatePreMulSum) fn = nullptr; resolve("ncclRedOpCreatePreMulSum", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(op, scalar, type, residence, comm);
}

extern "C" ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  resolve_all();
  static auto real = [] { decltype(&::ncclRedOpDestroy) fn = nullptr; resolve("ncclRedOpDestroy", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(op, comm);
}

extern "C" ncclResult_t ncclCommInitRankScalable(ncclComm_t* output, int nranks, int rank, int ids, ncclUniqueId* unique_ids, ncclConfig_t* config) {
  resolve_all();
  static auto real = [] { decltype(&::ncclCommInitRankScalable) fn = nullptr; resolve("ncclCommInitRankScalable", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  return prefix != ncclSuccess ? prefix : real(output, nranks, rank, ids, unique_ids, config);
}

extern "C" ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* info) {
  resolve_all();
  static auto real = [] { decltype(&::ncclGroupSimulateEnd) fn = nullptr;
    resolve("ncclGroupSimulateEnd", fn); return fn; }();
  const ncclResult_t prefix = force_passthrough();
  if (g_group.depth > 0) --g_group.depth;
  if (g_group.depth == 0) g_group.passthrough = false;
  const ncclResult_t result = real(info);
  g_group.initializations.clear();
  return prefix != ncclSuccess ? prefix : result;
}

#endif  // !NCCL_SHIM_PLAN_ONLY
