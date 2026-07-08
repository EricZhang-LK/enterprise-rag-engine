from dataclasses import asdict, dataclass, replace
from math import log2, log10
from typing import Literal

QuantizationMode = Literal["none", "scalar"]


@dataclass(frozen=True, slots=True)
class IndexParameterProfile:
    name: str
    hnsw_m: int
    ef_construct: int
    search_ef: int
    full_scan_threshold_kb: int
    payload_indexed: bool
    quantization: QuantizationMode = "none"

    def validate(self) -> None:
        if self.hnsw_m < 1:
            msg = "hnsw_m must be greater than 0"
            raise ValueError(msg)
        if self.ef_construct < 1:
            msg = "ef_construct must be greater than 0"
            raise ValueError(msg)
        if self.search_ef < 1:
            msg = "search_ef must be greater than 0"
            raise ValueError(msg)
        if self.full_scan_threshold_kb < 0:
            msg = "full_scan_threshold_kb must not be negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class IndexBenchmarkScenario:
    vector_count: int
    dimension: int
    top_k: int
    filter_selectivity: float = 1.0

    def __post_init__(self) -> None:
        if self.vector_count < 1:
            msg = "vector_count must be greater than 0"
            raise ValueError(msg)
        if self.dimension < 1:
            msg = "dimension must be greater than 0"
            raise ValueError(msg)
        if self.top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        if not 0 < self.filter_selectivity <= 1:
            msg = "filter_selectivity must be in the range (0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class IndexBenchmarkRow:
    profile_name: str
    hnsw_m: int
    ef_construct: int
    search_ef: int
    recall_at_k: float
    p95_latency_ms: float
    memory_mb: float
    build_cost_score: float
    filter_penalty_ms: float
    recommendation: str

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def default_index_profiles() -> tuple[IndexParameterProfile, ...]:
    """Return opinionated HNSW profiles for a first enterprise RAG benchmark."""

    return (
        IndexParameterProfile(
            name="low_latency",
            hnsw_m=12,
            ef_construct=64,
            search_ef=64,
            full_scan_threshold_kb=10_000,
            payload_indexed=True,
            quantization="scalar",
        ),
        IndexParameterProfile(
            name="balanced",
            hnsw_m=16,
            ef_construct=128,
            search_ef=128,
            full_scan_threshold_kb=10_000,
            payload_indexed=True,
        ),
        IndexParameterProfile(
            name="high_recall",
            hnsw_m=32,
            ef_construct=256,
            search_ef=256,
            full_scan_threshold_kb=10_000,
            payload_indexed=True,
        ),
        IndexParameterProfile(
            name="filtered_without_payload_index",
            hnsw_m=16,
            ef_construct=128,
            search_ef=128,
            full_scan_threshold_kb=10_000,
            payload_indexed=False,
        ),
    )


def benchmark_index_profiles(
    profiles: tuple[IndexParameterProfile, ...],
    scenario: IndexBenchmarkScenario,
) -> tuple[IndexBenchmarkRow, ...]:
    """Estimate recall, latency, memory, and build-cost tradeoffs for index profiles."""

    for profile in profiles:
        profile.validate()

    rows = tuple(_estimate_profile(profile, scenario) for profile in profiles)
    default_profile = _select_default_profile(rows)
    return tuple(
        row
        if row.profile_name != default_profile.profile_name
        else replace(row, recommendation="default")
        for row in rows
    )


def render_index_benchmark_markdown(rows: tuple[IndexBenchmarkRow, ...]) -> str:
    """Render benchmark rows as a compact Markdown table for reports and PRs."""

    lines = [
        "| profile | recall@5 | p95_latency_ms | memory_mb | recommendation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.profile_name} | "
            f"{row.recall_at_k:.3f} | "
            f"{row.p95_latency_ms:.3f} | "
            f"{row.memory_mb:.3f} | "
            f"{row.recommendation or '-'} |"
        )
    return "\n".join(lines)


def _estimate_profile(
    profile: IndexParameterProfile,
    scenario: IndexBenchmarkScenario,
) -> IndexBenchmarkRow:
    vector_bytes = 1 if profile.quantization == "scalar" else 4
    vector_memory_mb = scenario.vector_count * scenario.dimension * vector_bytes / (1024 * 1024)
    graph_memory_mb = scenario.vector_count * profile.hnsw_m * 8 / (1024 * 1024)
    recall_penalty = 0.025 if profile.quantization == "scalar" else 0.0
    filter_penalty_ms = _filter_penalty_ms(profile, scenario)

    recall = min(
        0.995,
        0.63
        + log2(profile.search_ef) * 0.032
        + log2(profile.hnsw_m) * 0.022
        + log2(profile.ef_construct) * 0.018
        - recall_penalty,
    )
    latency = (
        2.0
        + log10(scenario.vector_count) * 1.6
        + profile.search_ef * 0.045
        + scenario.top_k * 0.18
        + filter_penalty_ms
    )
    if profile.quantization == "scalar":
        latency *= 0.88

    build_cost = scenario.vector_count * profile.hnsw_m * profile.ef_construct / 1_000_000
    return IndexBenchmarkRow(
        profile_name=profile.name,
        hnsw_m=profile.hnsw_m,
        ef_construct=profile.ef_construct,
        search_ef=profile.search_ef,
        recall_at_k=round(recall, 3),
        p95_latency_ms=round(latency, 3),
        memory_mb=round(vector_memory_mb + graph_memory_mb, 3),
        build_cost_score=round(build_cost, 3),
        filter_penalty_ms=round(filter_penalty_ms, 3),
        recommendation="",
    )


def _filter_penalty_ms(
    profile: IndexParameterProfile,
    scenario: IndexBenchmarkScenario,
) -> float:
    if profile.payload_indexed:
        return 0.0
    filtered_ratio = 1 - scenario.filter_selectivity
    return 55.0 * filtered_ratio


def _select_default_profile(rows: tuple[IndexBenchmarkRow, ...]) -> IndexBenchmarkRow:
    max_recall = max(row.recall_at_k for row in rows)
    eligible_rows = tuple(
        row for row in rows if row.filter_penalty_ms == 0 and row.recall_at_k >= max_recall - 0.005
    )
    candidates = eligible_rows or rows
    return min(
        candidates,
        key=lambda row: row.p95_latency_ms + row.memory_mb * 0.005 + row.build_cost_score * 0.02,
    )
