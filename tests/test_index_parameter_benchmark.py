from enterprise_rag_engine.evals.index_parameter_benchmark import (
    IndexBenchmarkScenario,
    IndexParameterProfile,
    benchmark_index_profiles,
    default_index_profiles,
    render_index_benchmark_markdown,
)


def test_default_index_profiles_cover_latency_recall_and_filter_tradeoffs() -> None:
    profiles = default_index_profiles()

    assert [profile.name for profile in profiles] == [
        "low_latency",
        "balanced",
        "high_recall",
        "filtered_without_payload_index",
    ]
    assert profiles[0].search_ef < profiles[2].search_ef
    assert profiles[-1].payload_indexed is False


def test_benchmark_index_profiles_scores_expected_tradeoffs() -> None:
    scenario = IndexBenchmarkScenario(
        vector_count=50_000,
        dimension=1024,
        top_k=5,
        filter_selectivity=0.05,
    )

    rows = benchmark_index_profiles(default_index_profiles(), scenario)
    by_name = {row.profile_name: row for row in rows}

    assert by_name["low_latency"].p95_latency_ms < by_name["high_recall"].p95_latency_ms
    assert by_name["high_recall"].recall_at_k > by_name["low_latency"].recall_at_k
    assert by_name["filtered_without_payload_index"].filter_penalty_ms > 0
    assert by_name["balanced"].recommendation == "default"


def test_benchmark_rejects_invalid_profiles_and_scenarios() -> None:
    invalid_profile = IndexParameterProfile(
        name="invalid",
        hnsw_m=0,
        ef_construct=100,
        search_ef=100,
        full_scan_threshold_kb=10_000,
        payload_indexed=True,
    )
    scenario = IndexBenchmarkScenario(vector_count=1_000, dimension=768, top_k=5)

    try:
        benchmark_index_profiles((invalid_profile,), scenario)
    except ValueError as exc:
        assert "hnsw_m" in str(exc)
    else:
        raise AssertionError("invalid hnsw_m should be rejected")

    try:
        IndexBenchmarkScenario(vector_count=0, dimension=768, top_k=5)
    except ValueError as exc:
        assert "vector_count" in str(exc)
    else:
        raise AssertionError("invalid vector_count should be rejected")


def test_render_index_benchmark_markdown_outputs_table() -> None:
    scenario = IndexBenchmarkScenario(vector_count=10_000, dimension=768, top_k=5)
    rows = benchmark_index_profiles(default_index_profiles(), scenario)

    markdown = render_index_benchmark_markdown(rows)

    assert "| profile | recall@5 | p95_latency_ms | memory_mb | recommendation |" in markdown
    assert "| balanced |" in markdown
    assert "filtered_without_payload_index" in markdown
