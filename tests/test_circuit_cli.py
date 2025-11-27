from pathlib import Path
import sys
import argparse

import sympy as sp

sys.path.append(str(Path(__file__).resolve().parents[1]))

from circuit_cli import (  # noqa: E402
    CircuitGraph,
    ComponentSpec,
    UnitNormalizer,
    SESSION,
    _format_equations,
    export_to_csv,
    export_to_latex,
    nodal_equations,
    parse_component_line,
    parse_falstad_netlist,
    parse_substitutions,
    run,
    solve_equations,
)


def test_resistive_divider_voltage_half_supply():
    normalizer = UnitNormalizer()
    components = [
        ComponentSpec("V", "V1", "vin", "0", "10", normalizer=normalizer),
        ComponentSpec("R", "R1", "vin", "vout", "10k", normalizer=normalizer),
        ComponentSpec("R", "R2", "vout", "0", "10k", normalizer=normalizer),
    ]
    graph = CircuitGraph(components)
    equations, _ = nodal_equations(graph, "laplace")
    solutions = solve_equations(equations)
    assert sp.simplify(solutions["V_vout"] - 5) == 0


def test_rc_low_pass_transfer_function():
    s = sp.symbols("s")
    normalizer = UnitNormalizer()
    components = [
        ComponentSpec("V", "V1", "vin", "0", "5", normalizer=normalizer),
        ComponentSpec("R", "R1", "vin", "vout", "1k", normalizer=normalizer),
        ComponentSpec("C", "C1", "vout", "0", "1u", normalizer=normalizer),
    ]
    graph = CircuitGraph(components)
    equations, _ = nodal_equations(graph, "laplace")
    solutions = solve_equations(equations)
    expected = 5 / (1 + s * 1000 * 1e-6)
    assert sp.simplify(solutions["V_vout"] - expected) == 0


def test_wheatstone_bridge_balanced_nodes_match():
    normalizer = UnitNormalizer()
    components = [
        ComponentSpec("V", "V1", "vplus", "0", "10", normalizer=normalizer),
        ComponentSpec("R", "R1", "vplus", "n1", "1k", normalizer=normalizer),
        ComponentSpec("R", "R2", "vplus", "n2", "1k", normalizer=normalizer),
        ComponentSpec("R", "R3", "n1", "0", "1k", normalizer=normalizer),
        ComponentSpec("R", "R4", "n2", "0", "1k", normalizer=normalizer),
        ComponentSpec("R", "R5", "n1", "n2", "1k", normalizer=normalizer),
    ]
    graph = CircuitGraph(components)
    equations, _ = nodal_equations(graph, "laplace")
    solutions = solve_equations(equations)
    assert sp.simplify(solutions["V_n1"] - solutions["V_n2"]) == 0


def test_format_equations_returns_pretty_and_latex():
    s = sp.symbols("s")
    equation = sp.Eq(sp.symbols("V_out"), 1 / (s + 1))

    formatted = _format_equations([equation])

    assert formatted[0]["plain"] == "V_out = 1/(s + 1)"
    assert formatted[0]["latex"] == "V_{out} = \\frac{1}{s + 1}"
    assert " = " in formatted[0]["pretty"].splitlines()[0]


def test_export_functions_handle_formatted_equations(tmp_path):
    equation = sp.Eq(sp.symbols("V_out"), sp.symbols("V_in"))
    summary = {"equations": _format_equations([equation])}
    solution = {"V_out": sp.symbols("V_in")}

    csv_path = tmp_path / "summary.csv"
    latex_path = tmp_path / "summary.tex"

    export_to_csv(csv_path, summary, solution)
    export_to_latex(latex_path, summary, solution)

    csv_content = csv_path.read_text()
    latex_content = latex_path.read_text()
    equation_plain = summary["equations"][0]["plain"]
    equation_latex = summary["equations"][0]["latex"]

    assert f"ecuacion,{equation_plain}" in csv_content
    assert f"$${equation_latex}$$" in latex_content


def test_substitutions_parse_numeric_values():
    substitutions = parse_substitutions(["V1=10", "alpha=2*pi"])
    assert substitutions[sp.symbols("V1")] == 10
    assert substitutions[sp.symbols("alpha")].simplify() == 2 * sp.pi


def test_parse_falstad_netlist_builds_expected_topology():
    normalizer = UnitNormalizer()
    falstad_text = """
    $ 1 0.000005 10.20027730826997 50 5 50
    v 176 80 176 160 0 10 0 0 0 0
    r 176 80 256 80 0 1000
    r 256 80 336 80 0 1000
    w 336 80 336 160 0
    g 336 160 336 176 0
    g 176 160 176 176 0
    """

    components = parse_falstad_netlist(falstad_text, normalizer)

    assert [(c.type, c.node_a, c.node_b, c.value) for c in components] == [
        ("V", "n1", "0", "10"),
        ("R", "n1", "n2", "1000"),
        ("R", "n2", "0", "1000"),
    ]

    graph = CircuitGraph(components)
    assert graph.nodes == {"n1", "n2", "0"}
    assert not graph.warnings


def test_normalize_value_preserves_initial_condition_suffix():
    normalizer = UnitNormalizer()

    normalized = normalizer.normalize_value("1 ic=0.5", "L")

    assert normalized == "1H ic=0.5"


def test_falstad_netlist_with_initial_condition_parses_successfully():
    normalizer = UnitNormalizer()
    falstad_text = """
    $ 1 0.000005 10.20027730826997 50 5 50
    l 0 0 0 64 0 1 1 0.25 0
    g 0 0 0 16 0
    """

    components = parse_falstad_netlist(falstad_text, normalizer)

    assert any(comp.value.endswith("ic=0.25") for comp in components if comp.type == "L")
    inductor = next(comp for comp in components if comp.type == "L")
    assert inductor.netlist_line().endswith("ic=0.25")


def test_parse_component_line_detects_spice_separator():
    normalizer = UnitNormalizer()

    resistor = parse_component_line("R1 1 2 4", normalizer)

    assert resistor.type == "R"
    assert resistor.node_a == "1"
    assert resistor.node_b == "2"
    assert resistor.value == "4"
    assert resistor.netlist_line() == "R1 1 2 4"


def test_parse_component_line_handles_initial_condition_for_lc():
    normalizer = UnitNormalizer()

    capacitor = parse_component_line("C1 0 3 0.5 1", normalizer)

    assert capacitor.type == "C"
    assert capacitor.value == "0.5 ic=1"
    assert capacitor.netlist_line() == "C1 0 3 0.5 ic=1"


def _make_args(csv_paths, domain="laplace"):
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    return argparse.Namespace(
        mode="single" if len(csv_paths) == 1 else "double",
        csv=[str(path) for path in csv_paths],
        falstad=None,
        method="nodal",
        domain=domain,
        show_matrices=False,
        show_lcapy=False,
        solve=False,
        analyze=False,
        substitute=None,
        export_csv=None,
        export_latex=None,
        plot_time=None,
        plot_freq=None,
        query=None,
        query_time=None,
    )


def test_session_stores_single_topology_variants(tmp_path):
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("\n".join(["V,V1,n1,0,5", "R,R1,n1,0,1k"]), encoding="utf-8")

    args = _make_args(csv_path, domain="jw")
    result = run(args)

    session_summary = result["session"]
    assert session_summary["active_domain"] == "jw"
    single = session_summary["topologies"]["single"]
    assert single["references"]["circuit"]
    assert single["references"]["circuit_dc"]
    assert single["references"]["circuit_s"]
    assert single["references"]["circuit_jw"]
    assert "V1" in single["netlist"]
    assert SESSION.topologies["single"].circuit_jw is not None


def test_session_tracks_double_topology(tmp_path):
    pre_csv = tmp_path / "pre.csv"
    post_csv = tmp_path / "post.csv"
    pre_csv.write_text("\n".join(["V,V1,n1,0,5", "R,R1,n1,0,1k"]), encoding="utf-8")
    post_csv.write_text("\n".join(["V,V1,n1,0,10", "R,R1,n1,0,2k"]), encoding="utf-8")

    args = _make_args([pre_csv, post_csv])
    result = run(args)

    session_summary = result["session"]
    assert set(session_summary["topologies"].keys()) == {"pre", "post"}
    assert session_summary["active_domain"] == "laplace"
    assert SESSION.topologies["pre"].circuit_dc is not None
    assert SESSION.topologies["post"].circuit_s is not None
