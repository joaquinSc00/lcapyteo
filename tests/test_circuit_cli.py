import sympy as sp
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from circuit_cli import (  # noqa: E402
    CircuitGraph,
    ComponentSpec,
    UnitNormalizer,
    nodal_equations,
    parse_falstad_netlist,
    parse_substitutions,
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
