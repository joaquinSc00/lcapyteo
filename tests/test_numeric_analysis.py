import sympy as sp

from circuit_cli import (  # noqa: E402
    CircuitGraph,
    ComponentSpec,
    UnitNormalizer,
    perform_numeric_analysis,
)


def test_mesh_numeric_analysis_returns_expected_matrix_and_currents():
    normalizer = UnitNormalizer()
    components = [
        ComponentSpec("V", "V1", "n1", "0", "10", normalizer=normalizer),
        ComponentSpec("V", "V2", "n2", "0", "5", normalizer=normalizer),
        ComponentSpec("R", "R1", "n1", "n3", "10", normalizer=normalizer),
        ComponentSpec("R", "R2", "n2", "n3", "20", normalizer=normalizer),
        ComponentSpec("R", "R3", "n3", "0", "5", normalizer=normalizer),
    ]

    graph = CircuitGraph(components)
    analysis = perform_numeric_analysis(graph, "jw", "mesh")

    assert sp.Matrix([[15, 5], [5, 25]]) == analysis["matrix"]
    assert sp.Matrix([[10], [5]]) == analysis["vector"]

    solutions = analysis["solutions"]
    assert solutions["I_M1"] == sp.Rational(9, 14)
    assert solutions["I_M2"] == sp.Rational(1, 14)

    kvl = analysis.get("kvl_residuals", [])
    assert kvl == ["0 ∠ 0.00°", "0 ∠ 0.00°"]
