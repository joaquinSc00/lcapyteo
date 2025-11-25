import argparse
import csv
import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from lcapy import Circuit
from pint import UnitRegistry


DEFAULT_UNITS: Dict[str, str] = {
    "R": "ohm",
    "L": "henry",
    "C": "farad",
    "V": "volt",
    "I": "ampere",
    "G": "siemens",
}

VALID_COMPONENT_TYPES: Set[str] = {
    "R",
    "L",
    "C",
    "V",
    "I",
    "E",
    "F",
    "G",
    "H",
    "K",
    "D",
}


class UnitNormalizer:
    def __init__(
        self,
        unit_registry: Optional[UnitRegistry] = None,
        default_units: Optional[Dict[str, str]] = None,
    ) -> None:
        self.unit_registry = unit_registry or UnitRegistry(auto_reduce_dimensions=True)
        self.default_units = default_units or DEFAULT_UNITS

    def normalize_value(self, value: str, component_type: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El valor numérico no puede estar vacío.")

        try:
            quantity = self.unit_registry(value)
        except Exception as exc:  # noqa: BLE001 - Queremos mostrar el fallo de parseo
            raise ValueError(f"Valor '{value}' no reconocido: {exc}") from exc

        default_unit = self.default_units.get(component_type)
        if quantity.dimensionless and default_unit:
            quantity = quantity * self.unit_registry(default_unit)

        quantity = quantity.to_compact()
        magnitude = f"{quantity.magnitude:g}"
        if quantity.units == self.unit_registry.dimensionless:
            return magnitude

        unit_str = f"{quantity.units:~}"
        return f"{magnitude}{unit_str}"


@dataclass
class ComponentSpec:
    type: str
    name: str
    node_a: str
    node_b: str
    value: str
    orientation: Optional[str] = None
    normalizer: UnitNormalizer = field(default_factory=UnitNormalizer, repr=False)

    def __post_init__(self) -> None:
        self.type = self.type.strip().upper()
        self.name = self.name.strip()
        self.node_a = self.node_a.strip()
        self.node_b = self.node_b.strip()
        self.value = self.value.strip()
        if self.type not in VALID_COMPONENT_TYPES:
            raise ValueError(
                f"Tipo '{self.type}' inválido. Usa uno de: {sorted(VALID_COMPONENT_TYPES)}"
            )
        if not self.name:
            raise ValueError("El nombre del componente no puede estar vacío.")
        if not self.name.upper().startswith(self.type):
            raise ValueError(
                f"El nombre '{self.name}' debe comenzar con la letra de tipo '{self.type}'."
            )
        if not self.node_a or not self.node_b:
            raise ValueError("Los nodos A y B deben definirse.")
        if self.node_a.lower() == self.node_b.lower():
            raise ValueError("Los nodos A y B no pueden ser el mismo nodo.")

    @property
    def normalized_value(self) -> str:
        return self.normalizer.normalize_value(self.value, self.type)

    def netlist_line(self) -> str:
        orientation_suffix = f" {self.orientation}" if self.orientation else ""
        return f"{self.name} {self.node_a} {self.node_b} {self.normalized_value}{orientation_suffix}"

    def as_dict(self) -> Dict[str, str]:
        data = {
            "type": self.type,
            "name": self.name,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "value": self.value,
            "value_normalized": self.normalized_value,
        }
        if self.orientation:
            data["orientation"] = self.orientation
        return data


class CircuitGraph:
    def __init__(self, components: Sequence[ComponentSpec]):
        self.components: List[ComponentSpec] = list(components)
        self.nodes: Set[str] = set()
        self.edges: Dict[frozenset[str], List[ComponentSpec]] = defaultdict(list)
        self.reference_nodes: Set[str] = set()
        self.implicit_nodes: Set[str] = set()
        self.duplicate_references: Set[str] = set()
        self.name_collisions: Set[str] = set()
        self._build()
        self._validate()

    def _build(self) -> None:
        seen_names: Set[str] = set()
        reference_aliases: Set[str] = set()
        for component in self.components:
            if component.name in seen_names:
                self.name_collisions.add(component.name)
            seen_names.add(component.name)

            self.nodes.update({component.node_a, component.node_b})
            self.edges[frozenset({component.node_a, component.node_b})].append(component)

            for node in (component.node_a, component.node_b):
                normalized = node.lower()
                if normalized in {"0", "gnd"}:
                    if reference_aliases and normalized not in reference_aliases:
                        self.duplicate_references.update({node, *reference_aliases})
                    reference_aliases.add(normalized)
                    self.reference_nodes.add(node)

        if not self.reference_nodes:
            self.implicit_nodes.add("0")
            self.nodes.add("0")

    def _validate(self) -> None:
        if self.name_collisions:
            collisions = ", ".join(sorted(self.name_collisions))
            raise ValueError(f"Nombres de componente duplicados: {collisions}")
        if self.duplicate_references:
            duplicates = ", ".join(sorted(self.duplicate_references))
            raise ValueError(
                "Se detectaron múltiples alias de referencia (por ejemplo '0' y 'gnd'): "
                f"{duplicates}"
            )

    def to_lcapy_circuit(self) -> Circuit:
        circuit = Circuit()
        for component in self.components:
            circuit.add(component.netlist_line())
        return circuit


class SwitchedCircuit:
    def __init__(self, circuit_pre: Circuit, circuit_post: Circuit):
        self.circuit_pre = circuit_pre
        self.circuit_post = circuit_post

    def for_time(self, time: float) -> Circuit:
        return self.circuit_pre if time < 0 else self.circuit_post


def parse_component_line(line: str, normalizer: UnitNormalizer) -> ComponentSpec:
    parts = [piece.strip() for piece in line.split(",") if piece.strip()]
    if len(parts) not in (5, 6):
        raise ValueError(
            "Cada componente debe tener 5 o 6 campos: "
            "Tipo,Nombre,NodoA,NodoB,Valor[,Orientación]."
        )

    orientation = parts[5] if len(parts) == 6 else None
    return ComponentSpec(
        type=parts[0],
        name=parts[1],
        node_a=parts[2],
        node_b=parts[3],
        value=parts[4],
        orientation=orientation,
        normalizer=normalizer,
    )


def load_components_from_csv(path: str, normalizer: UnitNormalizer) -> list:
    components = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            line = ",".join(row)
            components.append(parse_component_line(line, normalizer))
    return components


def prompt_for_components(label: str, normalizer: UnitNormalizer) -> list:
    print(
        f"Introduce componentes para {label}. "
        "Sigue el formato Tipo,Nombre,NodoA,NodoB,Valor[,Orientación]."
    )
    print("Pulsa Enter en una línea vacía para terminar.\n")

    components = []
    while True:
        line = input(f"{label} > ").strip()
        if not line:
            break
        try:
            components.append(parse_component_line(line, normalizer))
        except ValueError as exc:
            print(f"Entrada no válida: {exc}")
    return components


def build_parser() -> argparse.ArgumentParser:
    description = (
        "Constructor sencillo de topologías usando líneas de texto o archivos CSV.\n"
        "El nodo de referencia debe llamarse '0' o 'gnd'."
    )

    examples = textwrap.dedent(
        """
        Ejemplos:
          python circuit_cli.py --mode single
          python circuit_cli.py --mode double --csv pre.csv post.csv
          python circuit_cli.py --mode single --csv topologia.csv

        Formato por línea:
          Tipo,Nombre,NodoA,NodoB,Valor[,Orientación]
          R,R1,n1,n2,5k
          C,C1,n2,0,10u
          V,V1,n0,n1,10,DC

        Reglas de nodos:
          - Usa '0' o 'gnd' como referencia.
          - No mezcles mayúsculas/minúsculas en el mismo nodo.
        """
    )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "double"],
        default="single",
        help="Topología única (single) o doble (double, t<0 y t>0).",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        metavar="FILE",
        help=(
            "Ruta(s) de archivo CSV con los componentes. "
            "En modo single se admite un archivo; en double, dos: primero t<0, segundo t>0."
        ),
    )
    return parser


def summarize_graph(graph: CircuitGraph, label: str) -> Dict[str, Iterable[str]]:
    return {
        f"implicit_nodes_{label}": sorted(graph.implicit_nodes),
        f"duplicate_references_{label}": sorted(graph.duplicate_references),
    }


def run(args: argparse.Namespace) -> dict:
    mode = args.mode
    normalizer = UnitNormalizer()

    if args.csv:
        if mode == "single" and len(args.csv) != 1:
            raise SystemExit("Modo single requiere exactamente un archivo CSV.")
        if mode == "double" and len(args.csv) != 2:
            raise SystemExit("Modo double requiere dos archivos CSV: t<0 y t>0.")

    if mode == "single":
        if args.csv:
            components = load_components_from_csv(args.csv[0], normalizer)
        else:
            components = prompt_for_components("todo t<0/t>0", normalizer)
        graph = CircuitGraph(components)
        circuit = graph.to_lcapy_circuit()
        result = {
            "mode": mode,
            "components": [component.as_dict() for component in components],
            "netlist": str(circuit),
        }
        result.update(summarize_graph(graph, "pre"))
        return result

    # Modo double
    if args.csv:
        components_pre = load_components_from_csv(args.csv[0], normalizer)
        components_post = load_components_from_csv(args.csv[1], normalizer)
    else:
        components_pre = prompt_for_components("topología t<0", normalizer)
        components_post = prompt_for_components("topología t>0", normalizer)

    graph_pre = CircuitGraph(components_pre)
    graph_post = CircuitGraph(components_post)
    circuit_pre = graph_pre.to_lcapy_circuit()
    circuit_post = graph_post.to_lcapy_circuit()
    switched = SwitchedCircuit(circuit_pre, circuit_post)

    result = {
        "mode": mode,
        "components_pre": [component.as_dict() for component in components_pre],
        "components_post": [component.as_dict() for component in components_post],
        "netlist_pre": str(circuit_pre),
        "netlist_post": str(circuit_post),
        "switched_mode": "pre" if switched.for_time(-1) is circuit_pre else "post",
    }
    result.update(summarize_graph(graph_pre, "pre"))
    result.update(summarize_graph(graph_post, "post"))
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    payload = run(args)
    print("\nResumen (JSON):")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
