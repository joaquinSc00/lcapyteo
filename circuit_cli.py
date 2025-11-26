import argparse
import csv
import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
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

    @staticmethod
    def _split_initial_condition(value: str) -> Tuple[str, str]:
        value = value.strip()
        lower = value.lower()
        ic_index = lower.find("ic=")
        if ic_index == -1:
            return value, ""

        base_value = value[:ic_index].strip()
        ic_suffix = value[ic_index:].strip()
        return base_value, ic_suffix

    def normalize_value(self, value: str, component_type: str) -> str:
        value, ic_suffix = self._split_initial_condition(value)
        if not value:
            raise ValueError("El valor numérico no puede estar vacío.")

        try:
            quantity = self.unit_registry.Quantity(value)
        except Exception as exc:  # noqa: BLE001 - Queremos mostrar el fallo de parseo
            raise ValueError(f"Valor '{value}' no reconocido: {exc}") from exc

        default_unit = self.default_units.get(component_type)
        if quantity.dimensionless and default_unit:
            quantity = quantity * self.unit_registry(default_unit)

        quantity = quantity.to_compact()
        magnitude = f"{quantity.magnitude:g}"
        if quantity.units == self.unit_registry.dimensionless:
            normalized = magnitude
        else:
            unit_str = f"{quantity.units:~}"
            normalized = f"{magnitude}{unit_str}"

        return f"{normalized} {ic_suffix}".strip() if ic_suffix else normalized

    def magnitude_in_base_units(self, value: str, component_type: str) -> float:
        value, _ = self._split_initial_condition(value)
        if not value:
            raise ValueError("El valor numérico no puede estar vacío.")

        default_unit = self.default_units.get(component_type)
        quantity = self.unit_registry.Quantity(value)
        if quantity.dimensionless:
            if default_unit:
                quantity = quantity * self.unit_registry(default_unit)
        elif default_unit:
            try:
                quantity = quantity.to(default_unit)
            except Exception:
                quantity = self.unit_registry.Quantity(f"{value}{default_unit}")
        if default_unit:
            quantity = quantity.to(default_unit)
        return float(quantity.magnitude)


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
        base_value, ic_suffix = self.normalizer._split_initial_condition(self.value)
        try:
            magnitude = self.normalizer.magnitude_in_base_units(
                base_value, self.type
            )
            value_token = f"{magnitude:g}"
            if ic_suffix:
                value_token = f"{value_token} {ic_suffix}"
        except Exception:
            value_token = self.normalized_value
        return f"{self.name} {self.node_a} {self.node_b} {value_token}"

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
        self.warnings: List[str] = []
        self._build()
        self._validate()
        self._detect_duplicate_nodes()
        self._check_connectivity()
        self._detect_wire_shorts()

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
            self.warnings.append(
                "Nombres de componente duplicados: "
                f"{collisions}. Usa identificadores únicos."
            )
        if self.duplicate_references:
            duplicates = ", ".join(sorted(self.duplicate_references))
            self.warnings.append(
                "Se detectaron múltiples alias de referencia (por ejemplo '0' y 'gnd'): "
                f"{duplicates}. Unifica todos los nodos de referencia con un único nombre."
            )

    def _detect_duplicate_nodes(self) -> None:
        normalized_to_originals: Dict[str, Set[str]] = defaultdict(set)
        for node in self.nodes:
            normalized_to_originals[node.lower()].add(node)
        duplicates = {
            normalized: originals
            for normalized, originals in normalized_to_originals.items()
            if len(originals) > 1
        }
        if duplicates:
            details = "; ".join(
                f"{', '.join(sorted(values))}"
                for values in duplicates.values()
            )
            self.warnings.append(
                "Nodos con el mismo nombre pero distinta capitalización: "
                f"{details}. Usa siempre la misma escritura para evitar cortos no deseados."
            )

    def _build_adjacency(self) -> Dict[str, Set[str]]:
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for component in self.components:
            adjacency[component.node_a].add(component.node_b)
            adjacency[component.node_b].add(component.node_a)
        return adjacency

    def _check_connectivity(self) -> None:
        if not self.nodes:
            return

        adjacency = self._build_adjacency()
        reference_candidates = {node for node in self.nodes if node.lower() in {"0", "gnd"}}
        if self.implicit_nodes and "0" in self.nodes:
            reference_candidates.add("0")

        if not reference_candidates:
            self.warnings.append(
                "No se encontró nodo de referencia explícito. Se usará '0' de forma implícita."
            )
            reference_candidates.add("0")

        visited: Set[str] = set()
        stack = list(reference_candidates)
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, set()) - visited)

        floating_nodes = self.nodes - visited
        if floating_nodes:
            floating_components = [
                comp
                for comp in self.components
                if comp.node_a in floating_nodes and comp.node_b in floating_nodes
            ]
            nodes_str = ", ".join(sorted(floating_nodes))
            comps_str = ", ".join(sorted({comp.name for comp in floating_components}))
            suggestion = (
                "Conecta estos nodos al nodo de referencia o revisa si faltan etiquetas "
                "consistentes."
            )
            if comps_str:
                suggestion += f" Componentes afectados: {comps_str}."
            self.warnings.append(
                f"Nodos sin camino al nodo de referencia: {nodes_str}. {suggestion}"
            )

    def _is_wire_like(self, component: ComponentSpec) -> bool:
        if component.type == "W":
            return True
        if component.type != "R":
            return False
        try:
            quantity = component.normalizer.unit_registry(component.value)
            if quantity.dimensionless:
                default_unit = component.normalizer.default_units.get(component.type)
                if default_unit:
                    quantity = quantity * component.normalizer.unit_registry(default_unit)
            return quantity.magnitude == 0
        except Exception:
            return False

    def _detect_wire_shorts(self) -> None:
        wire_components = [comp for comp in self.components if self._is_wire_like(comp)]
        if not wire_components:
            return

        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for comp in wire_components:
            adjacency[comp.node_a].add(comp.node_b)
            adjacency[comp.node_b].add(comp.node_a)

        visited: Set[str] = set()
        shorts: List[Set[str]] = []
        for node in adjacency:
            if node in visited:
                continue
            stack = [node]
            component_nodes: Set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_nodes.add(current)
                stack.extend(adjacency[current] - visited)
            if len(component_nodes) > 1:
                shorts.append(component_nodes)

        if shorts:
            shorts_str = "; ".join(
                ", ".join(sorted(short_nodes)) for short_nodes in shorts
            )
            self.warnings.append(
                "Posible corto accidental: los nodos "
                f"{shorts_str} están unidos solo por alambres/0Ω. "
                "Verifica que deban ser el mismo nodo o elimina el puente."
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


def _domain_operator(domain: str) -> sp.Expr:
    if domain == "laplace":
        return sp.symbols("s")
    if domain == "jw":
        return sp.I * sp.symbols("w")
    return sp.symbols("d_dt")


def _numeric_value(component: ComponentSpec) -> sp.Expr:
    try:
        magnitude = component.normalizer.magnitude_in_base_units(
            component.value, component.type
        )
        return sp.nsimplify(magnitude)
    except Exception:
        return sp.symbols(f"{component.name}_val")


def impedance_for_component(component: ComponentSpec, domain: str) -> Optional[sp.Expr]:
    operator = _domain_operator(domain)
    value = _numeric_value(component)
    if component.type == "R":
        return value
    if component.type == "L":
        return operator * value
    if component.type == "C":
        return 1 / (operator * value)
    if component.type in {"E", "F", "G", "H", "K", "D"}:
        return None
    return None


def source_value(component: ComponentSpec) -> sp.Expr:
    try:
        magnitude = component.normalizer.magnitude_in_base_units(
            component.value, component.type
        )
        return sp.nsimplify(magnitude)
    except Exception:
        return sp.symbols(component.value)


def _reference_node(graph: CircuitGraph) -> str:
    if "0" in graph.nodes:
        return "0"
    if graph.reference_nodes:
        return sorted(graph.reference_nodes)[0]
    return sorted(graph.nodes)[0]


def _node_voltage(node: str) -> sp.Symbol:
    return sp.symbols(f"V_{node}")


def nodal_equations(graph: CircuitGraph, domain: str) -> Tuple[list, List[str]]:
    reference = _reference_node(graph)
    node_currents: Dict[str, sp.Expr] = defaultdict(int)
    voltage_equations: List[sp.Eq] = []
    unsupported: List[str] = []

    for component in graph.components:
        z_value = impedance_for_component(component, domain)
        node_a = component.node_a
        node_b = component.node_b
        va = _node_voltage(node_a) if node_a != reference else sp.Integer(0)
        vb = _node_voltage(node_b) if node_b != reference else sp.Integer(0)

        if z_value is not None:
            current_ab = (va - vb) / z_value
            node_currents[node_a] += current_ab
            node_currents[node_b] -= current_ab
            continue

        if component.type == "I":
            current = source_value(component)
            node_currents[node_a] += current
            node_currents[node_b] -= current
            continue

        if component.type == "V":
            current_symbol = sp.symbols(f"I_{component.name}")
            node_currents[node_a] += current_symbol
            node_currents[node_b] -= current_symbol
            voltage_equations.append(sp.Eq(va - vb, source_value(component)))
            continue

        unsupported.append(component.name)

    kcl_equations = [
        sp.Eq(node_currents[node], 0)
        for node in sorted(graph.nodes)
        if node != reference
    ]

    warnings = []
    if unsupported:
        warnings.append(
            "Componentes no considerados en KCL (tipo no soportado para nodal): "
            f"{', '.join(sorted(set(unsupported)))}"
        )

    return kcl_equations + voltage_equations, warnings


def _build_edges(graph: CircuitGraph) -> List[Tuple[int, ComponentSpec]]:
    return list(enumerate(graph.components))


def _tree_adjacency(tree_edges: Set[int], edges: List[Tuple[int, ComponentSpec]]):
    adjacency: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for edge_id, component in edges:
        if edge_id not in tree_edges:
            continue
        adjacency[component.node_a].append((component.node_b, edge_id))
        adjacency[component.node_b].append((component.node_a, edge_id))
    return adjacency


def _find_tree_path(adjacency: Dict[str, List[Tuple[str, int]]], start: str, goal: str):
    stack = [(start, [])]
    visited: Set[str] = set()
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor, edge_id in adjacency.get(node, []):
            if neighbor not in visited:
                stack.append((neighbor, path + [(node, neighbor, edge_id)]))
    return []


def _fundamental_cycles(graph: CircuitGraph) -> List[list]:
    edges = _build_edges(graph)
    if not graph.nodes:
        return []

    visited_nodes: Set[str] = set()
    tree_edges: Set[int] = set()
    chords: List[int] = []
    start_node = next(iter(graph.nodes))
    queue = [start_node]
    visited_nodes.add(start_node)

    while queue:
        node = queue.pop(0)
        for edge_id, component in edges:
            if edge_id in tree_edges or edge_id in chords:
                continue
            if node not in {component.node_a, component.node_b}:
                continue
            other = component.node_b if component.node_a == node else component.node_a
            if other not in visited_nodes:
                visited_nodes.add(other)
                tree_edges.add(edge_id)
                queue.append(other)
            else:
                chords.append(edge_id)

    adjacency = _tree_adjacency(tree_edges, edges)
    cycles: List[list] = []
    for chord_id in chords:
        chord_component = edges[chord_id][1]
        start = chord_component.node_a
        end = chord_component.node_b
        path = _find_tree_path(adjacency, start, end)
        cycle_edges = []
        for frm, to, edge_id in path:
            comp = edges[edge_id][1]
            sign = 1 if (frm, to) == (comp.node_a, comp.node_b) else -1
            cycle_edges.append((edge_id, sign))
        chord_sign = 1 if (end, start) == (chord_component.node_a, chord_component.node_b) else -1
        cycle_edges.append((chord_id, chord_sign))
        cycles.append(cycle_edges)
    return cycles


def mesh_equations(graph: CircuitGraph, domain: str) -> Tuple[list, List[str]]:
    cycles = _fundamental_cycles(graph)
    edges = _build_edges(graph)
    mesh_currents = [sp.symbols(f"I_M{idx+1}") for idx in range(len(cycles))]
    edge_to_cycles: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    warnings: List[str] = []

    for cycle_idx, cycle_edges in enumerate(cycles):
        for edge_id, sign in cycle_edges:
            edge_to_cycles[edge_id].append((cycle_idx, sign))

    equations: List[sp.Eq] = []
    for idx, cycle_edges in enumerate(cycles):
        expression = 0
        for edge_id, sign in cycle_edges:
            component = edges[edge_id][1]
            if component.type == "I":
                warnings.append(
                    f"Fuente de corriente {component.name} en malla {idx+1} no se incluye en KVL."
                )
                continue

            if component.type == "V":
                expression += sign * source_value(component)
                continue

            impedance = impedance_for_component(component, domain)
            if impedance is None:
                warnings.append(
                    f"Componente {component.name} omitido en KVL (tipo no soportado)."
                )
                continue

            shared_current = 0
            for other_idx, other_sign in edge_to_cycles[edge_id]:
                shared_current += other_sign * mesh_currents[other_idx]
            expression += sign * impedance * shared_current
        equations.append(sp.Eq(expression, 0))

    return equations, sorted(set(warnings))


def parse_falstad_netlist(
    text: str, normalizer: UnitNormalizer
) -> List[ComponentSpec]:
    mapping = {
        "r": "R",
        "c": "C",
        "l": "L",
        "v": "V",
        "i": "I",
        "d": "D",
        "e": "E",
        "f": "F",
        "g": None,
        "h": "H",
        "k": "K",
        "w": None,
    }

    def coord_key(x: str, y: str) -> str:
        return f"{int(float(x))},{int(float(y))}"

    parents: Dict[str, str] = {}
    ground_coords: Set[str] = set()
    component_entries: List[Tuple[str, Tuple[str, str], str]] = []

    def find(item: str) -> str:
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("El netlist de Falstad no puede estar vacío.")

    for idx, line in enumerate(lines, start=1):
        if line.startswith("$"):
            continue
        tokens = line.split()
        if len(tokens) < 5:
            raise ValueError(
                f"Línea {idx} inválida en netlist Falstad: '{line}'."
            )

        prefix = tokens[0].lower()
        if prefix not in mapping:
            raise ValueError(f"Prefijo desconocido en Falstad: '{prefix}'.")

        comp_type = mapping[prefix]

        node_a = coord_key(tokens[1], tokens[2])
        node_b = coord_key(tokens[3], tokens[4])
        find(node_a)
        find(node_b)

        if prefix == "w":
            union(node_a, node_b)
            continue

        if prefix == "g":
            ground_coords.update({node_a, node_b})
            union(node_a, node_b)
            continue

        if len(tokens) < 7:
            raise ValueError(
                f"Valor numérico faltante en línea {idx} del netlist Falstad."
            )

        value = tokens[6]
        initial_condition: Optional[str] = None
        if comp_type == "C" and len(tokens) >= 9:
            initial_condition = tokens[-2]
        elif comp_type == "L" and len(tokens) >= 8:
            initial_condition = tokens[-2]

        if initial_condition is not None:
            value = f"{value} ic={initial_condition}"
        component_entries.append((prefix, (node_a, node_b), value))

    if ground_coords:
        ground_roots = {find(coord) for coord in ground_coords}
        primary_ground = next(iter(ground_roots))
        for root in ground_roots:
            union(primary_ground, root)

    node_names: Dict[str, str] = {}
    root_to_name: Dict[str, str] = {}
    ground_root = find(next(iter(ground_coords))) if ground_coords else None
    counter = 1
    for coord in sorted(parents):
        root = find(coord)
        if root not in root_to_name:
            if ground_root and root == ground_root:
                root_to_name[root] = "0"
            else:
                root_to_name[root] = f"n{counter}"
                counter += 1
        node_names[coord] = root_to_name[root]

    type_counters: Dict[str, int] = defaultdict(int)
    components: List[ComponentSpec] = []
    for prefix, (node_a, node_b), value in component_entries:
        comp_type = mapping[prefix]
        if comp_type is None:
            continue
        type_counters[comp_type] += 1
        name = f"{comp_type}{type_counters[comp_type]}"
        resolved_a = node_names[node_a]
        resolved_b = node_names[node_b]
        if resolved_a == resolved_b:
            raise ValueError(
                f"El componente {name} conecta el mismo nodo ({resolved_a}). Revisa las coordenadas."
            )
        components.append(
            ComponentSpec(
                type=comp_type,
                name=name,
                node_a=resolved_a,
                node_b=resolved_b,
                value=value,
                normalizer=normalizer,
            )
        )

    return components


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


def _prompt_yes_no(message: str, default: bool = False) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        choice = input(f"{message} {suffix}: ").strip().lower()
        if not choice:
            return default
        if choice in {"s", "si", "sí"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Responde con 's' o 'n'.")


def prompt_topology_mode(current_mode: str) -> str:
    wants_double = _prompt_yes_no(
        "¿Trabajar con doble topología (t<0 y t>0)?",
        default=current_mode == "double",
    )
    return "double" if wants_double else "single"


def prompt_input_style() -> str:
    print(
        "\n¿Cómo introducir el circuito?\n"
        "  1) Línea por línea (Tipo,Nombre,NodoA,NodoB,Valor[,Orientación])\n"
        "  2) Pegar netlist completo de Falstad/CircuitJS\n"
        "Pulsa Enter para usar la opción por defecto (1)."
    )
    while True:
        choice = input("Selecciona 1 o 2 [1]: ").strip()
        if choice in {"", "1"}:
            return "line"
        if choice == "2":
            return "falstad"
        print("Opción no válida. Elige 1 o 2.")


def prompt_falstad_text() -> str:
    print(
        "Pega aquí el netlist exportado desde Falstad/CircuitJS.\n"
        "Cuando termines, deja una línea vacía para finalizar."
    )
    lines: List[str] = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def _format_equations(equations: Sequence[sp.Eq]) -> List[str]:
    formatted = []
    for eq in equations:
        formatted.append(str(sp.simplify(eq)))
    return formatted


def _matrix_from_equations(equations: Sequence[sp.Eq]) -> Tuple[sp.Matrix, sp.Matrix, list]:
    parameter_names = {"s", "w", "d_dt"}
    variable_prefixes = ("V_", "I_", "I_M")
    symbols = sorted(
        {
            sym
            for eq in equations
            for sym in eq.free_symbols
            if sym.name not in parameter_names
            and sym.name.startswith(variable_prefixes)
        },
        key=lambda s: s.name,
    )
    if not symbols:
        return sp.Matrix(), sp.Matrix(), []
    matrix, vector = sp.linear_eq_to_matrix(equations, symbols)
    return matrix, vector, symbols


def parse_substitutions(pairs: Optional[Sequence[str]]) -> Dict[sp.Symbol, sp.Expr]:
    substitutions: Dict[sp.Symbol, sp.Expr] = {}
    if not pairs:
        return substitutions

    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"La sustitución '{item}' no es válida. Usa el formato nombre=valor."
            )
        name, raw_value = item.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if not name:
            raise ValueError("El nombre de la sustitución no puede estar vacío.")
        substitutions[sp.symbols(name)] = sp.sympify(raw_value)
    return substitutions


def solve_equations(
    equations: Sequence[sp.Eq], substitutions: Optional[Dict[sp.Symbol, sp.Expr]] = None
) -> Dict[str, sp.Expr]:
    substituted = [eq.subs(substitutions or {}) for eq in equations]
    matrix, vector, symbols = _matrix_from_equations(substituted)
    if not symbols or matrix.is_zero_matrix:
        return {}

    solution_set = sp.linsolve((matrix, vector), *symbols)
    if not solution_set:
        return {}

    solution = next(iter(solution_set))
    return {symbol.name: value for symbol, value in zip(symbols, solution)}


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _labelled_path(base_path: str, label: str) -> Path:
    path = Path(base_path)
    if not label:
        return path
    return path.with_name(f"{path.stem}_{label}{path.suffix}")


def export_to_csv(path: str, summary: dict, solution: Dict[str, sp.Expr]) -> None:
    csv_path = Path(path)
    _ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Tipo", "Contenido"])
        for equation in summary.get("equations", []):
            writer.writerow(["ecuacion", equation])
        for warning in summary.get("warnings", []):
            writer.writerow(["advertencia", warning])
        for var, expr in solution.items():
            writer.writerow(["solucion", f"{var}={sp.simplify(expr)}"])


def export_to_latex(path: str, summary: dict, solution: Dict[str, sp.Expr]) -> None:
    latex_path = Path(path)
    _ensure_parent_dir(latex_path)
    lines = ["\\section*{Ecuaciones y soluciones}"]
    if summary.get("equations"):
        lines.append("\\subsection*{Ecuaciones}")
        for equation in summary["equations"]:
            lines.append(f"$${sp.latex(sp.sympify(equation))}$$")
    if summary.get("warnings"):
        lines.append("\\subsection*{Advertencias}")
        for warning in summary["warnings"]:
            lines.append(f"\\textbf{{Aviso}}: {warning}\\\\")
    if solution:
        lines.append("\\subsection*{Soluciones}")
        for var, expr in solution.items():
            lines.append(f"$${var} = {sp.latex(sp.simplify(expr))}$$")
    latex_path.write_text("\n\n".join(lines), encoding="utf-8")


def _plot_response(
    expression: sp.Expr,
    symbol: sp.Symbol,
    start: float,
    stop: float,
    points: int,
    output_path: Path,
    label: str,
) -> None:
    output_path = Path(output_path)
    _ensure_parent_dir(output_path)
    samples = np.linspace(start, stop, points)
    func = sp.lambdify(symbol, expression, modules=["numpy"])
    values = func(samples)
    plt.figure()
    plt.plot(samples, np.real(values), label=f"Re{{{label}}}")
    if np.any(np.imag(values)):
        plt.plot(samples, np.imag(values), "--", label=f"Im{{{label}}}")
    plt.xlabel(symbol.name)
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def symbolic_analysis(graph: CircuitGraph, domain: str, method: str, show_matrices: bool):
    if method == "mesh":
        equations, warnings = mesh_equations(graph, domain)
    else:
        equations, warnings = nodal_equations(graph, domain)

    summary = {
        "equations": _format_equations(equations),
        "raw_equations": equations,
    }
    if warnings:
        summary["warnings"] = warnings

    if show_matrices:
        matrix, vector, symbols = _matrix_from_equations(equations)
        summary["matrix_G"] = [[str(item) for item in row] for row in matrix.tolist()]
        summary["matrix_B"] = [[str(item) for item in row] for row in vector.tolist()]
        summary["variables"] = [str(sym) for sym in symbols]

    return summary


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
    parser.add_argument(
        "--falstad",
        help=(
            "Pega un netlist exportado desde Falstad/CircuitJS. Solo se admite en modo single."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["nodal", "mesh"],
        default="nodal",
        help="Generar ecuaciones por método nodal o por mallas (KVL).",
    )
    parser.add_argument(
        "--domain",
        choices=["laplace", "jw", "time"],
        default="laplace",
        help="Dominio simbólico para impedancias: s, jω o derivadas en el tiempo.",
    )
    parser.add_argument(
        "--show-matrices",
        action="store_true",
        help="Mostrar matrices G y B (linealizadas) además de las ecuaciones simbólicas.",
    )
    parser.add_argument(
        "--show-lcapy",
        action="store_true",
        help="Imprimir el netlist generado por lcapy para la topología procesada.",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help=(
            "Resolver el sistema lineal (vías sympy) para obtener tensiones y corrientes."
        ),
    )
    parser.add_argument(
        "--substitute",
        nargs="+",
        metavar="NOMBRE=VALOR",
        help=(
            "Valores numéricos a usar en la resolución (fuentes o condiciones iniciales)."
        ),
    )
    parser.add_argument(
        "--export-csv",
        metavar="FILE",
        help="Guardar ecuaciones/soluciones en CSV (se añade sufijo _pre/_post en double).",
    )
    parser.add_argument(
        "--export-latex",
        metavar="FILE",
        help="Guardar ecuaciones/soluciones en LaTeX (con sufijo _pre/_post en double).",
    )
    parser.add_argument(
        "--plot-time",
        nargs=5,
        metavar=("VAR", "T0", "T1", "POINTS", "FILE"),
        help="Graficar una solución en el tiempo (variable, inicio, fin, muestras, archivo).",
    )
    parser.add_argument(
        "--plot-freq",
        nargs=5,
        metavar=("VAR", "W0", "W1", "POINTS", "FILE"),
        help="Graficar una solución en frecuencia (variable, w0, w1, muestras, archivo).",
    )
    return parser


def summarize_graph(graph: CircuitGraph, label: str) -> Dict[str, Iterable[str]]:
    return {
        f"implicit_nodes_{label}": sorted(graph.implicit_nodes),
        f"duplicate_references_{label}": sorted(graph.duplicate_references),
        f"warnings_{label}": graph.warnings,
    }


def print_warnings(graph: CircuitGraph, label: str) -> None:
    if not graph.warnings:
        return
    print(f"\n⚠️  Advertencias para {label}:")
    for warning in graph.warnings:
        print(f" - {warning}")


def print_symbolic(label: str, summary: dict, method: str, show_matrices: bool) -> None:
    print(f"\nEcuaciones simbólicas ({method}) para {label}:")
    for eq in summary.get("equations", []):
        print(f" - {eq}")

    for warning in summary.get("warnings", []):
        print(f" ⚠️  {warning}")

    if summary.get("solutions"):
        print("\nSoluciones:")
        for var, expr in summary["solutions"].items():
            print(f" {var} = {expr}")

    if summary.get("plots"):
        print("\nGráficas generadas:")
        for plot_path in summary["plots"]:
            print(f" - {plot_path}")

    if show_matrices and summary.get("variables"):
        print("\nMatriz G (coeficientes):")
        print(sp.Matrix(summary["matrix_G"]))
        print("Vector B:")
        print(sp.Matrix(summary["matrix_B"]))
        print(f"Variables: {', '.join(summary['variables'])}")


def _plot_instructions(args: argparse.Namespace) -> List[dict]:
    instructions: List[dict] = []
    if args.plot_time:
        var, start, stop, points, path = args.plot_time
        instructions.append(
            {
                "var": var,
                "start": float(start),
                "stop": float(stop),
                "points": int(points),
                "path": Path(path),
                "symbol": sp.symbols("t"),
            }
        )
    if args.plot_freq:
        var, start, stop, points, path = args.plot_freq
        instructions.append(
            {
                "var": var,
                "start": float(start),
                "stop": float(stop),
                "points": int(points),
                "path": Path(path),
                "symbol": sp.symbols("w"),
            }
        )
    return instructions


def enrich_analysis(
    summary: dict, label: str, args: argparse.Namespace, substitutions: dict
) -> Tuple[dict, Dict[str, sp.Expr]]:
    solutions: Dict[str, sp.Expr] = {}
    plots: List[str] = []
    extra_warnings: List[str] = []

    if args.solve:
        solutions = solve_equations(summary.get("raw_equations", []), substitutions)
        if solutions:
            summary["solutions"] = {k: str(sp.simplify(v)) for k, v in solutions.items()}
        else:
            extra_warnings.append(
                "No se pudo resolver el sistema con los parámetros proporcionados."
            )

    for instruction in _plot_instructions(args):
        var = instruction["var"]
        if var not in solutions:
            extra_warnings.append(
                f"No se puede graficar {var} en {label} porque no hay solución calculada."
            )
            continue
        try:
            _plot_response(
                solutions[var],
                instruction["symbol"],
                instruction["start"],
                instruction["stop"],
                instruction["points"],
                instruction["path"],
                var,
            )
            plots.append(str(instruction["path"]))
        except Exception as exc:  # noqa: BLE001 - queremos mostrar el error al usuario
            extra_warnings.append(f"Error al graficar {var} en {label}: {exc}")

    if plots:
        summary["plots"] = plots
    if extra_warnings:
        summary.setdefault("warnings", []).extend(extra_warnings)
    return summary, solutions


def run(args: argparse.Namespace) -> dict:
    mode = args.mode
    normalizer = UnitNormalizer()
    domain = args.domain
    method = args.method
    try:
        substitutions = parse_substitutions(args.substitute)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    interactive_entry: Optional[str] = None
    interactive_falstad_text: Optional[str] = None
    interactive_prompt = not args.csv and not args.falstad
    if interactive_prompt:
        mode = prompt_topology_mode(mode)
        interactive_entry = prompt_input_style()
        if mode == "double" and interactive_entry == "falstad":
            print("\nLa carga de netlists de Falstad solo está disponible para una topología.")
            interactive_entry = "line"
        if interactive_entry == "falstad":
            interactive_falstad_text = prompt_falstad_text()

    if args.falstad and args.csv:
        raise SystemExit("No combines --falstad con archivos CSV.")

    if args.falstad and args.mode == "double":
        raise SystemExit("--falstad solo está disponible en modo single.")

    if args.csv:
        if mode == "single" and len(args.csv) != 1:
            raise SystemExit("Modo single requiere exactamente un archivo CSV.")
        if mode == "double" and len(args.csv) != 2:
            raise SystemExit("Modo double requiere dos archivos CSV: t<0 y t>0.")

    components_from_falstad: Optional[List[ComponentSpec]] = None
    if args.falstad:
        try:
            components_from_falstad = parse_falstad_netlist(
                args.falstad, normalizer
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif interactive_falstad_text:
        components_from_falstad = parse_falstad_netlist(
            interactive_falstad_text, normalizer
        )

    if mode == "single":
        if components_from_falstad is not None:
            components = components_from_falstad
        elif args.csv:
            components = load_components_from_csv(args.csv[0], normalizer)
        else:
            components = prompt_for_components("todo t<0/t>0", normalizer)
        graph = CircuitGraph(components)
        print_warnings(graph, "topología única")
        circuit = graph.to_lcapy_circuit()
        analysis_summary = symbolic_analysis(graph, domain, method, args.show_matrices)
        analysis_summary, solutions = enrich_analysis(
            analysis_summary, "single", args, substitutions
        )
        if args.export_csv:
            export_to_csv(_labelled_path(args.export_csv, "single"), analysis_summary, solutions)
        if args.export_latex:
            export_to_latex(
                _labelled_path(args.export_latex, "single"), analysis_summary, solutions
            )
        print_symbolic("topología única", analysis_summary, method, args.show_matrices)
        if args.show_lcapy:
            print("\nNetlist lcapy:")
            print(circuit)
        result = {
            "mode": mode,
            "components": [component.as_dict() for component in components],
            "netlist": str(circuit),
            "analysis": analysis_summary,
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
    print_warnings(graph_pre, "topología t<0")
    print_warnings(graph_post, "topología t>0")
    circuit_pre = graph_pre.to_lcapy_circuit()
    circuit_post = graph_post.to_lcapy_circuit()
    switched = SwitchedCircuit(circuit_pre, circuit_post)
    analysis_pre = symbolic_analysis(graph_pre, domain, method, args.show_matrices)
    analysis_pre, solutions_pre = enrich_analysis(
        analysis_pre, "pre", args, substitutions
    )
    analysis_post = symbolic_analysis(graph_post, domain, method, args.show_matrices)
    analysis_post, solutions_post = enrich_analysis(
        analysis_post, "post", args, substitutions
    )

    if args.export_csv:
        export_to_csv(_labelled_path(args.export_csv, "pre"), analysis_pre, solutions_pre)
        export_to_csv(
            _labelled_path(args.export_csv, "post"), analysis_post, solutions_post
        )
    if args.export_latex:
        export_to_latex(
            _labelled_path(args.export_latex, "pre"), analysis_pre, solutions_pre
        )
        export_to_latex(
            _labelled_path(args.export_latex, "post"), analysis_post, solutions_post
        )

    print_symbolic("topología t<0", analysis_pre, method, args.show_matrices)
    print_symbolic("topología t>0", analysis_post, method, args.show_matrices)
    if args.show_lcapy:
        print("\nNetlist lcapy t<0:")
        print(circuit_pre)
        print("\nNetlist lcapy t>0:")
        print(circuit_post)

    result = {
        "mode": mode,
        "components_pre": [component.as_dict() for component in components_pre],
        "components_post": [component.as_dict() for component in components_post],
        "netlist_pre": str(circuit_pre),
        "netlist_post": str(circuit_post),
        "switched_mode": "pre" if switched.for_time(-1) is circuit_pre else "post",
        "analysis_pre": analysis_pre,
        "analysis_post": analysis_post,
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
