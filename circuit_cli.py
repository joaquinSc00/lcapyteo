import argparse
import csv
import json
import textwrap
import warnings
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


@dataclass
class SessionEntry:
    label: str
    domain: str
    circuit: Circuit
    circuit_dc: Optional[Circuit]
    circuit_s: Optional[Circuit]
    circuit_jw: Optional[Circuit]

    def describe(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "domain": self.domain,
            "netlist": str(self.circuit),
            "references": {
                "circuit": True,
                "circuit_dc": self.circuit_dc is not None,
                "circuit_s": self.circuit_s is not None,
                "circuit_jw": self.circuit_jw is not None,
            },
        }


class SessionContainer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active_domain: Optional[str] = None
        self.topologies: Dict[str, SessionEntry] = {}

    def store(self, label: str, circuit: Circuit, domain: str) -> SessionEntry:
        variants = _derive_circuit_variants(circuit)
        entry = SessionEntry(
            label=label,
            domain=domain,
            circuit=circuit,
            circuit_dc=variants.get("dc"),
            circuit_s=variants.get("laplace"),
            circuit_jw=variants.get("ac"),
        )
        self.topologies[label] = entry
        self.active_domain = domain
        return entry

    def describe(self) -> Dict[str, object]:
        return {
            "active_domain": self.active_domain,
            "topologies": {label: entry.describe() for label, entry in self.topologies.items()},
        }


SESSION = SessionContainer()

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


def _derive_circuit_variants(circuit: Circuit) -> Dict[str, Optional[Circuit]]:
    variants: Dict[str, Optional[Circuit]] = {}
    for key, builder in (
        ("dc", circuit.dc),
        ("laplace", circuit.laplace),
        ("ac", circuit.ac),
    ):
        try:
            variants[key] = builder()
        except Exception:
            variants[key] = None
    return variants


def _circuit_for_domain(entry: SessionEntry, domain: str) -> Circuit:
    if domain == "laplace" and entry.circuit_s is not None:
        return entry.circuit_s
    if domain == "jw" and entry.circuit_jw is not None:
        return entry.circuit_jw
    if domain == "time" and entry.circuit_dc is not None:
        return entry.circuit_dc
    return entry.circuit


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
    start_node = sorted(graph.nodes)[0]
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


@dataclass
class MeshSystem:
    equations: List[sp.Eq]
    warnings: List[str]
    mesh_currents: List[sp.Symbol]
    cycles: List[list]
    edges: List[Tuple[int, ComponentSpec]]
    edge_to_cycles: Dict[int, List[Tuple[int, int]]]


def _build_mesh_system(graph: CircuitGraph, domain: str) -> MeshSystem:
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
                expression -= sign * source_value(component)
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

    return MeshSystem(
        equations=equations,
        warnings=sorted(set(warnings)),
        mesh_currents=mesh_currents,
        cycles=cycles,
        edges=edges,
        edge_to_cycles=edge_to_cycles,
    )


def mesh_equations(graph: CircuitGraph, domain: str) -> Tuple[list, List[str]]:
    system = _build_mesh_system(graph, domain)
    return system.equations, system.warnings


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


def _parse_component_line_csv(line: str, normalizer: UnitNormalizer) -> ComponentSpec:
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


def _parse_component_line_spice(line: str, normalizer: UnitNormalizer) -> ComponentSpec:
    tokens = line.split()
    if len(tokens) not in (4, 5):
        raise ValueError(
            "Cada componente debe tener 4 o 5 campos: "
            "Nombre NodoA NodoB Valor [CondiciónInicial]."
        )

    name = tokens[0].strip()
    if not name:
        raise ValueError("El nombre del componente no puede estar vacío.")

    comp_type = name[0].upper()
    node_a, node_b, value = tokens[1], tokens[2], tokens[3]

    if len(tokens) == 5:
        if comp_type in {"L", "C"}:
            ic_value = tokens[4]
            value = f"{value} ic={ic_value}"
        else:
            raise ValueError(
                "Las condiciones iniciales solo se aceptan para inductores y capacitores."
            )

    return ComponentSpec(
        type=comp_type,
        name=name,
        node_a=node_a,
        node_b=node_b,
        value=value,
        normalizer=normalizer,
    )


def parse_component_line(line: str, normalizer: UnitNormalizer) -> ComponentSpec:
    if "," in line:
        return _parse_component_line_csv(line, normalizer)
    return _parse_component_line_spice(line, normalizer)


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


def _align_pretty(lhs: str, rhs: str) -> str:
    lhs_lines = lhs.splitlines() or [""]
    rhs_lines = rhs.splitlines() or [""]
    lhs_width = max(len(line) for line in lhs_lines)
    total_lines = max(len(lhs_lines), len(rhs_lines))
    lhs_lines += [""] * (total_lines - len(lhs_lines))
    rhs_lines += [""] * (total_lines - len(rhs_lines))
    aligned = [f"{lhs_lines[i].ljust(lhs_width)} = {rhs_lines[i]}" for i in range(total_lines)]
    return "\n".join(aligned)


def _format_equations(equations: Sequence[sp.Eq]) -> List[Dict[str, str]]:
    formatted = []
    for eq in equations:
        simplified = sp.simplify(eq)
        plain = f"{simplified.lhs} = {simplified.rhs}"
        pretty = _align_pretty(
            sp.pretty(simplified.lhs, use_unicode=False),
            sp.pretty(simplified.rhs, use_unicode=False),
        )
        latex = sp.latex(simplified)
        formatted.append({"plain": plain, "pretty": pretty, "latex": latex})
    return formatted


def _format_labelled_equations(
    equations: Sequence[sp.Eq], labels: Sequence[str]
) -> List[Dict[str, str]]:
    labelled = []
    formatted = _format_equations(equations)
    for idx, eq in enumerate(formatted):
        entry = dict(eq)
        entry["label"] = labels[idx] if idx < len(labels) else f"Eq{idx+1}"
        labelled.append(entry)
    return labelled


def _equation_repr(equation, style: str = "plain") -> str:
    if isinstance(equation, dict):
        return equation.get(style) or equation.get("plain", "")
    if style == "latex":
        try:
            return sp.latex(sp.sympify(equation))
        except Exception:  # noqa: BLE001 - usamos string como último recurso
            return str(equation)
    return str(equation)


def _matrix_from_equations(
    equations: Sequence[sp.Eq], explicit_symbols: Optional[Sequence[sp.Symbol]] = None
) -> Tuple[sp.Matrix, sp.Matrix, list]:
    if explicit_symbols:
        symbols = list(explicit_symbols)
    else:
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


def _extract_lcapy_equations(exprdict) -> Tuple[List[sp.Eq], List[str]]:
    equations: List[sp.Eq] = []
    labels: List[str] = []
    for idx, (lhs, rhs) in enumerate(exprdict.items(), start=1):
        if hasattr(rhs, "lhs") and hasattr(rhs, "rhs"):
            lhs_expr = rhs.lhs.expr if hasattr(rhs.lhs, "expr") else rhs.lhs
            rhs_expr = rhs.rhs.expr if hasattr(rhs.rhs, "expr") else rhs.rhs
            sympy_eq = sp.Eq(sp.simplify(lhs_expr), sp.simplify(rhs_expr))
        else:
            lhs_expr = lhs.expr if hasattr(lhs, "expr") else lhs
            rhs_expr = rhs.expr if hasattr(rhs, "expr") else rhs
            sympy_eq = sp.Eq(sp.simplify(lhs_expr), sp.simplify(rhs_expr))
        equations.append(sympy_eq)
        labels.append(str(lhs) if lhs is not None else f"Eq{idx}")
    return equations, labels


def _lcapy_matrix_equations(circuit: Circuit, method: str) -> Dict[str, object]:
    try:
        matrix_equation = circuit.matrix_equations(form="A y = b")
    except Exception as exc:  # noqa: BLE001 - queremos mostrar el fallo exacto
        message = str(exc)
        return {"matrix_error": message, "warnings": [message]}

    try:
        matrix, variables = matrix_equation.lhs.args
        vector = matrix_equation.rhs
    except Exception as exc:  # noqa: BLE001 - estructura inesperada
        message = f"Formato inesperado de matriz: {exc}"
        return {"matrix_error": message, "warnings": [message]}

    matrix_label = "Z" if method == "mesh" else "Y"
    matrix_entries = [
        f"{matrix_label}{row + 1}{col + 1} = {sp.simplify(matrix[row, col])}"
        for row in range(matrix.rows)
        for col in range(matrix.cols)
    ]
    vector_entries = [
        f"b{idx + 1} = {sp.simplify(vector[idx, 0])}"
        for idx in range(vector.rows)
    ]
    return {
        "lcapy_matrix": matrix,
        "lcapy_vector": vector,
        "lcapy_variables": [str(var) for var in variables],
        "lcapy_matrix_entries": matrix_entries,
        "lcapy_vector_entries": vector_entries,
    }


def lcapy_symbolic_equations(
    circuit: Circuit, method: str, include_matrix: bool
) -> Dict[str, object]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if method == "mesh":
                exprdict = circuit.mesh_analysis().mesh_equations()
            else:
                exprdict = circuit.nodal_analysis().nodal_equations()
        except Exception as exc:  # noqa: BLE001 - queremos propagar el error textual
            message = str(exc)
            return {"error": message, "warnings": [message]}

    equations, labels = _extract_lcapy_equations(exprdict)
    summary: Dict[str, object] = {
        "equations": equations,
        "formatted": _format_labelled_equations(equations, labels),
    }

    warning_messages = [str(warning.message) for warning in captured]
    if warning_messages:
        summary["warnings"] = warning_messages

    if include_matrix:
        matrix_summary = _lcapy_matrix_equations(circuit, method)
        summary.update(matrix_summary)
        if matrix_summary.get("warnings"):
            summary.setdefault("warnings", []).extend(matrix_summary["warnings"])

    return summary


def _map_unknown_functions_to_symbols(unknowns: Iterable[sp.Expr]) -> Dict[sp.Expr, sp.Symbol]:
    mapping: Dict[sp.Expr, sp.Symbol] = {}
    for unknown in unknowns:
        name = str(unknown)
        if name.endswith("(s)"):
            name = name[:-3]
        mapping[unknown] = sp.symbols(name)
    return mapping


def _functions_to_symbols_from_equations(
    equations: Sequence[sp.Eq], s_symbol: sp.Symbol
) -> Dict[sp.Expr, sp.Symbol]:
    functions = set()
    for eq in equations:
        functions.update(eq.atoms(sp.Function))
    filtered = [func for func in functions if len(func.args) == 1]
    return _map_unknown_functions_to_symbols(filtered)


def _extract_solution_mapping(unknowns: Iterable[sp.Expr], solutions: Dict[str, sp.Expr]):
    final: Dict[str, sp.Expr] = {}
    for unknown in unknowns:
        name = str(unknown)
        if name.endswith("(s)"):
            name = name[:-3]
        if name in solutions:
            final[name] = solutions[name]
    return final


def _superposition_to_expr(entry) -> Optional[sp.Expr]:
    expr_dict = getattr(entry, "expr", None)
    if isinstance(expr_dict, dict):
        if "s" in expr_dict:
            return expr_dict["s"]
        if expr_dict:
            return next(iter(expr_dict.values()))
    try:
        return sp.sympify(entry)
    except Exception:
        return None


def lcapy_solutions(
    circuit: Circuit,
    scopes: Iterable[str],
    requested_vars: Optional[Set[str]],
    simplify: bool,
    factor: bool,
    collect_symbol: Optional[str],
    to_time: bool,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    warnings_list: List[str] = []
    solutions: Dict[str, Dict[str, str]] = {}
    requested = None if requested_vars is None else set(requested_vars)

    cct_s = circuit.laplace()
    s_symbol = sp.symbols("s")

    if "node" in scopes:
        try:
            analysis = cct_s.nodal_analysis()
            equations_raw, _ = _extract_lcapy_equations(analysis.nodal_equations())
            mapping = _functions_to_symbols_from_equations(equations_raw, s_symbol)
            equations = [sp.Eq(eq.lhs.subs(mapping), eq.rhs.subs(mapping)) for eq in equations_raw]
            solved = solve_equations(equations, symbols=list(mapping.values()))
            node_solutions = _extract_solution_mapping(mapping.values(), solved)
            formatted: Dict[str, str] = {}
            for name, expr in node_solutions.items():
                if requested is not None and name not in requested:
                    continue
                transformed = _simplify_expression(expr, simplify, factor, collect_symbol)
                if to_time:
                    transformed = _inverse_laplace(transformed)
                    transformed = _simplify_expression(
                        transformed, simplify, factor, collect_symbol
                    )
                formatted[name] = str(transformed)
            if formatted:
                solutions["nodes"] = formatted
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"Error al resolver tensiones de nodo: {exc}")

    if "mesh" in scopes:
        try:
            loop_analysis = cct_s.loop_analysis()
            equations_raw, _ = _extract_lcapy_equations(loop_analysis.mesh_equations())
            mapping = _functions_to_symbols_from_equations(equations_raw, s_symbol)
            equations = [sp.Eq(eq.lhs.subs(mapping), eq.rhs.subs(mapping)) for eq in equations_raw]
            solved = solve_equations(equations, symbols=list(mapping.values()))
            mesh_solutions = _extract_solution_mapping(mapping.values(), solved)
            formatted: Dict[str, str] = {}
            for name, expr in mesh_solutions.items():
                if requested is not None and name not in requested:
                    continue
                transformed = _simplify_expression(expr, simplify, factor, collect_symbol)
                if to_time:
                    transformed = _inverse_laplace(transformed)
                    transformed = _simplify_expression(
                        transformed, simplify, factor, collect_symbol
                    )
                formatted[name] = str(transformed)
            if formatted:
                solutions["meshes"] = formatted
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"Error al resolver corrientes de malla: {exc}")

    if "branch" in scopes:
        branch_map: Dict[str, sp.Expr] = {}
        try:
            current_names = [str(name) for name in cct_s.branch_current_names()]
            branch_currents = cct_s.branch_currents()
            for name, current in zip(current_names, branch_currents):
                expr = _superposition_to_expr(current)
                if expr is not None:
                    branch_map[name] = expr
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"Error al obtener corrientes de rama: {exc}")

        try:
            voltage_names = [str(name) for name in cct_s.branch_voltage_names()]
            branch_voltages = cct_s.branch_voltages()
            for name, voltage in zip(voltage_names, branch_voltages):
                expr = _superposition_to_expr(voltage)
                if expr is not None:
                    branch_map[name] = expr
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"Error al obtener tensiones de rama: {exc}")

        if branch_map:
            formatted: Dict[str, str] = {}
            for name, expr in branch_map.items():
                if requested is not None and name not in requested:
                    continue
                transformed = _simplify_expression(expr, simplify, factor, collect_symbol)
                if to_time:
                    transformed = _inverse_laplace(transformed)
                    transformed = _simplify_expression(
                        transformed, simplify, factor, collect_symbol
                    )
                formatted[name] = str(transformed)
            if formatted:
                solutions["branches"] = formatted

    return solutions, warnings_list


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
    equations: Sequence[sp.Eq],
    substitutions: Optional[Dict[sp.Symbol, sp.Expr]] = None,
    symbols: Optional[Sequence[sp.Symbol]] = None,
) -> Dict[str, sp.Expr]:
    substituted = [eq.subs(substitutions or {}) for eq in equations]
    matrix, vector, solved_symbols = _matrix_from_equations(
        substituted, explicit_symbols=symbols
    )
    if not solved_symbols or matrix.is_zero_matrix:
        return {}

    solution_set = sp.linsolve((matrix, vector), *solved_symbols)
    if not solution_set:
        return {}

    solution = next(iter(solution_set))
    return {symbol.name: value for symbol, value in zip(solved_symbols, solution)}


def _simplify_expression(
    expr: sp.Expr,
    simplify: bool,
    factor: bool,
    collect_symbol: Optional[str],
) -> sp.Expr:
    result = expr
    if simplify:
        result = sp.simplify(result)
    if factor:
        result = sp.together(result)
        result = sp.factor(result)
    if collect_symbol:
        sym = sp.symbols(collect_symbol)
        result = sp.together(result)
        result = sp.collect(result, sym)
    return result


def _inverse_laplace(expr: sp.Expr) -> sp.Expr:
    t_symbol = sp.symbols("t")
    s_symbol = next((sym for sym in expr.free_symbols if sym.name == "s"), sp.symbols("s"))
    try:
        return sp.inverse_laplace_transform(expr, s_symbol, t_symbol, noconds=True)
    except Exception:
        return expr


def _eval_complex(expr: sp.Expr, substitutions: Optional[Dict[sp.Symbol, sp.Expr]] = None):
    try:
        numeric = expr.subs(substitutions or {})
        return complex(numeric.evalf())
    except Exception:
        return None


def _phasor_str(value: complex) -> str:
    magnitude = abs(value)
    angle_deg = np.degrees(np.angle(value))
    return f"{magnitude:.4g} ∠ {angle_deg:.2f}°"


def _evaluate_cycle_residuals(system: MeshSystem, substitutions: Dict[sp.Symbol, sp.Expr]):
    residuals = []
    for eq in system.equations:
        try:
            evaluated = eq.lhs.subs(substitutions).evalf() - eq.rhs.subs(substitutions).evalf()
            residuals.append(evaluated)
        except Exception:
            residuals.append(None)
    return residuals


def perform_numeric_analysis(
    graph: CircuitGraph,
    domain: str,
    method: str,
    substitutions: Optional[Dict[sp.Symbol, sp.Expr]] = None,
):
    substitutions = substitutions or {}
    if method == "mesh":
        mesh_system = _build_mesh_system(graph, domain)
        equations, warnings = mesh_system.equations, mesh_system.warnings
    else:
        equations, warnings = nodal_equations(graph, domain)
        mesh_system = None

    matrix, vector, symbols = _matrix_from_equations(
        [eq.subs(substitutions) for eq in equations]
    )
    solutions = solve_equations(equations, substitutions)

    numeric_solutions = {}
    for name, expr in solutions.items():
        value = _eval_complex(expr, substitutions)
        if value is not None:
            numeric_solutions[name] = value

    summary = {
        "matrix": matrix,
        "vector": vector,
        "variables": symbols,
        "warnings": warnings,
        "solutions": solutions,
        "numeric_solutions": numeric_solutions,
    }

    if mesh_system and numeric_solutions:
        phasors = {}
        for symbol in mesh_system.mesh_currents:
            if symbol.name in numeric_solutions:
                phasors[symbol.name] = _phasor_str(numeric_solutions[symbol.name])

        branch_voltages = []
        for edge_id, component in mesh_system.edges:
            impedance = impedance_for_component(component, domain)
            if impedance is None:
                continue
            current_expr = 0
            for mesh_idx, sign in mesh_system.edge_to_cycles.get(edge_id, []):
                current_expr += sign * mesh_system.mesh_currents[mesh_idx]
            voltage_expr = impedance * current_expr
            numeric_voltage = _eval_complex(voltage_expr, substitutions | solutions)
            if numeric_voltage is None:
                continue
            branch_voltages.append(
                {
                    "component": component.name,
                    "value": numeric_voltage,
                    "formatted": _phasor_str(numeric_voltage),
                }
            )

        residuals = _evaluate_cycle_residuals(mesh_system, substitutions | solutions)
        formatted_residuals = [
            _phasor_str(complex(res)) if res is not None else "N/A" for res in residuals
        ]

        summary.update(
            {
                "mesh_currents": phasors,
                "branch_voltages": branch_voltages,
                "kvl_residuals": formatted_residuals,
            }
        )

    return summary


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
            writer.writerow(["ecuacion", _equation_repr(equation, "plain")])
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
            latex_expr = _equation_repr(equation, "latex")
            if latex_expr == "":
                latex_expr = sp.latex(sp.sympify(equation))
            lines.append(f"$${latex_expr}$$")
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


def symbolic_analysis(
    graph: CircuitGraph,
    domain: str,
    method: str,
    show_matrices: bool,
    circuit: Optional[Circuit] = None,
):
    if method == "mesh":
        equations, warnings = mesh_equations(graph, domain)
    else:
        equations, warnings = nodal_equations(graph, domain)

    summary = {
        "equations": _format_equations(equations),
        "raw_equations": [str(eq) for eq in equations],
        "_sympy_equations": equations,
    }
    if warnings:
        summary["warnings"] = warnings

    if show_matrices:
        matrix, vector, symbols = _matrix_from_equations(equations)
        summary["matrix_G"] = [[str(item) for item in row] for row in matrix.tolist()]
        summary["matrix_B"] = [[str(item) for item in row] for row in vector.tolist()]
        summary["variables"] = [str(sym) for sym in symbols]

    if circuit is not None:
        lcapy_summary = lcapy_symbolic_equations(circuit, method, show_matrices)
        if lcapy_summary.get("formatted"):
            summary["lcapy_equations"] = lcapy_summary["formatted"]
            summary["_lcapy_sympy_equations"] = lcapy_summary.get("equations", [])
        if lcapy_summary.get("lcapy_matrix") is not None:
            summary["lcapy_matrix"] = lcapy_summary.get("lcapy_matrix")
            summary["lcapy_vector"] = lcapy_summary.get("lcapy_vector")
            summary["lcapy_matrix_entries"] = lcapy_summary.get("lcapy_matrix_entries")
            summary["lcapy_vector_entries"] = lcapy_summary.get("lcapy_vector_entries")
            summary["lcapy_variables"] = lcapy_summary.get("lcapy_variables")
        if lcapy_summary.get("error"):
            summary.setdefault("warnings", []).append(lcapy_summary["error"])
            summary["lcapy_error"] = lcapy_summary["error"]
        if lcapy_summary.get("warnings"):
            summary.setdefault("warnings", []).extend(lcapy_summary["warnings"])

    return summary


def _inject_lcapy_solutions(
    summary: dict,
    circuit: Circuit,
    args: argparse.Namespace,
    requested_vars: Optional[Set[str]],
) -> None:
    if not args.lcapy_solve:
        return
    scopes = set(args.lcapy_solve)
    lcapy_result, warnings_list = lcapy_solutions(
        circuit,
        scopes,
        requested_vars,
        simplify=args.solution_simplify,
        factor=args.solution_factor,
        collect_symbol=args.solution_collect,
        to_time=args.solution_time,
    )
    if lcapy_result:
        summary["lcapy_solutions"] = lcapy_result
    if warnings_list:
        summary.setdefault("warnings", []).extend(warnings_list)


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
        "--lcapy-solve",
        nargs="+",
        choices=["node", "branch", "mesh"],
        metavar="SCOPE",
        help=(
            "Resolver en el dominio s con Lcapy (cct_s.solve_laplace o equivalente) "
            "para tensiones de nodo, corrientes/tensiones de rama o corrientes de malla."
        ),
    )
    parser.add_argument(
        "--solution-vars",
        nargs="+",
        metavar="VAR",
        help="Variables concretas a mostrar al usar --lcapy-solve (por nombre de nodo/rama/malla).",
    )
    parser.add_argument(
        "--solution-time",
        action="store_true",
        help=(
            "Transformar las soluciones simbólicas al dominio del tiempo con inverse_laplace()/time() antes de mostrarlas."
        ),
    )
    parser.add_argument(
        "--solution-simplify",
        action="store_true",
        help="Aplicar sympy.simplify a las soluciones calculadas por Lcapy antes de imprimirlas.",
    )
    parser.add_argument(
        "--solution-factor",
        action="store_true",
        help="Factorizar denominadores/terminos para mejorar la legibilidad de las soluciones en consola.",
    )
    parser.add_argument(
        "--solution-collect",
        metavar="SYMBOL",
        help="Agrupar términos respecto a un símbolo (por ejemplo s o t) al mostrar las soluciones.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Modo de análisis numérico: construye la matriz de admitancias/impedancias y "
            "muestra fasores (usar --domain jw para AC)."
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
    parser.add_argument(
        "--query",
        nargs="+",
        metavar="VAR",
        help=(
            "Variables de interés a mostrar (por ejemplo, corrientes o tensiones de componentes)."
        ),
    )
    parser.add_argument(
        "--query-time",
        choices=["pre", "post"],
        help=(
            "En modo double, indica si la consulta aplica a t<0 (pre) o t>0 (post). "
            "Si no se indica y se usa --query, se pedirá de forma interactiva."
        ),
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


def print_session_summary() -> None:
    if not SESSION.topologies:
        return
    summary = SESSION.describe()
    active = summary.get("active_domain")
    print("\nSesión activa:")
    print(f" - Dominio activo: {active}")
    for label, entry in summary.get("topologies", {}).items():
        references = ", ".join(
            ref
            for ref, available in entry.get("references", {}).items()
            if available
        )
        print(
            " - {label}: netlist almacenado; referencias: {refs}".format(
                label=label, refs=references or "ninguna"
            )
        )


def print_symbolic(label: str, summary: dict, method: str, show_matrices: bool) -> None:
    print(f"\nEcuaciones simbólicas ({method}) para {label}:")
    for eq in summary.get("equations", []):
        print(textwrap.indent(_equation_repr(eq, "pretty"), " - "))

    if summary.get("lcapy_equations"):
        print("\nEcuaciones generadas por Lcapy:")
        for eq in summary["lcapy_equations"]:
            label_prefix = f"[{eq.get('label')}] " if eq.get("label") else ""
            print(textwrap.indent(f"{label_prefix}{_equation_repr(eq, 'pretty')}", " - "))

    for warning in summary.get("warnings", []):
        print(f" ⚠️  {warning}")

    if summary.get("solutions"):
        print("\nSoluciones:")
        for var, expr in summary["solutions"].items():
            print(f" {var} = {expr}")

    if summary.get("lcapy_solutions"):
        print("\nSoluciones (Lcapy):")
        for scope, entries in summary["lcapy_solutions"].items():
            print(f" [{scope}]")
            for var, expr in entries.items():
                print(f"  {var} = {expr}")

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

    if summary.get("lcapy_matrix_entries"):
        matrix_label = "Z" if method == "mesh" else "Y"
        print(f"\nMatriz {matrix_label} (según Lcapy):")
        for entry in summary["lcapy_matrix_entries"]:
            print(f" - {entry}")
        if summary.get("lcapy_vector_entries"):
            print("Vector b:")
            for entry in summary["lcapy_vector_entries"]:
                print(f"   • {entry}")
        if summary.get("lcapy_variables"):
            print(f"Variables: {', '.join(summary['lcapy_variables'])}")


def print_numeric(label: str, analysis: dict, method: str) -> None:
    matrix_label = "Z" if method == "mesh" else "Y"
    if analysis.get("matrix") is not None:
        print(f"\nMatriz {matrix_label} para {label}:")
        print(sp.Matrix(analysis["matrix"]))
        print("Vector de fuentes:")
        print(sp.Matrix(analysis["vector"]))
        if analysis.get("variables"):
            print(f"Variables: {', '.join(sym.name for sym in analysis['variables'])}")

    if analysis.get("mesh_currents"):
        print("\nCorrientes de malla (fasores mag/áng):")
        for name, formatted in analysis["mesh_currents"].items():
            print(f" - {name}: {formatted}")

    if analysis.get("branch_voltages"):
        print("\nTensiones por rama:")
        for item in analysis["branch_voltages"]:
            print(f" - {item['component']}: {item['formatted']}")

    if analysis.get("kvl_residuals"):
        print("\nVerificación de KVL (suma de caídas):")
        for idx, residual in enumerate(analysis["kvl_residuals"], start=1):
            print(f" - Malla {idx}: {residual}")


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


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {
            key: _sanitize_for_json(value)
            for key, value in obj.items()
            if not key.startswith("_")
        }
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, sp.MatrixBase):
        return [[_sanitize_for_json(item) for item in row] for row in obj.tolist()]
    if isinstance(obj, sp.Basic):
        return str(obj)
    return obj


def enrich_analysis(
    summary: dict,
    label: str,
    args: argparse.Namespace,
    substitutions: dict,
    requested_vars: Optional[Set[str]] = None,
) -> Tuple[dict, Dict[str, sp.Expr]]:
    solutions: Dict[str, sp.Expr] = {}
    plots: List[str] = []
    extra_warnings: List[str] = []

    equations = summary.get("_sympy_equations", [])

    requested = None if requested_vars is None else set(requested_vars)

    if args.solve:
        solved = solve_equations(equations, substitutions)
        missing_vars: Set[str] = set()
        if requested is None:
            solutions = solved
        else:
            solutions = {name: expr for name, expr in solved.items() if name in requested}
            missing_vars = requested - set(solved)

        if solutions:
            summary["solutions"] = {k: str(sp.simplify(v)) for k, v in solutions.items()}
        if missing_vars:
            extra_warnings.append(
                "No se encontraron soluciones para: " + ", ".join(sorted(missing_vars))
            )
        if not solved:
            extra_warnings.append(
                "No se pudo resolver el sistema con los parámetros proporcionados."
            )

    for instruction in _plot_instructions(args):
        var = instruction["var"]
        if requested is not None and var not in requested:
            extra_warnings.append(
                f"Se omitió la gráfica de {var} en {label} porque no está en la lista de consulta."
            )
            continue
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
    SESSION.reset()
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
    interactive_queries: Optional[List[str]] = None
    interactive_prompt = not args.csv and not args.falstad
    if interactive_prompt:
        mode = prompt_topology_mode(mode)
        interactive_entry = prompt_input_style()
        if mode == "double" and interactive_entry == "falstad":
            print("\nLa carga de netlists de Falstad solo está disponible para una topología.")
            interactive_entry = "line"
        if interactive_entry == "falstad":
            interactive_falstad_text = prompt_falstad_text()
        query_input = input(
            "\nIntroduce las variables que deseas consultar (separadas por espacio). "
            "Deja vacío para ver todas: "
        ).strip()
        if query_input:
            interactive_queries = query_input.split()

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

    query_vars: Optional[List[str]] = args.query or interactive_queries
    requested_set: Optional[Set[str]] = None
    query_time: Optional[str] = args.query_time

    if query_vars:
        requested_set = set(query_vars)
        if mode == "double" and not query_time:
            if interactive_prompt:
                while query_time not in {"pre", "post"}:
                    query_time = input(
                        "¿La consulta corresponde a t<0 (pre) o t>0 (post)? [pre/post]: "
                    ).strip().lower()
            else:
                raise SystemExit(
                    "En modo double usa --query-time pre|post para indicar si la consulta es para t<0 o t>0."
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
        entry = SESSION.store("single", circuit, domain)
        circuit_variant = _circuit_for_domain(entry, domain)
        analysis_summary = symbolic_analysis(
            graph, domain, method, args.show_matrices, circuit_variant
        )
        _inject_lcapy_solutions(
            analysis_summary,
            circuit_variant,
            args,
            set(args.solution_vars) if args.solution_vars else None,
        )
        analysis_summary, solutions = enrich_analysis(
            analysis_summary, "single", args, substitutions, requested_set
        )
        numeric_analysis = None
        if args.analyze:
            numeric_analysis = perform_numeric_analysis(graph, domain, method, substitutions)
            analysis_summary["numeric_analysis"] = _sanitize_for_json(numeric_analysis)
        if args.export_csv:
            export_to_csv(_labelled_path(args.export_csv, "single"), analysis_summary, solutions)
        if args.export_latex:
            export_to_latex(
                _labelled_path(args.export_latex, "single"), analysis_summary, solutions
            )
        print_symbolic("topología única", analysis_summary, method, args.show_matrices)
        if numeric_analysis:
            print_numeric("topología única", numeric_analysis, method)
        if args.show_lcapy:
            print(f"\nNetlist lcapy (dominio activo: {domain}):")
            print(circuit)
            print_session_summary()
        result = {
            "mode": mode,
            "components": [component.as_dict() for component in components],
            "netlist": str(circuit),
            "session": SESSION.describe(),
            "analysis": analysis_summary,
        }
        if numeric_analysis:
            result["numeric_analysis"] = _sanitize_for_json(numeric_analysis)
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
    entry_pre = SESSION.store("pre", circuit_pre, domain)
    circuit_variant_pre = _circuit_for_domain(entry_pre, domain)
    analysis_pre = symbolic_analysis(
        graph_pre, domain, method, args.show_matrices, circuit_variant_pre
    )
    _inject_lcapy_solutions(
        analysis_pre,
        circuit_variant_pre,
        args,
        set(args.solution_vars) if args.solution_vars else None,
    )
    analysis_pre, solutions_pre = enrich_analysis(
        analysis_pre,
        "pre",
        args,
        substitutions,
        requested_set if query_time == "pre" else (set() if requested_set else None),
    )
    numeric_pre = None
    if args.analyze:
        numeric_pre = perform_numeric_analysis(graph_pre, domain, method, substitutions)
        analysis_pre["numeric_analysis"] = _sanitize_for_json(numeric_pre)
    entry_post = SESSION.store("post", circuit_post, domain)
    circuit_variant_post = _circuit_for_domain(entry_post, domain)
    analysis_post = symbolic_analysis(
        graph_post, domain, method, args.show_matrices, circuit_variant_post
    )
    _inject_lcapy_solutions(
        analysis_post,
        circuit_variant_post,
        args,
        set(args.solution_vars) if args.solution_vars else None,
    )
    analysis_post, solutions_post = enrich_analysis(
        analysis_post,
        "post",
        args,
        substitutions,
        requested_set if query_time == "post" else (set() if requested_set else None),
    )
    numeric_post = None
    if args.analyze:
        numeric_post = perform_numeric_analysis(graph_post, domain, method, substitutions)
        analysis_post["numeric_analysis"] = _sanitize_for_json(numeric_post)

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
    if numeric_pre:
        print_numeric("topología t<0", numeric_pre, method)
    if numeric_post:
        print_numeric("topología t>0", numeric_post, method)
    if args.show_lcapy:
        print(f"\nNetlist lcapy t<0 (dominio activo: {domain}):")
        print(circuit_pre)
        print("\nNetlist lcapy t>0 (dominio activo: {domain}):")
        print(circuit_post)
        print_session_summary()

    result = {
        "mode": mode,
        "components_pre": [component.as_dict() for component in components_pre],
        "components_post": [component.as_dict() for component in components_post],
        "netlist_pre": str(circuit_pre),
        "netlist_post": str(circuit_post),
        "switched_mode": "pre" if switched.for_time(-1) is circuit_pre else "post",
        "session": SESSION.describe(),
        "analysis_pre": analysis_pre,
        "analysis_post": analysis_post,
        "numeric_pre": _sanitize_for_json(numeric_pre) if numeric_pre else None,
        "numeric_post": _sanitize_for_json(numeric_post) if numeric_post else None,
    }
    result.update(summarize_graph(graph_pre, "pre"))
    result.update(summarize_graph(graph_post, "post"))
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    payload = run(args)
    print("\nResumen (JSON):")
    print(json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
