import argparse
import csv
import json
import textwrap


def parse_component_line(line: str) -> dict:
    parts = [piece.strip() for piece in line.split(",") if piece.strip()]
    if len(parts) not in (5, 6):
        raise ValueError(
            "Cada componente debe tener 5 o 6 campos: "
            "Tipo,Nombre,NodoA,NodoB,Valor[,Orientación]."
        )

    component = {
        "type": parts[0],
        "name": parts[1],
        "node_a": parts[2],
        "node_b": parts[3],
        "value": parts[4],
    }
    if len(parts) == 6:
        component["orientation"] = parts[5]
    return component


def load_components_from_csv(path: str) -> list:
    components = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            line = ",".join(row)
            components.append(parse_component_line(line))
    return components


def prompt_for_components(label: str) -> list:
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
            components.append(parse_component_line(line))
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


def run(args: argparse.Namespace) -> dict:
    mode = args.mode
    components_pre = []
    components_post = []

    if args.csv:
        if mode == "single" and len(args.csv) != 1:
            raise SystemExit("Modo single requiere exactamente un archivo CSV.")
        if mode == "double" and len(args.csv) != 2:
            raise SystemExit("Modo double requiere dos archivos CSV: t<0 y t>0.")

    if mode == "single":
        if args.csv:
            components_pre = load_components_from_csv(args.csv[0])
        else:
            components_pre = prompt_for_components("todo t<0/t>0")
        return {"mode": mode, "components": components_pre}

    # Modo double
    if args.csv:
        components_pre = load_components_from_csv(args.csv[0])
        components_post = load_components_from_csv(args.csv[1])
    else:
        components_pre = prompt_for_components("topología t<0")
        components_post = prompt_for_components("topología t>0")

    return {
        "mode": mode,
        "components_pre": components_pre,
        "components_post": components_post,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    payload = run(args)
    print("\nResumen (JSON):")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
