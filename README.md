# lcapyteo

Herramienta de línea de comandos para capturar topologías de circuitos usando un formato de componente consistente.

## Formato de componente
Cada línea representa un componente con el formato:
```
Tipo,Nombre,NodoA,NodoB,Valor[,Orientación]
```
Ejemplos:
- `R,R1,n1,n2,5k`
- `C,C1,n2,0,10u`
- `V,V1,n0,n1,10,DC`

Reglas de nodos:
- Usa `0` o `gnd` como nodo de referencia.
- Evita mezclar mayúsculas y minúsculas para el mismo nodo.

## Uso
El script `circuit_cli.py` admite modo de topología única (`single`) o doble (`double`, para t<0 y t>0). También permite cargar datos desde archivos CSV para evitar introducirlos a mano.

### Ayuda
```
python circuit_cli.py --help
```
Muestra ejemplos de uso y recordatorios del formato de línea.

### Modo single
```
python circuit_cli.py --mode single
```
Se solicitarán líneas hasta dejar la entrada vacía. Para cargar desde CSV:
```
python circuit_cli.py --mode single --csv topologia.csv
```

### Modo double (t<0 y t>0)
```
python circuit_cli.py --mode double
```
Primero se piden los componentes para `t<0` y luego para `t>0`. Para usar archivos:
```
python circuit_cli.py --mode double --csv pre.csv post.csv
```

### Resultado
Al finalizar se imprime un resumen en JSON con la lista de componentes capturada (una sola lista en modo `single`, dos listas en modo `double`).
