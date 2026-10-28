"""
inspeccionar_dataset.py
-----------------------
Fase 2: primera inspección de application_train.csv.
Solo LEE y MUESTRA información. No modifica los datos ni entrena modelos.
"""

from pathlib import Path
import pandas as pd

# --- 1. Rutas relativas al proyecto --------------------------------------
# __file__ es este script (src/inspeccionar_dataset.py).
# .parent.parent sube dos niveles: src/ -> raíz del proyecto.
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "application_train.csv"

# --- 2. Comprobar que el archivo existe ----------------------------------
print("=" * 70)
print("1. COMPROBACIÓN DEL ARCHIVO")
print("=" * 70)
print(f"Ruta buscada: {RUTA_DATOS}")

if not RUTA_DATOS.exists():
    print("\n[ERROR] No se encontró el archivo.")
    print("Contenido real de la carpeta data/:")
    for archivo in sorted((RAIZ / "data").glob("*")):
        print("  -", archivo.name)
    raise SystemExit("Revisa el nombre/extensión del archivo y vuelve a ejecutar.")

print(f"Archivo encontrado. Tamaño: {RUTA_DATOS.stat().st_size / 1024**2:.1f} MB")

# --- 3. Cargar el CSV -----------------------------------------------------
df = pd.read_csv(RUTA_DATOS)

# Para que pandas no abrevie la salida en pantalla
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

# --- 4. Dimensiones -------------------------------------------------------
print("\n" + "=" * 70)
print("2. DIMENSIONES")
print("=" * 70)
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")

# --- 5. Primeras filas ----------------------------------------------------
print("\n" + "=" * 70)
print("3. PRIMERAS 5 FILAS (solo las 8 primeras columnas, por legibilidad)")
print("=" * 70)
print(df.iloc[:5, :8])

# --- 6. Nombres de columnas ----------------------------------------------
print("\n" + "=" * 70)
print("4. NOMBRES DE LAS COLUMNAS")
print("=" * 70)
for i, col in enumerate(df.columns, start=1):
    print(f"{i:>3}. {col}")

# --- 7. Tipos de datos ----------------------------------------------------
print("\n" + "=" * 70)
print("5. TIPOS DE DATOS")
print("=" * 70)
print("Cantidad de columnas por tipo:")
print(df.dtypes.value_counts())
print("\nColumnas de tipo texto (object):")
print(list(df.select_dtypes(include=["object", "string"]).columns))

# --- 8. Valores nulos -----------------------------------------------------
print("\n" + "=" * 70)
print("6. VALORES NULOS")
print("=" * 70)
nulos = df.isna().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)
porcentaje = (nulos / len(df) * 100).round(2)
tabla_nulos = pd.DataFrame({"nulos": nulos, "porcentaje_%": porcentaje})
print(f"Columnas con al menos un nulo: {len(tabla_nulos)} de {df.shape[1]}")
print("\nTop 20 columnas con más nulos:")
print(tabla_nulos.head(20))

# --- 9. Duplicados --------------------------------------------------------
print("\n" + "=" * 70)
print("7. DUPLICADOS")
print("=" * 70)
print(f"Filas completamente duplicadas: {df.duplicated().sum()}")
if "SK_ID_CURR" in df.columns:
    print(f"Identificadores SK_ID_CURR repetidos: {df['SK_ID_CURR'].duplicated().sum()}")
else:
    print("No existe la columna SK_ID_CURR en este archivo.")

# --- 10. Variable TARGET --------------------------------------------------
print("\n" + "=" * 70)
print("8. VARIABLE TARGET")
print("=" * 70)
if "TARGET" in df.columns:
    print("La columna TARGET existe.")
    print(f"Nulos en TARGET: {df['TARGET'].isna().sum()}")
    print("\nConteo por valor:")
    print(df["TARGET"].value_counts(dropna=False))
    print("\nProporción (%):")
    print((df["TARGET"].value_counts(normalize=True, dropna=False) * 100).round(2))
else:
    print("[AVISO] No existe la columna TARGET en este archivo.")

print("\nInspección terminada.")