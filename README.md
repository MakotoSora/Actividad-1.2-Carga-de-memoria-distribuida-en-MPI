# Actividad 1.2: Carga de Memoria Distribuida en MPI

Este repositorio contiene la implementación y comparación de dos enfoques de comunicación distribuida en memoria utilizando **Microsoft MPI (MS-MPI)** en C++:
1. **Comunicación Punto a Punto:** Utilizando `MPI_Send` y `MPI_Recv`.
2. **Comunicación Colectiva:** Utilizando `MPI_Scatter` y `MPI_Gather`.

---

## Datos del Equipo

**Equipo 11**

* **Hermosillo Prado Carlos**
* **Montes de Oca del Risco Rafael Alejandro**
* **Ramirez Andrade Uriel Ismael Guadalupe**

---

## Archivos del Repositorio

* `operaciones_send_recv.cpp`: Código fuente con implementación de comunicación punto a punto.
* `operaciones_scatter_gather.cpp`: Código fuente con implementación de comunicación colectiva optimizada.
* `.gitignore`: Configuración para excluir archivos ejecutables y temporales de compilación.
* `README.md`: Documentación e instrucciones de ejecución del proyecto.

---

## Requisitos Previos

1. **Microsoft MPI v10.1.2** o superior instalado en todos los nodos de la red.
2. Compilador de C++ (**x64 Native Tools Command Prompt for VS**).
3. Servicio de red `smpd` en ejecución en todas las computadoras participantes (`smpd -start`).
4. Reglas de Firewall habilitadas para los puertos de comunicación TCP de MPI.

---

## Compilación

Abre la consola **x64 Native Tools Command Prompt for VS** como Administrador en el nodo principal, dirígete a la carpeta del proyecto y ejecuta:

### Versión 1 (Send / Recv)
```cmd
cl /EHsc operaciones_send_recv.cpp /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /Fe:operaciones_send_recv.exe /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
