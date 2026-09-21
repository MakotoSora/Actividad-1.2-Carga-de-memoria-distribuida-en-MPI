#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <mpi.h>
#include <omp.h>

// Generador de números aleatorios thread-safe
inline int rand_thread_safe(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return static_cast<int>(*seed);
}

class OperacionesArreglos {
private:
    int rank;
    int totalProcesos;
    std::string infoEquipo;

public:
    OperacionesArreglos(int r, int totalP, const std::string& info)
        : rank(r), totalProcesos(totalP), infoEquipo(info) {}

    // Función para crear/llenar subarreglos asignados a cada proceso
    void crearArregloMPI(int* subA, int* subB, int subTamano, int posInicio, bool detallado) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            unsigned int seed = static_cast<unsigned int>(time(NULL)) + rank * 1000 + tid * 37;

            #pragma omp for schedule(static)
            for (int i = 0; i < subTamano; ++i) {
                int posGlobal = posInicio + i;
                if (detallado) {
                    subA[i] = posGlobal + 1;
                    subB[i] = (posGlobal + 1) * 2;
                    #pragma omp critical
                    {
                        std::cout << "[Equipo: " << infoEquipo << "] "
                                  << "[Proceso MPI: " << rank << "] "
                                  << "[Hilo OpenMP: " << tid << "] "
                                  << "[Posición: " << posGlobal << "] "
                                  << "[Operación: Creación | A=" << subA[i] << ", B=" << subB[i] << "]\n";
                        std::cout.flush();
                    }
                } else {
                    subA[i] = (rand_thread_safe(&seed) % 1000000) + 1;
                    subB[i] = (rand_thread_safe(&seed) % 1000000) + 1;
                }
            }
        }
    }

    void sumar(const int* subA, const int* subB, int* subR, int subTamano, int posInicio, bool detallado) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < subTamano; ++i) {
                subR[i] = subA[i] + subB[i];
                if (detallado) {
                    int posGlobal = posInicio + i;
                    #pragma omp critical
                    {
                        std::cout << "[Equipo: " << infoEquipo << "] "
                                  << "[Proceso MPI: " << rank << "] "
                                  << "[Hilo OpenMP: " << tid << "] "
                                  << "[Posición: " << posGlobal << "] "
                                  << "[Operación: Suma | " << subA[i] << " + " << subB[i] << " = " << subR[i] << "]\n";
                        std::cout.flush();
                    }
                }
            }
        }
    }

    void restar(const int* subA, const int* subB, int* subR, int subTamano, int posInicio, bool detallado) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < subTamano; ++i) {
                subR[i] = subA[i] - subB[i];
                if (detallado) {
                    int posGlobal = posInicio + i;
                    #pragma omp critical
                    {
                        std::cout << "[Equipo: " << infoEquipo << "] "
                                  << "[Proceso MPI: " << rank << "] "
                                  << "[Hilo OpenMP: " << tid << "] "
                                  << "[Posición: " << posGlobal << "] "
                                  << "[Operación: Resta | " << subA[i] << " - " << subB[i] << " = " << subR[i] << "]\n";
                        std::cout.flush();
                    }
                }
            }
        }
    }

    void multiplicar(const int* subA, const int* subB, int* subR, int subTamano, int posInicio, bool detallado) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < subTamano; ++i) {
                subR[i] = subA[i] * subB[i];
                if (detallado) {
                    int posGlobal = posInicio + i;
                    #pragma omp critical
                    {
                        std::cout << "[Equipo: " << infoEquipo << "] "
                                  << "[Proceso MPI: " << rank << "] "
                                  << "[Hilo OpenMP: " << tid << "] "
                                  << "[Posición: " << posGlobal << "] "
                                  << "[Operación: Multiplicación | " << subA[i] << " * " << subB[i] << " = " << subR[i] << "]\n";
                        std::cout.flush();
                    }
                }
            }
        }
    }

    void cuadrado(const int* subA, int* subR, int subTamano, int posInicio, bool detallado) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < subTamano; ++i) {
                subR[i] = subA[i] * subA[i];
                if (detallado) {
                    int posGlobal = posInicio + i;
                    #pragma omp critical
                    {
                        std::cout << "[Equipo: " << infoEquipo << "] "
                                  << "[Proceso MPI: " << rank << "] "
                                  << "[Hilo OpenMP: " << tid << "] "
                                  << "[Posición: " << posGlobal << "] "
                                  << "[Operación: Cuadrado | " << subA[i] << "^2 = " << subR[i] << "]\n";
                        std::cout.flush();
                    }
                }
            }
        }
    }
};

void mostrarMuestraArreglo(const int* arr, int tamano) {
    if (tamano <= 50) {
        std::cout << "Arreglo Resultado Completo:\n[ ";
        for (int i = 0; i < tamano; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << "]\n";
    } else {
        std::cout << "Arreglo Resultado (Muestra de primeros 10 y últimos 10 elementos):\n[ ";
        for (int i = 0; i < 10; ++i) std::cout << arr[i] << " ";
        std::cout << "... ";
        for (int i = tamano - 10; i < tamano; ++i) std::cout << arr[i] << " ";
        std::cout << "]\n";
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, totalProcesos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesos);

    char nombreHost[MPI_MAX_PROCESSOR_NAME];
    int lenHost;
    MPI_Get_processor_name(nombreHost, &lenHost);

    std::string nombreEquipo = "Equipo 11";
    std::string infoCompletaEquipo = nombreEquipo + " [Host: " + std::string(nombreHost) + "]";

    std::string integrantes[] = {
        "Hermosillo Prado Carlos",
        "Montes de Oca del Risco Rafael Alejandro",
        "Ramirez Andrade Uriel Ismael Guadalupe"
    };

    // IMPRESIÓN INICIAL OBLIGATORIA
    if (rank == 0) {
        std::cout << "=========================================================\n";
        std::cout << " ACTIVIDAD 1.2: CARGA DE MEMORIA DISTRIBUIDA EN MPI     \n";
        std::cout << " " << nombreEquipo << " (Versión: Send / Recv)\n";
        std::cout << " Integrantes:\n";
        for (const auto& integrante : integrantes) {
            std::cout << "  - " << integrante << "\n";
        }
        std::cout << "=========================================================\n\n";
        std::cout.flush();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int numTrabajadores = totalProcesos - 1;
    if (numTrabajadores < 1) {
        if (rank == 0) {
            std::cerr << "Error: Se requieren al menos 2 procesos (1 maestro y 1 trabajador).\n";
        }
        MPI_Finalize();
        return 1;
    }

    OperacionesArreglos opArreglos(rank, totalProcesos, infoCompletaEquipo);

    int opcion = 0;
    int tamanoTotal = 40;
    int modoDetalladoInt = 1; // 1 = 40 elems (detallado), 0 = 4,000,000 elems (rendimiento)

    // Punteros para el nodo maestro
    int* A = nullptr;
    int* B = nullptr;
    int* R = nullptr;

    // Punteros para los subarreglos
    int subTamano = 0;
    int* subA = nullptr;
    int* subB = nullptr;
    int* subR = nullptr;

    do {
        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "           MENÚ DE OPERACIONES          \n";
            std::cout << "========================================\n";
            std::cout << "Modo actual: " << (modoDetalladoInt == 1 ? "40 Elementos (Detallado)" : "4,000,000 Elementos (Rendimiento)") << "\n";
            std::cout << "1. Cambiar Modo (40 elems / 4,000,000 elems)\n";
            std::cout << "2. Crear Arreglos\n";
            std::cout << "3. Sumar Arreglos (A + B)\n";
            std::cout << "4. Restar Arreglos (A - B)\n";
            std::cout << "5. Multiplicar Arreglos (A * B)\n";
            std::cout << "6. Cuadrado de Arreglo (A^2)\n";
            std::cout << "7. Salir\n";
            std::cout << "Seleccione una opción: ";
            std::cin >> opcion;

            if (opcion == 1) {
                std::cout << "Seleccione tamaño (1 = 40 elems [Detallado], 2 = 4,000,000 elems [Rendimiento]): ";
                int sel;
                std::cin >> sel;
                modoDetalladoInt = (sel == 1) ? 1 : 0;
            }
        }

        MPI_Bcast(&opcion, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&modoDetalladoInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (opcion == 7) break;

        bool detallado = (modoDetalladoInt == 1);
        tamanoTotal = detallado ? 40 : 4000000;
        subTamano = tamanoTotal / numTrabajadores;

        // Gestión de memoria dinámica
        if (rank == 0) {
            if (!A) {
                A = new int[tamanoTotal];
                B = new int[tamanoTotal];
                R = new int[tamanoTotal];
            }
        } else {
            if (!subA) {
                subA = new int[subTamano];
                subB = new int[subTamano];
                subR = new int[subTamano];
            }
        }

        double tInicio = 0.0, tFin = 0.0;

        switch (opcion) {
            case 1:
                if (rank == 0) std::cout << "Modo actualizado correctamente.\n";
                // Reasignar memoria si cambió el tamaño
                if (rank == 0) {
                    delete[] A; delete[] B; delete[] R;
                    A = new int[tamanoTotal]; B = new int[tamanoTotal]; R = new int[tamanoTotal];
                } else {
                    delete[] subA; delete[] subB; delete[] subR;
                    subA = new int[subTamano]; subB = new int[subTamano]; subR = new int[subTamano];
                }
                break;

            case 2: // Crear arreglos
                if (rank == 0) tInicio = MPI_Wtime();

                if (rank != 0) {
                    int posInicio = (rank - 1) * subTamano;
                    opArreglos.crearArregloMPI(subA, subB, subTamano, posInicio, detallado);
                    MPI_Send(subA, subTamano, MPI_INT, 0, 101, MPI_COMM_WORLD);
                    MPI_Send(subB, subTamano, MPI_INT, 0, 102, MPI_COMM_WORLD);
                } else {
                    for (int w = 1; w <= numTrabajadores; ++w) {
                        int posInicio = (w - 1) * subTamano;
                        MPI_Recv(&A[posInicio], subTamano, MPI_INT, w, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&B[posInicio], subTamano, MPI_INT, w, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    tFin = MPI_Wtime();
                    std::cout << "\n[Nodo Maestro 0] Arreglos creados exitosamente.\n";
                    std::cout << "Tiempo de ejecución (Creación): " << (tFin - tInicio) << " segundos.\n";
                }
                break;

            case 3: // Suma
            case 4: // Resta
            case 5: // Multiplicación
            case 6: // Cuadrado
                if (rank == 0) tInicio = MPI_Wtime();

                // Distribución mediante MPI_Send desde el Maestro a los Trabajadores
                if (rank == 0) {
                    for (int w = 1; w <= numTrabajadores; ++w) {
                        int posInicio = (w - 1) * subTamano;
                        MPI_Send(&A[posInicio], subTamano, MPI_INT, w, 201, MPI_COMM_WORLD);
                        if (opcion != 6) {
                            MPI_Send(&B[posInicio], subTamano, MPI_INT, w, 202, MPI_COMM_WORLD);
                        }
                    }
                } else {
                    MPI_Recv(subA, subTamano, MPI_INT, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (opcion != 6) {
                        MPI_Recv(subB, subTamano, MPI_INT, 0, 202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }

                    int posInicio = (rank - 1) * subTamano;
                    if (opcion == 3) opArreglos.sumar(subA, subB, subR, subTamano, posInicio, detallado);
                    else if (opcion == 4) opArreglos.restar(subA, subB, subR, subTamano, posInicio, detallado);
                    else if (opcion == 5) opArreglos.multiplicar(subA, subB, subR, subTamano, posInicio, detallado);
                    else if (opcion == 6) opArreglos.cuadrado(subA, subR, subTamano, posInicio, detallado);

                    MPI_Send(subR, subTamano, MPI_INT, 0, 203, MPI_COMM_WORLD);
                }

                // Recopilación mediante MPI_Recv desde los Trabajadores al Maestro
                if (rank == 0) {
                    for (int w = 1; w <= numTrabajadores; ++w) {
                        int posInicio = (w - 1) * subTamano;
                        MPI_Recv(&R[posInicio], subTamano, MPI_INT, w, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    tFin = MPI_Wtime();
                    std::cout << "\n========================================\n";
                    std::cout << "        RESULTADO DE LA OPERACIÓN       \n";
                    std::cout << "========================================\n";
                    mostrarMuestraArreglo(R, tamanoTotal);
                    std::cout << "Tiempo de ejecución: " << (tFin - tInicio) << " segundos.\n";
                }
                break;
        }

        MPI_Barrier(MPI_COMM_WORLD);

    } while (opcion != 7);

    // Liberación de memoria
    if (rank == 0) {
        delete[] A; delete[] B; delete[] R;
    } else {
        delete[] subA; delete[] subB; delete[] subR;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // IMPRESIÓN FINAL OBLIGATORIA
    if (rank == 0) {
        std::cout << "\n=========================================================\n";
        std::cout << "   EJECUCIÓN FINALIZADA CORRECTAMENTE                    \n";
        std::cout << " " << nombreEquipo << "\n";
        std::cout << " Integrantes:\n";
        for (const auto& integrante : integrantes) {
            std::cout << "  - " << integrante << "\n";
        }
        std::cout << "=========================================================\n";
        std::cout.flush();
    }

    MPI_Finalize();
    return 0;
}