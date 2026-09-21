#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <mpi.h>
#include <omp.h>

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

    if (rank == 0) {
        std::cout << "=========================================================\n";
        std::cout << " ACTIVIDAD 1.2: CARGA DE MEMORIA DISTRIBUIDA EN MPI     \n";
        std::cout << " " << nombreEquipo << " (Versión: Scatter / Gather)\n";
        std::cout << " Integrantes:\n";
        for (const auto& integrante : integrantes) {
            std::cout << "  - " << integrante << "\n";
        }
        std::cout << "=========================================================\n\n";
        std::cout.flush();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    OperacionesArreglos opArreglos(rank, totalProcesos, infoCompletaEquipo);

    int opcion = 0;
    int tamanoTotal = 40;
    int modoDetalladoInt = 1;

    int* A = nullptr;
    int* B = nullptr;
    int* R = nullptr;

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
        
        // En Scatter/Gather dividimos entre todos los procesos de MPI_COMM_WORLD
        subTamano = tamanoTotal / totalProcesos;

        if (rank == 0) {
            if (!A) {
                A = new int[tamanoTotal];
                B = new int[tamanoTotal];
                R = new int[tamanoTotal];
            }
        }

        if (!subA) {
            subA = new int[subTamano];
            subB = new int[subTamano];
            subR = new int[subTamano];
        }

        double tInicio = 0.0, tFin = 0.0;

        switch (opcion) {
            case 1:
                if (rank == 0) std::cout << "Modo actualizado correctamente.\n";
                if (rank == 0) {
                    delete[] A; delete[] B; delete[] R;
                    A = new int[tamanoTotal]; B = new int[tamanoTotal]; R = new int[tamanoTotal];
                }
                delete[] subA; delete[] subB; delete[] subR;
                subA = new int[subTamano]; subB = new int[subTamano]; subR = new int[subTamano];
                break;

            case 2: // Crear arreglos
                if (rank == 0) tInicio = MPI_Wtime();

                {
                    int posInicio = rank * subTamano;
                    opArreglos.crearArregloMPI(subA, subB, subTamano, posInicio, detallado);
                }

                // Recolectar arreglos creados localmente en el maestro mediante MPI_Gather
                MPI_Gather(subA, subTamano, MPI_INT, A, subTamano, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Gather(subB, subTamano, MPI_INT, B, subTamano, MPI_INT, 0, MPI_COMM_WORLD);

                if (rank == 0) {
                    tFin = MPI_Wtime();
                    std::cout << "\n[Nodo Maestro 0] Arreglos creados mediante MPI_Gather.\n";
                    std::cout << "Tiempo de ejecución (Creación): " << (tFin - tInicio) << " segundos.\n";
                }
                break;

            case 3: // Suma
            case 4: // Resta
            case 5: // Multiplicación
            case 6: // Cuadrado
                if (rank == 0) tInicio = MPI_Wtime();

                // Distribución colectiva con MPI_Scatter
                MPI_Scatter(A, subTamano, MPI_INT, subA, subTamano, MPI_INT, 0, MPI_COMM_WORLD);
                if (opcion != 6) {
                    MPI_Scatter(B, subTamano, MPI_INT, subB, subTamano, MPI_INT, 0, MPI_COMM_WORLD);
                }

                {
                    int posInicio = rank * subTamano;
                    if (opcion == 3) opArreglos.sumar(subA, subB, subR, subTamano, posInicio, detallado);
                    else if (opcion == 4) opArreglos.restar(subA, subB, subR, subTamano, posInicio, detallado);
                    else if (opcion == 5) opArreglos.multiplicar(subA, subB, subR, subTamano, posInicio, detallado);
                    else if (opcion == 6) opArreglos.cuadrado(subA, subR, subTamano, posInicio, detallado);
                }

                // Recolección colectiva con MPI_Gather
                MPI_Gather(subR, subTamano, MPI_INT, R, subTamano, MPI_INT, 0, MPI_COMM_WORLD);

                if (rank == 0) {
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

    if (rank == 0) {
        delete[] A; delete[] B; delete[] R;
    }
    delete[] subA; delete[] subB; delete[] subR;

    MPI_Barrier(MPI_COMM_WORLD);

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