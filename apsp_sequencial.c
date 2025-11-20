#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF INT_MAX

// struct que representa o Grafo
typedef struct {
  int n;     // numero de vertices
  int **adj; // matriz de adjacencia
} Grafo;

// aloca o grafo e retorna o ponteiro para o grafo criado
Grafo *criar_grafo(int n) {
    Grafo *g = (Grafo *)malloc(sizeof(Grafo));
    g->n = n;
    // aloca a matriz de adjc
    g->adj = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        g->adj[i] = (int *)malloc(n * sizeof(int));
    }

    return g;
}

void liberar_grafo(Grafo *g) {
    for (int i = 0; i < g->n; i++) {
        free(g->adj[i]);
    }
    free(g->adj);
    free(g);
}

Grafo *ler_grafo(const char *nome_arquivo) {
    FILE *arquivo = fopen(nome_arquivo, "r");

    int n;
    fscanf(arquivo, "%d", &n);

    printf("Lendo grafo com %d vertices...\n", n);
    Grafo *g = criar_grafo(n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            char buffer[64]; // buffer para ler a linha
            fscanf(arquivo, "%s", buffer);

            // verifica se é INF
            if(buffer[0] == 'I' || buffer[0] == 'i'){
                g->adj[i][j] = INF;
            }else{
                g->adj[i][j] = atoi(buffer);
            }
        }
    }

    fclose(arquivo);
    printf("Grafo lido com sucesso!\n");
    return g;
}

void salva_matriz_distancias(int **matriz, int n, const char *nome_arquivo){
    FILE *arquivo = fopen(nome_arquivo, "w");

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            fprintf(arquivo, "%d ", matriz[i][j]);
        }
        fprintf(arquivo, "\n");
    }

    fclose(arquivo);
}

void imprimir_grafo(int **matriz, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matriz[i][j] == INF) {
                printf("  INF");
            } else {
                printf("%5d", matriz[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

int **copiar_matriz(int **origem, int n) {
    int **copia = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        copia[i] = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
        copia[i][j] = origem[i][j];
        }
    }
    return copia;
}

void liberar_matriz(int **matriz, int n) {
    for (int i = 0; i < n; i++) {
        free(matriz[i]);
    }
    free(matriz);
}

/*
 * O algoritmo funciona assim:
 *   - Para cada vertice k (intermediario)
 *     - Para cada par de vertices (i, j)
 *       - Verifica se o caminho i -> k -> j é menor que i -> j
 *       - Se sim, atualiza a distancia
 *
 * Parametros:
 *   - g: grafo de entrada
 * Retorno: matriz de distancia minimas
 */
int **floyd_warshall(Grafo *g) {
    int n = g->n;

    // cria uma copia da matriz de adj
    int **dist = copiar_matriz(g->adj, n);

    printf("\nececutando algoritmo Floyd-Warshall...\n");

    // loop principal: k é o vertice intermediario
    for (int k = 0; k < n; k++) {
        // para cada par de vertices (i, j)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // so vamos realizar a soma se ambas as entradas forem != de INF (sera que é melhor um OU?)
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    // se o caminho i->k->j é menor que i->j
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
    }

    printf("Floyd-Warshall concluido!\n");
    return dist;
}

int main(int argc, char *argv[]) {
    printf("=== APSP - Versão Sequencial ===\n\n");

    // Verificar argumentos da linha de comando
    if (argc < 2) {
        printf("Uso: %s <arquivo_grafo>\n", argv[0]);
        printf("Exemplo: %s grafo.txt\n", argv[0]);
        return 1;
    }

    // Ler o grafo do arquivo
    Grafo *g = ler_grafo(argv[1]);
    if (g == NULL) {
        return 1;
    }

    // printf("\n=== Matriz de Adjacência (Grafo de Entrada) ===\n");
    // imprimir_grafo(g->adj, g->n);

    // executa o algoritmo de floyd_warshall
    int **distancias = floyd_warshall(g);

    // imprimir_grafo(distancias, g->n);
    salva_matriz_distancias(distancias, g->n, argv[2]);


    liberar_matriz(distancias, g->n);
    liberar_grafo(g);

    printf("Programa finalizado com sucesso!\n");
    return 0;
}
