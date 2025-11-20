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

    imprimir_grafo(g->adj, g->n);

    // libera memoria
    liberar_grafo(g);

    return 0;
}
