#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
// #include <cuda_runtime.h>

#define INF INT_MAX

// struct que representa o Grafo
typedef struct {
  int n;     // numero de vertices
  int **adj; // matriz de adjacencia
} Grafo;

Grafo *criar_grafo(int n){
    Grafo *g = (Grafo*) malloc(sizeof(Grafo));
    g->n = n;

    g->adj = (int **) malloc(n * sizeof(int *));
    for(int i = 0; i < n; i++){
        g->adj[i] = (int *) malloc(n * sizeof(int));
    }

    return g;
}

void liberar_grafo(Grafo *g){
    for(int i = 0; i < g->n; i++){
        free(g->adj[i]);
    }
    free(g->adj);
    free(g);
}

Grafo *ler_grafo(const char *nome_arquivo){
    FILE *arquivo = fopen(nome_arquivo, "r");

    int n;
    fscanf(arquivo, "%d", &n);

    printf("Lendo grafo com %d vertices...\n", n);
    Grafo *g = criar_grafo(n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            char buffer[64];
            fscanf(arquivo, "%s", buffer);

            if(buffer[0] == 'I' || buffer[0] == 'i'){
                g->adj[i][j] = INF;
            }else{
                g->adj[i][j] = atoi(buffer);
            }
        }
    }

    fclose(arquivo);
    return g;
}

void salvar_matriz_distancias(int *matriz, int n, const char *nome_arquivo){
    FILE *arquivo = fopen(nome_arquivo, "w");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            fprintf(arquivo, "%d ", matriz[i * n + j]);
        }
        fprintf(arquivo, "\n");
    }
    fclose(arquivo);
}

// converte uma matriz 2D para um array linear. GPU trabalha melhor com arrays lineares
int *matriz_para_vetor(int **matriz, int n){
    int *vetor = (int *) malloc(n * n * sizeof(int));

    for(int i = 0; i < 0; i++){
        for(int j = 0; j < n; j++){
            vetor[i * n + j] = matriz[i][j];
        }
    }

    return vetor;
}

int main(int argc, char *argv[]){
    printf("=== VERSAO CUDA (GPU) ===\n\n");


    return 0;
}