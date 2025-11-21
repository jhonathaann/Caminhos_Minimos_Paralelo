#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>

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

/* KERNEL CUDA que executa na CPU

este codigo sera executado em PARALELO na GPU

__global__ = funcao que roda na GPU e é chamada pela CPU

- para cada vertice intermediario k, lançamos milhares de threads
- cada uma dessas threads calcula uma PAR (i, j)
- todas as threads trabalham ao mesmo tempo

parametros:
- dist: matriz de distancias (na forma linear)
- n: numero de vertices
- k: vertice intermediario atual
*/

__global__ void floyd_warshall_kernel(int *dist, int n, int k){

    // calcula o indide (i, j) dessa thread
    // blockIdx e threadIdx são variáveis automáticas do CUDA
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // verifica se estamos no limite da matriz
    if(i < n && j < n){
        // acessa os elementos: matriz[i][j] = vetor[i * n + j]
        int ik = dist[i * n + k]; // distancia i -> k
        int kj = dist[k * n + j]; // distancia k -> j
        int ij = dist[i * n + j]; // distancia i -> j

        // evita se dar ruim somando dois numeros muito grandes
        if(ik != INF && kj != INF){
            int novo_caminho = ik + kj;

            // se esse novo caminho é melhor: atualiza
            if(novo_caminho < ij){
                dist[i * n + j] = novo_caminho;
            }
        }
    }
}

int main(int argc, char *argv[]){
    printf("=== VERSAO CUDA (GPU) ===\n\n");


    return 0;
}