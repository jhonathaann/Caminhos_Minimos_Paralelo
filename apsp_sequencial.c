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

int main(int argc, char *argv[]) {
  printf("=== APSP - Versão Sequencial ===\n");

  int n = 4; //
  Grafo *g = criar_grafo(n);

  printf("Grafo com %d vertices criado com sucesso!\n", g->n);

  // libera memoria
  liberar_grafo(g);

  return 0;
}
