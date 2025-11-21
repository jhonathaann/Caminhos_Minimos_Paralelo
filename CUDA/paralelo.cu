#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>

#define INF 2147483647 // usa um valor muito grande ao inves de INF

// Macro para verificar erros CUDA
#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t error = call;                                   \
        if (error != cudaSuccess)                                   \
        {                                                           \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// struct que representa o Grafo
typedef struct
{
    int n;     // numero de vertices
    int **adj; // matriz de adjacencia
} Grafo;

Grafo *criar_grafo(int n)
{
    Grafo *g = (Grafo *)malloc(sizeof(Grafo));
    g->n = n;

    g->adj = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        g->adj[i] = (int *)malloc(n * sizeof(int));
    }

    return g;
}

void liberar_grafo(Grafo *g)
{
    for (int i = 0; i < g->n; i++)
    {
        free(g->adj[i]);
    }
    free(g->adj);
    free(g);
}

Grafo *ler_grafo(const char *nome_arquivo)
{
    FILE *arquivo = fopen(nome_arquivo, "r");

    int n;
    fscanf(arquivo, "%d", &n);

    printf("Lendo grafo com %d vertices...\n", n);
    Grafo *g = criar_grafo(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            char buffer[64];
            fscanf(arquivo, "%s", buffer);

            if (buffer[0] == 'I' || buffer[0] == 'i')
            {
                g->adj[i][j] = INF;
            }
            else
            {
                g->adj[i][j] = atoi(buffer);
            }
        }
    }

    fclose(arquivo);
    return g;
}

void salvar_matriz_distancias(int *matriz, int n, const char *nome_arquivo)
{
    FILE *arquivo = fopen(nome_arquivo, "w");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(arquivo, "%d ", matriz[i * n + j]);
        }
        fprintf(arquivo, "\n");
    }
    fclose(arquivo);
}

// converte uma matriz 2D para um array linear. GPU trabalha melhor com arrays lineares
int *matriz_para_vetor(int **matriz, int n)
{
    int *vetor = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
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

__global__ void floyd_warshall_kernel(int *dist, int n, int k)
{

    // calcula o indide (i, j) dessa thread
    // blockIdx e threadIdx são variáveis automáticas do CUDA
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // verifica se estamos no limite da matriz
    if (i < n && j < n)
    {
        // acessa os elementos: matriz[i][j] = vetor[i * n + j]
        int ik = dist[i * n + k]; // distancia i -> k
        int kj = dist[k * n + j]; // distancia k -> j
        int ij = dist[i * n + j]; // distancia i -> j

        // evita se dar ruim somando dois numeros muito grandes
        if (ik != INF && kj != INF)
        {
            int novo_caminho = ik + kj;

            // se esse novo caminho é melhor: atualiza
            if (novo_caminho < ij)
            {
                dist[i * n + j] = novo_caminho;
            }
        }
    }
}

// ALGORITMO PRINCIPAL FLOYD-WARSHALL (GPU) - retorna o tempo de execução do algoritmo
double floyd_warshall_cuda(Grafo *g, int **resultado)
{
    int n = g->n;
    int tamanho = n * n * sizeof(int);

    printf("\nPreparando execucao na GPU...\n");

    // converte a matriz de entrada para um vetor linear
    int *h_dist = matriz_para_vetor(g->adj, n);

    // aloca memoria na GPU
    int *d_dist;
    CUDA_CHECK(cudaMalloc(&d_dist, tamanho));

    // copia dados da CPU para GPU
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist, tamanho, cudaMemcpyHostToDevice));

    // configura dimensoes do bloco e da grade
    // cada bloco tem 16x16 threads (256 threads por bloco)
    dim3 threadsPerBlock(16, 16);

    // cada garde vai ter blocos suficientes para cobrir toda a matriz
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // printf("Configuração:\n");
    // printf("  - Threads por bloco: %dx%d = %d threads\n",threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.x * threadsPerBlock.y);
    // printf("  - Blocos na grade: %dx%d = %d blocos\n", numBlocks.x, numBlocks.y, numBlocks.x * numBlocks.y);
    // printf("  - Total de threads: %d\n", threadsPerBlock.x * threadsPerBlock.y * numBlocks.x * numBlocks.y);

    printf("\nExecutando Floyd-Warshall na GPU...\n");

    cudaEvent_t inicio, fim;
    CUDA_CHECK(cudaEventCreate(&inicio));
    CUDA_CHECK(cudaEventCreate(&fim));

    // INICIA CALCULO DO TEMPO
    CUDA_CHECK(cudaEventRecord(inicio));

    // loop principal: para cada vertice intermediario k
    for (int k = 0; k < n; k++)
    {
        // lança o kernel (executar na GPU)
        floyd_warshall_kernel<<<numBlocks, threadsPerBlock>>>(d_dist, n, k);

        // verificar erros de lançamento
        CUDA_CHECK(cudaGetLastError());
    }

    // faz uma sincronização para esperar todas as threads terminarem
    CUDA_CHECK(cudaDeviceSynchronize());

    // FINALIZA MEDIÇÃO DE TEMPO
    CUDA_CHECK(cudaEventRecord(fim));
    CUDA_CHECK(cudaEventSynchronize(fim));

    // calcula o tempo decorrido, em ms
    float tempo_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_ms, inicio, fim));
    double tempo_segundos = tempo_ms / 1000.0;

    printf("Floyd-Warshall (GPU) concluído!\n");

    // copiar resultado da GPU para CPU
    CUDA_CHECK(cudaMemcpy(h_dist, d_dist, tamanho, cudaMemcpyDeviceToHost));

    // converter resultado de volta para matriz 2D
    *resultado = (int *)malloc(tamanho);
    memcpy(*resultado, h_dist, tamanho);

    // libera memoria
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaEventDestroy(inicio));
    CUDA_CHECK(cudaEventDestroy(fim));
    free(h_dist);

    return tempo_segundos;
}

int main(int argc, char *argv[])
{
    printf("=== VERSAO CUDA (GPU) ===\n\n");

    // Verificar argumentos da linha de comando
    if (argc < 2)
    {
        printf("Uso: %s <arquivo_grafo>\n", argv[0]);
        printf("Exemplo: %s grafo.txt\n", argv[0]);
        return 1;
    }

    // verifica se ha GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("ERRO: Nenhuma GPU CUDA encontrada!\n");
        return 1;
    }

    // exibe algumas informações da GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU Detectada: %s\n", prop.name);
    printf("Memória Global: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    Grafo *g = ler_grafo(argv[1]);
    if (g == NULL)
    {
        return 1;
    }

    // Executar Floyd-Warshall na GPU
    int *distancias = NULL;
    double tempo_execucao = floyd_warshall_cuda(g, &distancias);

    printf("\n=== RESULTADOS ===\n");
    printf("Número de vértices: %d\n", g->n);
    printf("Tempo de execução (GPU): %.6f segundos\n", tempo_execucao);
    printf("Tempo de execução (GPU): %.2f ms\n", tempo_execucao * 1000);

    // salva o resultado se foi fornecido arquivo de saida
    if (argc >= 3)
    {
        salvar_matriz_distancias(distancias, g->n, argv[2]);
        printf("\nMatriz de distancias salva em: %s\n", argv[2]);
    }

    free(distancias);
    liberar_grafo(g);

    printf("\nPrograma finalizado com sucesso!\n");
    return 0;
}