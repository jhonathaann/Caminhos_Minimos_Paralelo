
import random
import os
import sys

def gerar_grafo(num_vertices, densidade, peso_min=1, peso_max=100):
    """
    gera um grafo com pesos nas arestas de forma aleatoria
    
    Paraetros:
        num_vertices: número de vértices do grafo
        densidade: porcentagem de arestas (0.0 a 1.0)
                   densidade = num_arestas / num_arestas_possiveis
        peso_min: peso mínimo das arestas
        peso_max: peso máximo das arestas
    
    Retorna:
        matriz de adj
    """
    # inicializar matriz com INF
    matriz = [['INF' for _ in range(num_vertices)] for _ in range(num_vertices)]
    
    # diagonal com zeros (distância de um vertice para ele mesmo)
    for i in range(num_vertices):
        matriz[i][i] = 0
    
    # numero maximo de arestas = n * (n-1) para grafos direcionados
    max_arestas = num_vertices * (num_vertices - 1)
    num_arestas = int(max_arestas * densidade)
    
    # cria lista de todas as posições possiveis (exceto diagonal)
    posicoes_possiveis = []
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                posicoes_possiveis.append((i, j))
    
    # seleciona de forma aleatoria as posicoes que terão arestas
    posicoes_com_arestas = random.sample(posicoes_possiveis, num_arestas)
    
    # atribui pesos aleatorios para as arestas selecionadas
    for i, j in posicoes_com_arestas:
        peso = random.randint(peso_min, peso_max)
        matriz[i][j] = peso
    
    return matriz

# salva o grafo em um arquivo
def salvar_grafo(matriz, nome_arquivo):
    num_vertices = len(matriz)
    
    with open(nome_arquivo, 'w') as f:
        # escreve o numero de vertices
        f.write(f"{num_vertices}\n")
        
        # escreve a matriz de adj
        for linha in matriz:
            linha_str = ' '.join(str(valor) for valor in linha)
            f.write(linha_str + '\n')
    
    print(f"Grafo salvo em: {nome_arquivo}")


def gerar_grafo_customizado():    
   
    num_vertices = int(input("\nNuemro de vertices: "))
    densidade = float(input("Densidade de arestas (0.0 a 1.0, ex: 0.5 para 50%): "))
    
    if densidade < 0 or densidade > 1:
        print("Erro: densidade deve estar entre 0 e 1")
        return
    
    # Nome do arquivo
    densidade_str = f"{int(densidade * 100)}"
    nome_arquivo = f"input/g_{num_vertices}_d{densidade_str}.txt"
    
    print(f"\nGerando grafo com {num_vertices} vertices e {int(densidade*100)}% de densidade...")
    
    # Gerar e salvar
    matriz = gerar_grafo(num_vertices, densidade)
    os.makedirs('input', exist_ok=True)
    salvar_grafo(matriz, nome_arquivo)
    
    print("\nGrafo gerado com sucesso!")
        

def main():
    random.seed(42)
    gerar_grafo_customizado

if __name__ == "__main__":
    main()
