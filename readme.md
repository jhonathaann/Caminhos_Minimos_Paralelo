## Trabalho 3 de Programação Paralela - Cálculo de Todos os Pares de Caminhos Míınimos em GPU com CUDA
### SEQUENCIAL
#### COMPILA: 
    gcc -o floyd_sequencial sequencial/floyd_sequencial.c -Wall

#### EXECUTA:
    ./floyd_sequencial input/arquivo_entrada.txt output/arquivo_saida.txt

OBS: caso não queira, não é necessário passar o arquivo de saida quando for executar. ele apenas salva a matriz de distâncias minimas calculada pelo algoritmo

### PARALELO
Para rodar no Google Colab, primeiro é necessário subir o arquivo presente no caminho CUDA/paralelo.cu dentro do ambiente criado, bem como algum arquivo .txt que contenha o grafo no formato de matriz de adjacência. Feito isso, basta rodar os seguintes comandos para compilar e executar, respectivamente:
    
    !nvcc -arch=sm_75 -o paralelo paralelo.cu

    !./paralelo arquivo_entrada.txt arquivo_saida.txt

OBS: No colab, é necessário que o ambiente de execução selecionado seja: GPUs: T4
