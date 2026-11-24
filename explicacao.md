# Semana 1 — resultados do GPT-2 vanilla

Os valores abaixo vêm das capturas da execução oficial no Colab, em uma Tesla T4 com precisão mista FP16. O código correspondente está no [notebook da solução](atividade_solucao_121210320.ipynb).

## Tokenizador

`treinar_bpe` aprende um BPE em nível de byte com 16.000 tokens usando somente `train.txt`. Depois, `medir_fertilidade` aplica os três tokenizadores aos mesmos 20.722 documentos de treino e divide o total de tokens pelo total de palavras. Os marcadores de fim de documento ficam fora dessa comparação.

| Tokenizador | Tokens por palavra |
| --- | ---: |
| BPE próprio | 1,3720 |
| BPE de referência | 1,3720 |
| GPT-2 do `tiktoken` | 2,4247 |

O BPE próprio reproduziu a fertilidade da referência na precisão exibida e usou cerca de **43,4% menos tokens por palavra** que o GPT-2 do `tiktoken`. Em “ação”, os dois BPEs da atividade usaram um token e o `tiktoken` usou três. Os símbolos da tabela representam bytes do UTF-8. A demonstração de decodificação reconstruiu o texto inicial com os acentos e o hífen.

O BPE próprio foi treinado separadamente. O corpus, o tipo de BPE e o tamanho do vocabulário em comum ajudam a explicar a coincidência na fertilidade e nos exemplos exibidos.

## Preparação e treino

`pre_tokenizar` transforma treino e validação em IDs usando o BPE próprio. `GPTDatasetTokens` constrói janelas em que cada alvo é a entrada deslocada em um token. A conferência mostrou **13 janelas idênticas** às do dataset do livro. `preparar_dataloaders` montou lotes de 32 sequências de 256 tokens. O orçamento efetivo foi **3.351 passos × 8.192 tokens = 27.451.392 tokens** em uma época.

O `GPTModel` usou quatro camadas, quatro cabeças e dimensão de embedding 256, com **11.414.016 parâmetros**. `configurar_otimizacao` e `treinar_lote` aplicaram AdamW, weight decay nas matrizes, warmup linear, cosine decay, gradient clipping e FP16 com `GradScaler`. `treinar_modelo` avaliou o checkpoint inicial e registrou a validação a cada 10% do treino.

| Passo | Loss de validação | BPB de validação |
| ---: | ---: | ---: |
| 0 | 9,8571 | 2,8415 |
| 336 | 6,5616 | 1,8920 |
| 671 | 5,9250 | 1,7084 |
| 1.006 | 5,6168 | 1,6195 |
| 3.351 | **4,9191** | **1,4183** |

## Resultados finais e interpretação

| Métrica | Valor |
| --- | ---: |
| Loss de treino no último intervalo | 5,0377 |
| Loss de validação | 4,9191 |
| Perplexidade de validação | 136,88 |
| Bits por byte de validação | 1,4183 |
| Throughput de treino | 80.961,6 tokens/s |
| Pico de memória GPU | 2.516,9 MiB |
| Tempo de parede registrado pelo loop | 415,5 s |

`avaliar` calcula a perplexidade como a exponencial da loss de validação. Para o BPB, divide a soma da loss dos alvos de texto por `ln(2) × bytes UTF-8 avaliados`, excluindo preenchimento e marcadores sintéticos. A queda de **2,8415 para 1,4183 BPB** indica que o vanilla passou a atribuir mais probabilidade aos tokens observados na validação.

A curva mostra uma queda rápida no início e ganhos menores perto do fim da época. A loss de treino reúne os lotes do último intervalo enquanto os pesos ainda mudam e o dropout está ativo. A loss de validação usa o modelo ao final desse intervalo, em modo de avaliação. Essas diferenças ajudam a interpretar por que a curva de validação ficou abaixo da de treino.

O throughput usa o tempo acumulado dos lotes de treino. O tempo de parede registrado pelo loop também inclui as avaliações e as gravações feitas durante o treino. A comparação futura deve repetir o mesmo BPE, a mesma semente e o mesmo orçamento de tokens, usando **1,4183 BPB** como referência do vanilla.

## Arquivos da execução

- `tokenizador/` contém `fertilidade.csv`, `segmentacoes.csv` e `tokenizer.json`.
- `semana1_vanilla/` contém `metricas.csv`, `configuracao.json`, `resumo.json`, `curvas_vanilla.png` e `checkpoint.pt`.

Os CSVs e JSONs da execução no Drive são a fonte para conferir os valores com precisão completa e documentar o experimento no relatório.
