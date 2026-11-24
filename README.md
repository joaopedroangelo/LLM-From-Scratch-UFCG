# GPT-2 do zero em português

Atividade 1 da UFCG. O [notebook da solução](atividade_solucao_121210320.ipynb) segue as seções do [notebook-base da disciplina](guide/atividade_base_original.ipynb) e será incrementado a cada semana. A Semana 1 treina um BPE de 16 mil tokens e um GPT-2 vanilla baseado no código de Sebastian Raschka em `guide/base-sebastian/`.

## Executar a Semana 1 no Colab

1. Obtenha o ZIP privado `corpus_ufcg_introducoes_alunos.zip` pelo canal indicado pelo professor e coloque-o em **Meu Drive**.
2. Abra `atividade_solucao_121210320.ipynb` no Colab. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione uma GPU.
3. Na primeira célula de código, ajuste `ZIP_NO_DRIVE` se o arquivo estiver em uma subpasta do Drive.
4. Clique em **Executar tudo** e autorize a montagem do Drive. A última célula compara os tokenizadores, treina ou retoma o vanilla e gera a avaliação em `val.txt`.

### Comando por experimento no Colab

Na seção **10. Conclusão da Semana 1**, a última célula chama o vanilla. Para executar outro experimento após registrar sua arquitetura, rode as células até a definição de `executar_experimento` e chame o comando correspondente.

| Experimento | Comando em uma célula do Colab | Saída em `PASTA_PROJETO_NO_DRIVE` |
| --- | --- | --- |
| GPT-2 vanilla | `executar_experimento("vanilla")` | `semana1_vanilla/` |
| GPT-2 modificado, após registrar a arquitetura na Semana 2 | `executar_experimento("modificado")` | `modificado/` |

Para retomar após uma desconexão, mantenha a configuração, o nome do experimento e `PASTA_PROJETO_NO_DRIVE`. Para iniciar uma rodada do zero, escolha outra pasta de projeto no Drive. Os checkpoints que já estavam em `Atividade1_UFCG/semana1_vanilla/` podem ser retomados na configuração padrão.

## Gerar texto com um checkpoint salvo

Em uma sessão nova, execute desde o início até a função `gerar_com_checkpoint` na seção **8. Checkpoint e retomada**. Se a sessão do treino ainda estiver aberta, vá direto à chamada abaixo. Em uma célula do Colab, rode:

```python
print(gerar_com_checkpoint("vanilla", "Este trabalho tem como objetivo"))
```

A geração usa o `checkpoint.pt` do experimento e o `tokenizador/tokenizer.json` com a assinatura registrada no checkpoint. Depois de registrar e treinar a arquitetura modificada, use `"modificado"` na mesma chamada. O carregamento e a geração funcionam independentemente da execução da seção de treino.

## Resultados no Drive

O BPE compartilhado e a comparação dos tokenizadores ficam em `tokenizador/`, com `tokenizer.json`, `fertilidade.csv` e `segmentacoes.csv`. Cada experimento salva `checkpoint.pt`, `configuracao.json`, `metricas.csv`, `resumo.json` e seu gráfico de loss na própria pasta. A tabela final exibe os valores gravados em `resumo.json`, incluindo BPB, perplexidade, parâmetros, throughput, memória e tempo.

No checkpoint da Semana 1, apresente a curva do vanilla e o BPB de validação. Versione as configurações, os CSVs e o resumo da execução oficial em uma pasta de resultados do repositório. Guarde os checkpoints no Drive. Na entrega final, registre no README os SHA-256 dos checkpoints finais e inclua `evaluate.py`, que recebe um arquivo de texto e calcula o BPB.

O corpus tem acesso restrito. Guarde cópias locais em `dataset/` e resultados de testes locais em `outputs/`. O Git ignora as duas pastas.
