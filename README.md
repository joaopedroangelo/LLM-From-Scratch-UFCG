# GPT-2 do zero em português

O [notebook da solução](atividade_solucao_121210320.ipynb) segue as seções do [notebook-base da disciplina](guide/atividade_base_original.ipynb) e será incrementado a cada semana. A Semana 1 treina um BPE de 16 mil tokens e um GPT-2 vanilla baseado no código de Sebastian Raschka em `guide/base-sebastian/`.

---
## Executar a Semana 1 no Colab

1. Obtenha o ZIP privado `corpus_ufcg_introducoes_alunos.zip`;
2. Abra `atividade_solucao_121210320.ipynb` no Colab. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione uma GPU;
3. Na primeira célula de código, ajuste `ZIP_NO_DRIVE` se o arquivo estiver em uma subpasta do Drive;
4. Clique em **Executar tudo** e autorize a montagem do Drive. A última célula compara os tokenizadores, treina ou retoma o vanilla e gera a avaliação em `val.txt`.

---
## Gerar texto com um checkpoint salvo

Em uma sessão nova, execute desde o início até a função `gerar_com_checkpoint` na seção **8. Checkpoint e retomada**. Se a sessão do treino ainda estiver aberta, vá direto à chamada abaixo. Em uma célula do Colab, rode:

```python
print(gerar_com_checkpoint("vanilla", "Este trabalho tem como objetivo"))
```

A geração usa o `checkpoint.pt` do experimento e o `tokenizador/tokenizer.json` com a assinatura registrada no checkpoint. Depois de registrar e treinar a arquitetura modificada, use `"modificado"` na mesma chamada. O carregamento e a geração funcionam independentemente da execução da seção de treino.

---
