# GPT-2 do zero em português

O [notebook da solução](atividade_solucao_121210320.ipynb) segue as seções do [notebook-base da disciplina](guide/atividade_base_original.ipynb) e será incrementado a cada semana. A Semana 1 treina um BPE de 16 mil tokens e um GPT-2 vanilla baseado no código de Sebastian Raschka em `guide/base-sebastian/`.

---
## Executar a Semana 1 no Colab

1. Coloque o ZIP privado `corpus_ufcg_introducoes_alunos.zip` em **Meu Drive → Colab Notebooks**.
2. Abra `atividade_solucao_121210320.ipynb` no Colab. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione uma GPU.
3. Na primeira célula, ajuste `NUM_EPOCAS` se quiser mudar as duas épocas padrão. Os caminhos padrão são `Colab Notebooks/corpus_ufcg_introducoes_alunos.zip` para o ZIP e `Colab Notebooks/Atividade1_UFCG` para o projeto.
4. Clique em **Executar tudo** e autorize a montagem do Drive. O treino avalia o vanilla em `val.txt`. A última célula compara gerações antes do treino e após as épocas 1 e 2.

## Checkpoints e escolha do resultado

Com duas épocas, a rodada do vanilla fica em `Colab Notebooks/Atividade1_UFCG/semana1_vanilla_2_epocas/`. Uma época usa `semana1_vanilla/`. Cada rodada guarda `checkpoint.pt` para retomada periódica e `checkpoint_epoca_001.pt`, `checkpoint_epoca_002.pt` e assim por diante para os finais de época.

Para retomar, rode `executar_experimento("vanilla")` com a mesma configuração. O total de épocas define o cronograma de learning rate. Ao alterá-lo, o notebook usa a pasta da nova rodada. Para acessar o treino anterior que já estava em **Meu Drive → Atividade1_UFCG**, configure `PASTA_PROJETO_NO_DRIVE = "Atividade1_UFCG"` e `NUM_EPOCAS = 1`.

O resultado selecionado é o checkpoint de fim de época com a menor loss de validação. `resumo.json` registra a época, o nome do arquivo, o hash SHA-256, o BPB e a perplexidade do mesmo checkpoint. `metricas.csv` e a curva de loss cobrem a rodada inteira. Para a comparação com a variante, mantenha o BPE, a semente, o orçamento e o cronograma de learning rate iguais.

---
## Gerar texto com um checkpoint salvo

Em uma sessão nova, execute desde o início até a função `gerar_com_checkpoint` na seção **8. Checkpoint e retomada**. Se a sessão do treino ainda estiver aberta, vá direto à chamada abaixo. Em uma célula do Colab, rode:

```python
print(gerar_com_checkpoint("vanilla", "Este trabalho tem como objetivo"))
print(gerar_com_checkpoint("vanilla", "Este trabalho tem como objetivo", epoca=1))
```

A primeira chamada usa o checkpoint selecionado em `resumo.json`. A segunda usa o da época 1 da rodada configurada em `NUM_EPOCAS`. Ambas carregam o BPE salvo. Após registrar e treinar a arquitetura modificada, use `"modificado"` na mesma chamada.

Na sessão em que as funções da seção 10 já estiverem definidas, também é possível repetir a comparação com outro prompt usando `comparar_geracoes("A UFCG é uma Universidade")`. Se a rodada tiver uma única época, o cartão da segunda informa que o checkpoint está indisponível.

---
