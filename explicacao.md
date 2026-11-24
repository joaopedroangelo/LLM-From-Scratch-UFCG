# Melhorias no loop de treinamento do GPT-2

Este texto explica as melhorias feitas no loop de treinamento da atividade, comparando três versões:

1. o loop simples do capítulo 5 do livro *Build a Large Language Model (From Scratch)*;
2. o loop melhorado apresentado no Apêndice D;
3. o loop implementado em `atividade_solucao_121210320.ipynb`.

A ideia principal é que o loop do capítulo 5 é adequado para ensinar o funcionamento básico do treinamento. O Apêndice D acrescenta técnicas que deixam a otimização mais estável. A solução da atividade aproveita essas técnicas e adiciona a infraestrutura necessária para treinar na GPU do Colab sem perder o trabalho quando a sessão é interrompida.

## 1. Como funciona o loop original

O loop do capítulo 5 segue este processo:

```python
for input_batch, target_batch in train_loader:
    optimizer.zero_grad()
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    loss.backward()
    optimizer.step()
```

Em cada lote, ele:

1. apaga os gradientes calculados no lote anterior;
2. executa o modelo e calcula o erro, chamado de `loss`;
3. calcula os gradientes por meio de `loss.backward()`;
4. usa o otimizador para atualizar os parâmetros do modelo.

O otimizador é criado desta forma:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1,
)
```

Esse loop funciona, mas possui algumas limitações:

- a taxa de aprendizado permanece constante;
- o `weight decay` é aplicado igualmente a todos os parâmetros;
- não existe proteção contra gradientes muito grandes;
- o treinamento usa precisão completa, consumindo mais memória;
- o loop não consegue retomar exatamente do ponto em que parou;
- não registra todas as métricas exigidas pela atividade.

## 2. O que é a taxa de aprendizado

A taxa de aprendizado, ou *learning rate*, controla o tamanho de cada alteração feita nos parâmetros do modelo.

Uma analogia simples é imaginar uma pessoa descendo uma montanha para encontrar o ponto mais baixo:

- uma taxa muito alta corresponde a passos grandes; a pessoa pode atravessar o ponto mais baixo e ficar oscilando de um lado para o outro;
- uma taxa muito baixa corresponde a passos pequenos; a pessoa avança com segurança, mas demora muito;
- uma taxa adequada permite avançar rapidamente no início e fazer ajustes menores quando já está perto de uma boa solução.

No treinamento, o otimizador usa os gradientes para decidir a direção da atualização. A taxa de aprendizado determina o tamanho dessa atualização.

## 3. Warmup linear

### O que é

O *warmup* começa o treinamento com uma taxa de aprendizado pequena e aumenta essa taxa gradualmente até o valor máximo planejado.

Na solução, os valores são:

```python
LR_INICIAL = 3e-5
LR_PICO = 4e-4
FRACAO_WARMUP = 0.10
```

Isso significa que, durante os primeiros 10% do treinamento, a taxa sobe linearmente de `0,00003` para `0,0004`.

```text
0,00003 ───────────────> 0,0004
          warmup
     primeiros 10% dos passos
```

### Por que é necessário aquecer a taxa no início

No começo do treinamento, os parâmetros do modelo são essencialmente aleatórios. As previsões são ruins e os gradientes podem variar bastante entre os primeiros lotes.

Se o treinamento começar imediatamente com uma taxa alta, uma dessas primeiras atualizações pode alterar demais os parâmetros. Isso pode causar:

- oscilações fortes na loss;
- instabilidade numérica;
- gradientes ou valores infinitos;
- perda de informações úteis que estavam presentes na inicialização dos pesos.

O warmup funciona como um período de adaptação. As primeiras atualizações são pequenas; à medida que o modelo começa a encontrar uma direção útil, a taxa aumenta.

### Código da solução

```python
if atualizacao < warmup:
    lr = inicial + (pico - inicial) * atualizacao / warmup
```

Quando `atualizacao` é zero, a taxa está próxima de `inicial`. A cada atualização, ela se aproxima de `pico`.

O loop original do capítulo 5 não possui warmup. Essa técnica vem do Apêndice D e foi adaptada na solução.

## 4. Cosine decay

### O que é

Depois do warmup, não é interessante manter a taxa no valor máximo até o fim. Conforme o modelo aprende, as atualizações devem ficar mais cuidadosas.

O *cosine decay* reduz suavemente a taxa de aprendizado usando o formato de uma curva de cosseno.

Na solução:

```python
LR_PICO = 4e-4
LR_MINIMO = 4e-5
```

Depois de chegar a `0,0004`, a taxa diminui gradualmente até `0,00004`.

```text
Taxa de aprendizado

0,0004              .
                    / \
                   /   \
                  /     `-.
0,00003  _________/         `------ 0,00004
          warmup             cosine decay
```

### Por que reduzir a taxa

No início, o modelo está longe de uma boa solução e pode dar passos maiores. Perto do final, passos grandes podem destruir pequenos ajustes que já foram aprendidos.

O cosine decay permite:

- aprender rapidamente na parte inicial;
- reduzir oscilações perto do final;
- fazer ajustes mais finos nos parâmetros;
- terminar o treinamento com uma taxa pequena, mas não necessariamente zero.

### Código da solução

```python
progresso = min(
    1.0,
    (atualizacao - warmup)
    / max(1, config["passos"] - 1 - warmup),
)

lr = minimo + (pico - minimo) * (
    1 + math.cos(math.pi * progresso)
) / 2
```

A função completa usada pelo scheduler é:

```python
def fator_lr(atualizacao, config):
    warmup = config["passos_warmup"]
    inicial = config["lr_inicial"]
    pico = config["lr_pico"]
    minimo = config["lr_minimo"]

    if atualizacao < warmup:
        lr = inicial + (pico - inicial) * atualizacao / warmup
    else:
        progresso = min(
            1.0,
            (atualizacao - warmup)
            / max(1, config["passos"] - 1 - warmup),
        )
        lr = minimo + (pico - minimo) * (
            1 + math.cos(math.pi * progresso)
        ) / 2

    return lr / pico
```

O retorno é dividido por `pico` porque o `LambdaLR` espera um fator multiplicativo. O otimizador possui `lr=pico`, e o scheduler multiplica esse valor pelo fator retornado.

## 5. Weight decay

### O que é

Durante o treinamento, os pesos podem crescer demais. Valores muito grandes podem fazer o modelo se adaptar excessivamente aos dados de treino e dificultar sua capacidade de funcionar bem em textos que não estavam no treinamento.

O `weight decay` desencoraja pesos muito grandes. A cada atualização, ele exerce uma pequena força que puxa os pesos em direção a zero.

Uma forma intuitiva de pensar nisso é:

```text
atualização normal:       aprende com o gradiente
weight decay:             evita que os pesos cresçam sem controle
resultado:                aprendizado com uma regularização suave
```

O objetivo não é zerar os pesos, mas evitar valores desnecessariamente grandes.

### Por que não aplicar em todos os parâmetros

Nem todos os parâmetros possuem a mesma função:

- matrizes de embeddings e camadas lineares representam os pesos principais do modelo;
- vieses deslocam os resultados das camadas;
- escalas e deslocamentos da normalização controlam a normalização das ativações.

Aplicar weight decay aos vieses e aos parâmetros da normalização normalmente não oferece o mesmo benefício e pode atrapalhar o trabalho dessas camadas.

Por isso, a solução separa os parâmetros em dois grupos:

```python
decay = [
    p for p in modelo.parameters()
    if p.ndim >= 2
]

sem_decay = [
    p for p in modelo.parameters()
    if p.ndim < 2
]
```

Depois cria o AdamW assim:

```python
otimizador = torch.optim.AdamW(
    [
        {
            "params": decay,
            "weight_decay": config["weight_decay"],
        },
        {
            "params": sem_decay,
            "weight_decay": 0.0,
        },
    ],
    lr=config["lr_pico"],
)
```

Na prática:

| Parâmetros | Exemplos | Weight decay |
|---|---|---:|
| Matrizes, `p.ndim >= 2` | embeddings e pesos das camadas lineares | `0.1` |
| Vetores, `p.ndim < 2` | vieses e parâmetros da normalização | `0.0` |

No loop original, `weight_decay=0.1` é aplicado a todos os parâmetros. O Apêndice D melhora o cronograma da taxa e os gradientes, mas também não faz essa separação. A separação explícita é uma melhoria adicional da solução e atende ao enunciado da atividade.

## 6. Gradient clipping

### O que são gradientes

O gradiente indica quanto e em qual direção cada parâmetro deve mudar para reduzir a loss.

Em alguns lotes, principalmente no início do treinamento, os gradientes podem ficar muito grandes. Nesse caso, uma única atualização pode modificar os pesos de maneira exagerada. Esse fenômeno é conhecido como *exploding gradients*.

### O que o clipping faz

O gradient clipping calcula o tamanho global dos gradientes. Se esse tamanho ultrapassar o limite configurado, todos os gradientes são reduzidos proporcionalmente.

Na solução, o limite é:

```python
CLIP_GRAD = 1.0
```

E a aplicação é:

```python
nn.utils.clip_grad_norm_(
    modelo.parameters(),
    config["clip_grad"],
)
```

O clipping não muda a direção conjunta dos gradientes. Ele apenas diminui o tamanho da atualização quando a norma ultrapassa o limite.

Uma analogia é um limitador de velocidade: o veículo continua indo na mesma direção, mas não pode ultrapassar uma velocidade considerada perigosa.

Na solução, o clipping começa depois do warmup:

```python
if estado["atualizacoes"] >= config["passos_warmup"]:
    ...
```

Esse comportamento acompanha a proposta do Apêndice D. O loop original do capítulo 5 não possui gradient clipping.

## 7. Precisão mista com FP16

### O que é

Normalmente, os números do modelo podem ser representados em 32 bits, usando `float32`. A precisão mista permite executar várias operações em 16 bits, usando `float16`, enquanto mantém em maior precisão as operações que precisam disso.

Na solução, o cálculo do modelo e da loss ocorre dentro de `autocast`:

```python
with autocast(dispositivo_modelo, scaler.is_enabled()):
    loss = calc_loss_batch(
        lote[0],
        lote[1],
        modelo,
        dispositivo_modelo,
    )
```

Os principais benefícios são:

- menor consumo de memória da GPU;
- possibilidade de usar lotes maiores;
- maior velocidade em GPUs que possuem suporte eficiente a FP16, como a T4.

### Por que é necessário o GradScaler

O FP16 representa uma faixa menor de valores. Gradientes muito pequenos podem ser arredondados para zero, fenômeno chamado de *underflow*.

O `GradScaler` multiplica temporariamente a loss por uma escala. Isso torna os gradientes maiores durante o `backward`, evitando que desapareçam. Antes da atualização, a escala é desfeita.

```python
scaler.scale(loss).backward()
scaler.unscale_(otimizador)
```

É importante desfazer a escala antes do gradient clipping. Caso contrário, o clipping mediria gradientes artificialmente grandes.

Depois disso, a atualização é feita por:

```python
scaler.step(otimizador)
scaler.update()
```

Se o scaler detectar `inf` ou `NaN`, ele pode ignorar a atualização dos pesos e reduzir sua escala.

O loop original e o Apêndice D não utilizam precisão mista. Essa é uma melhoria acrescentada pela solução e também uma exigência do enunciado.

## 8. Scheduler sincronizado com atualizações válidas

Como uma atualização pode ser ignorada pelo `GradScaler`, o scheduler não deve avançar automaticamente em todos os lotes. Se avançasse, o learning rate mudaria mesmo quando os pesos permanecessem iguais.

A solução compara a escala antes e depois do passo:

```python
escala_anterior = scaler.get_scale()

scaler.step(otimizador)
scaler.update()

if scaler.get_scale() >= escala_anterior:
    estado["atualizacoes"] += 1
    scheduler.step()
```

Quando a escala diminui, significa que o GradScaler encontrou um problema numérico e ignorou a atualização. Nesse caso, o contador de atualizações e o scheduler não avançam.

Isso mantém sincronizados:

- os pesos do modelo;
- o número de atualizações reais;
- o warmup;
- o cosine decay.

## 9. Código do treinamento de um lote

O núcleo implementado na solução é:

```python
def treinar_lote(
    modelo,
    lote,
    otimizador,
    scheduler,
    scaler,
    estado,
    config,
):
    dispositivo_modelo = next(modelo.parameters()).device
    lr = otimizador.param_groups[0]["lr"]

    otimizador.zero_grad(set_to_none=True)

    with autocast(dispositivo_modelo, scaler.is_enabled()):
        loss = calc_loss_batch(
            lote[0],
            lote[1],
            modelo,
            dispositivo_modelo,
        )

    scaler.scale(loss).backward()

    if estado["atualizacoes"] >= config["passos_warmup"]:
        scaler.unscale_(otimizador)
        nn.utils.clip_grad_norm_(
            modelo.parameters(),
            config["clip_grad"],
            error_if_nonfinite=not scaler.is_enabled(),
        )

    escala_anterior = scaler.get_scale()
    scaler.step(otimizador)
    scaler.update()

    if scaler.get_scale() >= escala_anterior:
        estado["atualizacoes"] += 1
        scheduler.step()

    return loss.detach().item(), lr
```

Em termos simples, essa função:

1. apaga os gradientes anteriores;
2. calcula a loss usando FP16 quando habilitado;
3. calcula gradientes com uma escala de segurança;
4. desfaz a escala;
5. limita gradientes muito grandes;
6. atualiza os pesos;
7. confirma que a atualização foi válida;
8. avança o cronograma do learning rate.

## 10. Checkpoint e retomada

Uma sessão gratuita do Colab pode ser encerrada antes do fim. Salvar somente os pesos do modelo não é suficiente para continuar exatamente do mesmo ponto.

A solução salva:

```python
checkpoint = {
    "config": config,
    "modelo": modelo.state_dict(),
    "otimizador": otimizador.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "estado": estado,
    "rng_cpu": torch.get_rng_state(),
    "rng_cuda": torch.cuda.get_rng_state(dispositivo_modelo),
}
```

Cada item é necessário:

- `modelo`: contém os parâmetros aprendidos;
- `otimizador`: contém os estados internos do AdamW;
- `scheduler`: contém a posição no warmup ou cosine decay;
- `scaler`: contém a escala atual usada pelo FP16;
- `estado`: contém passo, métricas, tempos e posição do treinamento;
- `rng_cpu` e `rng_cuda`: preservam os estados aleatórios para a retomada ser reprodutível.

Ao retomar, a solução também usa apenas os índices dos lotes que ainda não foram processados:

```python
indices = ordem_treino[inicio * lote:fim * lote]
```

Assim, o primeiro lote após a retomada é exatamente o próximo lote do treinamento interrompido.

O Apêndice D não implementa essa infraestrutura de checkpoint. O notebook-base apresenta funções simples de salvamento, mas a solução amplia essas funções para cobrir scheduler, scaler, aleatoriedade e histórico.

## 11. Avaliação periódica

Antes do primeiro passo, a solução mede o desempenho do modelo ainda não treinado. Depois, repete a validação a cada 10% do treinamento:

```python
marcos = {
    math.ceil(config["passos"] * i / 10)
    for i in range(1, 11)
}
```

Isso permite observar a curva completa, e não apenas o resultado final.

Durante a validação são calculados:

- loss de validação;
- perplexidade;
- bits por byte, ou BPB;
- quantidade de tokens e bytes avaliados.

A perplexidade é calculada por:

```python
perplexidade = math.exp(loss)
```

Ela fornece uma interpretação da incerteza média do modelo, mas depende do tokenizador. Por isso, quando tokenizadores diferentes são comparados, o BPB é uma medida mais apropriada.

O BPB mede quantos bits o modelo precisa, em média, para representar cada byte do texto. Quanto menor o BPB, melhor o modelo está atribuindo probabilidade ao texto observado.

## 12. Métricas de desempenho

Além da qualidade do modelo, a solução registra o custo do treinamento:

- tokens processados por segundo;
- tempo gasto apenas nos lotes de treino;
- tempo total de parede, incluindo validações e checkpoints;
- pico de memória alocada na GPU;
- atualizações ignoradas pelo GradScaler.

Essas métricas são importantes porque duas arquiteturas podem ter loss parecida, mas custos de execução muito diferentes.

## 13. Comparação direta

| Recurso | Capítulo 5 original | Apêndice D | Solução da atividade |
|---|:---:|:---:|:---:|
| AdamW | Sim | Sim | Sim |
| Weight decay apenas nas matrizes | Não | Não | Sim |
| Warmup linear | Não | Sim | Sim |
| Cosine decay | Não | Sim | Sim |
| Gradient clipping | Não | Sim | Sim |
| FP16 com `autocast` | Não | Não | Sim |
| `GradScaler` | Não | Não | Sim |
| Scheduler acompanha apenas atualizações válidas | Não | Não | Sim |
| Avaliação inicial e a cada 10% | Não | Não | Sim |
| Validação completa com a última janela parcial | Não | Não | Sim |
| Checkpoint do modelo e do otimizador | Fora do loop | Não | Sim |
| Checkpoint do scheduler e do scaler | Não | Não | Sim |
| Preservação do estado aleatório | Não | Não | Sim |
| Retomada no lote correto | Não | Não | Sim |
| Throughput e memória de pico | Não | Não | Sim |
| BPB | Não | Não | Sim |

## 14. Resumo final

O capítulo 5 apresenta o mecanismo fundamental de treinamento: calcular a loss, obter os gradientes e atualizar os parâmetros.

O Apêndice D melhora a otimização com:

- warmup linear;
- cosine decay;
- gradient clipping.

A solução da atividade mantém essas três técnicas e adiciona:

- weight decay somente nos parâmetros apropriados;
- precisão mista FP16;
- GradScaler para proteger gradientes pequenos;
- sincronização do scheduler com atualizações válidas;
- checkpoint completo;
- retomada exata do treinamento;
- validação periódica;
- loss, perplexidade e BPB;
- throughput, memória e tempo de parede.

Portanto, a melhoria não é apenas uma mudança na fórmula de atualização dos pesos. O novo loop foi construído para ser mais estável, mais rápido, reprodutível, mensurável e resistente às interrupções comuns das sessões gratuitas de GPU.
