# Registro experimental — PharmXAI-3D

Notas de bancada das medições feitas em 04/09/2026. O `README.md` descreve o que o
modelo é e faz; este documento registra **como os números foram obtidos, o que foi
testado e descartado, e o que cada resultado autoriza a afirmar**.

A ordem é cronológica porque cada etapa nasceu da anterior.

---

## Resumo

| | R (CASF-2016 core) | RMSE | origem |
|---|---|---|---|
| ponto de partida (mean pool global) | 0,680 | 1,612 | rodada única |
| **após a correção do readout** | **0,754** | **1,486** | rodada de referência¹ |
| o mesmo, média de 3 sementes | 0,753 ± 0,016 | 1,463 ± 0,040 | 3 sementes |
| **o mesmo modelo, sem homólogos no treino** | **0,692** ± 0,014 | 1,600 ± 0,028 | 3 sementes |
| baseline linear sobre 8 contagens | 0,592 | 1,837 | determinístico |
| mediana das 34 funções clássicas do CASF-2016 | 0,537 | — | pacote CASF |

¹ o checkpoint em `pharm_model_weights_best.pth`, que é o que o `README.md` reporta.
Ele cai dentro da dispersão das três sementes, como se espera.

As duas últimas linhas são as referências que dão sentido às primeiras.

**Os dois números honestos do modelo são 0,753 e 0,692.** O primeiro mede prever um
ligante novo numa proteína **conhecida**; o segundo, num alvo **novo**. Eles diferem
porque o benchmark não contém nenhum alvo novo — ver a seção 3.

---

## 1. O gargalo era o readout, não overfitting

### O sintoma enganoso

A loss de treino caía para 0,57 enquanto a validação estacionava em 2,4. A leitura
óbvia era overfitting, e ela estava errada. Duas medições derrubaram essa hipótese:

1. **O R de *treino*, em modo `eval`, era 0,720.** Memorização daria 0,95 ou mais. O
   valor de 0,57 do log é a loss corrente com dropout ativo, de uma época posterior à
   que o early stopping selecionou — não descreve o checkpoint que se usa.
2. **Não havia shift de distribuição**: treino 6,401 ± 1,797 contra validação
   6,381 ± 1,942.

Dropout (0,2) e weight decay (1e-4) já estavam ativos. Aumentá-los teria derrubado o R
de treino sem levantar o de validação.

### O teste que fechou o caso

Uma regressão linear sobre **8 features de contagem** — número de átomos do ligante e
do bolso, número de contatos, distância média e mínima, contatos por átomo, e os logs
de dois desses — atinge **R = 0,592** no CASF-2016 core.

O GNN inteiro fazia 0,680. **Toda a maquinaria geométrica valia 0,088 de R.** O
gargalo era representacional.

### A causa

`global_mean_pool` sobre a união dos átomos, com dois defeitos:

- **Diluição.** O ligante tem 33 átomos e o bolso 130 (razão 1:3,9). Numa média sobre
  a união, o fármaco — a parte que varia entre complexos de um mesmo alvo — entra com
  peso ¼.
- **A média destrói o preditor mais forte disponível.** Afinidade cresce com o número
  de contatos favoráveis, e `log(n_contatos)` sozinho correlaciona +0,349 com pKd. Uma
  média normaliza exatamente por tamanho, obrigando o modelo a reconstruir esse sinal
  por vias indiretas.

### A correção

A classe `ReadoutPorPapel` em `gnn_model.py`: média **e** soma, separadas por papel
(ligante / proteína), mais três descritores em log (n_lig, n_prot, n_contatos). São
259 dimensões contra 64, com a cabeça alargada para 128→64→1. De 83 mil para 127 mil
parâmetros. A soma passa por `LayerNorm` porque escala com o tamanho do grafo (90 a
256 átomos) e alimentaria o MLP com ativações de magnitude muito variável.

| split | R antes | R depois | RMSE antes | RMSE depois |
|---|---|---|---|---|
| treino | 0,720 | 0,786 | 1,260 | 1,127 |
| validação | 0,616 | 0,667 | 1,534 | 1,448 |
| **CASF-2016 core** | **0,680** | **0,754** | **1,612** | **1,486** |

O ganho aparece nos três splits, o que descarta sorte na seleção do checkpoint. E a
margem sobre o baseline de contagem quase dobrou: de 0,088 para 0,162 de R — a
geometria passou a carregar o dobro do peso.

Métricas completas do modelo resultante no CASF core: Pearson R 0,754 · Spearman ρ
0,746 · RMSE 1,486 · MAE 1,185 · slope 0,424.

---

## 2. IC50 no conjunto de treino: duas correções, ambas rejeitadas

**O problema.** 51,3% do conjunto de treino é IC50, e validação e teste têm 0% — o
refined set exclui IC50 por critério de qualidade. IC50 depende das condições do
ensaio (concentração de substrato, pH, tempo) e não é constante termodinâmica, então
metade do sinal de treino carrega um viés sistemático ausente onde o modelo é avaliado.

Isso escapa a qualquer verificação de shift no alvo: as distribuições marginais de pKd
são quase idênticas.

| variante | treino R | val R | CASF R | CASF RMSE |
|---|---|---|---|---|
| readout por papel (referência) | 0,786 | 0,667 | **0,754** | **1,486** |
| IC50 descartado do treino | 0,658 | 0,643 | 0,726 | 1,521 |
| token de tipo de ensaio | 0,762 | 0,691 | 0,775 | 1,373 |
| *o mesmo, com o token forçado a Kd* | 0,743 | 0,672 | 0,756 | 1,478 |

**Descartar piora.** O treino cai para 6.678 complexos e o CASF vai a 0,726. Perder
7.033 complexos custa mais cobertura química do que o viés de rótulo custa em ruído.

**O token parece ganhar, mas não ganha.** As duas últimas linhas são o **mesmo
modelo**, mudando apenas o token usado na inferência. Com o tipo verdadeiro dá 0,775;
com tudo forçado a Kd dá 0,756 — praticamente o 0,754 de não ter token nenhum.

O embedding não melhorou a generalização: ele **calibra um deslocamento quando o tipo
de ensaio já é conhecido**. Medido no CASF, Ki desloca +0,32 e IC50 −0,11 em relação a
Kd. O viés de ensaio existe e tem o sinal esperado, mas isso não é capacidade
preditiva nova.

E o caso de uso decide: em `predict_custom.py`, com um ligante novo, **não existe tipo
de ensaio a informar** — o composto ainda não foi medido. Por isso o número reportado
é 0,754, sem token, e `usar_assay` fica desligado por padrão.

---

## 3. Homologia entre treino e teste: o benchmark não tem alvo novo

### A pergunta

O R medido reflete prever afinidade, ou reconhecer a proteína? O `train.py` faz o
corte hermético corretamente — o core set fica fora do treino e da seleção de
checkpoint — mas o general set pode conter a **mesma proteína** com outro ligante.

### A medição

Para cada um dos 285 complexos do core, a similaridade à proteína mais parecida entre
os 13.711 do treino, por *containment* de 4-mers de aminoácidos (a fração dos 4-mers
da proteína de teste que também aparecem na de treino; containment em vez de Jaccard
porque é robusto a diferença de comprimento).

| percentil | similaridade |
|---|---|
| **mínimo** | **0,803** |
| 25 | 0,985 |
| mediana | 0,996 |
| 75 e acima | 1,000 |

99% acima de 0,90. **62% acima de 0,99.** O complexo mais isolado do benchmark inteiro
ainda compartilha 80% dos seus 4-mers com uma proteína de treino.

**Não existe subgrupo "sem parente no treino".** O teste estratificado por faixa de
homologia é impossível neste split — e essa impossibilidade é a resposta.

### Validação da métrica

Containment alto poderia ser saturação do espaço de k-mers, não homologia. Dois
controles negativos, ambos passados com folga:

| | k=4 | k=6 | k=8 |
|---|---|---|---|
| core vs. melhor do treino (valor real) | 0,996 | 0,993 | 0,991 |
| core **embaralhado** vs. treino | 0,044 | 0,004 | 0,000 |
| pares aleatórios treino–treino | 0,003 | 0,000 | 0,000 |

O embaralhamento preserva a composição de aminoácidos e destrói a ordem — é o controle
forte. Com 8-mers, ambos os controles vão a zero exato enquanto o valor real fica em
0,991.

### Quanto o modelo aproveita disso

Removendo do treino os 2.812 complexos homólogos ao core (restam 10.899) e comparando
com um **controle de 10.899 sorteados ao acaso**. O controle é indispensável: sem ele,
qualquer queda se confundiria com ter 20% menos dados.

Três seeds, comparação pareada — a mesma seed dá a mesma inicialização e a mesma ordem
de lotes nas três configurações, de modo que a diferença entre elas é a configuração,
não o sorteio.

| configuração | seed 1 | seed 2 | seed 3 | média |
|---|---|---|---|---|
| treino completo (13.711) | 0,736 | 0,768 | 0,755 | 0,753 |
| controle aleatório (10.899) | 0,726 | 0,749 | 0,759 | 0,745 |
| sem homólogos (10.899) | 0,676 | 0,700 | 0,699 | **0,692** |

| efeito | por seed | média | IC95% | p | veredito |
|---|---|---|---|---|---|
| 20% menos dados | +0,010 · +0,019 · −0,004 | +0,008 ± 0,012 | [−0,020, +0,037] | 0,34 | **nulo** (sinal inconsistente) |
| **homologia** | +0,050 · +0,049 · +0,060 | **+0,053 ± 0,006** | [+0,038, +0,068] | **0,004** | **significativo** |

### Duas conclusões que não se confundem

**O benchmark é viciado.** Nenhum dos 285 alvos é novo para o modelo. Isso vale para
todo método treinado em PDBbind, e por isso a comparação com as 34 funções clássicas
do CASF — que nunca viram esses dados — favorece indevidamente os métodos de ML. A
comparação *entre* métodos de ML permanece justa, porque todos herdam o mesmo viés.

**Mas o modelo não depende do vício.** Sem nenhum homólogo no treino ele ainda entrega
R = 0,692, acima do baseline de contagem (0,592) e da mediana das funções clássicas
(0,537). O que ele aprendeu sobrevive à remoção do atalho mais óbvio.

**Efeito colateral útil:** remover 20% dos complexos ao acaso não muda nada
detectável. O modelo já saturou nos dados disponíveis — mais complexos do PDBbind não
levariam mais longe.

---

## 4. Regras de medição que emergiram

Registradas porque cada uma custou uma conclusão errada antes de ser aprendida.

**Diferença abaixo de ~0,03 em R exige múltiplas seeds pareadas.** A mesma
configuração, mudando apenas a seed, varia de 0,736 a 0,768 — amplitude de 0,032, da
mesma ordem dos efeitos investigados. A rodada única estimou o efeito da homologia em
−0,026; o valor real é −0,053, **metade**. O pareamento é o que salva a medição: o
desvio das diferenças pareadas é ±0,006, contra ±0,016 dos valores absolutos.

**Antes de chamar um gap treino/validação de overfitting, meça o R de treino em modo
`eval`.** A loss de treino do log é computada com dropout ativo e pesos em movimento;
ela não descreve o checkpoint selecionado.

**Compare sempre contra um baseline trivial.** Foi uma regressão linear sobre contagem
de átomos que localizou o gargalo real, depois de o diagnóstico intuitivo ter apontado
para o lugar errado.

**Toda remoção de dados precisa de um controle de mesmo tamanho.** Sem ele, o efeito
da variável se confunde com o efeito de ter menos exemplos.

**Um ganho que só aparece quando uma informação privilegiada é fornecida na inferência
não é um ganho.** Foi o caso do token de ensaio: teste forçando o valor que estaria
disponível no uso real.

---

## 5. Como reproduzir

`train.py` é controlado por variáveis de ambiente, todas opcionais:

| variável | efeito |
|---|---|
| `PHARM_CKPT` | nome do checkpoint de saída (padrão `pharm_model_weights_best.pth`) |
| `PHARM_SEED` | semente de inicialização e de ordem dos lotes (padrão 0) |
| `PHARM_IC50` | `keep` (padrão), `drop` ou `token` |
| `PHARM_TREINO_IDS` | caminho de um JSON com a lista de `pdb_id` a que o treino fica restrito |

```bash
# referência
python train.py

# sem os homólogos do CASF core, semente 1
PHARM_SEED=1 PHARM_TREINO_IDS=treino_sem_homologos_0.3.json \
  PHARM_CKPT=exp_sem_homologos_s1.pth python train.py

# token de tipo de ensaio
PHARM_IC50=token PHARM_CKPT=exp_token.pth python train.py
```

Os logs de todas as rodadas citadas estão em `logs/`. Os checkpoints de experimento
não são versionados (`*.pth` no `.gitignore`); apenas
`pharm_model_weights_best.pth`, que corresponde à linha de referência.

---

## 6. O que fica em aberto

**O slope de 0,424** é a limitação mais séria que resta. Um preditor ideal daria 1,0;
a 0,424 o modelo comprime as previsões na direção da média, subestimando ligantes
fortes e superestimando fracos — prevendo 2,83–10,67 onde a verdade vai de 2,07 a
11,82. É por isso que o `README.md` diz que o modelo serve para **ordenar uma
biblioteca, não para reportar afinidade absoluta**. Atacar a compressão seria o próximo
ganho real, e é independente de tudo o que foi feito aqui.

**Avaliação em alvo novo, de forma sistemática.** O número de 0,692 vem de uma única
configuração de remoção. Um protocolo de validação cruzada por cluster de proteína
daria uma estimativa com incerteza, em vez de um ponto.

**Eficiência de parâmetros.** 127 mil parâmetros contra os milhões de Pafnucy
(≈1,5 M) e OnionNet (>10 M), que reportam 0,78–0,82 — mas na condição fácil, sem terem
sido testados na difícil. Essa comparação, feita em pé de igualdade, é o resultado
potencialmente mais interessante do projeto e ainda não foi feita.
