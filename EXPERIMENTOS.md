# Registro experimental — PharmXAI-3D

Notas de bancada das medições feitas em 04/09/2026 (seções 1–3) e 09/09/2026
(seções 4 a 6). O `README.md` descreve o que o modelo é e faz; este documento registra
**como os números foram obtidos, o que foi testado e descartado, e o que cada
resultado autoriza a afirmar**.

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

Tudo acima é **scoring power**, e ele não implica as demais capacidades. Medido em
09/09/2026, o **docking power** conta outra história:

| | top1 (CASF-2016 core) | posição |
|---|---|---|
| **PharmXAI-3D** | **47,0%** | **30ª de 35** |
| mediana das 34 funções clássicas | 64,9% | — |
| AutodockVina | 90,1% | 1ª |
| baseline `n_contatos` | 30,9% | — |
| acaso | 25,2% | — |

**2ª de 35 em scoring power, 30ª de 35 em docking power.** Nenhuma das 34 referências
tem perfil tão assimétrico. É a assinatura de um modelo treinado só em poses
cristalográficas: sabe dizer *quão forte* liga, não sabe dizer *onde* — ver a seção 4.

A seção 5 fecha essa lacuna. Com uma cabeça de pose separada, treinada por
estacionariedade mais ranking contra perturbações rígidas:

| | antes | depois |
|---|---|---|
| top1 (docking power) | 47,0% · 30ª de 35 | **79,3%** · **14ª de 35** |
| rho entre poses ≤ 3 Å | 0,249 | **0,600** |
| R (afinidade) | 0,754 | **0,750** |

**+32 pontos de docking power sem custo mensurável na afinidade** — a queda de 0,004
em R cai dentro da dispersão de três sementes.

A seção 6 acrescenta features direcionais (ângulo, não só distância). Em rodada única
elas pareciam entregar o melhor R do projeto (0,768) e a primeira melhora do slope
(0,499). **Três sementes pareadas derrubam quase tudo:**

| métrica | escalar + física | direcional + física | p | |
|---|---|---|---|---|
| R | 0,752 ± 0,013 | 0,770 ± 0,012 | 0,324 | nulo |
| slope | 0,458 ± 0,044 | 0,489 ± 0,032 | 0,554 | nulo |
| top1 | 78,5 ± 1,7 | 79,0 ± 0,5 | 0,625 | nulo |
| **rho (poses)** | **0,486 ± 0,011** | **0,580 ± 0,041** | **0,037** | **significativo** |

Sobra um único efeito — a ordenação global de poses — que não é a métrica de uso, ao
custo de 2,3× os parâmetros. **O resultado sustentado do projeto continua sendo o da
seção 5.**

Um ensemble heterogêneo dos seis checkpoints chega a **R = 0,819**, mas com 4× a
inferência e sem o argumento de eficiência.

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

## 4. Docking power: o modelo é um re-scorer, não uma scoring function

### A pergunta

As seções anteriores medem **scoring power** — dada a pose cristalográfica, quão bem
o modelo prevê a afinidade. É uma das quatro capacidades que o CASF-2016 avalia, e
não implica as outras. A pergunta independente é o **docking power**: dado um sítio e
~79 poses candidatas do mesmo ligante, o modelo põe a pose correta no topo?

Ela decide o que o modelo é. Com scoring power e sem docking power ele é um
**re-scorer**: precisa que outro programa ache a pose, e só então estima a afinidade.
É exatamente o papel que ele cumpre hoje em `predict_custom.py`, onde quem docka é o
smina e a rede entra depois.

O `decoys_docking/` do pacote CASF-2016 traz as poses prontas com RMSD anotado, e o
`power_docking/examples/` traz o score das 34 funções de referência para as mesmas
poses. Isso permite para docking power a mesma comparação recomputada que o
`README.md` faz para scoring power.

### O resultado

285 complexos, 22.482 poses (79 por complexo, 98,7% das listadas), near-native
definida como RMSD ≤ 2 Å.

| score | top1 % | top2 % | top3 % | rho |
|---|---|---|---|---|
| **PharmXAI-3D** | **47,0** | 63,2 | 72,3 | 0,374 |
| baseline `n_contatos` | 30,9 | 47,0 | 57,5 | 0,591 |
| baseline LJ (proxy) | 29,1 | 41,8 | 53,7 | 0,124 |
| acaso | 25,2 | — | — | — |

Contra as 34 referências, recomputadas com a mesma métrica e restritas às mesmas
poses:

| posição | função | top1 % |
|---|---|---|
| 1 | AutodockVina | 90,1 |
| 2 | deltaVinaRF20 | 89,1 |
| 17 | X-ScoreHM | 65,6 |
| — | *mediana das 34* | *64,9* |
| **30** | **PharmXAI-3D** | **47,0** |
| 35 | D-Score@SYBYL | 26,3 |

**2ª de 35 em scoring power, 30ª de 35 em docking power.** Nenhuma referência tem
perfil tão assimétrico, e a causa não é misteriosa: todo complexo do PDBbind é uma
pose correta. O modelo nunca viu, no treino, o mesmo ligante no mesmo bolso na
posição errada, então não há razão para a superfície de score ter mínimo na pose
nativa. Medido diretamente: na pose cristalográfica de 1a30, a força resultante sobre
o ligante é 1,52 e o torque 2,01, onde uma energia daria ~0 em ambos.

O modelo bate os dois baselines triviais com folga (47,0 contra 30,9 e 29,1), então
aprendeu geometria de verdade — só não o bastante.

### Onde exatamente está o déficit

Separando tendência global de discriminação fina:

| | rho global | rho entre poses ≤ 3 Å | AUC |
|---|---|---|---|
| PharmXAI-3D | **0,374** | 0,249 | 0,690 |
| AutodockVina | 0,334 | **0,614** | 0,791 |
| X-Score | 0,601 | 0,609 | 0,856 |

O PharmXAI **supera** o AutodockVina na tendência global e perde 2,5× na vizinhança
da pose nativa. Os 43 pontos de diferença no top1 (47,0 contra 90,1) saem inteiros
dessa vizinhança: o modelo sabe separar pose plausível de pose absurda, e não sabe
escolher entre duas poses plausíveis.

Isso localiza a intervenção. Não adianta melhorar a tendência global — ali ele já
está à frente do primeiro colocado. O que falta é estrutura na vizinhança do mínimo,
que é justamente onde uma condição de estacionariedade opera.

### Três armadilhas de medição, todas encontradas por baseline trivial

**1. O bolso não pode seguir o ligante.** `build_hetero_graph` define o bolso como os
átomos de proteína a 6 Å do ligante, recalculado a cada chamada. Para afinidade em
pose cristalográfica está correto; para docking power é fatal, porque mover o ligante
move o bolso junto e uma pose deslocada ganha um bolso perfeitamente formado ao redor
de si. A informação "esta pose está no lugar errado" é apagada na construção do
grafo, antes de a rede ver qualquer coisa. Em `docking_power.py` o conjunto de átomos
do bolso é escolhido uma vez por complexo e reusado por todas as poses; só as arestas
de interação são recalculadas.

**2. A pose cristalográfica é candidata.** Ela consta do `_rmsd.dat` como
`{pdb}_ligand` com RMSD 0,0 e as 34 referências a pontuam, mas mora em `coreset/` e
não no `_decoys.mol2`. Deixá-la de fora dava a cada referência um acerto garantido
que o modelo não tinha: 45,3% sem ela, 47,0% com ela.

**3. Ela precisa ser lida pelo mesmo parser dos decoys.** Lida do `.sdf`, chega com
49 átomos onde o decoy do mesmo ligante tem 26 — sob sanitização tolerante o
`RemoveHs` não remove os hidrogênios do `.sdf`. Com quase o dobro de átomos ela ganha
quase o dobro de contatos e passa a ser identificável por **tamanho**, não por
geometria: o baseline `n_contatos` subiu a **96,1%** de top1, contra 30,9% depois da
correção. O modelo foi de 45,3% a 60,4% na mesma rodada — um salto que era artefato
de parsing, e que teria sido registrado como ganho se só o número do modelo estivesse
sendo olhado.

### Controle: a escolha do bolso

O bolso fixo pode ser definido pela pose nativa (fiel à distribuição de treino, mas
poses distantes perdem contato por truncamento) ou pela união de todas as poses (sem
truncamento, mas bolso maior que o do treino). Rodar os dois separa geometria de
artefato:

| score | bolso `nativa` | bolso `uniao` | Δ |
|---|---|---|---|
| PharmXAI-3D | 47,0 | 41,8 | −5,2 |
| `n_contatos` | 30,9 | 15,8 | −15,1 |

O modelo perde pouco onde o baseline de contagem despenca, ou seja, depende bem menos
do artefato de truncamento. A conclusão da seção não muda de um modo para o outro.

### Limitação conhecida

295 poses (1,3%) falham no parse por kekulização (`non-ring atom marked aromatic`),
concentradas em 14 dos 285 complexos — `3utu` perde 49 de 78. Os complexos afetados
têm resultado individual pouco confiável; no agregado de 285 o efeito é pequeno, mas
a correção continua pendente.

---

## 5. Física de pose: docking power de 47,0% para 79,3%

Medido em 09/09/2026, logo depois da seção 4. A pergunta que a seção anterior deixou:
o déficit está na vizinhança da pose nativa — dá para atacá-lo sem perder o R = 0,754?

### A intervenção

Uma **terceira cabeça**, treinada com dois termos, e a separação não é detalhe de
implementação:

**Por que cabeça separada.** pKd é energia **livre** — entropia e dessolvatação
dentro — enquanto a pose é determinada por energia **potencial**. As duas tarefas
puxam em direções diferentes, e a própria tabela da seção 4 mostra o trade-off
operando entre as referências: o AutodockVina faz 90,1% de top1 com R = 0,604, contra
47,0% e R = 0,754 aqui. Impor comportamento de energia sobre a cabeça de afinidade
arriscaria o resultado que já existe.

**(a) Estacionariedade.** Se o score vai se comportar como energia, a força e o torque
resultantes sobre o ligante se anulam na pose cristalográfica. É uma condição
variacional, e é o análogo bem-posto aqui do resíduo de EDP de um PINN — não existe
EDP cuja solução seja pKd, mas existe esta condição. Imposta sobre os 6 graus de
liberdade de **corpo rígido**, não sobre os 3N cartesianos: a pose cristalográfica não
é o mínimo da geometria *interna* do ligante segundo o modelo, e exigir isso injetaria
ruído.

**(b) Ranking contra perturbação rígida.** Rotação e translação do ligante com o bolso
parado, mantendo as **arestas fixas** e recomputando só as distâncias — assim
`n_contatos` fica idêntico entre nativa e perturbada, o que fecha o atalho de contagem
antes que ele exista.

Custo: 143.795 parâmetros contra 127.090 (+16.705), 2,2× por lote, ~30 s/época em uma
RTX 3050 6 GB.

### Duas correções que o desenho exigiu

**Estacionariedade sozinha é degenerada.** Uma função constante zera força e torque
perfeitamente sem ter aprendido nada. Medido na inicialização: a cabeça nasce
praticamente constante, com |F| = 1e-4 e a loss de estacionariedade em 0,000000. É por
isso que o ranking existe — ele dá **conteúdo** ao mínimo, e a estacionariedade lhe dá
**forma**. Pela mesma razão a estacionariedade só entra na época 6: ligada desde o
início, competiria com o ranking justamente enquanto este tenta tirar a cabeça da
constante.

**Margem fixa satura em quatro épocas.** Na primeira versão a perturbação era sempre
de 1,5 Å e a margem constante:

| época | loss de ranking, margem fixa | margem proporcional |
|---|---|---|
| 1 | 0,0418 | 0,0435 |
| 2 | 0,0006 | 0,0024 |
| 4 | **0,0003** | 0,0019 |

Separar a nativa de uma perturbação sempre do mesmo tamanho é fácil, e uma vez
resolvida a tarefa para de dar gradiente — restando só a estacionariedade, que sozinha
puxa de volta para a constante. A correção: **sortear a magnitude** entre 0,3 e 3,0 Å
por grafo, com margem **proporcional ao RMSD** da perturbação. Assim a cabeça precisa
aprender o quanto o score decai com a distância, não apenas o sinal da diferença, e as
perturbações pequenas continuam difíceis depois que as grandes ficaram fáceis.

### Resultado

Treino parou na época 49, melhor modelo na 19.

| métrica | antes | depois |
|---|---|---|
| **top1 (docking power)** | 47,0% | **79,3%** |
| posição entre as 35 | 30ª | **14ª** |
| top2 / top3 | 63,2 / 72,3 | **88,4 / 92,3** |
| rho entre poses ≤ 3 Å | 0,249 | **0,600** |
| rho global | 0,374 | **0,484** |
| AUC | 0,690 | **0,835** |
| RMSD da pose escolhida | 2,91 Å | **1,46 Å** |
| **R (afinidade, CASF core)** | **0,754** | **0,750** |
| RMSE | 1,486 | 1,483 |

O R cai 0,004, dentro da dispersão de três sementes (0,753 ± 0,016). **A cabeça
separada cumpriu o que prometia: o docking power subiu 32 pontos sem custo mensurável
na afinidade.**

E o ganho apareceu onde o diagnóstico apontava. O rho fino mais que dobrou e passou a
empatar com AutodockVina (0,614) e X-Score (0,609); a AUC ficou acima do Vina. A
tendência global, onde o modelo já liderava, subiu pouco. A intervenção acertou o alvo
que a medição tinha identificado, e não outro.

### Três controles

Um salto de 32 pontos pede que se tente derrubá-lo antes de acreditar.

| controle | pergunta | resultado |
|---|---|---|
| cabeça de **afinidade** do mesmo checkpoint | o ganho é difuso pelo modelo? | **40,4%** — não é |
| baseline `n_contatos` | virou atalho de contagem? | **30,9%**, inalterado — não |
| **sem** a pose cristalográfica no conjunto | só aprendeu a reconhecer o cristal? | **75,8%** contra 45,6% — não |

O terceiro é o mais importante. A cabeça foi treinada contra perturbações rígidas
sintéticas de poses de treino, e a capacidade **transferiu para decoys gerados por
docking real**, num conjunto de proteínas que nunca viu. Isso não era garantido: o
desenho poderia ter produzido um detector de "coordenadas cristalográficas" em vez de
um juiz de geometria.

### Limitações desta rodada

**Uma semente.** A diferença de 32 pontos é uma ordem de grandeza acima do ruído de
semente do projeto, mas a regra da seção 6 pede pareamento para qualquer número que se
queira defender com precisão.

**O checkpoint é selecionado só pela loss de afinidade.** Não há validação de qualidade
de pose: o early stopping olha `val pKd`, e a cabeça de pose vai junto no estado em que
estiver. Isso protege o 0,754 por construção e foi deliberado nesta primeira rodada,
mas significa que o 79,3% é o que se obteve, não o que se otimizou.

**A perturbação rígida não é um decoy realista.** Não tem clash, nem mudança
conformacional do ligante, nem a distribuição de poses que um programa de docking
produz. Que tenha transferido é o resultado; que transferiria não era previsível.

---

## 6. Features direcionais: um único efeito sobrevive ao pareamento

A seção 5 resolveu a metade *pose* da discriminação fina impondo a condição de
equilíbrio, mas por uma via que **contorna** a causa suspeita em vez de removê-la: o
modelo continua vendo apenas `||r_i − r_j||`.

Ligação de hidrogênio é **direcional** — N–H···O vale muito a 180° e quase nada a 90°,
com a mesma distância. A hipótese era que essa cegueira angular explicava as duas
limitações restantes, e que corrigi-la atacaria ambas de uma vez.

**A hipótese estava parcialmente certa, pelo motivo errado.**

### A implementação

`CamadaDirecional` em `gnn_model.py` (flag `usar_direcional`, `PHARM_DIRECIONAL=1`):
blocos estilo PaiNN em que cada átomo carrega escalares `s` [N, F] **e** vetores
`v` [N, 3, F]. Os vetores giram com o complexo, mas só entram nos escalares por
quantidades invariantes — a norma `||Vv||` e o produto interno `⟨Uv, Vv⟩` — e o readout
consome apenas `s`. **Invariância verificada: desvio de 2,2e-08 sob rotação e
translação aleatórias**, que é a precisão do float32.

Escolhido em vez de DimeNet/GemNet por custo: enumerar tripletos é O(E·k) e mataria o
argumento de eficiência. Aqui a direcionalidade emerge do produto interno. Custa
127.090 → 276.338 parâmetros (293.043 com a cabeça de pose) e 1,7× por lote.

### A matriz, em rodada única

| configuração | R | slope | top1 | rho global | rho fino |
|---|---|---|---|---|---|
| escalar | 0,754 | 0,424 | 47,0% | 0,374 | 0,249 |
| direcional | 0,746 | 0,465 | 46,0% | 0,459 | 0,236 |
| escalar + física | 0,750 | 0,436 | 79,3% | 0,484 | **0,600** |
| direcional + física | **0,768** | **0,499** | 79,6% | **0,599** | 0,592 |

Lida assim, a conclusão seria: o ângulo sozinho não entrega nada, mas combinado com a
física dá o melhor R do projeto e a primeira melhora do slope. **Essa conclusão está
errada, e três sementes pareadas mostram por quê.**

### O que sobrevive ao pareamento

Três sementes, escalar+física contra direcional+física, mesma inicialização e mesma
ordem de lotes em cada par.

| métrica | escalar + física | direcional + física | efeito | IC95% | p | |
|---|---|---|---|---|---|---|
| R | 0,752 ± 0,013 | 0,770 ± 0,012 | +0,018 | [−0,043, +0,079] | 0,324 | **nulo** |
| slope | 0,458 ± 0,044 | 0,489 ± 0,032 | +0,031 | [−0,158, +0,220] | 0,554 | **nulo** |
| RMSE | 1,470 ± 0,042 | 1,426 ± 0,045 | −0,045 | [−0,256, +0,167] | 0,459 | **nulo** |
| top1 | 78,5 ± 1,7 | 79,0 ± 0,5 | +0,57 | [−3,70, +4,83] | 0,625 | **nulo** |
| top3 | 93,2 ± 2,3 | 95,1 ± 1,1 | +1,83 | [−5,99, +9,66] | 0,420 | **nulo** |
| **rho (poses)** | **0,486 ± 0,011** | **0,580 ± 0,041** | **+0,095** | **[+0,014, +0,176]** | **0,037** | **significativo** |

**Um único efeito sobrevive.** As features direcionais melhoram a *ordenação global* das
poses e nada mais. Os sinais das diferenças contam a história:

| métrica | diferenças por semente |
|---|---|
| R | +0,018 · +0,043 · **−0,006** |
| slope | +0,063 · +0,086 · **−0,056** |
| rho | +0,115 · +0,112 · +0,057 |

O R = 0,768 e o slope de 0,499 vieram da semente 0. Na semente 2 o escalar tem slope
0,509 contra 0,453 do direcional — **maior**. Escolher qual semente citar decidiria a
conclusão, que é exatamente o que o pareamento existe para impedir.

### Sozinho, o ângulo não entrega — e não por falta de uso

A razão `||v||/|s|` cresce de 0,10 na primeira camada para 0,55 na última, e uma
ablação zerando as projeções `U` e `V` — o canal por onde os vetores voltam para os
escalares — desloca o pKd em **2,14 unidades** e derruba a correlação com a saída
original para 0,57. O modelo usa a informação angular intensamente. O resultado nulo
não é falha de implementação nem de otimização: a informação entra, é processada, e
não se converte em acerto.

### Quem consertou o quê

| | rho fino (≤ 3 Å) |
|---|---|
| escalar | 0,249 |
| + ângulo | 0,236 |
| **+ física** | **0,600** |
| + ângulo + física | 0,592 |

**A discriminação fina foi resolvida pela restrição física, não pelo ângulo** — e a
hipótese que motivou esta seção previa o contrário. Adicionar ângulo por cima da física
não muda nada (0,600 contra 0,592).

O que o ângulo faz é melhorar a correlação global entre score e RMSD (+0,095, p = 0,037),
levando o rho a 0,580 — perto do X-Score (0,601) e bem acima do AutodockVina (0,334).
Mas rho não é a métrica de uso: quem escolhe a pose é o top1, e ali o efeito é nulo.
**É um ganho real e de valor prático pequeno.**

### O custo escondido das features vetoriais

**Três treinos perdidos por instabilidade numérica que só existe no caminho
direcional.** Nenhuma delas aparece no modelo escalar, e nenhuma é mencionada na
literatura de PaiNN:

1. **Arestas de comprimento zero.** 72 em 3,7 milhões (0,0019%, atingindo 1,1% dos
   grafos) — átomos sobrepostos, provavelmente conformação alternativa do PDB. A
   derivada segunda do versor escala com 1/d², e a estacionariedade usa
   `create_graph=True`: com piso de 1e-6 isso chega a 1e12. A loss de estacionariedade
   foi a **703.684** e virou NaN na 6ª época. No caminho escalar essas arestas passam
   despercebidas porque o RBF satura em vez de explodir. Corrigido com piso físico de
   0,5 Å — nenhuma ligação química real mede menos que ~0,9 Å.

2. **Cauda pesada na norma do gradiente.** Treino saudável até a época 20
   (`Estac` ~0,007), explosão na 21 (`Estac` 813, `pKd` 523), NaN na 22, sem aviso
   antes. Corrigido com recorte em 500 — valor **medido**, não convencionado: as normas
   em treino saudável têm mediana 53,6 e máximo 149,0 no escalar, mediana 54,1 e máximo
   325,3 no direcional. O default folclórico de 10 cortaria **100% dos lotes** e
   mudaria o treino por completo. O recorte é aplicado só quando `usar_direcional`,
   para que o resultado da seção 5 continue reproduzível.

3. **Loss finita com gradiente NaN.** Descartar lotes pela loss não basta: `sqrt(0)`
   tem valor finito e derivada infinita. Um lote assim injeta NaN nos pesos e a partir
   daí toda loss é NaN — nas épocas 11 a 19 caía 1 lote por época, e na 20 caíram os
   429, com a validação congelada. O descarte passou a proteger um modelo já morto. A
   verificação correta é na **norma do gradiente, depois do backward**.

Isso é uma desvantagem prática real das features vetoriais, e pesa na decisão de
adotá-las: o ganho de 0,014 em R está na fronteira do ruído, o de docking power é
marginal (79,6% contra 79,3%), e o custo é 2,3× os parâmetros mais três modos de falha
numérica.

### Ensemble

A média simples dos quatro checkpoints:

| | R | RMSE | slope |
|---|---|---|---|
| melhor individual (direcional + física, semente 0) | 0,768 | 1,435 | **0,499** |
| ensemble dos 4 (arquiteturas distintas) | 0,817 | 1,380 | 0,456 |
| **ensemble dos 6 (com as sementes)** | **0,819** | **1,357** | 0,473 |

R = 0,817 contra 0,816 do deltaVinaRF20, primeiro colocado entre as 35. Três ressalvas
que precisam acompanhar o número: é um ensemble **heterogêneo** (quatro arquiteturas,
não quatro sementes), o que rende mais porque os erros são menos correlacionados; custa
840 mil parâmetros somados e 4× a inferência, o que desfaz o argumento de eficiência; e
o slope **cai** de 0,499 para 0,456, porque a média comprime mais — R e slope não sobem
juntos aqui.

### Veredito

**As features direcionais não se justificam neste modelo.** O único ganho sustentado é
o rho de ordenação de poses (+0,095), que não é a métrica de uso. Em troca custam 2,3×
os parâmetros (127.090 → 293.043), 1,7× o tempo por lote e três modos de falha numérica
que não existem no caminho escalar.

Isso **não** põe em dúvida a seção 5: os +32 pontos de docking power da física de pose
são uma ordem de grandeza acima do ruído de semente, e foram medidos com os mesmos
controles.

Registrado como resultado negativo porque é útil: evita que a tentativa seja refeita, e
o diagnóstico de *por que* falhou — a informação angular é usada intensamente e ainda
assim não converte — é mais informativo que o número em si.

---

## 7. Regras de medição que emergiram

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

**Uma métrica de correlação e uma métrica de topo medem coisas diferentes.** Em
docking power o `n_contatos` tem rho 0,591 contra 0,374 do modelo, e mesmo assim
acerta 30,9% de top1 contra 47,0%. Correlação alta ao longo de toda a faixa convive
com incapacidade de escolher entre as melhores candidatas — e é o topo que decide o
uso. Uma leitura preliminar em 8 complexos concluiu, pelo rho, que a contagem de
contatos ordenava poses melhor que a rede; nos 285 o oposto é verdade.

**Uma restrição que a solução trivial satisfaz não treina nada.** A estacionariedade
sozinha — força e torque nulos — é zerada perfeitamente por uma função constante, e a
cabeça de pose *nasce* constante (|F| = 1e-4 na inicialização). Antes de adicionar um
termo físico à loss, pergunte qual função degenerada o satisfaz de graça; se existir
uma, é preciso um segundo termo que a exclua.

**Uma tarefa auxiliar de dificuldade fixa satura e para de ensinar.** Com perturbação
sempre de 1,5 Å, a loss de ranking caiu de 0,042 para 0,0003 em quatro épocas. Sortear
a magnitude e escalar a margem com ela mantém a tarefa viva: as perturbações pequenas
continuam difíceis depois que as grandes ficaram fáceis.

**Faltar representação e faltar objetivo predizem o mesmo agregado, e se separam nas
partes.** A hipótese era que a cegueira angular explicava a discriminação fina ruim.
Quem a consertou foi a restrição física (rho fino 0,249 → 0,600); adicionar ângulo por
cima não mudou nada (0,592). Medir só o número agregado teria creditado o ganho à
hipótese errada.

**Um efeito que muda de sinal entre sementes é ruído, por maior que pareça na melhor
delas.** O direcional deu +0,043 em R na semente 1 e −0,006 na 2; o slope, +0,086 e
−0,056. Citar a semente 0 teria produzido duas alegações falsas. A regra da amplitude
vale aqui em cheio: só o efeito cujo sinal se repete nas três (rho, p = 0,037)
sobreviveu.

**Uma capacidade pode ser usada sem se converter em acerto.** Features direcionais são
processadas intensamente — `||v||/|s|` cresce de 0,10 para 0,55 entre as camadas, e
ablacioná-las desloca o pKd em 2,14 unidades — e mesmo assim não melhoram R, slope nem
top1 de forma sustentada. "O modelo usa a informação" não é evidência de que a
informação ajuda; são duas medições diferentes.

**Limiar numérico é para medir, não para herdar.** O recorte de gradiente "padrão" de
10 cortaria 100% dos lotes deste modelo, cujas normas em treino saudável têm mediana
53,6. O valor certo (500) veio de medir a distribuição das normas.

**Descartar lotes pela loss não protege contra NaN.** Uma loss finita pode ter
gradiente NaN — `sqrt(0)` é o caso canônico. O check tem de ser na norma do gradiente,
depois do backward; caso contrário o NaN entra nos pesos e o descarte passa a proteger
um modelo já morto, com a validação congelada e todos os lotes caindo.

**Um salto no número principal pede conferir o baseline trivial na mesma rodada.**
Incluir a pose cristalográfica levou o modelo de 45,3% a 60,4% de top1 — e o
`n_contatos` a 96,1%, denunciando que a pose fora identificável por tamanho. O ganho
do modelo era o mesmo artefato. Sem o baseline rodando lado a lado, ele teria sido
registrado como resultado.

---

## 8. Como reproduzir

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

### Docking power (seção 4)

Não treina nada — só lê o checkpoint. Requer os subdiretórios `decoys_docking`,
`power_docking` e `coreset` do `CASF-2016.tar.gz`, que somam 563 MB:

```bash
tar xzf CASF-2016.tar.gz CASF-2016/decoys_docking CASF-2016/power_docking \
                         CASF-2016/coreset CASF-2016/README

# ~2 min em 14 processos
python3 docking_power.py --bolso nativa --jobs 14 --saida logs/dock_nativa.csv
python3 docking_power.py --bolso uniao  --jobs 14 --saida logs/dock_uniao.csv

# posição entre as 34 funções de referência, restrita às mesmas poses
python3 comparar_docking.py logs/dock_nativa.csv
```

Não extraia o pacote inteiro sem conferir o espaço livre: `decoys_screening` sozinho
ocupa 8,3 GB descomprimido, e ele só é necessário para *screening power*, que ainda
não foi medido.

### Física de pose (seção 5)

| variável | efeito |
|---|---|
| `PHARM_POSE` | `1` liga a cabeça de pose e os dois termos físicos (padrão desligado) |
| `PHARM_LAMBDA_RANK` | peso do ranking (padrão 1,0) |
| `PHARM_LAMBDA_ESTA` | peso da estacionariedade (padrão 0,1) |
| `PHARM_MARGEM` | margem por Å de RMSD (padrão 0,5) |
| `PHARM_ESTA_WARMUP` | épocas antes de ligar a estacionariedade (padrão 5) |

```bash
# ~25 min numa RTX 3050 6 GB
PHARM_POSE=1 PHARM_CKPT=exp_pose_fisica.pth python3 -u train.py

# docking power pela cabeça de pose, e o controle pela de afinidade
python3 docking_power.py --pesos exp_pose_fisica.pth --cabeca pose \
        --jobs 14 --saida logs/dock_fisica_pose.csv
python3 docking_power.py --pesos exp_pose_fisica.pth --cabeca afinidade --jobs 14

# afinidade de um ou mais checkpoints, e a média deles
python3 ensemble.py pharm_model_weights_best.pth exp_pose_fisica.pth
```

`python3 train.py` sem variáveis continua reproduzindo exatamente a linha de
referência: a cabeça de pose não é sequer construída.

### Features direcionais (seção 6)

`PHARM_DIRECIONAL=1` troca as convoluções escalares por blocos com vetores
equivariantes. Combina com todas as demais variáveis.

```bash
# a matriz da seção 6 (~25 min escalar, ~50 min direcional, RTX 3050 6 GB)
PHARM_DIRECIONAL=1 PHARM_CKPT=exp_direcional.pth python3 -u train.py
PHARM_DIRECIONAL=1 PHARM_POSE=1 PHARM_CKPT=exp_direcional_fisica.pth python3 -u train.py

# ensemble: R, RMSE, slope de cada um e da média
python3 ensemble.py pharm_model_weights_best.pth exp_pose_fisica.pth \
                    exp_direcional.pth exp_direcional_fisica.pth
```

O recorte de gradiente (500) e o descarte de lotes com gradiente não-finito só entram
quando `PHARM_DIRECIONAL=1`, de propósito: o caminho escalar chegou aos seus resultados
sem eles, e ativá-los tornaria aqueles números irreprodutíveis.

---

## 9. O que fica em aberto

Estado em 09/09/2026, depois das seções 5 e 6.

### Concluídos

**Física de pose** (seção 5): o maior déficit medido, docking power em 30ª de 35,
passou a 14ª. Sustentado em três sementes.

**Features direcionais** (seção 6): rejeitadas. Três sementes pareadas mostram efeito
nulo em R, slope, RMSE, top1 e top3; sobra apenas o rho de ordenação de poses
(+0,095, p = 0,037), que não é a métrica de uso, ao custo de 2,3× os parâmetros.

**Sementes pareadas** para as duas seções acima.

O slope segue em **0,458 ± 0,044**, praticamente onde estava. A tentativa de atacá-lo
por representação falhou, e ele volta a ser o item aberto mais sério.

### 1. Ensemble, decidido com cuidado

R = 0,819 com seis checkpoints — acima do deltaVinaRF20 (0,816), primeiro colocado do
CASF-2016. Mas é ensemble **heterogêneo**, e o custo desfaz o argumento de eficiência,
que é o mais forte do projeto. O que falta medir antes de adotar: quanto rende um
ensemble só de sementes da **mesma** arquitetura escalar — mais honesto de reportar,
provavelmente menos ganho, e sem os 2,3× de parâmetros que a seção 6 rejeitou.

Registre-se que o slope **cai** no ensemble (0,499 → 0,473): a média comprime mais. R e
slope não sobem juntos.

### 2. Decoys reais por redocking

A perturbação rígida da seção 5 não tem clash, nem mudança conformacional, nem a
distribuição de poses que um programa de docking produz. Que tenha transferido é o
resultado mais surpreendente do projeto. Redocking do conjunto de treino com smina
(~7–14 h em 16 núcleos) diria quanto ainda há para ganhar.

### 3. Decomposição física do score

Substituir a predição crua por termos de forma funcional conhecida — LJ, ligação de H
com dependência angular, hidrofóbico, dessolvatação, entropia torcional — com a GNN
aprendendo coeficientes e correções. É de onde vem o docking power de X-Score e Vina,
e o único item que ataca a raiz em vez de compensá-la.

A seção 6 dá um argumento novo a favor, por via negativa: dar ao modelo a *capacidade*
de ler ângulo não bastou — ele usa a informação e não a converte em acerto. Uma
decomposição física não oferece a capacidade, **impõe a forma funcional**, o que é uma
aposta diferente e ainda não testada.

### 4. Validação de qualidade de pose na seleção do checkpoint

O early stopping olha só a loss de afinidade. A cabeça de pose vai junto no estado em
que estiver — o 79,6% é o que se obteve, não o que se otimizou.

### Pendentes desde as primeiras seções

**Avaliação em alvo novo, de forma sistemática.** O 0,692 vem de uma única
configuração de remoção. Validação cruzada por cluster de proteína daria uma estimativa
com incerteza.

**Eficiência de parâmetros, em pé de igualdade.** 144 mil (o modelo da seção 5) contra Pafnucy (≈1,5 M) e OnionNet (>10 M), que reportam 0,78–0,82 na condição fácil,
sem terem sido testados na difícil. Continua sendo o resultado potencialmente mais
interessante do projeto, e continua não feito.

**Screening power nunca foi medido.** A quarta capacidade do CASF-2016, e a mais
próxima do uso real de triagem virtual. Precisa dos 8,3 GB de `decoys_screening`.
