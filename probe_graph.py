"""
Grafo de sondas: bolso → campo farmacofórico
=============================================
Monta a entrada do modelo que responde "dado o bolso, que característica um
ligante precisaria apresentar em cada ponto do espaço".

Dois tipos de nó:

  **átomos do bolso** — com todas as features químicas de `data_processor`
  **sondas**          — pontos no espaço, SEM nenhuma feature química

A rede nunca vê a química do ligante. Ela vê o bolso e uma posição, e responde
o que caberia ali. É essa restrição que separa este modelo de um preditor de
afinidade: se as features do ligante entrassem, a tarefa viraria "classifique
este átomo", que é trivial e inútil.

Topologia das arestas, e por que ela é assim
---------------------------------------------
  bolso ↔ bolso   : contexto local. Sem isso a sonda vê átomos crus, sem
                    ambiente, e não distingue uma carbonila de backbone de uma
                    de cadeia lateral.
  bolso → sonda   : direcionada, só neste sentido.

**Sondas não se conectam entre si, nem devolvem mensagem ao bolso.** Duas razões:
na inferência a grade tem milhares de sondas, e deixá-las interagir tornaria a
predição de cada ponto dependente de onde as outras foram colocadas — o campo
deixaria de ser uma propriedade do bolso. E torna a inferência em grade
paralelizável e reprodutível: cada sonda é independente das demais.

Sondas de treino
----------------
  positivas: nas posições reais dos átomos pesados do ligante, rotuladas por
             `pharmacophore_labels.rotular_ligante`
  negativas: pontos da cavidade que nenhum átomo do ligante ocupa — plausíveis
             geometricamente (não colidem com a proteína, não estão no solvente
             distante) mas vazios. Sem elas o modelo aprende que todo ponto
             oferecido é ocupado, e o campo perde a capacidade de dizer "aqui
             não cabe nada".

Ressalva sobre a truncagem do bolso
------------------------------------
`PharmGraphBuilder` define o bolso como a casca em torno do ligante daquele
complexo. Para o teste de invariância entre os cinco ligantes de um cluster do
CASF isso é apertado demais: um ligante maior sai da casca definida por outro.
Use `raio_bolso` maior, ou construa o bolso a partir da união das poses do
cluster — ver `bolso_uniao`.
"""

import torch

from pharmacophore_labels import N_TIPOS, rotular_ligante

# Raios em ångströms
R_BOLSO_BOLSO = 4.0    # contexto entre átomos do bolso
R_BOLSO_SONDA = 6.0    # alcance com que uma sonda enxerga o bolso
R_CLASH       = 2.2    # mais perto que isto de um átomo pesado é colisão
R_OCUPADO     = 1.8    # sonda a esta distância de um átomo do ligante conta como ele
R_SOLVENTE    = 5.0    # mais longe que isto do bolso é solvente, não cavidade

N_FEAT_ATOMO  = 13     # colunas de data_processor.get_atom_features


def _raio(pos_a, pos_b, r):
    """Arestas de todo ponto de `a` a todo ponto de `b` a menos de `r`."""
    d = torch.cdist(pos_a, pos_b)
    idx = (d <= r).nonzero(as_tuple=False).t()
    return idx, d[idx[0], idx[1]].view(-1, 1)


def amostrar_negativas(pos_lig, pos_bolso, n, margem=2.0, gerador=None):
    """Pontos da cavidade que o ligante NÃO ocupa.

    Amostra na caixa que contém o ligante, expandida por `margem`, e descarta:
      * o que colide com a proteína     (< R_CLASH de um átomo do bolso)
      * o que o ligante já ocupa        (< R_OCUPADO de um átomo do ligante)
      * o que caiu no solvente          (> R_SOLVENTE de todo átomo do bolso)

    O resultado são posições onde um átomo de ligante *poderia* estar e não está —
    que é exatamente o negativo informativo. Amostrar em qualquer lugar da caixa
    daria negativos triviais (dentro da proteína, ou no vácuo) e o modelo
    aprenderia a detectar colisão em vez de complementaridade química.
    """
    if pos_lig.numel() == 0 or n <= 0:
        return torch.zeros(0, 3)
    lo = pos_lig.min(0).values - margem
    hi = pos_lig.max(0).values + margem
    aceitos = []
    for _ in range(12):                       # tentativas; a taxa de aceite é ~10-30%
        if sum(a.shape[0] for a in aceitos) >= n:
            break
        cand = torch.rand(4 * n, 3, generator=gerador) * (hi - lo) + lo
        ok = torch.ones(cand.shape[0], dtype=torch.bool)
        if pos_bolso.numel():
            d = torch.cdist(cand, pos_bolso)
            ok &= (d.min(dim=1).values > R_CLASH) & (d.min(dim=1).values < R_SOLVENTE)
        ok &= torch.cdist(cand, pos_lig).min(dim=1).values > R_OCUPADO
        aceitos.append(cand[ok])
    if not aceitos:
        return torch.zeros(0, 3)
    return torch.cat(aceitos)[:n]


def montar(data, pos_sonda, y_sonda=None, ocupacao=None):
    """Grafo homogêneo [bolso | sondas] pronto para o modelo.

    Args:
        data:      HeteroData de PharmGraphBuilder (usa só 'protein')
        pos_sonda: (n_sonda, 3) posições onde perguntar
        y_sonda:   (n_sonda, 6) rótulos de tipo, ou None na inferência
        ocupacao:  (n_sonda, 1) 1 onde há átomo de ligante, ou None

    Returns:
        dict com x, pos, edge_index, edge_attr, mask_sonda, y, ocupacao
    """
    px, pp = data['protein'].x, data['protein'].pos
    n_p, n_s = px.shape[0], pos_sonda.shape[0]

    # Coluna 0 marca o tipo de nó; as 13 seguintes são químicas e ficam em zero
    # nas sondas — a rede não recebe nenhuma pista sobre o que há ali.
    x = torch.zeros(n_p + n_s, 1 + N_FEAT_ATOMO)
    x[:n_p, 1:] = px
    x[n_p:, 0] = 1.0

    ei_pp, _ = _raio(pp, pp, R_BOLSO_BOLSO)
    ei_pp = ei_pp[:, ei_pp[0] != ei_pp[1]]                 # sem laços
    ea_pp = torch.norm(pp[ei_pp[0]] - pp[ei_pp[1]], dim=1, keepdim=True)

    ei_ps, ea_ps = _raio(pp, pos_sonda, R_BOLSO_SONDA)
    ei_ps = torch.stack([ei_ps[0], ei_ps[1] + n_p])        # bolso → sonda

    saida = {
        "x": x,
        "pos": torch.cat([pp, pos_sonda]),
        "edge_index": torch.cat([ei_pp, ei_ps], dim=1),
        "edge_attr": torch.cat([ea_pp, ea_ps]),
        "mask_sonda": torch.cat([torch.zeros(n_p, dtype=torch.bool),
                                 torch.ones(n_s, dtype=torch.bool)]),
    }
    if y_sonda is not None:
        saida["y"] = y_sonda
    if ocupacao is not None:
        saida["ocupacao"] = ocupacao.view(-1, 1).float()
    return saida


def montar_treino(data, razao_negativas=1.0, gerador=None):
    """Grafo de treino: sondas positivas nos átomos do ligante, mais negativas."""
    pos_lig = data['ligand'].pos
    y_pos = data.y_pharm if hasattr(data, 'y_pharm') else rotular_ligante(data)
    n_neg = int(round(pos_lig.shape[0] * razao_negativas))
    pos_neg = amostrar_negativas(pos_lig, data['protein'].pos, n_neg, gerador=gerador)

    pos = torch.cat([pos_lig, pos_neg])
    y = torch.cat([y_pos, torch.zeros(pos_neg.shape[0], N_TIPOS)])

    # Ocupação é "há átomo de ligante aqui", NÃO "há átomo que interage aqui".
    # As duas coisas se separam: um átomo de ligante pode estar presente e não
    # fazer interação nenhuma — medido, ~16% deles. Conflatá-las ensinaria o
    # modelo a considerar vazio um ponto ocupado, e a cabeça de forma prediria
    # um ligante com buracos.
    ocup = torch.cat([torch.ones(pos_lig.shape[0]),
                      torch.zeros(pos_neg.shape[0])])
    return montar(data, pos, y, ocup)


def grade_cavidade(data, passo=1.0, margem=3.0):
    """Sondas numa grade regular sobre a cavidade — o modo de inferência.

    Devolve só os pontos que passam nos mesmos filtros geométricos das negativas
    (não colidem, não estão no solvente), porque perguntar "o que cabe aqui" só
    faz sentido onde algo caberia.
    """
    pp = data['protein'].pos
    ref = data['ligand'].pos if 'ligand' in data.node_types else pp
    lo, hi = ref.min(0).values - margem, ref.max(0).values + margem
    eixos = [torch.arange(float(lo[i]), float(hi[i]) + passo, passo) for i in range(3)]
    g = torch.stack(torch.meshgrid(*eixos, indexing='ij'), dim=-1).reshape(-1, 3)
    if pp.numel():
        d = torch.cdist(g, pp).min(dim=1).values
        g = g[(d > R_CLASH) & (d < R_SOLVENTE)]
    return g
