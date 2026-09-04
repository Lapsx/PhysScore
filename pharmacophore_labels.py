"""
Rotulagem geométrica de pontos farmacofóricos
==============================================
O PDBbind não traz farmacóforo anotado. Este módulo deriva os rótulos das
estruturas cristalográficas, aplicando os critérios geométricos padrão de
detecção de interação — os mesmos que ferramentas como o PLIP usam.

O rótulo é do lado do LIGANTE: para cada átomo pesado do ligante, quais tipos de
interação ele efetivamente faz com o bolso. É esse o objeto que um farmacóforo
descreve, e é o alvo do modelo bolso→farmacóforo: dada só a proteína, que
característica um ligante precisaria apresentar em cada ponto do espaço.

Trabalha sobre o `HeteroData` que `PharmGraphBuilder.build_hetero_graph` devolve,
e não sobre os objetos do RDKit. Assim reaproveita toda a leitura, a truncagem do
bolso e a destilação dos hidrogênios em flags que já estão validadas ali.

Colunas de `x`, conforme `data_processor.get_atom_features`:
    0 Z | 1 massa | 2 grau | 3 aromático | 4 carga formal | 5 doador | 6 aceptor
    7-12 hibridização (SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED)
"""

import torch

# Índices das colunas de features
Z, AROM, CARGA, DOADOR, ACEPTOR = 0, 3, 4, 5, 6

TIPOS = ["doador", "aceptor", "hidrofobico", "aromatico", "cationico", "anionico"]
N_TIPOS = len(TIPOS)

# Critérios geométricos, em ångströms. Valores de consenso da literatura de
# detecção de interação; o PLIP usa 4.1 para hidrofóbico e 5.5 para empilhamento.
# Adotamos o limite mais frouxo de 4.5 para hidrofóbico porque aqui não há
# filtro de ângulo, e um corte apertado sem ângulo perde contatos reais.
D_HBOND      = 3.5    # doador–aceptor, entre átomos pesados
D_HIDROFOB   = 4.5    # carbono/enxofre apolar a carbono/enxofre apolar
D_AROMATICO  = 5.5    # centro a centro de anéis; aqui, átomo aromático a átomo aromático
D_IONICO     = 4.5    # cargas formais opostas

APOLARES = {6, 16}    # C, S


def _apolar(x):
    """Carbono ou enxofre que não seja doador nem aceptor."""
    elem = torch.isin(x[:, Z].long(), torch.tensor(list(APOLARES)))
    return elem & (x[:, DOADOR] < 0.5) & (x[:, ACEPTOR] < 0.5)


def rotular_ligante(data, retornar_parceiros=False):
    """Rótulos multi-classe por átomo pesado do ligante.

    Args:
        data: HeteroData com 'ligand'/'protein', cada um com .x e .pos
        retornar_parceiros: se True, devolve também o índice do átomo do bolso
            mais próximo que justifica cada rótulo (útil para inspeção visual)

    Returns:
        y: (n_ligante, 6) float — 1.0 onde o átomo faz aquele tipo de interação.
           Um átomo pode ter mais de um tipo (uma hidroxila aromática é doadora,
           aceptora e aromática ao mesmo tempo), daí ser multi-rótulo e não
           classificação exclusiva.
    """
    lx, lp = data['ligand'].x,  data['ligand'].pos
    px, pp = data['protein'].x, data['protein'].pos
    nl = lx.shape[0]
    y = torch.zeros(nl, N_TIPOS)
    if px.shape[0] == 0:
        return (y, None) if retornar_parceiros else y

    d = torch.cdist(lp, pp)                                  # (n_lig, n_prot)

    lig_doa, lig_ace = lx[:, DOADOR] > 0.5, lx[:, ACEPTOR] > 0.5
    pro_doa, pro_ace = px[:, DOADOR] > 0.5, px[:, ACEPTOR] > 0.5
    lig_apo, pro_apo = _apolar(lx), _apolar(px)
    lig_aro, pro_aro = lx[:, AROM] > 0.5, px[:, AROM] > 0.5
    lig_cat, lig_ani = lx[:, CARGA] > 0.5, lx[:, CARGA] < -0.5
    pro_cat, pro_ani = px[:, CARGA] > 0.5, px[:, CARGA] < -0.5

    def contato(mask_l, mask_p, corte):
        """Existe algum átomo do bolso com mask_p a menos de `corte` deste átomo?"""
        if not mask_l.any() or not mask_p.any():
            return torch.zeros(nl, dtype=torch.bool)
        viz = (d <= corte) & mask_p.unsqueeze(0)
        return mask_l & viz.any(dim=1)

    # Doador do ligante precisa de ACEPTOR na proteína, e vice-versa: a
    # complementaridade é o ponto, não a coincidência de tipo.
    y[:, 0] = contato(lig_doa, pro_ace, D_HBOND).float()      # doador
    y[:, 1] = contato(lig_ace, pro_doa, D_HBOND).float()      # aceptor
    y[:, 2] = contato(lig_apo, pro_apo, D_HIDROFOB).float()   # hidrofóbico
    y[:, 3] = contato(lig_aro, pro_aro, D_AROMATICO).float()  # aromático
    y[:, 4] = contato(lig_cat, pro_ani, D_IONICO).float()     # catiônico
    y[:, 5] = contato(lig_ani, pro_cat, D_IONICO).float()     # aniônico

    if not retornar_parceiros:
        return y
    parceiro = torch.where(d.min(dim=1).values < 6.0, d.argmin(dim=1),
                           torch.full((nl,), -1))
    return y, parceiro


def resumo(y):
    """Contagens por tipo e fração de átomos sem interação nenhuma."""
    tem = y.sum(dim=1) > 0
    return {
        "n_atomos": int(y.shape[0]),
        "com_interacao": int(tem.sum()),
        "frac_com_interacao": float(tem.float().mean()),
        "por_tipo": {t: int(y[:, i].sum()) for i, t in enumerate(TIPOS)},
        "tipos_por_atomo": float(y.sum(dim=1).mean()),
    }
