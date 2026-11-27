---
marp: true
theme: default
paginate: true
math: katex
html: true
size: 16:9
title: Inferência de preferências do oponente em MO-Connect4
style: |
  section { font-size: 26px; }
  h1 { color: #1f3a5f; }
  h2 { color: #1f3a5f; }
  table { font-size: 22px; }
  code { font-size: 0.9em; }
  .two-col { display: grid; grid-template-columns: 1.25fr 1fr; gap: 36px; align-items: center; font-size: 22px; }
  .two-col img { width: 100%; }
  .caption { font-size: 18px; color: #555; }
---

<!-- _class: lead -->

# Inferência de preferências do oponente em ambientes multiobjetivo e multiagente com soma zero.



Anderson Gonçalves, Diogo Rezende, Luiz Felipe e João Augusto · 27/11/2026

---

## Problema

- Em jogos multiobjetivo, cada jogador pondera objetivos por um vetor de preferências $\omega$.
- A modelagem de oponentes existente prevê a **próxima ação** do oponente.
- Este projeto tenta inferir **a preferência $\hat{\omega}$ do oponente** a partir dos seus lances e condiciona a política do agente em ($\omega, \hat{\omega}$).

[Artigo base](https://openreview.net/pdf?id=UW4K5qTUfA): *Opponent Modelling for Multi-Objective Multi-Agent Reinforcement Learning in Sequential Settings (2026; Obiang, Rădulescu)*

---

## Hipótese (H1)

Contra oponentes com $\omega$ **não vistos no treino**, um agente que infere $\hat{\omega}$ tem utilidade maior do que o MO-PPO sem modelo de oponente.

Ordem esperada:

```
melhor resposta exata ≥ oráculo ≥ agente ω̂ ≥ MO-PPO
```

- Gap melhor resposta ↔ oráculo: custo da aproximação por RL.
- Gap oráculo ↔ $\hat{\omega}$: custo de não conhecer ω do oponente.

---

## Ambiente, preferências e oponentes

<div class="two-col">
<div>

No MO-Connect4, dois jogadores alternam peças em um tabuleiro 4×4 e tentam fazer quatro em linha. Cada jogador avalia o resultado por três objetivos: vencer, vencer rápido e controlar colunas.

A preferência de um jogador é um vetor ω = (ω₁, ω₂, ω₃) de pesos não negativos que somam 1. Os vetores possíveis formam um triângulo, o **simplex**, cujos vértices são os objetivos puros: (1, 0, 0), (0, 1, 0) e (0, 0, 1).

Usamos 15 pontos desse triângulo, espaçados de 0,25. Cada ponto define um oponente, que joga segundo a política $\mathrm{softmax}(Q_k / \tau)$, calculada de forma exata. Com τ = 0,1, ele quase sempre escolhe o melhor lance, mas comete erros ocasionais.

</div>
<div>

![](assets/moconnect4.gif)

<p class="caption">Animação do ambiente MO-Connect4, da documentação do MOMAland (momaland.farama.org).</p>

</div>
</div>

---

## Baseline: MO-PPO

- Política MLP *actor-critic* com máscara de ações; entrada = tabuleiro + ω próprio.
- Hiperparâmetros do Apêndice A do artigo base; 5 seeds × 2M passos.
- Avaliação no **split de teste** (3 oponentes, 15 ω próprios, dois lados, 100 jogos por célula).

| Agente | Utilidade média | Vitória | Empate | Derrota |
|---|---|---|---|---|
| Aleatório | −0,134 | 0,098 | 0,462 | 0,440 |
| MO-PPO | +0,146 ± 0,014 | 0,355 ± 0,037 | 0,556 ± 0,031 | 0,089 ± 0,045 |

IC 95% sobre 5 seeds.

---

## Agente ω̂: filtro bayesiano

$$
P(k \mid h_t) \propto P(k)\prod_{\tau \le t} \pi_k(a^{opp}_\tau \mid s_\tau)
$$

**Para que serve:** estimar, a partir dos lances observados, qual dos 15 oponentes está jogando. **Como se comporta:** começa com a mesma probabilidade para todos e, a cada lance do oponente, multiplica o peso de cada hipótese pela probabilidade que aquele oponente daria ao lance observado; assim, as hipóteses que explicam os lances ganham peso e as demais perdem.

- Hipóteses: os 15 índices do grid (não grupos, porque com τ > 0 ω's do mesmo grupo diferem).
- Atualização em log-espaço, a cada lance do oponente.
- Entrada da política: tabuleiro + ω próprio + **posterior** sobre k.
- Treino usa só os oponentes de treino (D1).

---

## Resultado: agente ω̂ vs MO-PPO (teste)

| Agente | Utilidade média | Vitória | Empate | Derrota |
|---|---|---|---|---|
| MO-PPO | +0,146 ± 0,014 | 0,355 ± 0,037 | 0,556 ± 0,031 | 0,089 ± 0,045 |
| ω̂ (hipóteses = treino) | +0,152 ± 0,013 | 0,376 ± 0,055 | 0,541 ± 0,043 | 0,083 ± 0,053 |

- As duas médias estão dentro do IC uma da outra: **sem ganho detectável** nesta versão.
- H1 **não** está confirmada.

---

## Por que o filtro não ajuda: teste de identificação (fase 0)

Para saber se o problema estava nos lances, medimos o filtro sozinho. Um agente aleatório jogou 100 partidas contra cada um dos 15 oponentes, e registramos quanta probabilidade o filtro dava ao grupo verdadeiro depois de cada lance. Em média, essa probabilidade vai de 0,10 após o primeiro lance para 0,16 após o oitavo: o filtro aprende, mas muito devagar.

O critério de aprovação era chegar a 0,9. Nenhum oponente chega perto: após oito lances, a probabilidade fica entre 0,08 e 0,28, e nenhuma partida passa do limiar com τ = 0,1. Em um tabuleiro 4×4, portanto, os lances do oponente não revelam a sua preferência, e trocar τ não muda isso.

Isso explica o resultado do slide anterior. A crença que chega à política traz pouca informação, então o agente ω̂ se comporta quase como o MO-PPO.

---

## Limitações

O limite mais importante é que, no teste, o filtro conhece apenas os oponentes de treino. O oponente verdadeiro nunca faz parte das hipóteses (TD-19), então o agente ω̂ é avaliado com um modelo que não contém justamente o que deveria identificar. Uma versão com o grid completo ainda precisa ser rodada para comparar.

O modelo do oponente também tem limites. Sua tabela supõe que o oponente tem a mesma preferência que o agente, que é uma simplificação conhecida (TD-17). O parâmetro τ é um valor provisório (TD-02), e ainda não temos a melhor resposta exata, então não há medida de regret (TD-01).

Por fim, o MO-PPO apareceu com 0,366 de vitória na primeira rodada e com 0,355 na repetição. As duas estão registradas, e a diferença fica dentro do intervalo de confiança.

---

## Próximos passos

1. **Decidir o gate da fase 0** (TD-18): mudar a observação ou o horizonte, ou reportar que ω não é identificado em uma partida e apoiar a claim no gap até a melhor resposta.
2. Rodar a sensibilidade com o grid completo como hipótese (D1, TD-19).
3. Calibrar τ (TD-02) e implementar a melhor resposta exata com regret (TD-01).
4. Agente oráculo (TD-04) e métrica de erro de ω̂ por lance (TD-05).
5. Figuras, tabela final e texto do paper (TD-06, TD-07).

---

<!-- _class: lead -->

# Obrigado
