---
marp: true
theme: default
paginate: true
math: katex
size: 16:9
title: Inferência de preferências do oponente em MO-Connect4
style: |
  section { font-size: 26px; }
  h1 { color: #1f3a5f; }
  h2 { color: #1f3a5f; }
  table { font-size: 22px; }
  code { font-size: 0.9em; }
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

Contra oponentes com $$\omega$$ **não vistos no treino**, um agente que infere $$\hat{\omega}$$ tem utilidade maior do que o MO-PPO sem modelo de oponente.

Ordem esperada:

```
melhor resposta exata ≥ oráculo ≥ agente ω̂ ≥ MO-PPO
```

- Gap melhor resposta ↔ oráculo: custo da aproximação por RL.
- Gap oráculo ↔ $$\hat{\omega}$$: custo de não conhecer ω do oponente.

---

## Ambiente e solver

- **MO-Connect4 4×4**, três objetivos: vencer, vencer rápido, controle de colunas.
- **Utilidade linear:** u = ω·R.
- **Solver exato** por indução reversa: ~134k estados não terminais, resolvido em segundos.
- **Oponentes:** política softmax(Q_k / τ) sobre 15 pontos do simplex (passo 0,25), com τ = 0,1).
- **Split por grupo de comportamento** (12 grupos): treino e teste nunca compartilham grupo.

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

## Por que não há ganho: teste de identificação (fase 0)

- Medido com política aleatória, 100 jogos por oponente, 8 lances.
- Massa posterior no grupo verdadeiro após 8 lances: **0,07 a 0,28**.
- Taxa de passagem (massa ≥ 0,9): **0,00** para τ = 0,1; até 0,04 para τ = 0.
- Em um jogo 4×4, os lances do oponente **não identificam ω**. Trocar τ não resolve.

Consequência: o agente recebe pouca informação útil no vetor de crença.

---

## Limitações

- **Filtro mal especificado no teste:** as hipóteses são só os oponentes de treino (TD-19), então o verdadeiro oponente nunca está no suporte do filtro.
- **Modelo do oponente:** a tabela é `solve_spe(ω, ω)`, que ignora o ω do agente (TD-17).
- **τ placeholder** (TD-02); sem métrica de regret ainda (TD-01).
- **Uma versão do baseline:** o MO-PPO re-rodado dá vitória 0,355, contra 0,366 na primeira tabela; as duas estão registradas.

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
