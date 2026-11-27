# Oponentes de teste: quem são e em que diferem (`test_opponents.md`)

Os oponentes de teste são os de índice 9, 10 e 11 no grid de 15 pontos do simplex. Cada um é uma política fixa, definida por um vetor de preferência ω. Eles nunca entram no treino.

## Quem são

O vetor ω tem três pesos, na ordem **(vencer; vencer rápido; controle de colunas)**. Os pesos são não negativos e somam 1.

| Oponente | ω (vencer; vencer rápido; colunas) | O que ele valoriza |
|---|---|---|
| 9 | (0,25; 0; 0,75) | Sobretudo controle de colunas, com um quarto de peso em vencer |
| 10 | (0; 1; 0) | Apenas vencer rápido |
| 11 | (0; 0,75; 0,25) | Sobretudo vencer rápido, com um quarto de peso em colunas |

Cada um pertence a um grupo de comportamento próprio (grupos 9, 10 e 11). Ou seja, nenhum dos três joga exatamente igual a outro oponente do grid, em nenhum tabuleiro, quando o jogo é determinístico (τ = 0).

## Eles variam entre si?

Sim. Com τ = 0,1, a política de cada oponente é uma distribuição sobre colunas. Para medir o quanto duas políticas diferem, usamos a **distância de variação total** entre as duas distribuições, média sobre todas as 134.289 posições não terminais do jogo. A distância vai de 0 (idênticas) a 1 (completamente diferentes).

| Par | Distância |
|---|---|
| 9 × 10 | 0,12 |
| 9 × 11 | 0,12 |
| 10 × 11 | 0,05 |

Os oponentes 10 e 11 são os mais parecidos entre si, porque ambos valorizam vencer rápido. O 9 é o mais diferente dos dois, porque dá mais peso ao controle de colunas.

Em termos de lances, com τ = 0 a melhor jogada muda em cerca de 9% das posições quando comparamos o oponente 10 com os demais, e cerca de 10% para o 11. Para o 9, a média é de 7,5%.

## Parecidos com oponentes de treino

Como o filtro bayesiano não conhece esses três, é útil saber qual oponente de treino mais se parece com cada um:

| Oponente de teste | Oponente de treino mais parecido | ω do treino | Distância |
|---|---|---|---|
| 9 | 8 | (0,25; 0,25; 0,5) | 0,03 |
| 10 | 12 | (0; 0,5; 0,5) | 0,09 |
| 11 | 12 | (0; 0,5; 0,5) | 0,05 |

Esse é o motivo de a avaliação no teste ser uma pergunta de generalização. O agente vê oponentes próximos dos de treino, mas não os exatamente iguais.

## Como foi calculado

- Política de cada oponente: softmax(Q_k / τ) sobre as tabelas exatas (`data/q_tables_4x4.npz`), com τ = 0,1.
- Distância de variação total entre duas políticas, em uma posição: metade da soma dos valores absolutos das diferenças entre as duas distribuições sobre as colunas.
- Média sobre as 134.289 posições não terminais.
- Diferença na melhor jogada: fração das posições em que a coluna de maior Q difere entre os dois oponentes (τ = 0).

## Limitações

- A média inclui posições que nenhum jogo de teste visitaria. Ela mede a diferença estrutural das políticas, não a diferença no jogo real.
- A distância não diz qual dos oponentes é mais forte. Um oponente pode ser diferente e ainda assim ter utilidade parecida para o agente.
