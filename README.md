# World Model com Transição Difusiva para Dinâmica de Cripto

Pipeline completo, em um único notebook, que compara duas cabeças de transição probabilísticas para previsão de retornos de **BTC/USDT (1h)**:

- **WM-Gauss** — baseline: gaussiana diagonal `(μ, log σ)` sobre a próxima observação.
- **WM-Diff** — difusão condicional (DDPM) que modela a mesma distribuição condicional via denoising.

Ambos compartilham o mesmo encoder recorrente (GRU), de modo que a diferença medida vem da cabeça de transição, não da recorrência.

> Projeto de pesquisa/estudo. Nada aqui é recomendação de investimento.

---

## Visão geral

```
Binance klines (1h) ──► features ──► janelas (24h) ──► GRU encoder ──► h_t ─┬─► Cabeça gaussiana  (WM-Gauss)
                                                                            └─► Cabeça DDPM       (WM-Diff)
                                                                                      │
                                      avaliação: 1 passo · horizontes fixos · rollout ◄┘
```

| Etapa | Descrição |
|---|---|
| Dados | Klines públicos da Binance (sem chave), paginados e cacheados em CSV |
| Features | `log_ret`, `range_norm`, `log_volume`, `log_trades`, `rv_24h` |
| Split | Temporal estrito 70/15/15 (treino/val/teste), sem embaralhar |
| Normalização | z-score com estatísticas **apenas do treino** |
| Modelos | WM-Gauss, WM-Diff (DDPM no treino, DDIM na avaliação) |
| Baselines | Trivial (drift + gaussiana fixa), GARCH(1,1), LSTM de 1 passo |
| Avaliação | NLL/ELBO 1 passo · CRPS/Pinball/PICP · Wasserstein-1 de rollout |
| Robustez | IC bootstrap, Wilcoxon pareado, estratificação por regime, validação em ETHUSDT |

---

## Arquivos

```
wm_diff_cripto_v2.ipynb   # pipeline completo
data/                     # klines brutos (cache CSV)
checkpoints/              # wm_gauss.pt, wm_diff.pt
figures/                  # figuras e tabela-resumo
```

`data/`, `checkpoints/` e `figures/` são criados automaticamente na primeira execução.

## Requisitos

- Python 3.10+
- `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`, `requests`
- `arch` (opcional, para o baseline GARCH)

```bash
pip install torch numpy pandas matplotlib scipy requests arch
```

GPU é detectada automaticamente (`cuda`); sem GPU roda em CPU.

## Como executar

1. Abra `wm_diff_cripto_v2.ipynb` e rode as células em ordem.
2. Na primeira execução os klines são baixados da API pública da Binance e salvos em `data/`; reexecuções usam o cache.
3. O modo de execução é controlado por `TEST_MODE` (seção 1):

| Constante | `TEST_MODE = True` | `TEST_MODE = False` |
|---|---|---|
| `HISTORY_DAYS` | 60 | 730 (~2 anos) |
| `BATCH_SIZE` | 64 | 256 |
| `N_EPOCHS` | 3 | 100 |
| `DIFFUSION_STEPS_TRAIN` | 20 | 50 |
| `DIFFUSION_STEPS_DDIM_EVAL` | 5 | 10 |
| `ROLLOUT_HORIZONS` | 1, 6, 24 | 1, 6, 24, 168 |
| `N_ROLLOUT_SAMPLES` | 32 | 200 |
| `N_BOOTSTRAP` | 200 | 2000 |

Use `TEST_MODE = True` para validar o pipeline ponta a ponta rapidamente e `False` para o treino real. A seção 12 (sanity check sintético) é recomendada antes do treino real.

---

## Arquitetura

**Encoder (compartilhado).** `GRU(obs_dim=5 → hidden_dim=128)`, 1 camada; consome as últimas `CONTEXT_LEN = 24` observações e devolve o estado final `h_t`.

**WM-Gauss.** MLP `h_t → (μ, log σ)` com `log σ` limitado a `[-7, 3]`; treinado por NLL.

**WM-Diff.** DDPM condicional com schedule linear de β (`1e-4 → 0.02`). A rede ε recebe `(x_k, h_t, embedding sinusoidal de k)` e é treinada com o objetivo de denoising padrão (MSE de ε). Amostragem:
- **DDPM ancestral** (`n_steps` completos) — amostragem completa.
- **DDIM** com poucos passos — usado na avaliação e no rollout para viabilizar o custo.

**Treino.** Adam, `lr = 1e-3`, `ReduceLROnPlateau`, early stopping (`patience = 5`) e restauração do melhor checkpoint por loss de validação.

## Baselines

Todos seguem a mesma interface de `WorldModel`, então reaproveitam exatamente as mesmas funções de avaliação.

- **Trivial:** `N(0, 1)` i.i.d. por feature no espaço normalizado (equivale a drift = média do treino e variância = variância do treino).
- **GARCH(1,1):** variância de `log_ret` pela recursão `σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁`, filtrada sobre o contexto e depois sobre os próprios valores gerados. As demais features usam a distribuição trivial. Requer `arch`; se ausente, o baseline é omitido.
- **LSTM de 1 passo:** mesma cabeça gaussiana com recorrência LSTM, para isolar se o ganho vem da difusão ou só da recorrência.

---

## Avaliação

Três níveis:

1. **Um passo** — NLL (gaussianos e baselines) ou proxy de ELBO (WM-Diff) no teste.
2. **Horizontes fixos** (1, 6, 24, 168 passos) — CRPS empírico, pinball (mediana) e PICP do intervalo de 90%, sobre pontos de partida amostrados do teste.
3. **Qualidade de rollout** — distância de Wasserstein-1 entre valores gerados e realizados por horizonte, além de volatilidade, assimetria e curtose.

Análises complementares:

- **IC por bootstrap** do CRPS médio por horizonte (reamostrando janelas de partida, não amostras do modelo).
- **Teste de Wilcoxon pareado** WM-Gauss vs. WM-Diff, por ponto de partida e horizonte.
- **Estratificação por regime** de volatilidade (alta/baixa, split pela mediana de `rv_24h`).
- **Validação cruzada em ETHUSDT**, reaplicando os modelos treinados em BTC, sem refit.
- **Ablações:** passos DDIM (tempo de execução) e decomposição de erro 1 passo vs. rollout.
- **Sanity check sintético:** o DDPM deve recuperar uma mistura bimodal que uma gaussiana única não captura.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `checkpoints/wm_gauss.pt`, `wm_diff.pt` | Pesos treinados |
| `figures/curvas_treino.png` | Curvas de loss de validação |
| `figures/painel_histogramas.png` | Real vs. WM-Gauss vs. WM-Diff |
| `figures/wasserstein_vs_horizonte.png` | Wasserstein-1 por horizonte, todos os modelos |
| `figures/preco_faixa_incerteza.png` | Preço com faixa de 90% e mediana prevista |
| `figures/tabela_resumo_metricas.csv` | CRPS e PICP por modelo e horizonte |

---

## Limitações conhecidas

- **ELBO ≠ NLL.** A NLL exata não é tratável para o DDPM; o WM-Diff usa o MSE de denoising por exemplo como proxy. É comparável em ranking/tendência, **não em escala absoluta** com a NLL dos demais.
- **Poucas janelas.** São ~`len(test)//20` pontos de partida por horizonte, e nos horizontes longos as janelas se sobrepõem no tempo. Mesmo p-valores baixos do Wilcoxon devem ser lidos com cautela.
- **GARCH sem refit por janela.** O burn-in da variância é só o contexto (`CONTEXT_LEN` passos); pode subestimar a persistência se `α + β` estiver muito próximo de 1.
- **Mediana achatada no gráfico de preço.** Com drift de retorno próximo de zero, a mediana do preço integrado fica quase constante; a informação útil está na faixa de incerteza, não na direção da mediana.
- **Backtest/trading fora de escopo.** O projeto avalia a qualidade probabilística da previsão, não uma estratégia.

## Histórico da revisão (v2)

- Baselines (Trivial/GARCH/LSTM) passam a entrar nas mesmas métricas dos world models.
- IC por bootstrap e teste de Wilcoxon pareado para WM-Gauss vs. WM-Diff.
- Correção na quebra por regime: o ELBO do WM-Diff era o mesmo valor repetido para todo o batch; agora é calculado por exemplo (`reduction="none"`).
- Comentário explicativo sobre a mediana achatada no gráfico de preço.

## Reprodutibilidade

Seed global fixa (`SEED = 42`) para `random`, `numpy` e `torch`. Os dados dependem da data de execução (janela de `HISTORY_DAYS` até a hora atual), então resultados mudam entre downloads; o cache em `data/` fixa o conjunto usado.