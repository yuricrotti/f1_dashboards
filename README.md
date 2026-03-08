# F1 Intelligence Studio

Dashboard analítico de Fórmula 1 construído em **Streamlit** com dados da API **OpenF1**.

O projeto foi pensado para análise técnica e apresentação executiva, com foco em:
- leitura rápida de sessão (treino, quali, corrida)
- comparativos entre pilotos
- análises estratégicas (pneus, consistência, long-run)
- narrativa visual pronta para uso comercial

---

## 1. Visão Geral

O sistema consulta dados públicos da OpenF1 e transforma em painéis interativos com:
- scorecards de performance por piloto e equipe
- insights automáticos
- comparações entre pilotos e teammates
- evolução de posição por volta (em corrida)
- heatmaps e visualização setorial 3D
- bloco de texto pronto para post

### Importante sobre sessões de corrida
Em `Race`, o app diferencia:
- **Vencedor oficial**: via `session_result`
- **Melhor volta (fastest lap)**: via `laps`

Ou seja, o vencedor e o dono da melhor volta podem ser pilotos diferentes.

---

## 2. Stack

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Plotly
- Requests

Dependências no arquivo [requirements.txt](/home/galaxia/Documents/Code/f1_project/requirements.txt).

---

## 3. Estrutura do Projeto

```text
f1_project/
├── app.py                    # Entry point (launcher da aplicação)
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py             # Configurações globais (API, cache, defaults)
    └── dashboard_app.py      # Lógica principal (dados + analytics + UI)
```

### Responsabilidade de cada arquivo
- [app.py](/home/galaxia/Documents/Code/f1_project/app.py): apenas importa e executa `main()`.
- [config.py](/home/galaxia/Documents/Code/f1_project/src/config.py): concentra constantes e defaults (`BASE`, `AppConfig`, `PLOT_CONFIG`).
- [dashboard_app.py](/home/galaxia/Documents/Code/f1_project/src/dashboard_app.py): implementação completa da aplicação.

---

## 4. Como Rodar

## Pré-requisitos
1. Python 3.10+ instalado
2. Ambiente virtual recomendado

## Instalação
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução
```bash
streamlit run app.py
```

A aplicação abrirá no navegador com URL local (geralmente `http://localhost:8501`).

---

## 5. Fluxo de Dados

1. Usuário seleciona ano, país e sessão na sidebar.
2. App consulta `sessions` para resolver `session_key`.
3. Com `session_key`, carrega endpoints principais:
   - `drivers`
   - `laps`
   - `stints`
   - `weather`
   - `race_control`
   - `session_result`
   - `position`
   - (quando aplicável) `intervals`
4. Pipeline de preparação:
   - best laps por piloto
   - rankings setoriais
   - resumo de equipe
   - classificação de voltas (push/traffic/inlap/outlap)
   - métricas de long-run
   - scorecards
5. Render das abas analíticas.

### Caching
A aplicação usa `@st.cache_data` com TTL para reduzir chamadas repetidas à OpenF1 e acelerar navegação.

---

## 6. Organização das Abas

A navegação foi reorganizada para não misturar assuntos:

## Visão Executiva
- infográfico principal
- scorecards por piloto/equipe
- classificação oficial (corrida) ou top laps (não-corrida)
- insights automáticos

## Corrida (aparece só em `Race`)
- evolução de posição por volta
- linha do tempo da corrida

## Pilotos
- comparativo entre 2-4 pilotos
- delta entre 2 pilotos
- teammate intelligence

## Setores
- heatmap setorial
- gráfico 3D (S1 x S2 x S3)

## Estratégia
- tyre analytics
- classificação automática de voltas
- ritmo e consistência dos pilotos foco
- long-run analytics

## Dados & Conteúdo
- texto pronto para post
- visualização de dados brutos por endpoint

---

## 7. Métricas e Regras Analíticas

## Best lap vs resultado final
- best lap: menor `lap_duration`
- resultado final (corrida): `session_result`

## Scorecard (0-100)
Combinação ponderada de:
- pace score
- consistency score
- degradation score
- execution score

## Classificação automática de voltas
As voltas podem ser classificadas como:
- `push`
- `traffic`
- `outlap`
- `inlap`

## Long-run
Agrupamento por stint para estimar:
- pace médio
- consistência (desvio)
- degradação (`s/lap`)

---

## 8. Decisões de Produto Importantes

- Em corrida, o dashboard evita confundir "vencedor" com "melhor volta".
- A evolução de posição por volta preenche voltas faltantes por piloto para evitar linhas truncadas quando não há update de posição em todas as voltas.
- Se alguns endpoints vierem incompletos, os gráficos usam fallback e mensagens informativas na UI.

---

## 9. Troubleshooting

## Erro de merge/coluna (ex.: `KeyError: driver_number`)
- Já tratado no pipeline de evolução de posição por volta.
- Se ocorrer novamente, valide tipos e presença das colunas em `position` e `laps`.

## Erro de `fillna` com valor inválido
- Já corrigido com checagem explícita de colunas antes de `fillna`.

## Dados inesperados na corrida
- Verifique se a sessão selecionada é realmente `Race` e se `session_result` está populado.
- Lembre-se: fastest lap pode não ser do vencedor.

## API sem dados para sessão específica
- Algumas sessões podem ter dados incompletos ou atrasados.
- Troque sessão/ano/país e compare.

---

## 10. Boas Práticas Adotadas

- configuração centralizada (`src/config.py`)
- entrypoint enxuto (`app.py`)
- funções separadas por responsabilidade (fetch, prep, analytics, chart)
- cache para endpoints caros
- tratamento de cenários com dados ausentes
- organização por seções de negócio na UI

---

## 11. Próximas Melhorias Sugeridas

- exportação PDF executiva
- modo apresentação (client-facing)
- comparação multi-sessão (FP1/FP2/Quali/Race)
- score de confiança por insight
- otimização de pit window com simulação probabilística

---

## 12. Licença e Dados

- Este projeto usa dados de terceiros via OpenF1.
- Verifique termos de uso da API para uso comercial/redistribuição.

