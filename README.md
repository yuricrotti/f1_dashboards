# F1 Intelligence Studio

Dashboard de análise de sessões de F1 com dados da OpenF1.

## Stack
- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Plotly
- Requests

## Estrutura
```text
f1_project/
├── app.py
├── requirements.txt
└── src/
    ├── config.py
    ├── dashboard_app.py
    ├── helpers.py
    ├── data_layer.py
    ├── analytics.py
    ├── charts.py
    └── ui_components.py
```

## Arquivos principais
- `app.py`: entrypoint.
- `src/dashboard_app.py`: fluxo principal e montagem das abas.
- `src/data_layer.py`: coleta/preparo de dados da OpenF1.
- `src/analytics.py`: métricas, rankings, scorecards, insights.
- `src/charts.py`: gráficos Plotly.
- `src/ui_components.py`: estilo e componentes visuais.
- `src/config.py`: constantes e configurações globais.

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

## Fontes OpenF1 usadas
- `sessions`
- `drivers`
- `laps`
- `stints`
- `weather`
- `race_control`
- `session_result`
- `position`
- `intervals` (quando disponível)

## Organização da UI
- Visão Executiva
- Corrida (somente quando `session_type == race`)
- Pilotos
- Setores
- Estratégia
- Dados & Conteúdo

## Regras importantes
- Em corrida, **vencedor oficial** vem de `session_result`.
- **Melhor volta** vem de `laps`.
- Vencedor e melhor volta podem ser pilotos diferentes.

## Problemas comuns
- Dados faltando em alguns endpoints: comportamento esperado da API para certas sessões.
- Diferenças entre classificação e fastest lap: comportamento correto.

## Roadmap curto
- Exportação de relatório (PDF)
- Comparação multi-sessão
- Modo apresentação
