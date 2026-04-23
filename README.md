# Modelagem Avancada - Reducao de Dimensionalidade

Projeto para comparar tecnicas de selecao/reducao de atributos sobre o dataset
`train.csv` ja limpo pelo projeto de pre-processamento.

## Objetivo

Comparar o desempenho de um modelo supervisionado binario usando diferentes
valores de `k` para:

- metodos filter;
- metodos wrapper;
- metodos embedded;
- PCA.

O dataset limpo esperado e o arquivo:

```text
modelagem-avancada-pre-processing/outputs/cleaned_train.csv
```

## Estrutura

```text
.
├── main.py
├── requirements.txt
├── README.md
└── reducao_dimensionalidade/
    ├── analysis.py
    ├── cleanup_manager.py
    ├── config.py
    ├── data_loader.py
    ├── experiment_runner.py
    ├── feature_pipeline.py
    ├── metrics_reporter.py
    ├── training_types.py
    └── trainers/
        ├── base.py
        ├── embedded_trainer.py
        ├── filter_trainer.py
        ├── pca_trainer.py
        └── wrapper_trainer.py
```

## Acoes da CLI

Atualizar o projeto de cleanup local:

```bash
python3 main.py --clone-cleanup
```

O comando acima remove a pasta local `modelagem-avancada-pre-processing/`,
quando ela existir, e clona novamente:

```text
https://github.com/felipesidooski/modelagem-avancada-pre-processing.git
```

Gerar analise inicial dos dados:

```bash
python3 main.py --analise
```

Executar os metodos separadamente:

```bash
python3 main.py --filter
python3 main.py --wrapper
python3 main.py --embedded
python3 main.py --pca
```

Executar tudo:

```bash
python3 main.py --todos
```

Consolidar metricas ja geradas:

```bash
python3 main.py --metricas
```

Gerar graficos a partir dos resultados:

```bash
python3 main.py --chart
```

Exibir os graficos na tela, sem salvar em `outputs/charts/`:

```bash
python3 main.py --chart -show
```

Tambem existem aliases com acento para `--análise` e `--métricas`.

## Parametros uteis

Definir valores de `k`:

```bash
python3 main.py --todos --k-values 5,10,15,20,30,45,60
```

Usar outro caminho para o dataset limpo:

```bash
python3 main.py --analise --data-path /caminho/para/cleaned_train.csv
```

Incluir variaveis sensiveis nos experimentos:

```bash
python3 main.py --todos --include-sensitive
```

Por padrao, `ORIENTACAO_SEXUAL` e `RELIGIAO` sao removidas dos experimentos,
mas continuam disponiveis para analise.

Executar filtros adicionais ou estrategias embedded mais pesadas:

```bash
python3 main.py --filter --filter-methods f_classif,mutual_info
python3 main.py --embedded --embedded-strategies l1,forest
```

## Saidas

```text
outputs/
├── analysis/
│   └── analysis_summary.json
├── experiments/
│   ├── results.csv
│   └── metrics_ranking.csv
└── charts/
    ├── comparativo_metricas_por_k.png
    ├── ranking_top_resultados.png
    └── melhor_resultado_metricas.png
```

Alem de salvar os arquivos, a CLI imprime um relatorio rapido no terminal com:

- dimensoes, distribuicao do target e alertas principais em `--analise`;
- melhor resultado e top metodos apos cada execucao de treino;
- ranking resumido quando `--metricas` e executado.
- caminhos dos graficos quando `--chart` e executado.
- nomes dos graficos exibidos quando `--chart -show` e executado.

## Metodos implementados no esboco

- `FilterTrainer`: `SelectKBest` com `f_classif` por padrao e `mutual_info_classif` opcional.
- `WrapperTrainer`: `RFE` com regressao logistica.
- `EmbeddedTrainer`: selecao via regressao logistica L1 por padrao e `ExtraTreesClassifier` opcional.
- `PCATrainer`: PCA com `n_components=k`.

Todos os metodos usam o mesmo classificador base no final:
`LogisticRegression(class_weight="balanced", solver="lbfgs")`, adequado para a
base desbalanceada. A penalidade L1 usa `liblinear` apenas como estimador
interno de selecao, porque esse solver e necessario para essa estrategia.
