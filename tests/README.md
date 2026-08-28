# Numerical validation of the neural-network core

## Italiano

Questa suite confronta il nucleo matematico di **Dentro la rete** con PyTorch usando calcoli in `float64`.

Il test non contiene una seconda implementazione manuale della backpropagation. `generate_fixtures.mjs` legge direttamente il file HTML dell'applicazione, individua il blocco `<script>` che contiene `forward`, `computeStep` e `applyStep`, lo esegue in un contesto Node isolato e salva i risultati di una serie di casi deterministici. `test_engine.py` ricostruisce le stesse reti con operazioni PyTorch e usa `autograd` per calcolare le derivate di riferimento.

Vengono confrontati:

- somme pesate `z`;
- attivazioni di tutti gli strati;
- output della rete;
- loss del singolo esempio;
- termini `delta` della backpropagation;
- derivate rispetto a tutti i pesi e bias;
- variazioni dei parametri prodotte dal learning rate;
- pesi e bias dopo l'aggiornamento.

I casi coprono classificazione binaria, regressione, classificazione multiclasse con softmax e cross-entropy, reti con zero, uno e due strati nascosti e attivazioni `sigmoid`, `tanh` e `ReLU`. È incluso anche un caso `9 → 9 → 6`, che esercita contemporaneamente il massimo di 9 input e 6 classi previsto per i dataset importati. Un test separato verifica esplicitamente la convenzione `ReLU'(0) = 0` usata dall'applicazione e da PyTorch.

Durante la generazione delle fixture viene inoltre verificato, per ogni caso a singolo passo, che `trainingStep` — l'entry point usato dal ciclo di allenamento dell'applicazione — produca esattamente lo stesso oggetto di passo e gli stessi parametri finali della composizione esplicita `computeStep` + `applyStep`. Questo controllo avviene interamente nello stesso motore JavaScript e richiede quindi uguaglianza esatta.

Per alcuni parametri le derivate vengono controllate anche con differenze finite centrali. Questo controllo ha una tolleranza più larga di quello con `autograd`, perché la derivazione numerica introduce errore di troncamento e arrotondamento.

Sono incluse anche due brevi sequenze di sei aggiornamenti. Servono come controllo dell'interazione tra calcolo delle derivate e aggiornamento dei parametri; non cercano di imporre identità numerica dopo allenamenti lunghi.

### Esecuzione locale

Requisiti: Node.js, Python, `pytest` e PyTorch.

Dalla radice del repository:

```bash
python -m pip install pytest torch
node tests/generate_fixtures.mjs
pytest -q tests/test_engine.py
```

Nel repository del progetto il generatore usa automaticamente `en/index.html`; se quel file non è presente prova `it/index.html`. Come fallback, continua a riconoscere anche eventuali file HTML standalone versionati presenti nella radice. È sempre possibile indicare esplicitamente il file da validare:

```bash
node tests/generate_fixtures.mjs --html en/index.html
```

Per controllare manualmente anche la versione italiana:

```bash
node tests/generate_fixtures.mjs --html it/index.html
pytest -q tests/test_engine.py
```

Il JSON generato si trova in `tests/fixtures/generated_reference.json` ed è ignorato da Git grazie al `.gitignore` locale della cartella.

La tolleranza principale è `rtol = 1e-12`, `atol = 1e-13`. Non si richiede uguaglianza bit per bit: JavaScript e PyTorch possono eseguire operazioni matematicamente equivalenti in ordini leggermente diversi.

Il workflow `.github/workflows/numerical-validation.yml` rigenera le fixture e confronta con PyTorch sia `en/index.html` sia `it/index.html`. Si avvia automaticamente quando cambiano l’app, la suite o il workflow, e può essere avviato anche manualmente da GitHub Actions.

---

## English

This suite cross-checks the mathematical core of **Inside the Network** against PyTorch using `float64` calculations.

The test does not contain a second hand-written implementation of backpropagation. `generate_fixtures.mjs` reads the application's actual HTML file, locates the `<script>` block containing `forward`, `computeStep`, and `applyStep`, runs it in an isolated Node context, and saves the results of a set of deterministic cases. `test_engine.py` rebuilds the same networks with PyTorch operations and uses `autograd` to obtain the reference derivatives.

The suite compares:

- weighted sums `z`;
- activations at every layer;
- network outputs;
- per-example loss;
- backpropagation `delta` terms;
- derivatives with respect to every weight and bias;
- parameter changes produced by the learning rate;
- weights and biases after the update.

The cases cover binary classification, regression, multiclass classification with softmax and cross-entropy, networks with zero, one, and two hidden layers, and `sigmoid`, `tanh`, and `ReLU` hidden activations. A `9 → 9 → 6` case also exercises the maximum supported imported-data shape of 9 inputs and 6 classes. A separate test explicitly checks the `ReLU'(0) = 0` convention used by both the application and PyTorch.

While generating the fixtures, the suite also checks every single-step case to ensure that `trainingStep` — the entry point used by the application's training loop — produces exactly the same step object and final parameters as the explicit `computeStep` + `applyStep` composition. Because both paths run in the same JavaScript engine, this check requires exact equality.

For selected parameters, the gradients are also checked with centered finite differences. This check uses a looser tolerance than the autograd comparison because numerical differentiation introduces truncation and rounding error.

Two short six-update trajectories are included as well. They check the interaction between gradient calculation and parameter updates without requiring two implementations to remain numerically identical over long training runs.

### Running locally

Requirements: Node.js, Python, `pytest`, and PyTorch.

From the repository root:

```bash
python -m pip install pytest torch
node tests/generate_fixtures.mjs
pytest -q tests/test_engine.py
```

In this repository the generator automatically uses `en/index.html`; if that file is not present, it tries `it/index.html`. As a fallback, it still recognizes versioned standalone HTML files placed in the repository root. You can always select the application file explicitly:

```bash
node tests/generate_fixtures.mjs --html en/index.html
```

To check the Italian version manually as well:

```bash
node tests/generate_fixtures.mjs --html it/index.html
pytest -q tests/test_engine.py
```

The generated JSON is written to `tests/fixtures/generated_reference.json` and is ignored by Git through the local `.gitignore` in that directory.

The main tolerance is `rtol = 1e-12`, `atol = 1e-13`. Bit-for-bit equality is not required: JavaScript and PyTorch may evaluate mathematically equivalent expressions in slightly different orders.

The `.github/workflows/numerical-validation.yml` workflow regenerates the fixtures and cross-checks both `en/index.html` and `it/index.html` against PyTorch. It runs automatically when the application, test suite, or workflow changes, and it can also be started manually from GitHub Actions.
