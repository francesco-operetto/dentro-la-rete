# Numerical validation of the neural-network engine

## Italiano

Questa suite confronta il motore numerico di **Dentro la rete** con PyTorch usando calcoli in `float64`.

La suite usa direttamente il motore JavaScript dell'applicazione per generare le fixture. `generate_fixtures.mjs` legge il file HTML, individua il blocco `<script>` dedicato al motore numerico, ne controlla l'isolamento dalle definizioni dei problemi e dal DOM e lo esegue in un contesto Node isolato. `test_engine.py` ricostruisce le stesse reti in PyTorch usando `float64` e usa `autograd` per calcolare le derivate di riferimento.

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

Sono incluse anche due brevi sequenze di sei aggiornamenti, usate per controllare l'interazione tra calcolo delle derivate e aggiornamento dei parametri.

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

Il confronto principale usa `rtol = 1e-12` e `atol = 1e-13`, così da ammettere le piccole differenze dovute all'ordine delle operazioni floating-point tra JavaScript e PyTorch.

Il workflow `.github/workflows/numerical-validation.yml` rigenera le fixture e confronta con PyTorch sia `en/index.html` sia `it/index.html`; può anche essere avviato manualmente da GitHub Actions.

---

## English

This suite cross-checks the numerical engine of **Inside the Network** against PyTorch using `float64` calculations.

The suite uses the application's JavaScript numerical engine directly to generate the fixtures. `generate_fixtures.mjs` reads the HTML file, locates the numerical-engine `<script>` block, checks its isolation from problem definitions and the DOM, and runs it in an isolated Node context. `test_engine.py` rebuilds the same networks in PyTorch using `float64` and uses `autograd` to compute the reference derivatives.

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

While generating the fixtures, the suite also checks every single-step case to ensure that `trainingStep` — the entry point used by the application's training loop — produces exactly the same step object and final parameters as the explicit `computeStep` + `applyStep` composition. Since both paths run in the same JavaScript engine, this check asserts exact equality.

For selected parameters, the gradients are also checked with centered finite differences. This check uses a looser tolerance than the autograd comparison because numerical differentiation introduces truncation and rounding error.

The suite also includes two short six-update trajectories to check the interaction between gradient calculation and parameter updates.

### Running locally

Requirements: Node.js, Python, `pytest`, and PyTorch.

From the repository root:

```bash
python -m pip install pytest torch
node tests/generate_fixtures.mjs
pytest -q tests/test_engine.py
```

In the project repository, the generator automatically uses `en/index.html`; if that file is not present, it tries `it/index.html`. As a fallback, it still recognizes versioned standalone HTML files in the repository root. You can always select a file explicitly:

```bash
node tests/generate_fixtures.mjs --html en/index.html
```

To check the Italian version manually as well:

```bash
node tests/generate_fixtures.mjs --html it/index.html
pytest -q tests/test_engine.py
```

The generated JSON is written to `tests/fixtures/generated_reference.json` and is ignored by Git through the local `.gitignore` in that directory.

The main comparison uses `rtol = 1e-12` and `atol = 1e-13` to allow for small floating-point differences caused by the order of operations in JavaScript and PyTorch.

The `.github/workflows/numerical-validation.yml` workflow regenerates the fixtures and compares both `en/index.html` and `it/index.html` with PyTorch; it can also be started manually from GitHub Actions.
