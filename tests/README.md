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

### Probabilità molto piccole e underflow

L'errore multiclasse dell'applicazione usa `-Math.log(Math.max(Number.MIN_VALUE, p_y))`. Ogni probabilità positiva rappresentabile viene lasciata invariata; se la softmax restituisce zero per underflow, il valore dell'errore è circa `744,4400719213812`. `Number.MIN_VALUE` è il minimo positivo **subnormale**, non `Number.EPSILON` né il minimo numero normalizzato.

La suite include dieci controlli diretti della loss, da probabilità 1 e 0,5 fino a `Number.MIN_VALUE` e zero, compresi valori sopra, sulla e sotto la precedente soglia `1e-12`. Un nuovo caso a singolo passo verifica anche loss, gradienti, aggiornamenti e differenze finite con una probabilità bersaglio dell'ordine di `1e-100`.

Otto ulteriori casi coprono probabilità ordinarie, molto piccole, subnormali, arrotondate a zero e arrotondate a uno. Si verificano anche `sampleLoss`, `meanLoss`, la composizione di `trainingStep` e i parametri dopo l'aggiornamento. Per questi casi il valore dell'errore viene confrontato con il logaritmo della probabilità effettivamente prodotta da JavaScript, con la sola protezione sullo zero; i gradienti vengono invece confrontati con `torch.nn.functional.cross_entropy` sulle pre-attivazioni. Questa distinzione è intenzionale: in presenza di underflow la loss visualizzata può differire da quella teorica, mentre la backpropagation continua a usare `p - t`. Non si deriva la funzione con soglia usando autograd, perché non rappresenterebbe il gradiente adottato dall'applicazione.

Le probabilità estreme vengono confrontate con PyTorch con tolleranza assoluta pari al minimo positivo subnormale: la tolleranza ordinaria della suite nasconderebbe errori su valori così piccoli. Le differenze finite dei casi ordinari, compreso il nuovo caso con probabilità piccola ma normalizzata, usano la loss teorica calcolata dalle pre-attivazioni; non vengono applicate alla loss arrotondata nei casi estremi.

Il generatore estrae inoltre dal commento nel file HTML la funzione didattica `crossEntropyFromLogits`, verifica che sia assente dal codice attivo e la esegue in un contesto separato. La sua loss viene confrontata con PyTorch anche in underflow e dopo traslazioni comuni delle pre-attivazioni. In questo modo viene controllato l'esempio effettivamente presente nel sorgente, senza attivarlo nell'applicazione. Non viene imposto il limite artificiale di circa 744,44 all'alternativa basata sulle pre-attivazioni.

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

### Very small probabilities and underflow

The application's multiclass loss uses `-Math.log(Math.max(Number.MIN_VALUE, p_y))`. Every representable positive probability is left unchanged; if softmax returns zero through underflow, the reported loss is about `744.4400719213812`. `Number.MIN_VALUE` is the smallest positive **subnormal**, not `Number.EPSILON` or the smallest normal number.

The suite includes ten direct loss checks, from probabilities 1 and 0.5 down to `Number.MIN_VALUE` and zero, including values above, at, and below the former `1e-12` floor. A new single-step case also checks loss, gradients, updates, and finite differences with a target probability of the order of `1e-100`.

Eight additional cases cover ordinary, very small, subnormal, rounded-to-zero, and rounded-to-one probabilities. They also check `sampleLoss`, `meanLoss`, the composition of `trainingStep`, and the updated parameters. For these cases, the reported loss is compared with the logarithm of the probability actually produced by JavaScript, with only the zero guard; gradients are compared with `torch.nn.functional.cross_entropy` applied to logits. This distinction is intentional: under underflow the displayed loss can differ from the theoretical loss, while backpropagation still uses `p - t`. Autograd does not differentiate the clamped loss, since that would not represent the gradient used by the application.

Extreme probabilities are compared with PyTorch using an absolute tolerance equal to the smallest positive subnormal: the suite's ordinary tolerance would conceal errors in such tiny values. Finite differences for the ordinary cases, including the new small-but-normal target probability case, use the theoretical loss computed from logits; they are not applied to the rounded loss in the extreme cases.

The generator also extracts the teaching function `crossEntropyFromLogits` from its comment in the HTML file, checks that it is absent from the active code, and executes it in a separate context. Its loss is compared with PyTorch even under underflow and after common shifts of all logits. This tests the actual example in the source without enabling it in the application. The artificial limit of about 744.44 is not imposed on the logits-based alternative.

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
