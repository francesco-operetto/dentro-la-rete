# Dataset di test — 9 input, 6 classi, 2000 esempi

File associato: `test_import_9_input_6_classi_2000_difficolta_intermedia.csv`

## Italiano

### Scopo

Questo dataset è stato creato per testare la funzione di importazione di **Dentro la rete** vicino ai limiti previsti dall’applicazione:

- 2000 esempi;
- 9 colonne numeriche di input;
- 6 classi;
- nessun valore mancante;
- tutte le colonne di input variabili;
- classi quasi perfettamente bilanciate.

A differenza di un dataset composto da cluster molto separati, questo file è stato progettato per produrre un problema di classificazione di **difficoltà intermedia**. L’obiettivo è osservare un apprendimento graduale, non una convergenza quasi immediata al 100%.

Il comportamento preciso dipende comunque dal seed usato nell’applicazione, dalla suddivisione fra allenamento e verifica, dall’architettura della rete, dalla funzione di attivazione e dal tasso di apprendimento.

### Struttura del file

Il CSV contiene una riga di intestazione seguita da 2000 esempi.

Le colonne sono:

`input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, classe`

Distribuzione delle classi:

| Classe | Esempi |
|---|---:|
| C1 | 334 |
| C2 | 334 |
| C3 | 333 |
| C4 | 333 |
| C5 | 333 |
| C6 | 333 |

Le righe sono state mescolate casualmente, quindi l’ordine nel file non contiene informazioni sulla classe.

### Come sono stati generati gli input

La generazione parte da sei centri di classe in uno spazio latente a sei dimensioni:

| Classe | z1 | z2 | z3 | z4 | z5 | z6 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.00 | 0.30 | -0.40 | 0.00 | 0.25 | -0.15 |
| C2 | 0.60 | 0.90 | 0.10 | -0.20 | -0.10 | 0.20 |
| C3 | -0.20 | 1.00 | 0.50 | 0.30 | 0.15 | 0.05 |
| C4 | -0.90 | 0.20 | 0.70 | 0.70 | -0.20 | -0.10 |
| C5 | -0.70 | -0.80 | 0.00 | 0.80 | 0.05 | 0.25 |
| C6 | 0.20 | -0.90 | -0.60 | 0.30 | -0.15 | -0.20 |

Per ogni esempio:

- `z1`–`z4` sono estratti da distribuzioni gaussiane centrate sui valori della classe, con deviazione standard `0.34`. Sono gli input che contengono la parte principale dell’informazione utile alla classificazione.
- `z5`–`z6` sono estratti da gaussiane centrate sui valori della classe, ma con deviazione standard `0.90`. Il segnale di classe è quindi molto più debole e le distribuzioni si sovrappongono maggiormente.
- `z7`–`z9` sono estratti indipendentemente da una normale standard `N(0,1)` e non dipendono dalla classe. Funzionano come variabili di disturbo.

I nove valori salvati nel CSV vengono poi ottenuti applicando scale numeriche molto diverse:

```text
input_1 = 100 + 50·z1
input_2 = -3 + 2·z2
input_3 = 5000 + 1000·z3
input_4 = 0.8 + 0.2·z4
input_5 = 10·z5
input_6 = 20 + z6
input_7 = 100·z7
input_8 = 0.01·z8
input_9 = -50 + 5·z9
```

Queste scale differenti sono intenzionali: permettono di verificare anche la normalizzazione separata delle colonne eseguita durante l’importazione.

### Riproducibilità

Il file è stato generato con un generatore pseudocasuale inizializzato con il seed:

```text
20260827
```

A parità di algoritmo e seed, il dataset può quindi essere rigenerato in modo deterministico.

Dopo la generazione, le 2000 righe sono state mescolate usando lo stesso generatore pseudocasuale.

### Controlli effettuati

Durante la creazione sono stati verificati automaticamente:

- esattamente 2000 esempi;
- 9 colonne di input più la colonna `classe`;
- esattamente 6 classi;
- distribuzione delle classi prevista;
- stesso numero di colonne in ogni riga;
- tutte le colonne di input interpretabili come numeri;
- nessuna colonna di input costante.

Il file usa la virgola come separatore CSV e il punto come separatore decimale.

---

## English

### Purpose

This dataset was created to test the import feature of **Inside the Network** close to the application's supported limits:

- 2,000 examples;
- 9 numerical input columns;
- 6 classes;
- no missing values;
- every input column varies;
- almost perfectly balanced classes.

Unlike a dataset made of widely separated clusters, this file was designed as a **moderately difficult** classification problem. Its purpose is to make learning progress gradually rather than allowing the network to reach 100% accuracy almost immediately.

The exact behavior still depends on the seed used in the application, the training/validation split, the network architecture, the activation function, and the learning rate.

### File structure

The CSV contains one header row followed by 2,000 examples.

The columns are:

`input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, classe`

Class distribution:

| Class | Examples |
|---|---:|
| C1 | 334 |
| C2 | 334 |
| C3 | 333 |
| C4 | 333 |
| C5 | 333 |
| C6 | 333 |

The rows were shuffled randomly, so file order carries no information about class membership.

### How the inputs were generated

Generation starts from six class centers in a six-dimensional latent space:

| Class | z1 | z2 | z3 | z4 | z5 | z6 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.00 | 0.30 | -0.40 | 0.00 | 0.25 | -0.15 |
| C2 | 0.60 | 0.90 | 0.10 | -0.20 | -0.10 | 0.20 |
| C3 | -0.20 | 1.00 | 0.50 | 0.30 | 0.15 | 0.05 |
| C4 | -0.90 | 0.20 | 0.70 | 0.70 | -0.20 | -0.10 |
| C5 | -0.70 | -0.80 | 0.00 | 0.80 | 0.05 | 0.25 |
| C6 | 0.20 | -0.90 | -0.60 | 0.30 | -0.15 | -0.20 |

For each example:

- `z1`–`z4` are sampled from Gaussian distributions centered on the values for that class, with standard deviation `0.34`. These variables carry most of the useful class signal.
- `z5`–`z6` are sampled from Gaussian distributions centered on the class values, but with standard deviation `0.90`. Their class signal is therefore much weaker and the distributions overlap much more.
- `z7`–`z9` are sampled independently from a standard normal distribution `N(0,1)` and do not depend on the class. They act as nuisance variables.

The nine values written to the CSV are then placed on deliberately different numerical scales:

```text
input_1 = 100 + 50·z1
input_2 = -3 + 2·z2
input_3 = 5000 + 1000·z3
input_4 = 0.8 + 0.2·z4
input_5 = 10·z5
input_6 = 20 + z6
input_7 = 100·z7
input_8 = 0.01·z8
input_9 = -50 + 5·z9
```

These different scales are intentional: they also exercise the per-column normalization performed during import.

### Reproducibility

The file was generated with a pseudorandom number generator initialized with the seed:

```text
20260827
```

With the same algorithm and seed, the dataset can therefore be regenerated deterministically.

After generation, all 2,000 rows were shuffled using the same pseudorandom generator.

### Validation performed

The generation script automatically checked that the file contains:

- exactly 2,000 examples;
- 9 input columns plus the `classe` column;
- exactly 6 classes;
- the intended class distribution;
- the same number of columns in every row;
- numerical values in every input column;
- no constant input columns.

The file uses commas as CSV separators and decimal points for numerical values.
