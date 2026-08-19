# Dentro la rete / Inside the Network

**Versione / Version: 7.3.23 beta**

*Dentro la rete* (*Inside the Network*) è un simulatore didattico interattivo per esplorare il funzionamento di piccole reti neurali direttamente nel browser.

L'applicazione è costituita da HTML, CSS e JavaScript autonomi, senza librerie esterne. Può essere eseguita online tramite GitHub Pages oppure aperta localmente e usata offline.

*Inside the Network* is an interactive educational simulator for exploring how small neural networks work directly in the browser.

The application is self-contained HTML, CSS and JavaScript, with no external libraries. It can run online through GitHub Pages or be opened locally and used offline.

## Versioni / Languages

Le due versioni linguistiche sono organizzate allo stesso livello:

- **Italiano:** `it/index.html`
- **English:** `en/index.html`

La pagina `index.html` nella radice del repository è una pagina bilingue di ingresso che permette di scegliere la lingua.

The two language versions are organized symmetrically:

- **Italian:** `it/index.html`
- **English:** `en/index.html`

The root `index.html` is a bilingual landing page for choosing the language.

## Cosa permette di esplorare / What you can explore

- architettura della rete e numero di strati e neuroni;
- funzioni di attivazione;
- tasso di apprendimento e andamento dell'errore;
- pesi, bias, attivazioni e valori interni della rete;
- forward pass, funzione di errore, backpropagation e aggiornamento dei parametri, anche un esempio alla volta;
- mappe delle attivazioni;
- diversi esperimenti e dataset didattici;
- importazione di dataset di classificazione in formato CSV/TSV;
- ripetibilità degli esperimenti tramite seme casuale fissato.

---

- network architecture, hidden layers and neurons;
- activation functions;
- learning rate and loss evolution;
- weights, biases, activations and internal network values;
- forward pass, loss, backpropagation and parameter updates, including a guided single-example view;
- activation maps;
- several educational experiments and datasets;
- import of classification datasets in CSV/TSV format;
- reproducible experiments through a fixed random seed.

## Uso locale / Local use

Non è richiesta alcuna installazione.

1. Scarica il repository.
2. Apri `index.html` per scegliere la lingua, oppure:
   - `it/index.html` per la versione italiana;
   - `en/index.html` per la versione inglese.

No installation is required.

1. Download the repository.
2. Open `index.html` to choose the language, or open:
   - `it/index.html` for the Italian version;
   - `en/index.html` for the English version.

## Struttura del repository / Repository structure

```text
dentro-la-rete/
├── index.html
├── it/
│   └── index.html
├── en/
│   └── index.html
├── README.md
├── LICENSE
├── LICENSE-CONTENT.md
├── CITATION.cff
├── .gitignore
└── .nojekyll
```

## Licenze / Licenses

Il **software**, compresi i commenti tecnici al codice, è distribuito secondo la **GNU General Public License v3.0 o successiva (GPL-3.0-or-later)**. Il testo completo è nel file [`LICENSE`](LICENSE).

I **contenuti didattici ed esplicativi originali destinati all'utente** — tra cui guida, spiegazioni, descrizioni dei problemi, attività e osservazioni — sono disponibili anche secondo la **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

Per i dettagli sulla distinzione tra le due licenze, vedi [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).

The **software**, including technical source-code comments, is released under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. The complete license text is in [`LICENSE`](LICENSE).

The original **educational and explanatory material intended for users** — including the guide, explanations, problem descriptions, activities and observations — is also available under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

See [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md) for details on the distinction between the two licenses.

## Attribuzione / Attribution

> “Dentro la rete”, © Francesco Operetto, 2026.

## Stato del progetto / Project status

La versione corrente è **7.3.23 beta**.

The current version is **7.3.23 beta**.
