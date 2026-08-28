#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..');
const DEFAULT_OUTPUT = path.join(SCRIPT_DIR, 'fixtures', 'generated_reference.json');

function parseArgs(argv) {
  const args = { html: null, out: DEFAULT_OUTPUT };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--html') args.html = argv[++i];
    else if (arg === '--out') args.out = argv[++i];
    else if (arg === '--help' || arg === '-h') args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function versionTuple(name) {
  const match = name.match(/^dentro_la_rete_v(\d+)\.(\d+)\.(\d+)(?:_beta)?_(EN|IT)\.html$/i);
  if (!match) return null;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    language: match[4].toUpperCase(),
  };
}

function compareCandidates(a, b) {
  for (const key of ['major', 'minor', 'patch']) {
    if (a.version[key] !== b.version[key]) return b.version[key] - a.version[key];
  }
  // Prefer the English file when both language versions have the same code.
  if (a.version.language !== b.version.language) return a.version.language === 'EN' ? -1 : 1;
  return a.name.localeCompare(b.name);
}

function discoverHtml() {
  // Repository layout: prefer the English published app, then the Italian one.
  // The fallback keeps the script usable with versioned standalone HTML files.
  for (const relativePath of ['en/index.html', 'it/index.html']) {
    const candidate = path.join(REPO_ROOT, relativePath);
    if (fs.existsSync(candidate)) return candidate;
  }

  const candidates = fs.readdirSync(REPO_ROOT)
    .map(name => ({ name, version: versionTuple(name) }))
    .filter(item => item.version)
    .sort(compareCandidates);

  if (!candidates.length) {
    throw new Error(
      'No application HTML was found at en/index.html, it/index.html, or as a ' +
      'versioned standalone file in the repository root. Pass it explicitly ' +
      'with --html <path>.'
    );
  }
  return path.join(REPO_ROOT, candidates[0].name);
}

function extractMathematicalCore(html) {
  const scripts = [];
  const scriptPattern = /<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi;
  let match;
  while ((match = scriptPattern.exec(html)) !== null) scripts.push(match[1]);

  const candidates = scripts.filter(script =>
    script.includes('function forward(network, x)') &&
    script.includes('function computeStep(network, x, y)') &&
    script.includes('function applyStep(network, stepDetails, learningRate)') &&
    script.includes('const ACTIVATIONS')
  );

  if (candidates.length !== 1) {
    throw new Error(
      `Expected exactly one mathematical-core <script>, found ${candidates.length}. ` +
      'The application structure may have changed.'
    );
  }
  return candidates[0];
}

function loadAppApi(htmlPath) {
  const html = fs.readFileSync(htmlPath, 'utf8');
  const core = extractMathematicalCore(html);
  const context = vm.createContext({ console, Math });
  const expose = `\n;globalThis.__networkTestApi = {\n` +
    `  ACTIVATIONS, forward, computeStep, prepareUpdates, applyStep,\n` +
    `  trainingStep, lossFromOutputs, softmax\n` +
    `};\n`;

  vm.runInContext(core + expose, context, {
    filename: `${path.basename(htmlPath)}#mathematical-core`,
  });

  return {
    api: context.__networkTestApi,
    coreHash: crypto.createHash('sha256').update(core).digest('hex'),
  };
}

function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

function deterministicNetwork(arch, activation, output, phase = 0) {
  const weights = [];
  const bias = [];

  for (let l = 0; l < arch.length - 1; l++) {
    const layerWeights = [];
    const layerBias = [];
    for (let j = 0; j < arch[l + 1]; j++) {
      const row = [];
      for (let i = 0; i < arch[l]; i++) {
        const angle = phase + (l + 1) * 1.31 + (j + 1) * 0.73 + (i + 1) * 0.41;
        row.push(0.62 * Math.sin(angle));
      }
      layerWeights.push(row);
      layerBias.push(0.19 * Math.cos(phase + (l + 1) * 0.89 + (j + 1) * 0.57));
    }
    weights.push(layerWeights);
    bias.push(layerBias);
  }

  return { arch: [...arch], weights, bias, att: activation, output: deepCopy(output) };
}

function parameters(network) {
  return { weights: deepCopy(network.weights), bias: deepCopy(network.bias) };
}

function assertTrainingStepComposition(api, spec) {
  const base = spec.network
    ? deepCopy(spec.network)
    : deterministicNetwork(spec.arch, spec.activation, spec.output, spec.phase);
  const composedNetwork = deepCopy(base);
  const entryPointNetwork = deepCopy(base);

  const composedStep = api.computeStep(composedNetwork, spec.x, spec.y);
  api.applyStep(composedNetwork, composedStep, spec.learningRate);
  const entryPointStep = api.trainingStep(entryPointNetwork, spec.x, spec.y, spec.learningRate);

  if (JSON.stringify(entryPointStep) !== JSON.stringify(composedStep) ||
      JSON.stringify(parameters(entryPointNetwork)) !== JSON.stringify(parameters(composedNetwork))) {
    throw new Error(`${spec.name}: trainingStep diverges from computeStep + applyStep.`);
  }
}

function makeSingleCase(api, spec) {
  const network = spec.network
    ? deepCopy(spec.network)
    : deterministicNetwork(spec.arch, spec.activation, spec.output, spec.phase);
  const initialParameters = parameters(network);
  const step = api.computeStep(network, spec.x, spec.y);
  api.applyStep(network, step, spec.learningRate);

  return {
    name: spec.name,
    note: spec.note || '',
    arch: deepCopy(network.arch),
    activation: network.att,
    output: deepCopy(network.output),
    x: deepCopy(spec.x),
    y: spec.y,
    learningRate: spec.learningRate,
    finiteDifference: spec.finiteDifference !== false,
    initialParameters,
    step: deepCopy(step),
    updatedParameters: parameters(network),
  };
}

function makeTrajectory(api, spec) {
  const network = deterministicNetwork(spec.arch, spec.activation, spec.output, spec.phase);
  const initialParameters = parameters(network);
  const steps = [];

  for (const sample of spec.samples) {
    const beforeParameters = parameters(network);
    const step = api.computeStep(network, sample.x, sample.y);
    api.applyStep(network, step, spec.learningRate);
    steps.push({
      x: deepCopy(sample.x),
      y: sample.y,
      beforeParameters,
      step: deepCopy(step),
      afterParameters: parameters(network),
    });
  }

  return {
    name: spec.name,
    arch: deepCopy(network.arch),
    activation: network.att,
    output: deepCopy(network.output),
    learningRate: spec.learningRate,
    initialParameters,
    steps,
  };
}

function buildFixtures(api, sourceInfo) {
  const binary = { type: 'binary', neurons: 1 };
  const regression = { type: 'regression', neurons: 1 };

  const reluZeroNetwork = {
    arch: [1, 1, 1],
    weights: [[[1.0]], [[0.7]]],
    bias: [[-0.25], [-0.10]],
    att: 'relu',
    output: deepCopy(binary),
  };

  const singleSpecs = [
    {
      name: 'binary_no_hidden', arch: [3, 1], activation: 'tanh', output: binary,
      phase: 0.10, x: [0.17, 0.63, 0.91], y: 1, learningRate: 0.08,
    },
    {
      name: 'binary_tanh_one_hidden', arch: [3, 4, 1], activation: 'tanh', output: binary,
      phase: 0.35, x: [0.12, 0.58, 0.83], y: 0, learningRate: 0.05,
    },
    {
      name: 'binary_sigmoid_two_hidden', arch: [2, 3, 2, 1], activation: 'sigmoid', output: binary,
      phase: 0.70, x: [0.27, 0.74], y: 1, learningRate: 0.11,
    },
    {
      name: 'binary_relu_one_hidden', arch: [3, 4, 1], activation: 'relu', output: binary,
      phase: 1.05, x: [0.21, 0.69, 0.44], y: 0, learningRate: 0.06,
    },
    {
      name: 'regression_tanh_one_hidden', arch: [3, 3, 1], activation: 'tanh', output: regression,
      phase: 1.40, x: [0.31, 0.52, 0.88], y: 0.37, learningRate: 0.04,
    },
    {
      name: 'multiclass_no_hidden', arch: [4, 3], activation: 'sigmoid',
      output: { type: 'multiclass', neurons: 3 },
      phase: 1.75, x: [0.15, 0.39, 0.67, 0.92], y: 2, learningRate: 0.09,
    },
    {
      name: 'multiclass_relu_one_hidden', arch: [3, 5, 4], activation: 'relu',
      output: { type: 'multiclass', neurons: 4 },
      phase: 2.10, x: [0.18, 0.61, 0.79], y: 1, learningRate: 0.07,
    },
    {
      name: 'multiclass_tanh_two_hidden', arch: [4, 4, 3, 3], activation: 'tanh',
      output: { type: 'multiclass', neurons: 3 },
      phase: 2.45, x: [0.11, 0.33, 0.72, 0.86], y: 0, learningRate: 0.05,
    },
    {
      name: 'multiclass_max_import_shape', arch: [9, 9, 6], activation: 'tanh',
      output: { type: 'multiclass', neurons: 6 }, phase: 2.80,
      x: [0.08, 0.19, 0.31, 0.44, 0.56, 0.68, 0.77, 0.89, 0.96],
      y: 5, learningRate: 0.04,
      note: 'Exercises the maximum imported-data shape: 9 inputs and 6 classes.',
    },
    {
      name: 'relu_zero_convention', network: reluZeroNetwork,
      x: [0.25], y: 1, learningRate: 0.10, finiteDifference: false,
      note: 'The hidden pre-activation is exactly zero; both implementations are expected to use ReLU derivative 0 there.',
    },
  ];

  const trajectorySpecs = [
    {
      name: 'multiclass_tanh_six_updates', arch: [3, 4, 3], activation: 'tanh',
      output: { type: 'multiclass', neurons: 3 }, phase: 0.92, learningRate: 0.07,
      samples: [
        { x: [0.14, 0.52, 0.81], y: 0 },
        { x: [0.76, 0.24, 0.43], y: 1 },
        { x: [0.37, 0.88, 0.19], y: 2 },
        { x: [0.62, 0.57, 0.12], y: 1 },
        { x: [0.25, 0.31, 0.93], y: 0 },
        { x: [0.48, 0.73, 0.55], y: 2 },
      ],
    },
    {
      name: 'binary_relu_six_updates', arch: [2, 3, 1], activation: 'relu',
      output: binary, phase: 1.28, learningRate: 0.06,
      samples: [
        { x: [0.18, 0.82], y: 1 },
        { x: [0.71, 0.29], y: 0 },
        { x: [0.43, 0.67], y: 1 },
        { x: [0.88, 0.16], y: 0 },
        { x: [0.36, 0.41], y: 1 },
        { x: [0.59, 0.94], y: 0 },
      ],
    },
  ];

  // trainingStep is the entry point used by the application's training loop.
  // It must remain exactly equivalent to the explicit computeStep + applyStep composition.
  for (const spec of singleSpecs) assertTrainingStepComposition(api, spec);

  const cases = singleSpecs.map(spec => makeSingleCase(api, spec));
  const trajectories = trajectorySpecs.map(spec => makeTrajectory(api, spec));

  // Keep the ordinary ReLU cases away from the non-differentiable point.
  for (const fixture of cases.filter(item => item.activation === 'relu' && item.name !== 'relu_zero_convention')) {
    const hiddenZ = fixture.step.zs.slice(1, -1).flat();
    if (hiddenZ.some(value => Math.abs(value) < 1e-6)) {
      throw new Error(`${fixture.name} accidentally places a ReLU pre-activation too close to zero.`);
    }
  }

  return {
    formatVersion: 1,
    generatedBy: 'tests/generate_fixtures.mjs',
    source: sourceInfo,
    numericalContract: {
      javascriptNumber: 'IEEE-754 binary64',
      comparisonReference: 'PyTorch float64 autograd',
      singleStepRtol: 1e-12,
      singleStepAtol: 1e-13,
    },
    cases,
    trajectories,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log('Usage: node tests/generate_fixtures.mjs [--html path] [--out path]');
    return;
  }

  const htmlPath = path.resolve(args.html || discoverHtml());
  const outputPath = path.resolve(args.out);
  const { api, coreHash } = loadAppApi(htmlPath);
  const fixtures = buildFixtures(api, {
    htmlFile: path.relative(REPO_ROOT, htmlPath).replaceAll(path.sep, '/'),
    mathematicalCoreSha256: coreHash,
  });

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(fixtures, null, 2) + '\n', 'utf8');

  console.log(`Application: ${htmlPath}`);
  console.log(`Mathematical core SHA-256: ${coreHash}`);
  console.log(`Single-step cases: ${fixtures.cases.length}`);
  console.log(`trainingStep composition checks: ${fixtures.cases.length} exact matches`);
  console.log(`Short trajectories: ${fixtures.trajectories.length}`);
  console.log(`Fixture file: ${outputPath}`);
}

main();
