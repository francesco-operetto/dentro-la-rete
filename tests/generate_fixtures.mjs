#!/usr/bin/env node

import assert from 'node:assert/strict';
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
  // Prefer the English file when both language files have the same version.
  if (a.version.language !== b.version.language) return a.version.language === 'EN' ? -1 : 1;
  return a.name.localeCompare(b.name);
}

function discoverHtml() {
  // Repository layout: prefer the published English app, then the Italian one.
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

function extractNumericalEngine(html) {
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
      `Expected exactly one numerical-engine <script>, found ${candidates.length}. ` +
      'The application structure may have changed.'
    );
  }

  const engine = candidates[0];
  if (engine.includes('const DATASET') || engine.includes('document.') || engine.includes('const el =')) {
    throw new Error(
      'The numerical-engine <script> also contains problem definitions or DOM code. ' +
      'Keep the numerical engine isolated from data/UI concerns.'
    );
  }
  return engine;
}

function loadAppApi(htmlPath) {
  const html = fs.readFileSync(htmlPath, 'utf8');
  const engine = extractNumericalEngine(html);
  const context = vm.createContext({ console, Math });
  const expose = `\n;globalThis.__networkTestApi = {\n` +
    `  ACTIVATIONS, forward, computeStep, prepareUpdates, applyStep,\n` +
    `  trainingStep, lossFromOutputs, sampleLoss, meanLoss, softmax\n` +
    `};\n`;

  vm.runInContext(engine + expose, context, {
    filename: `${path.basename(htmlPath)}#numerical-engine`,
  });

  // Exercise the actual teaching example, without enabling it in the app.
  assert.equal(vm.runInContext('typeof crossEntropyFromLogits', context), 'undefined',
    'The logits-based alternative must remain commented out in the application.');
  const examples = [...engine.matchAll(/\/\*([\s\S]*?)\*\//g)]
    .map(match => match[1])
    .filter(comment => comment.includes('function crossEntropyFromLogits(z, y)'));
  assert.equal(examples.length, 1, 'Expected one commented crossEntropyFromLogits example.');
  const exampleCode = examples[0].slice(examples[0].indexOf('function crossEntropyFromLogits'));
  const commentedLoss = vm.runInNewContext(
    exampleCode + '\n;crossEntropyFromLogits;', { Math },
    { filename: `${path.basename(htmlPath)}#commented-loss-example` },
  );

  return {
    api: context.__networkTestApi,
    commentedLoss,
    engineHash: crypto.createHash('sha256').update(engine).digest('hex'),
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

  assert.deepStrictEqual(
    entryPointStep,
    composedStep,
    `${spec.name}: trainingStep result diverges from computeStep + applyStep.`,
  );
  assert.deepStrictEqual(
    parameters(entryPointNetwork),
    parameters(composedNetwork),
    `${spec.name}: trainingStep parameters diverge from computeStep + applyStep.`,
  );
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

function buildFixtures(api, commentedLoss, sourceInfo) {
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
      note: 'The hidden pre-activation is exactly zero, so both implementations use 0 for the ReLU derivative at this point.',
    },
    {
      name: 'multiclass_probability_below_old_floor',
      network: {
        arch: [1, 3], att: 'linear', output: { type: 'multiclass', neurons: 3 },
        weights: [[[0.6], [-0.4], [0.2]]], bias: [[0, -230, -1]],
      },
      x: [0.5], y: 1, learningRate: 0.05,
      note: 'A positive target probability far below 1e-12 must retain its full loss and gradient.',
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

  // trainingStep is the entry point used by the application's training loop;
  // these cases verify exact equivalence with computeStep + applyStep.
  for (const spec of singleSpecs) assertTrainingStepComposition(api, spec);

  const cases = singleSpecs.map(spec => makeSingleCase(api, spec));
  const trajectories = trajectorySpecs.map(spec => makeTrajectory(api, spec));

  // These cases deliberately distinguish the reported probability-based loss
  // from the theoretical cross-entropy gradient once probabilities underflow.
  const numericalSpecs = [
    { name: 'ordinary', z: [2, 0, -1], y: 2, probabilityState: 'normal' },
    { name: 'below_old_floor', z: [0, -40, -2], y: 1, probabilityState: 'normal' },
    { name: 'very_small_positive', z: [0, -230, -2], y: 1, probabilityState: 'normal' },
    { name: 'subnormal_positive', z: [0, -740, -2], y: 1, probabilityState: 'subnormal' },
    { name: 'underflow_zero', z: [0, -1000, -2], y: 1, probabilityState: 'zero' },
    { name: 'underflow_shifted_up', z: [1000, 0, 998], y: 1, probabilityState: 'zero' },
    { name: 'underflow_shifted_down', z: [-1000, -2000, -1002], y: 1, probabilityState: 'zero' },
    { name: 'rounded_to_one', z: [-1000, 0, -1002], y: 1, probabilityState: 'one' },
  ];
  const multiclassNumerics = numericalSpecs.map(spec => {
    const network = {
      arch: [1, spec.z.length], att: 'linear',
      output: { type: 'multiclass', neurons: spec.z.length },
      weights: [spec.z.map(() => [0])], bias: [spec.z.slice()],
    };
    const stepSpec = { ...spec, network, x: [0.25], learningRate: 0.05, finiteDifference: false };
    assertTrainingStepComposition(api, stepSpec);
    const fixture = makeSingleCase(api, stepSpec);
    const sample = { x: stepSpec.x, y: spec.y };
    return {
      ...fixture, probabilityState: spec.probabilityState,
      commentedLogitsLoss: commentedLoss(spec.z, spec.y),
      sampleLoss: api.sampleLoss(network, sample),
      meanLoss: api.meanLoss(network, [sample, sample]),
    };
  });

  const lossBoundaries = [
    ['one', 1], ['half', 0.5], ['above_old_floor', 1e-11], ['old_floor', 1e-12],
    ['below_old_floor', 1e-13], ['tiny_positive', 1e-100], ['near_normal_limit', 1e-300],
    ['twice_minimum', 2 * Number.MIN_VALUE], ['minimum_positive', Number.MIN_VALUE], ['zero', 0],
  ].map(([name, probability]) => ({
    name, probability,
    loss: api.lossFromOutputs({ output: { type: 'multiclass' } }, [1 - probability, probability], 1),
  }));

  // Verify that the other ReLU fixtures stay away from the non-differentiable point at zero.
  for (const fixture of cases.filter(item => item.activation === 'relu' && item.name !== 'relu_zero_convention')) {
    const hiddenZ = fixture.step.zs.slice(1, -1).flat();
    if (hiddenZ.some(value => Math.abs(value) < 1e-6)) {
      throw new Error(`${fixture.name}: a ReLU pre-activation is too close to zero.`);
    }
  }

  return {
    formatVersion: 2,
    generatedBy: 'tests/generate_fixtures.mjs',
    source: sourceInfo,
    numericalContract: {
      javascriptNumber: 'IEEE-754 binary64',
      comparisonReference: 'PyTorch float64 autograd',
      singleStepRtol: 1e-12,
      singleStepAtol: 1e-13,
      minimumPositiveNumber: Number.MIN_VALUE,
      reportedMulticlassLoss: '-log(max(Number.MIN_VALUE, p_c))',
      multiclassGradient: 'theoretical cross-entropy from logits (p - t)',
    },
    cases,
    trajectories,
    multiclassNumerics,
    lossBoundaries,
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
  const { api, commentedLoss, engineHash } = loadAppApi(htmlPath);
  const fixtures = buildFixtures(api, commentedLoss, {
    htmlFile: path.relative(REPO_ROOT, htmlPath).replaceAll(path.sep, '/'),
    numericalEngineSha256: engineHash,
  });

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(fixtures, null, 2) + '\n', 'utf8');

  console.log(`Application: ${htmlPath}`);
  console.log(`Numerical engine SHA-256: ${engineHash}`);
  console.log(`Single-step cases: ${fixtures.cases.length}`);
  console.log(`trainingStep composition checks: ${fixtures.cases.length + fixtures.multiclassNumerics.length} exact matches`);
  console.log(`Short trajectories: ${fixtures.trajectories.length}`);
  console.log(`Multiclass numerical edge cases: ${fixtures.multiclassNumerics.length}`);
  console.log(`Probability loss boundary cases: ${fixtures.lossBoundaries.length}`);
  console.log(`Fixture file: ${outputPath}`);
}

main();
