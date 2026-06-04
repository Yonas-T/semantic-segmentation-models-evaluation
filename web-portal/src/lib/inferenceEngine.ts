/**
 * Server-side ONNX inference engine for semantic segmentation models.
 *
 * Uses a singleton pattern to lazily load ONNX sessions and keep them
 * cached across requests. Handles image preprocessing with `sharp` and
 * post-processing (argmax → coloured mask → base64 PNG).
 */

import path from 'path';
import { CITYSCAPES_COLORS, NUM_CLASSES } from './cityscapesPalette';

// ─── Types ──────────────────────────────────────────────────────────────────

export type ModelName = 'segformer' | 'deeplabv3';

export interface InferenceResult {
  /** Base-64 encoded PNG of the coloured segmentation mask. */
  mask: string;
  /** Inference latency in milliseconds. */
  latencyMs: number;
  /** [height, width] of the output mask. */
  resolution: [number, number];
}

// ─── Constants ──────────────────────────────────────────────────────────────

const INPUT_HEIGHT = 512;
const INPUT_WIDTH = 1024;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// ─── Singleton sessions ─────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sessions: Record<string, any> = {};

async function getOrt() {
  if (!ort) {
    ort = await import('onnxruntime-node');
  }
  return ort;
}

async function getSession(modelName: ModelName) {
  if (sessions[modelName]) return sessions[modelName];

  const runtime = await getOrt();
  const modelFileName = modelName === 'segformer'
    ? 'segformer_b2.onnx'
    : 'deeplabv3_resnet101.onnx';

  const modelPath = path.join(process.cwd(), 'public', 'models', modelFileName);

  try {
    const session = await runtime.InferenceSession.create(modelPath, {
      executionProviders: ['cpu'],
    });
    sessions[modelName] = session;
    console.log(`[InferenceEngine] Loaded ${modelName} from ${modelPath}`);
    return session;
  } catch (err) {
    console.error(`[InferenceEngine] Failed to load ${modelName}: ${err}`);
    return null;
  }
}

// ─── Preprocessing ──────────────────────────────────────────────────────────

async function preprocessImage(imageBuffer: Buffer): Promise<Float32Array> {
  const sharp = (await import('sharp')).default;

  const { data, info } = await sharp(imageBuffer)
    .resize(INPUT_WIDTH, INPUT_HEIGHT, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixels = info.width * info.height;
  const float32 = new Float32Array(3 * pixels);

  // HWC → CHW + ImageNet normalisation
  for (let i = 0; i < pixels; i++) {
    const r = data[i * 3] / 255.0;
    const g = data[i * 3 + 1] / 255.0;
    const b = data[i * 3 + 2] / 255.0;

    float32[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    float32[pixels + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    float32[2 * pixels + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return float32;
}

// ─── Postprocessing ─────────────────────────────────────────────────────────

function argmaxMask(
  outputData: Float32Array,
  numClasses: number,
  height: number,
  width: number,
): Uint8Array {
  const pixels = height * width;
  const mask = new Uint8Array(pixels);

  for (let i = 0; i < pixels; i++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let c = 0; c < numClasses; c++) {
      const val = outputData[c * pixels + i];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = c;
      }
    }
    mask[i] = maxIdx;
  }

  return mask;
}

async function maskToColorPng(mask: Uint8Array, width: number, height: number): Promise<string> {
  const sharp = (await import('sharp')).default;

  const rgbBuffer = Buffer.alloc(width * height * 3);
  for (let i = 0; i < mask.length; i++) {
    const classId = mask[i];
    const color = classId < NUM_CLASSES ? CITYSCAPES_COLORS[classId] : [0, 0, 0];
    rgbBuffer[i * 3] = color[0];
    rgbBuffer[i * 3 + 1] = color[1];
    rgbBuffer[i * 3 + 2] = color[2];
  }

  const pngBuffer = await sharp(rgbBuffer, {
    raw: { width, height, channels: 3 },
  }).png().toBuffer();

  return pngBuffer.toString('base64');
}

// ─── Public API ─────────────────────────────────────────────────────────────

export async function runInference(
  imageBuffer: Buffer,
  modelName: ModelName,
): Promise<InferenceResult | null> {
  const session = await getSession(modelName);

  if (!session) {
    return null;
  }

  const runtime = await getOrt();

  // Preprocess
  const inputData = await preprocessImage(imageBuffer);
  const inputTensor = new runtime.Tensor(
    'float32',
    inputData,
    [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
  );

  // Run inference
  const inputName = session.inputNames[0];
  const start = performance.now();
  const results = await session.run({ [inputName]: inputTensor });
  const latencyMs = Math.round(performance.now() - start);

  // Postprocess
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];
  const outputData = outputTensor.data as Float32Array;

  // Determine output spatial dimensions
  const dims = outputTensor.dims as number[];
  const outH = dims[2];
  const outW = dims[3];

  const mask = argmaxMask(outputData, NUM_CLASSES, outH, outW);
  const maskBase64 = await maskToColorPng(mask, outW, outH);

  return {
    mask: maskBase64,
    latencyMs,
    resolution: [outH, outW],
  };
}

/**
 * Check whether ONNX model files exist on disk.
 */
export function modelsExist(): { segformer: boolean; deeplabv3: boolean } {
  const fs = require('fs');
  const modelsDir = path.join(process.cwd(), 'public', 'models');
  return {
    segformer: fs.existsSync(path.join(modelsDir, 'segformer_b2.onnx')),
    deeplabv3: fs.existsSync(path.join(modelsDir, 'deeplabv3_resnet101.onnx')),
  };
}
