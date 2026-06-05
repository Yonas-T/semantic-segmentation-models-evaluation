/**
 * POST /api/inference
 *
 * Accepts a multipart/form-data request with an 'image' field,
 * runs dual-model segmentation (SegFormer + DeepLabV3), and returns
 * coloured mask PNGs as base64 alongside latency metrics.
 */

import { NextRequest, NextResponse } from 'next/server';
import { runInference, modelsExist, type InferenceResult } from '@/lib/inferenceEngine';

export const runtime = 'nodejs';

// Allow up to 10MB uploads
export const maxDuration = 60;

// ─── Mock response for demo when models aren't available ────────────────────

function getMockResponse() {
  return {
    segformer: {
      mask: null,
      latencyMs: 22,
      resolution: [512, 1024] as [number, number],
      mockMode: true,
    },
    deeplabv3: {
      mask: null,
      latencyMs: 15,
      resolution: [512, 1024] as [number, number],
      mockMode: true,
    },
    message: 'ONNX models not found. Place segformer_b2.onnx and deeplabv3_resnet101.onnx in public/models/ to enable real inference.',
  };
}

// ─── Handler ────────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const imageFile = formData.get('image');

    if (!imageFile || !(imageFile instanceof File)) {
      return NextResponse.json(
        { error: 'Missing "image" field in form data.' },
        { status: 400 },
      );
    }

    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(imageFile.type)) {
      return NextResponse.json(
        { error: `Invalid file type: ${imageFile.type}. Accepted: JPEG, PNG, WebP.` },
        { status: 400 },
      );
    }

    // Validate file size (10MB max)
    if (imageFile.size > 10 * 1024 * 1024) {
      return NextResponse.json(
        { error: 'File too large. Maximum size: 10 MB.' },
        { status: 400 },
      );
    }

    // Check if models exist
    const models = modelsExist();
    if (!models.segformer && !models.deeplabv3) {
      console.log('model does not exist')
      return NextResponse.json(getMockResponse(), { status: 200 });
    }

    // Read image buffer
    const arrayBuffer = await imageFile.arrayBuffer();
    const imageBuffer = Buffer.from(arrayBuffer);

    console.log({imageBuffer})

    // Run inference on both models in parallel
    const [segformerResult, deeplabResult] = await Promise.all([
      models.segformer
        ? runInference(imageBuffer, 'segformer')
        : Promise.resolve(null),
      models.deeplabv3
        ? runInference(imageBuffer, 'deeplabv3')
        : Promise.resolve(null),
    ]);

    const response: Record<string, InferenceResult | null> = {
      segformer: segformerResult,
      deeplabv3: deeplabResult,
    };

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error('[API /inference] Error:', error);
    const message = error instanceof Error ? error.message : 'Internal server error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
