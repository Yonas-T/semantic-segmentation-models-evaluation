/**
 * Cityscapes 19-class colour palette, class names, and visualisation helpers.
 *
 * Shared between the server-side inference engine (for encoding coloured
 * masks) and the client-side components (for tooltips and legends).
 */

// ─── Class names (trainId order) ────────────────────────────────────────────

export const CITYSCAPES_CLASSES: string[] = [
  'road',          // 0
  'sidewalk',      // 1
  'building',      // 2
  'wall',          // 3
  'fence',         // 4
  'pole',          // 5
  'traffic light', // 6
  'traffic sign',  // 7
  'vegetation',    // 8
  'terrain',       // 9
  'sky',           // 10
  'person',        // 11
  'rider',         // 12
  'car',           // 13
  'truck',         // 14
  'bus',           // 15
  'train',         // 16
  'motorcycle',    // 17
  'bicycle',       // 18
];

export const NUM_CLASSES = 19;

// ─── Official RGB colour palette ────────────────────────────────────────────

export const CITYSCAPES_COLORS: [number, number, number][] = [
  [128, 64, 128],   // road
  [244, 35, 232],   // sidewalk
  [70, 70, 70],     // building
  [102, 102, 156],  // wall
  [190, 153, 153],  // fence
  [153, 153, 153],  // pole
  [250, 170, 30],   // traffic light
  [220, 220, 0],    // traffic sign
  [107, 142, 35],   // vegetation
  [152, 251, 152],  // terrain
  [70, 130, 180],   // sky
  [220, 20, 60],    // person
  [255, 0, 0],      // rider
  [0, 0, 142],      // car
  [0, 0, 70],       // truck
  [0, 60, 100],     // bus
  [0, 80, 100],     // train
  [0, 0, 230],      // motorcycle
  [119, 11, 32],    // bicycle
];

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Get the RGB colour tuple for a given class ID. */
export function classIdToColor(id: number): [number, number, number] {
  if (id >= 0 && id < NUM_CLASSES) {
    return CITYSCAPES_COLORS[id];
  }
  return [0, 0, 0]; // void / unknown → black
}

/** Convert an RGB tuple to a CSS `rgb()` string. */
export function colorToRgbString(color: [number, number, number]): string {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

/** Convert an RGB tuple to a hex string (for charts). */
export function colorToHex(color: [number, number, number]): string {
  return '#' + color.map(c => c.toString(16).padStart(2, '0')).join('');
}

/**
 * Convert a flat class-ID mask into an RGBA ImageData buffer.
 *
 * This runs on the **client** to render masks onto a `<canvas>`.
 *
 * @param maskData - Flat array of class IDs, length = width × height.
 * @param width    - Image width in pixels.
 * @param height   - Image height in pixels.
 * @param alpha    - Alpha value for non-void pixels (0-255), default 255.
 * @returns A new `ImageData` object ready for `ctx.putImageData()`.
 */
export function createColoredMaskImageData(
  maskData: number[],
  width: number,
  height: number,
  alpha: number = 255,
): ImageData {
  const imageData = new ImageData(width, height);
  const data = imageData.data;

  for (let i = 0; i < maskData.length; i++) {
    const classId = maskData[i];
    const [r, g, b] = classIdToColor(classId);
    const offset = i * 4;
    data[offset] = r;
    data[offset + 1] = g;
    data[offset + 2] = b;
    data[offset + 3] = alpha;
  }

  return imageData;
}
