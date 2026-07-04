// MediaPipe 21-landmark hand topology (pairs of landmark indices)
export const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                 // palm base
];

/**
 * Draw a hand skeleton on a 2D context.
 * @param ctx    canvas 2D context
 * @param points array of [x, y] in canvas pixel coordinates (21 entries)
 * @param color  stroke color for bones
 */
export function drawSkeleton(ctx, points, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
    ctx.stroke();
  }
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], i % 4 === 0 ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = i % 4 === 0 ? color : "#ffffff";
    ctx.fill();
  });
}

// Cubic in-out — matches the server's easing for baked frames.
export const easeInOutCubic = (t) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export const lerpPose = (a, b, t) =>
  a.map((p, i) => [p[0] + (b[i][0] - p[0]) * t, p[1] + (b[i][1] - p[1]) * t]);
