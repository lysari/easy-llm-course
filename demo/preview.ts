import { Canvas, createCanvas } from "canvas";
import * as d3 from "d3";
import { JSDOM } from "jsdom";
import sharp from "sharp";
import pkg from "fs-extra";
import path from "path";
const { writeFileSync } = pkg;

export function drawGraph(x: number[], y: number[]) {
  const width = 500;
  const height = 500;
  const canvas = createCanvas(width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "white";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "blue";

  const xMax = Math.max(...x);
  const yMax = Math.max(...y);

  //draw foundations graph x,y
  context.strokeStyle = "black";
  context.lineWidth = 2;
  // x axis
  context.beginPath();
  context.moveTo(40, height - 40);
  context.lineTo(width - 20, height - 40);
  context.stroke();
  // y axis
  context.beginPath();
  context.moveTo(40, height - 40);
  context.lineTo(40, 20);
  context.stroke();

  // plot points

  const padding = 40;
  for (let i = 0; i < x.length; i++) {
    const xPos = (x[i] / xMax) * (width - 2 * padding) + padding;
    const yPos = height - ((y[i] / yMax) * (height - 2 * padding) + padding);
    context.beginPath();
    context.arc(xPos, yPos, 5, 0, Math.PI * 2);
    context.fill();
  }
  const buffer = canvas.toBuffer("image/png");
  writeFileSync("linear_regression_plot.png", buffer);
}

export function drawGraphD3(xs: number[], ys: number[], w: number, b: number) {
  const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>");
  global.window = dom.window as unknown as Window & typeof globalThis;
  global.document = dom.window.document as unknown as Document;

  // Declare the chart dimensions and margins.
  const width = 640;
  const height = 400;
  const marginTop = 20;
  const marginRight = 20;
  const marginBottom = 30;
  const marginLeft = 40;

  // --- D3 Scales (same as SVG version) ---
  const x = d3
    .scaleLinear()
    .domain([0, (d3.max(xs) ?? 10) + 1])
    .range([marginLeft, width - marginRight]);

  const y = d3
    .scaleLinear()
    .domain([0, (d3.max(ys) ?? 100) + 10])
    .range([height - marginBottom, marginTop]);

  // --- Create Canvas ---
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  // Background
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, width, height);

  // --- Draw Gridlines (optional, mimics SVG) ---
  ctx.strokeStyle = "#ddd";
  ctx.lineWidth = 1;

  const xTicks = x.ticks(10);
  const yTicks = y.ticks(10);

  xTicks.forEach((tick) => {
    const px = x(tick);
    ctx.beginPath();
    ctx.moveTo(px, marginTop);
    ctx.lineTo(px, height - marginBottom);
    ctx.stroke();
  });

  yTicks.forEach((tick) => {
    const py = y(tick);
    ctx.beginPath();
    ctx.moveTo(marginLeft, py);
    ctx.lineTo(width - marginRight, py);
    ctx.stroke();
  });

  // --- Draw X-Axis ---
  ctx.strokeStyle = "black";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(marginLeft, height - marginBottom);
  ctx.lineTo(width - marginRight, height - marginBottom);
  ctx.stroke();

  // X-Axis Label
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  xTicks.forEach((tick) => {
    const px = x(tick);
    ctx.fillStyle = "black";
    ctx.fillText(tick.toString(), px, height - marginBottom + 6);
  });

  // --- Draw Y-Axis ---
  ctx.beginPath();
  ctx.moveTo(marginLeft, height - marginBottom);
  ctx.lineTo(marginLeft, marginTop);
  ctx.stroke();

  // Y-Axis Label
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  yTicks.forEach((tick) => {
    const py = y(tick);
    ctx.fillStyle = "black";
    ctx.fillText(tick.toString(), marginLeft - 8, py);
  });

  // --- Optional: Plot data points (example) ---
  ctx.fillStyle = "steelblue";
  xs.forEach((val, i) => {
    const px = x(val);
    const py = y(ys[i]);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // --- Draw Regression Line ---
  ctx.strokeStyle = "red";
  ctx.lineWidth = 1;
  console.log("Drawing line with w:", w, "b:", b);
  ctx.beginPath();
  ys.forEach((_, i) => {
    const x1 = xs[i];
    const y1 = w * x1 + b;

    const px1 = x(x1);
    const py1 = y(y1);

    ctx.lineTo(px1, py1);
    ctx.stroke();
    ctx.moveTo(px1, py1);
  });

  // Save to PNG
  saveCanvasToPng(canvas, "linear_regression_d3_canvas_plot.png");
}

function saveSvg(svgElement: SVGSVGElement, filename: string) {
  if (!svgElement) {
    throw new Error("SVG element is null or undefined.");
  }
  const svgString = svgElement.outerHTML;
  const buffer = Buffer.from(svgString);
  sharp(buffer)
    .png()
    .toFile("output/" + filename)
    .then(() => {
      console.log(`SVG saved as ${filename}`);
    })
    .catch((err) => {
      console.error("Error saving SVG:", err);
    });
}

function saveCanvasToPng(canvas: Canvas, filename: string) {
  const buffer = canvas.toBuffer("image/png");
  const filepath = path.join(process.cwd(), "output", filename);
  writeFileSync(filepath, buffer);
  console.log(`Saved: ${filepath}`);
}
