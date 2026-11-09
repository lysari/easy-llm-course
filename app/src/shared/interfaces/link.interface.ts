export interface LinkPath {
  id: string;
  d: string;
}

export interface LinkDot {
  id: string;
  x: number;
  y: number;
}

export interface LinkDrawing {
  paths: LinkPath[];
  dots: LinkDot[];
}
