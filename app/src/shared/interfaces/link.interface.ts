export interface LinkPath {
  id: string;
  fromId: string;
  d: string;
  startX: number;
  endX: number;
}

export interface LinkDot {
  id: string;
  fromId: string;
  x: number;
  y: number;
}

export interface LinkDrawing {
  paths: LinkPath[];
  dots: LinkDot[];
}
