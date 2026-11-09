export interface FormulaTerm {
  term: string;
  meaning: string;
}

export interface Formula {
  lines: string[];
  legend?: FormulaTerm[];
}
