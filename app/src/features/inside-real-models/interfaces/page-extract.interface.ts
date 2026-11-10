export interface PageExtract {
  title: string;
  text: string;
  pageLength: number;
  keptLength: number;
  otherTextLength: number;
  codeLength: number;
  markupLength: number;
  scriptCount: number;
  styleCount: number;
  droppedText: string[];
}
