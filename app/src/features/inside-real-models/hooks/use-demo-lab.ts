import { useReducer, useState } from "react";
import { ModelLab } from "@/core/model/lab";
import { type AdamTrace, START_TRACE, adamStep } from "@/features/inside-real-models/story/training/training-math";

interface NumberChange {
  before: number;
  after: number;
  slope: number;
}

export function useDemoLab() {
  const [lab] = useState(() => new ModelLab());
  const [adam, setAdam] = useState<AdamTrace>(START_TRACE);
  const [change, setChange] = useState<NumberChange | null>(null);
  const [, redraw] = useReducer((rounds: number) => rounds + 1, 0);

  function train(rounds: number) {
    const table = lab.live.tokenTable;
    let trace = adam;
    let last: NumberChange | null = change;
    for (let i = 0; i < rounds && !lab.isTrained; i++) {
      const before = table.data[0];
      lab.trainRounds(1);
      trace = adamStep(trace, table.grad[0]);
      last = { before, after: table.data[0], slope: table.grad[0] };
    }
    setAdam(trace);
    setChange(last);
    redraw();
  }

  function reset() {
    lab.reset();
    setAdam(START_TRACE);
    setChange(null);
    redraw();
  }

  return { lab, adam, change, train, reset };
}
