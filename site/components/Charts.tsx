"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useEffect, useRef, useState } from "react";
import type { CurvePoint } from "@/lib/data";

/** Renders its chart only once the (fixed-height) box is near the viewport: the box itself never changes size. */
function useNearViewport<T extends Element>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || near) return;
    if (typeof IntersectionObserver === "undefined") {
      setNear(true);
      return;
    }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setNear(true);
        io.disconnect();
      }
    }, { rootMargin: "300px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [near]);
  return [ref, near];
}

const AXIS = { stroke: "var(--muted)", fontSize: 12 };
const tooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--rule)",
  borderRadius: 4,
  color: "var(--ink)",
  fontSize: 12,
};

/** One tick per January (every other year on long series), labelled by year: no duplicate labels. */
function yearTicks(points: CurvePoint[]): string[] {
  const jan = points.map((p) => p.date).filter((d) => d.slice(5, 7) === "01");
  const ticks = jan.length ? jan : [points[0]?.date].filter(Boolean);
  return ticks.length > 7 ? ticks.filter((_, i) => i % 2 === 0) : ticks;
}

/** Growth of $1, strategy vs benchmark. The box has a fixed height, so nothing shifts when it renders. */
export function EquityChart({ points, benchmark, label }: { points: CurvePoint[]; benchmark: string; label: string }) {
  const ticks = yearTicks(points);
  const [ref, near] = useNearViewport<HTMLElement>();
  return (
    <figure ref={ref} className="h-64 sm:h-72 w-full" role="img" aria-label={label}>
      {near && (
      <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 600, height: 288 }}>
        <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="date" ticks={ticks} interval={0} tick={AXIS} tickFormatter={(d: string) => d.slice(0, 4)} stroke="var(--rule)" />
          <YAxis tick={AXIS} width={44} stroke="var(--rule)" tickFormatter={(v: number) => v.toFixed(1)} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
            labelFormatter={(d) => `Month ending ${String(d)}`}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "var(--muted)" }} />
          <ReferenceLine y={1} stroke="var(--rule)" />
          <Line type="monotone" dataKey="strategy" name="Strategy" stroke="var(--series-strategy)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="benchmark" name={benchmark} stroke="var(--series-benchmark)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      )}
    </figure>
  );
}

type ICPoint = { year: number; model: number | null; momentum: number | null };

export function ICByYearChart({ data, label }: { data: ICPoint[]; label: string }) {
  const [ref, near] = useNearViewport<HTMLElement>();
  return (
    <figure ref={ref} className="h-64 sm:h-72 w-full" role="img" aria-label={label}>
      {near && (
      <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 600, height: 288 }}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="year" tick={AXIS} stroke="var(--rule)" />
          <YAxis tick={AXIS} width={52} stroke="var(--rule)" tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
          <Legend wrapperStyle={{ fontSize: 12, color: "var(--muted)" }} />
          <ReferenceLine y={0} stroke="var(--muted)" />
          <Bar dataKey="model" name="LightGBM model" fill="var(--series-strategy)" isAnimationActive={false} />
          <Bar dataKey="momentum" name="12-1 momentum" fill="var(--series-benchmark)" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      )}
    </figure>
  );
}
