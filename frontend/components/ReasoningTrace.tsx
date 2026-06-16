"use client";

import { useEffect, useRef, useState } from "react";

import type { TraceItem } from "@/lib/sse";

type ReasoningTraceProps = {
  traces: TraceItem[];
  isLoading: boolean;
};

const EVENT_COLORS: Record<string, string> = {
  node_start: "border-slate-600 bg-slate-800/50",
  query_ready: "border-blue-500/40 bg-blue-500/10",
  retrieval_results: "border-cyan-500/40 bg-cyan-500/10",
  rerank_results: "border-indigo-500/40 bg-indigo-500/10",
  answer_generated: "border-violet-500/40 bg-violet-500/10",
  validation_result: "border-amber-500/40 bg-amber-500/10",
  retry_triggered: "border-orange-500/40 bg-orange-500/10",
  final_answer: "border-emerald-500/40 bg-emerald-500/10",
};

const EVENT_LABELS: Record<string, string> = {
  query_ready: "Query ready",
  retrieval_results: "Retrieval",
  rerank_results: "Rerank",
  answer_generated: "Generation",
  validation_result: "Validation",
  retry_triggered: "Self-healing retry",
  final_answer: "Complete",
};

function labelForEvent(item: TraceItem): string {
  if (item.message) {
    return item.message;
  }
  if (item.node && item.event !== "node_start") {
    return `${item.node}: ${item.event}`;
  }
  return EVENT_LABELS[item.event] ?? item.event;
}

function shouldShowTrace(item: TraceItem): boolean {
  return item.event !== "node_start";
}

function TraceCard({ item }: { item: TraceItem }) {
  const [showJson, setShowJson] = useState(false);
  const hasData = item.data && Object.keys(item.data).length > 0;

  return (
    <div
      className={`rounded-xl border px-4 py-3.5 ${
        EVENT_COLORS[item.event] ?? "border-border bg-surface/70"
      }`}
    >
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-white">
          {EVENT_LABELS[item.event] ?? item.event.replace(/_/g, " ")}
        </span>
        <time className="shrink-0 text-xs text-slate-500">
          {new Date(item.timestamp).toLocaleTimeString()}
        </time>
      </div>
      <p className="text-sm leading-relaxed text-slate-300">{labelForEvent(item)}</p>
      {hasData && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setShowJson((current) => !current)}
            className="text-xs font-medium text-slate-400 transition hover:text-slate-200"
          >
            {showJson ? "Hide details" : "Show details"}
          </button>
          {showJson && (
            <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-black/40 p-3 text-xs leading-relaxed text-slate-400">
              {JSON.stringify(item.data, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function ReasoningTrace({ traces, isLoading }: ReasoningTraceProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const visibleTraces = traces.filter(shouldShowTrace);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [traces, isLoading]);

  return (
    <section className="flex h-full min-h-0 flex-col rounded-2xl border border-border bg-panel shadow-xl shadow-black/20">
      <header className="shrink-0 border-b border-border px-6 py-5">
        <h2 className="text-xl font-semibold text-white">Reasoning Trace</h2>
        <p className="mt-1 text-sm text-slate-400">Live agent steps via SSE</p>
      </header>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-6 py-5">
        {visibleTraces.length === 0 ? (
          <p className="text-base text-slate-500">
            Trace events will appear here as agents run.
          </p>
        ) : (
          visibleTraces.map((item) => <TraceCard key={item.id} item={item} />)
        )}
        {isLoading && visibleTraces.length > 0 && (
          <div className="flex items-center gap-2 text-sm text-accent">
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
            Processing...
          </div>
        )}
      </div>
    </section>
  );
}
