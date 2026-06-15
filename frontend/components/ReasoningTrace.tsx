"use client";

import type { TraceItem } from "@/lib/sse";

type ReasoningTraceProps = {
  traces: TraceItem[];
  isLoading: boolean;
};

const EVENT_COLORS: Record<string, string> = {
  node_start: "text-slate-300",
  query_ready: "text-blue-300",
  retrieval_results: "text-cyan-300",
  rerank_results: "text-indigo-300",
  answer_generated: "text-violet-300",
  validation_result: "text-amber-300",
  retry_triggered: "text-orange-300",
  final_answer: "text-success",
};

function labelForEvent(item: TraceItem): string {
  if (item.message) {
    return item.message;
  }
  if (item.node) {
    return `${item.node}: ${item.event}`;
  }
  return item.event;
}

export default function ReasoningTrace({ traces, isLoading }: ReasoningTraceProps) {
  return (
    <section className="flex h-full flex-col rounded-2xl border border-border bg-panel">
      <header className="border-b border-border px-5 py-4">
        <h2 className="text-base font-semibold text-white">Reasoning Trace</h2>
        <p className="text-xs text-slate-400">Live SSE events from the agent graph</p>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
        {traces.length === 0 ? (
          <p className="text-sm text-slate-500">
            Trace events will appear here as agents run.
          </p>
        ) : (
          traces.map((item) => (
            <div
              key={item.id}
              className="rounded-xl border border-border bg-surface/70 px-4 py-3"
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span
                  className={`text-xs font-semibold uppercase tracking-wide ${
                    EVENT_COLORS[item.event] ?? "text-slate-300"
                  }`}
                >
                  {item.event}
                </span>
                <time className="text-[10px] text-slate-500">
                  {new Date(item.timestamp).toLocaleTimeString()}
                </time>
              </div>
              <p className="text-sm text-slate-200">{labelForEvent(item)}</p>
              {item.data && Object.keys(item.data).length > 0 && (
                <pre className="mt-2 overflow-x-auto rounded-lg bg-black/30 p-2 text-[11px] text-slate-400">
                  {JSON.stringify(item.data, null, 2)}
                </pre>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="rounded-xl border border-dashed border-accent/40 px-4 py-3 text-sm text-accent">
            Agents are running...
          </div>
        )}
      </div>
    </section>
  );
}
