"use client";

import { useEffect, useRef } from "react";

import type { ChatMessage } from "@/lib/sse";

type ChatPanelProps = {
  messages: ChatMessage[];
  input: string;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onSubmit: () => void;
};

export default function ChatPanel({
  messages,
  input,
  isLoading,
  onInputChange,
  onSubmit,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <section className="flex h-full min-h-0 flex-col rounded-2xl border border-border bg-panel shadow-xl shadow-black/20">
      <header className="shrink-0 border-b border-border px-6 py-5">
        <h2 className="text-xl font-semibold text-white">Chat</h2>
        <p className="mt-1 text-sm text-slate-400">
          Ask questions about your ingested documents
        </p>
      </header>

      <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto px-6 py-5">
        {messages.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-surface/60 p-8 text-base leading-relaxed text-slate-400">
            Try: <span className="text-slate-300">&quot;What is LangGraph?&quot;</span> or{" "}
            <span className="text-slate-300">&quot;How does self-healing RAG work?&quot;</span>
          </div>
        ) : (
          messages.map((message) => (
            <article
              key={message.id}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                  message.role === "user"
                    ? "bg-accent/25 text-blue-50"
                    : "border border-border bg-surface text-slate-100"
                }`}
              >
                <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  {message.role}
                </p>
                <p className="whitespace-pre-wrap text-base leading-7">{message.content}</p>
                {message.role === "assistant" && message.validationPassed !== undefined && (
                  <div
                    className={`mt-4 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${
                      message.validationPassed
                        ? "bg-success/15 text-success"
                        : "bg-warning/15 text-warning"
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        message.validationPassed ? "bg-success" : "bg-warning"
                      }`}
                    />
                    Validation {message.validationPassed ? "passed" : "failed"}
                    {typeof message.retryCount === "number" && ` · ${message.retryCount} retries`}
                  </div>
                )}
              </div>
            </article>
          ))
        )}
        {isLoading && (
          <p className="text-sm text-slate-500">Agents are running...</p>
        )}
      </div>

      <form
        className="shrink-0 border-t border-border px-6 py-5"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <div className="flex gap-3">
          <input
            className="flex-1 rounded-xl border border-border bg-surface px-4 py-3.5 text-base text-white outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/30"
            placeholder="Ask about LangGraph, RAG, or your documents..."
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="shrink-0 rounded-xl bg-accent px-6 py-3.5 text-base font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? "Running..." : "Send"}
          </button>
        </div>
      </form>
    </section>
  );
}
