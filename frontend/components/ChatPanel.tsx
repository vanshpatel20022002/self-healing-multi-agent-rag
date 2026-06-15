"use client";

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
  return (
    <section className="flex h-full flex-col rounded-2xl border border-border bg-panel">
      <header className="border-b border-border px-6 py-4">
        <h1 className="text-lg font-semibold text-white">Self-Healing RAG</h1>
        <p className="text-sm text-slate-400">
          Multi-agent retrieval with live reasoning trace
        </p>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
        {messages.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-surface/60 p-6 text-sm text-slate-400">
            Ask a question about your ingested documents. The agent graph will
            retrieve, rerank, generate, validate, and self-heal if needed.
          </div>
        ) : (
          messages.map((message) => (
            <article
              key={message.id}
              className={`max-w-3xl rounded-2xl px-4 py-3 text-sm leading-6 ${
                message.role === "user"
                  ? "ml-auto bg-accent/20 text-blue-100"
                  : "bg-surface text-slate-200"
              }`}
            >
              <p className="mb-1 text-xs uppercase tracking-wide text-slate-400">
                {message.role}
              </p>
              <p className="whitespace-pre-wrap">{message.content}</p>
              {message.role === "assistant" && message.validationPassed !== undefined && (
                <p
                  className={`mt-3 text-xs ${
                    message.validationPassed ? "text-success" : "text-warning"
                  }`}
                >
                  Validation: {message.validationPassed ? "passed" : "failed"}
                  {typeof message.retryCount === "number" && ` · retries: ${message.retryCount}`}
                </p>
              )}
            </article>
          ))
        )}
      </div>

      <form
        className="border-t border-border px-6 py-4"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <div className="flex gap-3">
          <input
            className="flex-1 rounded-xl border border-border bg-surface px-4 py-3 text-sm text-white outline-none ring-accent focus:ring-2"
            placeholder="Ask about LangGraph, RAG, or your documents..."
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="rounded-xl bg-accent px-5 py-3 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? "Running..." : "Send"}
          </button>
        </div>
      </form>
    </section>
  );
}
