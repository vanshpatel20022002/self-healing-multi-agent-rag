"use client";

import { useState } from "react";

import ChatPanel from "@/components/ChatPanel";
import ReasoningTrace from "@/components/ReasoningTrace";
import {
  streamQuery,
  toTraceItem,
  type ChatMessage,
  type TraceItem,
} from "@/lib/sse";

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    const query = input.trim();
    if (!query || isLoading) {
      return;
    }

    setError(null);
    setIsLoading(true);
    setInput("");
    setTraces([]);

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
    };
    setMessages((current) => [...current, userMessage]);

    try {
      await streamQuery(query, (event, data) => {
        setTraces((current) => [...current, toTraceItem(event, data)]);

        if (event === "final_answer") {
          const answer = typeof data.answer === "string" ? data.answer : "";
          const assistantMessage: ChatMessage = {
            id: `assistant-${Date.now()}`,
            role: "assistant",
            content: answer || "No answer generated.",
            validationPassed:
              typeof data.validation_passed === "boolean"
                ? data.validation_passed
                : undefined,
            retryCount:
              typeof data.retry_count === "number" ? data.retry_count : undefined,
          };
          setMessages((current) => [...current, assistantMessage]);
        }
      });
    } catch (streamError) {
      const message =
        streamError instanceof Error
          ? streamError.message
          : "Failed to stream query.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-surface">
      <header className="shrink-0 border-b border-border bg-panel/80 px-6 py-4 backdrop-blur md:px-10">
        <div className="mx-auto flex w-full max-w-[1600px] items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Self-Healing RAG</h1>
            <p className="mt-0.5 text-sm text-slate-400">
              Multi-agent retrieval with live reasoning trace
            </p>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-border bg-surface px-4 py-1.5 text-xs text-slate-400 sm:flex">
            <span className="h-2 w-2 rounded-full bg-success" />
            API connected
          </div>
        </div>
      </header>

      <main className="flex flex-1 flex-col px-4 py-6 md:px-10 md:py-8">
        <div className="mx-auto grid h-full w-full max-w-[1600px] flex-1 gap-6 lg:grid-cols-2 lg:gap-8">
          <ChatPanel
            messages={messages}
            input={input}
            isLoading={isLoading}
            onInputChange={setInput}
            onSubmit={handleSubmit}
          />
          <ReasoningTrace traces={traces} isLoading={isLoading} />
        </div>
        {error && (
          <p className="mx-auto mt-6 w-full max-w-[1600px] rounded-xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-red-200">
            {error}
          </p>
        )}
      </main>
    </div>
  );
}
