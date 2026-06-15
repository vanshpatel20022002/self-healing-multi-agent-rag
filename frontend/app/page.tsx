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
    <main className="min-h-screen bg-surface px-4 py-6 md:px-8">
      <div className="mx-auto grid h-[calc(100vh-3rem)] max-w-7xl gap-6 lg:grid-cols-[1.4fr_1fr]">
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
        <p className="mx-auto mt-4 max-w-7xl rounded-xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      )}
    </main>
  );
}
