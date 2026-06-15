const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export type TraceItem = {
  id: string;
  event: string;
  node?: string;
  message?: string;
  data?: Record<string, unknown>;
  timestamp: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  validationPassed?: boolean;
  retryCount?: number;
};

function parseSseChunk(buffer: string): {
  events: Array<{ event: string; data: Record<string, unknown> }>;
  remainder: string;
} {
  const events: Array<{ event: string; data: Record<string, unknown> }> = [];
  const blocks = buffer.split("\n\n");

  for (let index = 0; index < blocks.length - 1; index += 1) {
    const block = blocks[index];
    const lines = block.split("\n");
    let eventName = "message";
    let dataLine = "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLine = line.slice(5).trim();
      }
    }

    if (dataLine) {
      try {
        events.push({ event: eventName, data: JSON.parse(dataLine) });
      } catch {
        events.push({ event: eventName, data: { raw: dataLine } });
      }
    }
  }

  return { events, remainder: blocks[blocks.length - 1] ?? "" };
}

export async function streamQuery(
  query: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming is not supported in this browser.");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseChunk(buffer);
    buffer = parsed.remainder;

    for (const item of parsed.events) {
      onEvent(item.event, item.data);
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseChunk(`${buffer}\n\n`);
    for (const item of parsed.events) {
      onEvent(item.event, item.data);
    }
  }
}

export function toTraceItem(event: string, data: Record<string, unknown>): TraceItem {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    event,
    node: typeof data.node === "string" ? data.node : undefined,
    message: typeof data.message === "string" ? data.message : undefined,
    data,
    timestamp: new Date().toISOString(),
  };
}
