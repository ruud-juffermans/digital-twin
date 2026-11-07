export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export type ChatEvent =
  | { type: "text"; text: string }
  | {
      type: "meta";
      model: string;
      inputTokens: number;
      outputTokens: number;
      cachedTokens: number;
      costUsd: number;
      latencyMs: number;
    }
  | { type: "error"; message: string }
  | { type: "citation"; title: string }
  | { type: "trace"; nodes: { node: string; ms: number | null }[]; guardFlags: string[] }
  | { type: "tool"; name: string }
  | { type: "approval"; status: "pending" }
  | { type: "done" };

// Mirrors APPROVAL_NOTICE in server/app/chat.py: the server appends it to the
// answer text; the client strips it and renders a proper notice instead.
export const APPROVAL_NOTICE =
  "This request needs Ruud's personal approval — it has been queued for his review.";

export const MAX_QUESTION_CHARS = 4000;

export class ChatHttpError extends Error {
  status: number;
  constructor(status: number) {
    super(`HTTP ${status}`);
    this.status = status;
  }
}

/** Visitor-facing copy for a failed request. Never surfaces raw status codes. */
export function errorCopy(err: unknown): string {
  if (err instanceof ChatHttpError) {
    if (err.status === 429) {
      return "You're sending messages quickly. Give it a moment and try again.";
    }
    if (err.status === 400 || err.status === 413 || err.status === 422) {
      return "That message is too long. Keep it under 4,000 characters.";
    }
    return "Something went wrong on my side. Try again in a moment.";
  }
  if (err instanceof TypeError) {
    // fetch() rejects with a TypeError when the network is unreachable.
    return "I can't reach the server. Check your connection and try again.";
  }
  return "Something went wrong on my side. Try again in a moment.";
}

export async function* streamChat(
  messages: ChatTurn[],
  threadId: string,
  signal: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages,
      thread_id: threadId,
      // Admin stats only: where the question came from.
      locale: navigator.language?.slice(0, 16) || null,
      embed: new URLSearchParams(window.location.search).has("embed"),
    }),
    signal,
  });
  if (!response.ok || !response.body) throw new ChatHttpError(response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (part.startsWith("data: ")) {
        yield JSON.parse(part.slice("data: ".length)) as ChatEvent;
      }
    }
  }
}

export interface Feedback {
  threadId: string;
  verdict: "up" | "down";
  question: string;
  answer: string;
  comment?: string;
}

/** 👍/👎 on an answer. Resolves true when the server stored it. */
export async function sendFeedback(feedback: Feedback): Promise<boolean> {
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        thread_id: feedback.threadId,
        verdict: feedback.verdict,
        question: feedback.question.slice(0, 4000),
        answer: feedback.answer.slice(0, 16000),
        comment: feedback.comment?.trim() || undefined,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
