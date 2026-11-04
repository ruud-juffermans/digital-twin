import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import Markdown, { type Components } from "react-markdown";
import {
  APPROVAL_NOTICE,
  MAX_QUESTION_CHARS,
  errorCopy,
  sendFeedback,
  streamChat,
  type ChatTurn,
} from "./chat";
import {
  AlertIcon,
  ArrowDownIcon,
  CalendarIcon,
  CheckIcon,
  ClockIcon,
  CloseIcon,
  CogIcon,
  ExpandIcon,
  PaperPlaneIcon,
  SearchIcon,
  SendIcon,
  ShrinkIcon,
  StopIcon,
  ThumbDownIcon,
  ThumbUpIcon,
} from "./icons";
import "./App.css";

interface FeedbackState {
  verdict?: "up" | "down";
  status: "idle" | "pending" | "sent" | "failed";
}
const NO_FEEDBACK: FeedbackState = { status: "idle" };

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "done" | "error" | "stopped";
  tools: string[];
  citations: string[];
  approval?: "pending";
  error?: string;
  feedback: FeedbackState;
  // The visitor's question this answer responds to (feedback and retry).
  question?: string;
}

// Embedded in another site via widget.js: the parent page owns the launcher
// and the iframe chrome; we render the chat full-bleed and signal close /
// resize intents up via postMessage.
const EMBED = new URLSearchParams(window.location.search).has("embed");

const STARTERS = [
  "What kind of projects does Ruud take on?",
  "Is Ruud available for a new assignment?",
  "What did Ruud do at the Dutch Police and NLR?",
  "How do I get in touch or book an intro call?",
];

const TOOL_COPY: Record<string, { active: string; done: string; Icon: typeof SearchIcon }> = {
  search_knowledge: {
    active: "Checking Ruud's notes…",
    done: "Checked Ruud's notes",
    Icon: SearchIcon,
  },
  get_availability: {
    active: "Checking availability…",
    done: "Checked availability",
    Icon: CalendarIcon,
  },
  draft_contact_message: {
    active: "Preparing your message for Ruud…",
    done: "Prepared your message for Ruud",
    Icon: PaperPlaneIcon,
  },
};
const UNKNOWN_TOOL = { active: "Working on it…", done: "Done", Icon: CogIcon };

const SCROLL_STICK_PX = 48;

function postToParent(message: { type: string; expanded?: boolean }) {
  // The signals carry nothing sensitive, so "*" is fine here; widget.js
  // validates the message *source* before acting on it.
  window.parent.postMessage(message, "*");
}

function reducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function toolCopy(name: string) {
  return TOOL_COPY[name] ?? UNKNOWN_TOOL;
}

/** Roughly de-markdown an answer for the screen-reader announcement. */
function plainText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[`*_#>]/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 1000);
}

const EMAIL_RE = /\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b/g;
const URL_RE = /https?:\/\/[^\s<>)]+/g;
// Segments to leave untouched: fenced code, inline code, existing links.
const PROTECTED_RE = /```[\s\S]*?```|`[^`\n]*`|\[[^\]]*\]\([^)]*\)|<[^>\s]+>/g;

/** Make bare email addresses and URLs clickable; the model rarely emits link syntax. */
function autolink(markdown: string): string {
  let out = "";
  let last = 0;
  for (const match of markdown.matchAll(PROTECTED_RE)) {
    out += linkify(markdown.slice(last, match.index)) + match[0];
    last = match.index + match[0].length;
  }
  return out + linkify(markdown.slice(last));
}

function linkify(text: string): string {
  return text
    .replace(URL_RE, (url) => {
      const trimmed = url.replace(/[.,;:!?]+$/, "");
      return `[${trimmed}](${trimmed})${url.slice(trimmed.length)}`;
    })
    .replace(EMAIL_RE, (addr) => `[${addr}](mailto:${addr})`);
}

// Markdown rendering rules for answers: links leave the iframe in a new tab,
// headings are demoted (the panel has one <h1>), images are dropped.
function ProseHeading({ children }: { children?: ReactNode }) {
  return <p className="prose-heading">{children}</p>;
}
const MARKDOWN_COMPONENTS: Components = {
  a: (props) => {
    const { node, ...rest } = props;
    void node;
    return <a {...rest} target="_blank" rel="noopener noreferrer" />;
  },
  h1: ProseHeading,
  h2: ProseHeading,
  h3: ProseHeading,
  h4: ProseHeading,
  h5: ProseHeading,
  h6: ProseHeading,
  img: () => null,
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [live, setLive] = useState("");
  const [showJump, setShowJump] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // One conversation thread per page load; the server checkpoints state by it.
  const threadRef = useRef(crypto.randomUUID());
  const listRef = useRef<HTMLDivElement>(null);
  const fieldRef = useRef<HTMLTextAreaElement>(null);
  // Follow the stream only while the visitor is reading the bottom.
  const stickRef = useRef(true);

  useEffect(() => () => abortRef.current?.abort(), []);

  function scrollToBottom(smooth: boolean) {
    const el = listRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth && !reducedMotion() ? "smooth" : "auto" });
  }

  // Streaming appends content; keep the bottom in view only when stuck there.
  useLayoutEffect(() => {
    if (stickRef.current) scrollToBottom(false);
  }, [messages, expanded]);

  function onListScroll() {
    const el = listRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickRef.current = distance <= SCROLL_STICK_PX;
    setShowJump(!stickRef.current && busy);
  }

  function jumpToLatest() {
    stickRef.current = true;
    setShowJump(false);
    scrollToBottom(true);
  }

  function toggleExpanded() {
    const next = !expanded;
    setExpanded(next);
    postToParent({ type: "ask-ruud:resize", expanded: next });
  }

  useEffect(() => {
    if (!EMBED) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      // Escape steps back: first out of the expanded view, then closes.
      if (expanded) {
        setExpanded(false);
        postToParent({ type: "ask-ruud:resize", expanded: false });
      } else {
        postToParent({ type: "ask-ruud:close" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  function patchMessage(id: string, update: Partial<Message> | ((m: Message) => Partial<Message>)) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...(typeof update === "function" ? update(m) : update) } : m)),
    );
  }

  async function ask(question: string, base: Message[] = messages) {
    question = question.trim();
    if (!question || busy) return;
    setBusy(true);
    setLive("Thinking…");
    stickRef.current = true;
    setShowJump(false);

    const history: ChatTurn[] = [
      ...base
        .filter((m) => m.content && m.status !== "error")
        .map((m): ChatTurn => ({ role: m.role, content: m.content })),
      { role: "user", content: question },
    ];
    const userId = crypto.randomUUID();
    const answerId = crypto.randomUUID();
    setMessages([
      ...base,
      { id: userId, role: "user", content: question, status: "done", tools: [], citations: [], feedback: NO_FEEDBACK },
      {
        id: answerId,
        role: "assistant",
        content: "",
        status: "streaming",
        tools: [],
        citations: [],
        feedback: NO_FEEDBACK,
        question,
      },
    ]);
    const patch = (update: Partial<Message> | ((m: Message) => Partial<Message>)) =>
      patchMessage(answerId, update);

    const abort = new AbortController();
    abortRef.current = abort;
    let content = "";
    try {
      for await (const event of streamChat(history, threadRef.current, abort.signal)) {
        if (event.type === "text") {
          content += event.text;
          patch({ content });
        } else if (event.type === "tool") {
          const name = event.name;
          setLive(toolCopy(name).active);
          patch((m) => (m.tools.at(-1) === name ? {} : { tools: [...m.tools, name] }));
        } else if (event.type === "citation") {
          patch((m) => (m.citations.includes(event.title) ? {} : { citations: [...m.citations, event.title] }));
        } else if (event.type === "approval") {
          content = content.replace(APPROVAL_NOTICE, "").trimEnd();
          patch({ content, approval: "pending" });
        } else if (event.type === "error") {
          patch({ status: "error", error: errorCopy(null) });
        } else if (event.type === "done") {
          patch((m) => (m.status === "streaming" ? { status: "done" } : {}));
          setLive(`Answer ready. ${plainText(content)}`);
        }
      }
    } catch (err) {
      if (abort.signal.aborted) {
        if (content) {
          patch({ status: "stopped" });
        } else {
          // Nothing arrived yet: drop the pair and hand the question back.
          setMessages((prev) => prev.filter((m) => m.id !== userId && m.id !== answerId));
          setInput(question);
        }
        setLive("Stopped.");
      } else {
        patch({ status: "error", error: errorCopy(err) });
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
      setShowJump(false);
      // Still streaming here means the stream ended without a done event.
      patch((m) => (m.status === "streaming" ? { status: "done" } : {}));
    }
  }

  function retry(answer: Message) {
    const index = messages.findIndex((m) => m.id === answer.id);
    const previous = messages[index - 1];
    const base = messages.filter(
      (m) => m.id !== answer.id && !(previous?.role === "user" && m.id === previous.id),
    );
    void ask(answer.question ?? previous?.content ?? "", base);
  }

  function stop() {
    abortRef.current?.abort();
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const question = input;
    setInput("");
    if (fieldRef.current) fieldRef.current.style.height = "";
    void ask(question);
  }

  function onFieldKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing) return;
    e.preventDefault();
    if (!busy && input.trim()) submit(e);
  }

  function autoGrow(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  }

  async function giveFeedback(answer: Message, verdict: "up" | "down", comment?: string) {
    patchMessage(answer.id, { feedback: { verdict, status: "pending" } });
    const ok = await sendFeedback({
      threadId: threadRef.current,
      verdict,
      question: answer.question ?? "",
      answer: answer.content,
      comment,
    });
    if (!ok) {
      patchMessage(answer.id, { feedback: { verdict, status: "failed" } });
    } else if (verdict === "up" || comment !== undefined) {
      patchMessage(answer.id, { feedback: { verdict, status: "sent" } });
    }
  }

  const showCounter = input.length >= MAX_QUESTION_CHARS - 400;

  return (
    <main className={EMBED ? "chat embed" : "chat"}>
      <header className="header">
        <div className="header-brand">
          <img src="/favicon.svg" alt="" width="28" height="28" />
          <div className="header-text">
            <h1 className="header-title">Ask Ruud</h1>
            <p className="header-subtitle">AI assistant · work, availability, contact</p>
          </div>
        </div>
        {EMBED && (
          <div className="header-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={toggleExpanded}
              aria-label={expanded ? "Shrink chat window" : "Expand chat window"}
              aria-pressed={expanded}
            >
              {expanded ? <ShrinkIcon /> : <ExpandIcon />}
            </button>
            <button
              type="button"
              className="icon-btn"
              onClick={() => postToParent({ type: "ask-ruud:close" })}
              aria-label="Close chat"
            >
              <CloseIcon />
            </button>
          </div>
        )}
      </header>

      <div
        className={messages.length === 0 ? "messages is-empty" : "messages"}
        ref={listRef} onScroll={onListScroll} role="log" aria-label="Conversation">
        {messages.length === 0 ? (
          <div className="empty">
            <h2>Ask about Ruud's work and availability.</h2>
            <p>I answer from his own notes, so I'll say when I don't know.</p>
            <p className="starters-label">Try one of these</p>
            <div className="starters">
              {STARTERS.map((text) => (
                <button key={text} type="button" className="starter" onClick={() => void ask(text)}>
                  {text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) =>
            message.role === "user" ? (
              <div key={message.id} className="msg msg-user">
                <span className="sr-only">You:</span>
                <div className="bubble">{message.content}</div>
              </div>
            ) : (
              <AssistantMessage
                key={message.id}
                message={message}
                onRetry={() => retry(message)}
                onFeedback={(verdict, comment) => void giveFeedback(message, verdict, comment)}
              />
            ),
          )
        )}
        {showJump && (
          <button type="button" className="jump-latest" onClick={jumpToLatest}>
            <ArrowDownIcon size="1rem" /> Jump to latest
          </button>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        {showCounter && (
          <p
            className={input.length >= MAX_QUESTION_CHARS ? "composer-counter at-limit" : "composer-counter"}
            aria-live="polite"
          >
            {input.length.toLocaleString("en")} / {MAX_QUESTION_CHARS.toLocaleString("en")}
          </p>
        )}
        <div className="composer-row">
          <textarea
            ref={fieldRef}
            className="composer-field"
            rows={1}
            value={input}
            maxLength={MAX_QUESTION_CHARS}
            onChange={(e) => {
              setInput(e.target.value);
              autoGrow(e.target);
            }}
            onKeyDown={onFieldKeyDown}
            placeholder="Ask about Ruud's work…"
            aria-label="Your question"
            enterKeyHint="send"
            autoComplete="off"
          />
          {busy ? (
            <button type="button" className="btn-round btn-stop" onClick={stop} aria-label="Stop generating">
              <StopIcon />
            </button>
          ) : (
            <button type="submit" className="btn-round btn-send" disabled={!input.trim()} aria-label="Send">
              <SendIcon />
            </button>
          )}
        </div>
      </form>
      <p className="disclaimer">
        AI answers can be wrong. For pricing, planning or anything binding, contact Ruud directly.
      </p>
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {live}
      </div>
    </main>
  );
}

function Dots({ small = false }: { small?: boolean }) {
  return (
    <span className={small ? "dots dots-small" : "dots"} aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
}

function AssistantMessage({
  message,
  onRetry,
  onFeedback,
}: {
  message: Message;
  onRetry: () => void;
  onFeedback: (verdict: "up" | "down", comment?: string) => void;
}) {
  const streaming = message.status === "streaming";
  const toolsActive = streaming && !message.content;
  const showFeedback = message.status === "done" && message.content && !message.error;

  return (
    <div className="msg msg-assistant">
      <span className="sr-only">Assistant:</span>
      {message.tools.map((name, i) => {
        const copy = toolCopy(name);
        const active = toolsActive && i === message.tools.length - 1;
        return (
          <p key={`${name}-${i}`} className={active ? "status-line" : "status-line is-done"}>
            {active ? <copy.Icon size="1rem" /> : <CheckIcon size="1rem" className="status-check" />}
            <span>{active ? copy.active : copy.done}</span>
            {active && <Dots small />}
          </p>
        );
      })}
      {toolsActive && message.tools.length === 0 && <Dots />}
      {message.content && (
        <div className="prose">
          <Markdown components={MARKDOWN_COMPONENTS}>{autolink(message.content)}</Markdown>
        </div>
      )}
      {message.status === "stopped" && <p className="stopped">Stopped.</p>}
      {message.approval && (
        <div className="notice notice-approval" role="status">
          <ClockIcon />
          <div>
            <p className="notice-title">Sent to Ruud for approval</p>
            <p>
              Ruud reads every message himself before it's recorded. He'll reply to the contact
              details you gave.
            </p>
          </div>
        </div>
      )}
      {message.citations.length > 0 && (
        <p className="sources">
          <span className="sources-label">Sources:</span>{" "}
          {message.citations.map((title, i) => (
            <span key={title}>
              {i > 0 && <span aria-hidden="true"> · </span>}
              {title}
            </span>
          ))}
        </p>
      )}
      {message.error && (
        <div className="notice notice-error" role="alert">
          <AlertIcon />
          <div>
            <p>{message.error}</p>
            <button type="button" className="btn-retry" onClick={onRetry}>
              Retry
            </button>
          </div>
        </div>
      )}
      {showFeedback && <FeedbackBar state={message.feedback} onFeedback={onFeedback} />}
    </div>
  );
}

function FeedbackBar({
  state,
  onFeedback,
}: {
  state: FeedbackState;
  onFeedback: (verdict: "up" | "down", comment?: string) => void;
}) {
  const [comment, setComment] = useState("");
  const [commentOpen, setCommentOpen] = useState(false);
  const decided = state.status !== "idle" && state.status !== "failed";
  const chosen = state.status === "failed" ? undefined : state.verdict;

  function down() {
    setCommentOpen(true);
    onFeedback("down");
  }

  function sendNote(e: FormEvent) {
    e.preventDefault();
    setCommentOpen(false);
    if (comment.trim()) onFeedback("down", comment);
  }

  return (
    <div className="feedback">
      <div className="feedback-row">
        <button
          type="button"
          className="icon-btn feedback-btn"
          aria-label="Good answer"
          aria-pressed={chosen === "up"}
          disabled={decided}
          onClick={() => onFeedback("up")}
        >
          <ThumbUpIcon size="1.125rem" />
        </button>
        <button
          type="button"
          className="icon-btn feedback-btn"
          aria-label="Poor answer"
          aria-pressed={chosen === "down"}
          disabled={decided}
          onClick={down}
        >
          <ThumbDownIcon size="1.125rem" />
        </button>
        {state.status === "sent" && !commentOpen && (
          <span className="feedback-note">Thanks, that helps.</span>
        )}
        {state.status === "failed" && (
          <span className="feedback-note feedback-failed">Couldn't send feedback right now.</span>
        )}
      </div>
      {commentOpen && state.status !== "failed" && (
        <form className="feedback-comment" onSubmit={sendNote}>
          <label className="sr-only" htmlFor="feedback-comment">
            What was off?
          </label>
          <textarea
            id="feedback-comment"
            rows={1}
            maxLength={1000}
            value={comment}
            autoFocus
            placeholder="Tell Ruud what was off (optional)"
            onChange={(e) => {
              setComment(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 96)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                setCommentOpen(false);
              }
            }}
          />
          <button type="submit" className="btn-secondary">
            Send note
          </button>
        </form>
      )}
    </div>
  );
}
