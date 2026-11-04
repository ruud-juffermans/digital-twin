/*
 * Ask Ruud — embeddable chat widget loader.
 *
 * Usage on any site (one line, place before </body>):
 *   <script src="https://ask.ruudjuffermans.nl/widget.js" defer></script>
 *
 * Dependency-free. Injects a floating launcher button; the chat itself runs
 * inside an iframe on the assistant's own origin (?embed=1), so the host page
 * never talks to the API directly — no CORS exposure, no style bleed, and the
 * conversation stays in the iframe's origin. The iframe is created on first
 * open, so visitors who never click pay nothing.
 *
 * Theme: the chat follows the visitor's OS colour scheme. If the host site has
 * its own light/dark switch, name the <html> attribute that holds the active
 * scheme ("light" or "dark") and the chat follows that instead, live:
 *   <script src=".../widget.js" data-theme-attribute="data-theme" defer></script>
 *
 * postMessage protocol (iframe -> parent, source-checked):
 *   {type: "ask-ruud:close"}                    close the panel
 *   {type: "ask-ruud:resize", expanded: bool}   grow/shrink the iframe
 * postMessage protocol (parent -> iframe):
 *   {type: "ask-ruud:theme", theme: "light" | "dark" | null}   null = follow OS
 */
(function () {
  "use strict";

  if (window.__askRuudWidget) return; // double-include guard
  window.__askRuudWidget = true;

  var script = document.currentScript;
  var origin = new URL(script.src).origin;
  var themeAttr = script.getAttribute("data-theme-attribute");
  var Z = 2147483000;

  var style = document.createElement("style");
  style.textContent = [
    ".ask-ruud-launcher{position:fixed;right:16px;bottom:16px;z-index:" + Z + ";",
    "width:56px;height:56px;border-radius:50%;border:none;background:#7e14ff;color:#fff;",
    "display:flex;align-items:center;justify-content:center;cursor:pointer;",
    "box-shadow:0 0 2rem rgba(0,0,0,.075),0 1rem 1rem -1rem rgba(0,0,0,.1);",
    "transition:transform .15s ease}",
    ".ask-ruud-launcher:hover{transform:scale(1.06)}",
    ".ask-ruud-launcher:focus-visible{outline:2px solid #7e14ff;outline-offset:2px}",
    ".ask-ruud-frame{position:fixed;right:16px;bottom:84px;z-index:" + Z + ";",
    "width:clamp(384px,40vw,512px);max-width:calc(100vw - 32px);",
    "height:min(608px,calc(100dvh - 112px));",
    "border-radius:16px;overflow:hidden;background:#fff;",
    "box-shadow:0 0 8rem rgba(0,0,0,.1),0 2rem 4rem -3rem rgba(0,0,0,.5);",
    "opacity:0;transform:translateY(8px);pointer-events:none;",
    "transition:opacity .2s ease,transform .2s ease,width .2s ease,height .2s ease}",
    ".ask-ruud-frame.dark{background:#15131c}",
    "@media (prefers-color-scheme:dark){.ask-ruud-frame:not(.light){background:#15131c}}",
    ".ask-ruud-frame.open{opacity:1;transform:none;pointer-events:auto}",
    ".ask-ruud-frame.expanded{width:min(900px,calc(100vw - 32px));",
    "height:calc(100dvh - 32px);bottom:16px}",
    ".ask-ruud-frame iframe{width:100%;height:100%;border:0;display:block}",
    "@media (max-width:480px){.ask-ruud-frame,.ask-ruud-frame.expanded{",
    "right:0;bottom:0;width:100vw;max-width:100vw;height:100dvh;border-radius:0}}",
  ].join("");
  document.head.appendChild(style);

  var CHAT_ICON =
    '<svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">' +
    '<path d="M12 3C7 3 3 6.6 3 11c0 2.2 1 4.2 2.7 5.6-.1 1.2-.5 2.4-1.4 3.4 ' +
    "1.8 0 3.3-.6 4.4-1.3 1 .3 2.1.4 3.3.4 5 0 9-3.6 9-8.1S17 3 12 3z" +
    '" fill="currentColor"/></svg>';
  var CLOSE_ICON =
    '<svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">' +
    '<path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2.2" ' +
    'stroke-linecap="round" fill="none"/></svg>';

  var launcher = document.createElement("button");
  launcher.type = "button";
  launcher.className = "ask-ruud-launcher";
  launcher.setAttribute("aria-label", "Open chat — Ask Ruud");
  launcher.setAttribute("aria-expanded", "false");
  launcher.innerHTML = CHAT_ICON;

  var wrap = null; // created on first open
  var frame = null;
  var open = false;

  // The host's active scheme, or null when it has none (chat follows the OS).
  function hostTheme() {
    if (!themeAttr) return null;
    var value = document.documentElement.getAttribute(themeAttr);
    return value === "dark" || value === "light" ? value : null;
  }

  function ensureFrame() {
    if (wrap) return;
    var theme = hostTheme();
    wrap = document.createElement("div");
    wrap.className = "ask-ruud-frame";
    wrap.classList.toggle("dark", theme === "dark");
    wrap.classList.toggle("light", theme === "light");
    frame = document.createElement("iframe");
    // In the URL so the first paint is already right; changes go by message.
    frame.src = origin + "/?embed=1" + (theme ? "&theme=" + theme : "");
    frame.title = "Ask Ruud — AI chat about Ruud Juffermans";
    frame.setAttribute("allow", "clipboard-write");
    wrap.appendChild(frame);
    document.body.appendChild(wrap);
  }

  function setOpen(next) {
    open = next;
    if (open) {
      ensureFrame();
      // next frame so the transition runs on first open too
      requestAnimationFrame(function () {
        wrap.classList.add("open");
      });
    } else if (wrap) {
      wrap.classList.remove("open");
    }
    launcher.innerHTML = open ? CLOSE_ICON : CHAT_ICON;
    launcher.setAttribute("aria-label", open ? "Close chat" : "Open chat — Ask Ruud");
    launcher.setAttribute("aria-expanded", String(open));
  }

  launcher.addEventListener("click", function () {
    setOpen(!open);
  });

  window.addEventListener("message", function (event) {
    // Only trust messages from our own iframe.
    if (!frame || event.origin !== origin || event.source !== frame.contentWindow) return;
    var data = event.data || {};
    if (data.type === "ask-ruud:close") {
      setOpen(false);
      launcher.focus();
    } else if (data.type === "ask-ruud:resize") {
      wrap.classList.toggle("expanded", Boolean(data.expanded));
    }
  });

  if (themeAttr) {
    new MutationObserver(function () {
      if (!frame) return;
      var theme = hostTheme();
      wrap.classList.toggle("dark", theme === "dark");
      wrap.classList.toggle("light", theme === "light");
      frame.contentWindow.postMessage({ type: "ask-ruud:theme", theme: theme }, origin);
    }).observe(document.documentElement, { attributes: true, attributeFilter: [themeAttr] });
  }

  function mount() {
    document.body.appendChild(launcher);
  }
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount);
})();