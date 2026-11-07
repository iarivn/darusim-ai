// DaruSim AI — Cloudflare Worker (free tier)
// ---------------------------------------------------------------
// POST /ask  { "question": "..." }  ->  { "answer": "...", "sources": ["نام دارو", ...] }
//
// The Gemini API key lives ONLY here, as a Worker Secret named
// GEMINI_API_KEY (Dashboard -> Workers -> darusim-ai -> Settings ->
// Variables -> Add variable, type Secret). It never reaches the browser,
// so visitors ask questions without any key of their own.
//
// How it works:
//   1. load data.json (241 drugs + 768-dim vectors) from the Pages site,
//      cached in memory and in caches.default
//   2. embed the question with gemini-embedding-001
//   3. cosine retrieval, top 4
//   4. generate the answer with gemini-3.5-flash, grounded in the sources
//
// Abuse guard: 20 requests per minute per IP (best effort, per isolate).
// If the key is ever abused, rotate it in AI Studio and update the Secret.

const DATA_URL = "https://iarivn.github.io/darusim-ai/data.json";
const ALLOWED_ORIGIN = "https://iarivn.github.io";
const EMBED_MODEL = "gemini-embedding-001";
const GEN_MODEL = "gemini-3.5-flash";
const DIMS = 768;
const TOP_K = 4;
const DISCLAIMER = "این اطلاعات صرفاً آموزشی است و جایگزین نظر پزشک یا داروساز نیست.";
const RATE_LIMIT = 20;
const WINDOW_MS = 60_000;

let DATA = null;
const hits = new Map();

function corsHeaders(allowed, origin) {
  const h = {};
  if (allowed) {
    h["Access-Control-Allow-Origin"] = origin;
    h["Access-Control-Allow-Methods"] = "POST, OPTIONS";
    h["Access-Control-Allow-Headers"] = "Content-Type";
  }
  return h;
}

function json(obj, status, origin, allowed) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(allowed, origin) },
  });
}

async function loadData() {
  if (DATA) return DATA;
  const cache = caches.default;
  const cached = await cache.match(DATA_URL);
  if (cached) {
    DATA = await cached.json();
    return DATA;
  }
  const res = await fetch(DATA_URL);
  if (!res.ok) throw new Error("data fetch failed: " + res.status);
  const text = await res.text();
  cache
    .put(DATA_URL, new Response(text, { headers: { "Content-Type": "application/json" } }))
    .catch(() => {});
  DATA = JSON.parse(text);
  return DATA;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

async function gemini(path, body, key) {
  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/" + path + "?key=" + key,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
  );
  if (!res.ok) {
    const err = new Error("gemini " + res.status);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

async function embed(text, key) {
  const out = await gemini(
    "models/" + EMBED_MODEL + ":embedContent",
    { model: "models/" + EMBED_MODEL, content: { parts: [{ text }] } },
    key
  );
  return out.embedding.values.slice(0, DIMS);
}

async function generate(prompt, key) {
  let lastErr = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const out = await gemini(
        "models/" + GEN_MODEL + ":generateContent",
        { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.2 } },
        key
      );
      return out.candidates[0].content.parts[0].text;
    } catch (e) {
      lastErr = e;
      if (e.status === 429 || e.status === 503) {
        await new Promise((r) => setTimeout(r, 8000 * (attempt + 1)));
      } else {
        throw e;
      }
    }
  }
  throw lastErr;
}

function buildPrompt(question, retrieved) {
  const sources = retrieved
    .map((d, i) => "--- منبع " + (i + 1) + ": " + d.fa + " (" + d.en + ") ---\n" + d.text)
    .join("\n\n");
  return (
    "تو یک دستیار دارویی هستی. فقط بر اساس منابع زیر به سؤال جواب بده و به زبان فارسی پاسخ بده.\n" +
    "اگر جواب در منابع نیست، صریح بگو که در منابع موجود نیست و حدس نزن.\n" +
    "نام داروهایی که از آن‌ها استفاده کردی را در جواب ذکر کن.\n" +
    "در پایان جواب، دقیقاً این جمله را اضافه کن: «" + DISCLAIMER + "»\n\n" +
    "منابع:\n" + sources + "\n\nسؤال: " + question
  );
}

function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > RATE_LIMIT;
}

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get("Origin") || "";
    const allowed = origin === ALLOWED_ORIGIN;

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(allowed, origin) });
    }

    const url = new URL(request.url);
    if (request.method !== "POST" || url.pathname !== "/ask") {
      return json({ error: "not found" }, 404, origin, allowed);
    }
    if (!allowed) {
      return json({ error: "forbidden origin" }, 403, origin, false);
    }

    const ip = request.headers.get("CF-Connecting-IP") || "unknown";
    if (rateLimited(ip)) {
      return json({ error: "rate_limited" }, 429, origin, allowed);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "bad json" }, 400, origin, allowed);
    }
    const question = (body.question || "").trim();
    if (!question || question.length > 500) {
      return json({ error: "bad question" }, 400, origin, allowed);
    }
    if (!env.GEMINI_API_KEY) {
      return json({ error: "no key configured" }, 500, origin, allowed);
    }

    try {
      const data = await loadData();
      const qvec = await embed(question, env.GEMINI_API_KEY);
      const ranked = data
        .map((d) => ({ d, s: cosine(qvec, d.vec) }))
        .sort((a, b) => b.s - a.s)
        .slice(0, TOP_K)
        .map((x) => x.d);
      let answer = await generate(buildPrompt(question, ranked), env.GEMINI_API_KEY);
      if (!answer.includes("صرفاً آموزشی")) answer += "\n\n" + DISCLAIMER;
      return json({ answer, sources: ranked.map((d) => d.fa) }, 200, origin, allowed);
    } catch (e) {
      console.error(e);
      const status = e.status === 429 || e.status === 503 ? 503 : 500;
      return json({ error: "upstream", status }, status, origin, allowed);
    }
  },
};
