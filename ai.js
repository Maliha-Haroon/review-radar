
// Vercel serverless function: keeps the Gemini API key on the server.
const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash"; // change via env var if needed

module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const key = process.env.GEMINI_API_KEY;
  if (!key) return res.status(503).json({ error: "no_key" });

  const { kind, review = "", sentiment = "", topic = "", facts = "" } = req.body || {};
  const cut = (s, n) => String(s).slice(0, n);
  let prompt;
  if (kind === "reply") {
    prompt = `You are the customer care manager of Zaiqa Bites, a Karachi restaurant. Write a short, warm reply (max 60 words) to this ${cut(sentiment, 20)} review about ${cut(topic, 20)}. Apologise and offer a fix if negative. Reply in the same language style as the review. Output only the reply.\n\nReview: ${cut(review, 600)}`;
  } else if (kind === "report") {
    prompt = `You are a restaurant operations analyst. From these facts write a manager report: 1) top 3 problems with evidence, 2) one concrete action for each, 3) one quick win. Be concise and use plain text with short bullet lines.\n\n${cut(facts, 6000)}`;
  } else {
    return res.status(400).json({ error: "bad request" });
  }

  try {
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`, {
      method: "POST",
      headers: { "x-goog-api-key": key, "content-type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 2048, temperature: 0.7 },
      }),
    });
    const j = await r.json();
    if (!r.ok) return res.status(502).json({ error: (j.error && j.error.message) || "Gemini API error" });
    const parts = (j.candidates && j.candidates[0] && j.candidates[0].content && j.candidates[0].content.parts) || [];
    const text = parts.filter((p) => p.text && !p.thought).map((p) => p.text).join("").trim();
    if (!text) return res.status(502).json({ error: "empty response" });
    return res.status(200).json({ text });
  } catch (e) {
    return res.status(502).json({ error: e.message });
  }
};
