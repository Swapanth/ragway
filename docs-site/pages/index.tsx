import Head from 'next/head'
import Link from 'next/link'
import { useState, useCallback, useEffect } from 'react'

/* ─────────────────────────────────────────
   DATA
───────────────────────────────────────── */

const YAML_EXAMPLES = {
  groq: `version: "1.0"
pipeline: hybrid

plugins:
  llm:
    provider: groq
    model: llama-3.3-70b-versatile
    api_key: \${GROQ_API_KEY}

  embedding:
    provider: bge
    model: BAAI/bge-large-en-v1.5

  vectorstore:
    provider: qdrant
    index_path: .ragway/index

  reranker:
    enabled: true
    provider: bge

  chunking:
    strategy: recursive
    chunk_size: 512`,

  anthropic: `version: "1.0"
pipeline: naive

plugins:
  llm:
    provider: anthropic
    model: claude-sonnet-4-6
    api_key: \${ANTHROPIC_API_KEY}

  embedding:
    provider: openai
    model: text-embedding-3-small
    api_key: \${OPENAI_API_KEY}

  vectorstore:
    provider: pinecone
    index_name: my-project
    api_key: \${PINECONE_API_KEY}

  reranker:
    enabled: true
    provider: cohere
    api_key: \${COHERE_API_KEY}

  chunking:
    strategy: semantic
    chunk_size: 512`,

  local: `version: "1.0"
pipeline: naive

plugins:
  llm:
    provider: llama
    model_path: ./models/llama-3.1-8b.gguf

  embedding:
    provider: bge
    model: BAAI/bge-large-en-v1.5

  vectorstore:
    provider: faiss
    index_path: .ragway/index

  reranker:
    enabled: false

  chunking:
    strategy: fixed
    chunk_size: 512`,
}

const REGISTRY_CATEGORIES = [
  { cat: 'pipeline', label: 'Pipeline', items: ['naive', 'hybrid', 'self', 'long_context', 'agentic'], default: 'naive' },
  { cat: 'llm', label: 'LLM provider', items: ['anthropic', 'openai', 'mistral', 'groq', 'llama', 'local'], default: 'groq' },
  { cat: 'embedding', label: 'Embedding', items: ['bge', 'openai', 'cohere', 'sentence_transformer'], default: 'bge' },
  { cat: 'vectorstore', label: 'Vectorstore', items: ['faiss', 'qdrant', 'chroma', 'pinecone', 'weaviate', 'pgvector'], default: 'faiss' },
  { cat: 'retrieval', label: 'Retrieval', items: ['vector', 'hybrid', 'bm25', 'multi_query', 'parent_document'], default: 'vector' },
  { cat: 'reranker', label: 'Reranker', items: ['none', 'bge', 'cohere', 'cross_encoder'], default: 'none' },
  { cat: 'chunking', label: 'Chunking', items: ['recursive', 'fixed', 'semantic', 'sliding_window', 'hierarchical'], default: 'recursive' },
] as const

type CatKey = typeof REGISTRY_CATEGORIES[number]['cat']
type Selections = Record<CatKey, string>

const LLM_MODELS: Record<string, string> = {
  anthropic: 'claude-sonnet-4-6', openai: 'gpt-4o',
  mistral: 'mistral-large-latest', groq: 'llama-3.3-70b-versatile',
  llama: 'llama-3.1-8b-instruct', local: 'local-default-model',
}
const EMB_MODELS: Record<string, string> = {
  openai: 'text-embedding-3-small', bge: 'BAAI/bge-large-en-v1.5',
  cohere: 'embed-english-v3.0', sentence_transformer: 'all-MiniLM-L6-v2',
}
const NEEDS_KEY: Record<string, string[]> = {
  llm: ['anthropic', 'openai', 'mistral', 'groq'],
  embedding: ['openai', 'cohere'],
  vectorstore: ['pinecone', 'weaviate'],
  reranker: ['cohere'],
}

function buildYaml(s: Selections): string {
  const llmKey = NEEDS_KEY.llm.includes(s.llm)
    ? `\n    api_key: \${${s.llm.toUpperCase()}_API_KEY}`
    : s.llm === 'llama' ? '\n    model_path: ./models/llama-3.1-8b.gguf' : ''
  const embKey = NEEDS_KEY.embedding.includes(s.embedding)
    ? `\n    api_key: \${${s.embedding.toUpperCase()}_API_KEY}` : ''
  const vsKey = NEEDS_KEY.vectorstore.includes(s.vectorstore)
    ? `\n    api_key: \${${s.vectorstore.toUpperCase()}_API_KEY}` : ''
  const vsExtra =
    ['faiss', 'chroma', 'qdrant'].includes(s.vectorstore) ? '\n    index_path: .ragway/index' :
      s.vectorstore === 'pinecone' ? '\n    index_name: my-project' :
        s.vectorstore === 'pgvector' ? '\n    connection: ${PGVECTOR_CONNECTION_STRING}' :
          s.vectorstore === 'weaviate' ? '\n    url: ${WEAVIATE_URL}' : ''
  const rerankerBlock = s.reranker === 'none'
    ? `  reranker:\n    enabled: false`
    : `  reranker:\n    enabled: true\n    provider: ${s.reranker}${NEEDS_KEY.reranker.includes(s.reranker)
      ? `\n    api_key: \${${s.reranker.toUpperCase()}_API_KEY}` : ''
    }\n    top_k: 3`
  const hybridExtra = s.retrieval === 'hybrid' ? '\n    hybrid_alpha: 0.5' : ''

  return `version: "1.0"\npipeline: ${s.pipeline}\n\nplugins:\n  llm:\n    provider: ${s.llm}\n    model: ${LLM_MODELS[s.llm] ?? s.llm}${llmKey}\n    temperature: 0.2\n    max_tokens: 1024\n\n  embedding:\n    provider: ${s.embedding}\n    model: ${EMB_MODELS[s.embedding] ?? s.embedding}${embKey}\n    batch_size: 32\n\n  vectorstore:\n    provider: ${s.vectorstore}${vsExtra}${vsKey}\n\n  retrieval:\n    strategy: ${s.retrieval}\n    top_k: 5${hybridExtra}\n\n  ${rerankerBlock}\n\n  chunking:\n    strategy: ${s.chunking}\n    chunk_size: 512\n    overlap: 50`
}

const FAQS = [
  { q: 'How is ragway different from LangChain?', a: 'ragway is RAG-only and config-driven. Swap any component — LLM, vectorstore, reranker — with one YAML line. No Python code changes needed. LangChain is a general framework; ragway is focused entirely on RAG and is far easier to configure and debug.' },
  { q: 'Do I need all providers installed?', a: 'No. ragway uses optional dependencies. pip install ragway[groq,faiss] installs only what you need. Each provider is completely independent.' },
  { q: 'Can I use ragway without any API keys?', a: 'Yes. Use the fully_local config: Llama for LLM, BGE for embeddings and reranking, FAISS for vectorstore. Everything runs on your machine — zero cost, zero keys.' },
  { q: 'Can I switch providers without re-ingesting?', a: 'You can switch the LLM and reranker without re-ingesting. Switching vectorstore or embedding model requires re-ingesting because vector representations change.' },
  { q: 'What document types can ragway ingest?', a: 'PDF, Markdown, DOCX, Excel, HTML, plain text, URLs, YouTube transcripts, and Notion pages.' },
]

const CODE_HTML = `<span class="tok-kw">from</span> ragway <span class="tok-kw">import</span> <span class="tok-cls">RAG</span>
<span class="tok-kw">import</span> asyncio

<span class="tok-cm"># load config — swap anything via yaml</span>
rag = <span class="tok-cls">RAG</span>.<span class="tok-fn">from_config</span>(<span class="tok-str">"rag.yaml"</span>)

<span class="tok-kw">async def</span> <span class="tok-fn">main</span>():
    <span class="tok-cm"># ingest your documents</span>
    count = <span class="tok-kw">await</span> rag.<span class="tok-fn">ingest</span>(<span class="tok-str">"./docs/"</span>)
    <span class="tok-fn">print</span>(<span class="tok-str">f"ingested {count} chunks"</span>)

    answer = <span class="tok-kw">await</span> rag.<span class="tok-fn">query</span>(
        <span class="tok-str">"What is in my documents?"</span>
    )
    <span class="tok-fn">print</span>(answer)

    <span class="tok-cm"># switch LLM — one line, no re-ingest</span>
    fast = rag.<span class="tok-fn">switch</span>(<span class="tok-str">llm=</span><span class="tok-str">"groq"</span>)
    answer2 = <span class="tok-kw">await</span> fast.<span class="tok-fn">query</span>(
        <span class="tok-str">"same question, faster"</span>
    )
    <span class="tok-fn">print</span>(answer2)

asyncio.<span class="tok-fn">run</span>(<span class="tok-fn">main</span>())<span class="cursor"></span>`

/* ─────────────────────────────────────────
   COMPONENT
───────────────────────────────────────── */

export default function Home() {
  const [activeYaml, setActiveYaml] = useState<keyof typeof YAML_EXAMPLES>('groq')
  const [openFaq, setOpenFaq] = useState<number | null>(null)
  const [isDark, setIsDark] = useState(true)
  const themeClass = isDark ? 'dark-mode' : 'light-mode'

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const storedTheme = window.localStorage.getItem('theme')
    if (storedTheme === 'light' || storedTheme === 'dark') {
      setIsDark(storedTheme === 'dark')
      document.documentElement.setAttribute('data-theme', storedTheme)
    }
  }, [])

  const toggleTheme = useCallback(() => {
    setIsDark(prev => {
      const nextIsDark = !prev
      const nextTheme = nextIsDark ? 'dark' : 'light'

      if (typeof window !== 'undefined') {
        window.localStorage.setItem('theme', nextTheme)
        document.documentElement.setAttribute('data-theme', nextTheme)
      }

      return nextIsDark
    })
  }, [])

  const defaultSel = Object.fromEntries(
    REGISTRY_CATEGORIES.map(c => [c.cat, c.default])
  ) as Selections
  const [selections, setSelections] = useState<Selections>(defaultSel)

  const select = useCallback((cat: CatKey, val: string) => {
    setSelections(prev => ({ ...prev, [cat]: val }))
  }, [])

  const yaml = buildYaml(selections)

  const downloadYaml = useCallback(() => {
    const blob = new Blob([yaml], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'rag.yaml'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [yaml])

  return (
    <>
      <Head>
        <title>ragway — The way to build RAG</title>
        <meta name="description" content="Modular RAG library. Swap any component via YAML config." />
        <link rel="icon" href="/favicon.ico" />
        <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet" />
      </Head>

      <style jsx global>{`
        /* ── variables ── */
        :root {
          --accent:        #38b6ff;
          --accent-dim:    rgba(56,182,255,0.10);
          --accent-border: rgba(56,182,255,0.28);
        }
        .dark-mode {
          --bg:      #080c10;
          --bg2:     #0c1620;
          --bg3:     #0f1e2e;
          --border:  #0e1e2e;
          --border2: #1a2d42;
          --text:    #ddeaf8;
          --text2:   #5a7a96;
          --text3:   #2a4a62;
          --heading: #f0f8ff;
          --header-bg: rgba(8,12,16,0.88);
        }
        .light-mode {
          --bg:      #f7fafd;
          --bg2:     #ffffff;
          --bg3:     #eef4fa;
          --border:  #dde8f0;
          --border2: #c0d4e4;
          --text:    #1a2a38;
          --text2:   #5a7a96;
          --text3:   #94afc4;
          --heading: #0a1824;
          --header-bg: rgba(247,250,253,0.90);
        }

        /* ── reset + base ── */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { font-size: 17px; scroll-behavior: smooth; }
        body {
          font-family: 'DM Sans', sans-serif;
          background: #080c10;
          color: #ddeaf8;
          min-height: 100vh;
        }
        a { color: inherit; text-decoration: none; }
        .page-shell {
          background: var(--bg);
          color: var(--text);
          min-height: 100vh;
          transition: background .3s, color .3s;
        }

        /* ── keyframes ── */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(22px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(36px); }
          to   { opacity: 1; transform: translateX(0);    }
        }
        @keyframes float {
          0%,100% { transform: translateY(0);   }
          50%      { transform: translateY(-9px); }
        }
        @keyframes glow {
          0%,100% { box-shadow: 0 0 28px rgba(56,182,255,0.07); }
          50%      { box-shadow: 0 0 52px rgba(56,182,255,0.18); }
        }
        @keyframes shimmer {
          0%   { left: -130%; }
          100% { left: 230%;  }
        }
        @keyframes pulse {
          0%,100% { opacity: 1;   transform: scale(1);    }
          50%      { opacity: 0.4; transform: scale(0.72); }
        }
        @keyframes blink {
          0%,100% { opacity: 1; }
          50%      { opacity: 0; }
        }
        @keyframes revealUp {
          from { opacity: 0; transform: translateY(24px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
        @keyframes chipPop {
          0%  { transform: scale(1);    }
          40% { transform: scale(0.93); }
          100%{ transform: scale(1);    }
        }

        /* ── header ── */
        .header {
          position: fixed; top: 0; left: 0; right: 0; z-index: 100;
          display: flex; align-items: center; justify-content: space-between;
          padding: 0 2.5rem; height: 60px;
          border-bottom: 1px solid var(--border);
          background: var(--header-bg);
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
          transition: background .3s;
        }
        .header-logo {
          font-family: 'DM Mono', monospace;
          font-size: 18px; font-weight: 500; letter-spacing: -0.5px;
          color: var(--heading);
        }
        .header-nav { display: flex; align-items: center; gap: 1.75rem; }
        .header-nav a {
          font-family: 'DM Mono', monospace;
          font-size: 13px; color: var(--text2); transition: color .15s;
        }
        .header-nav a:hover { color: var(--text); }
        .gh-btn {
          display: flex !important; align-items: center; gap: 6px;
          font-size: 12px !important; color: var(--text2) !important;
          padding: 6px 14px;
          border: 1px solid var(--border2); border-radius: 7px;
          transition: all .18s !important;
        }
        .gh-btn:hover { color: var(--text) !important; border-color: var(--accent) !important; }
        .theme-btn {
          display: flex; align-items: center; justify-content: center;
          width: 36px; height: 36px;
          border: 1px solid var(--border2); border-radius: 7px;
          background: transparent; cursor: pointer;
          color: var(--text2); transition: all .18s; flex-shrink: 0;
        }
        .theme-btn:hover { color: var(--text); border-color: var(--accent); }

        /* ── layout ── */
        .container { max-width: 980px; margin: 0 auto; padding: 0 2.5rem; }

        /* ── hero ── */
        .hero-wrap { padding-top: 60px; }
        .hero-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 4rem;
          align-items: center;
          min-height: calc(100vh - 60px);
          padding: 4.5rem 0;
        }
        @media (max-width: 820px) {
          .hero-grid { grid-template-columns:1fr; min-height:auto; padding:3rem 0 4rem; gap:2.5rem; }
          .code-editor { display:none; }
        }

        .hero-pill {
          display: inline-flex; align-items: center; gap: 7px;
          font-family: 'DM Mono', monospace; font-size: 12px;
          color: var(--accent);
          border: 1px solid var(--accent-border); border-radius: 20px;
          padding: 5px 15px; margin-bottom: 1.75rem;
          background: var(--accent-dim);
          animation: fadeUp .55s ease both;
        }
        .pill-dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: var(--accent); flex-shrink: 0;
          animation: pulse 2.2s ease-in-out infinite;
        }

        .hero-h1 {
          font-family: 'DM Mono', monospace;
          font-size: clamp(40px, 5.5vw, 62px);
          font-weight: 500; letter-spacing: -2.5px; line-height: 1.04;
          color: var(--heading); margin-bottom: 1.25rem;
          animation: fadeUp .65s .1s ease both;
        }
        .hero-h1 span { color: var(--accent); }

        .hero-p {
          font-size: 17px; color: var(--text2); line-height: 1.75;
          max-width: 400px; margin-bottom: 2.25rem;
          animation: fadeUp .65s .18s ease both;
        }

        .hero-actions {
          display: flex; align-items: center; gap: 12px;
          flex-wrap: wrap; margin-bottom: 2.25rem;
          animation: fadeUp .65s .26s ease both;
        }

        .btn-primary {
          font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500;
          background: var(--accent); color: #fff;
          padding: 11px 26px; border-radius: 8px; border: none;
          cursor: pointer; position: relative; overflow: hidden;
          transition: background .2s, transform .15s;
          display: inline-block;
        }
        .btn-primary::after {
          content: ''; position: absolute; top: 0; left: -130%;
          width: 55%; height: 100%;
          background: rgba(255,255,255,0.22);
          transform: skewX(-22deg);
          animation: shimmer 3.2s ease-in-out infinite;
        }
        .btn-primary:hover { background: #2aa6ef; transform: translateY(-2px); }
        .btn-primary:active { transform: translateY(0); }

        .btn-secondary {
          font-family: 'DM Mono', monospace; font-size: 13px;
          color: var(--text2); padding: 11px 26px;
          border-radius: 8px; border: 1px solid var(--border2);
          cursor: pointer; transition: all .18s;
          background: transparent; display: inline-block;
        }
        .btn-secondary:hover { color: var(--text); border-color: var(--accent); }

        .hero-install {
          display: flex; align-items: center; gap: 10px;
          font-family: 'DM Mono', monospace; font-size: 13px; color: var(--text3);
          animation: fadeUp .65s .34s ease both;
        }
        .hero-install code {
          background: var(--bg2); border: 1px solid var(--border2);
          border-radius: 7px; padding: 9px 18px;
          color: var(--accent); font-size: 13px;
          transition: border-color .2s;
        }
        .hero-install code:hover { border-color: var(--accent); }

        /* ── code editor card ── */
        .code-editor {
          background: var(--bg2);
          border: 1px solid var(--border2);
          border-radius: 14px; overflow: hidden;
          animation:
            slideInRight .75s .15s ease both,
            float 5.5s 1.2s ease-in-out infinite,
            glow  5.5s 1.2s ease-in-out infinite;
        }
        .editor-header {
          display: flex; align-items: center; gap: 7px;
          padding: 13px 18px; border-bottom: 1px solid var(--border);
          background: var(--bg);
        }
        .editor-dot  { width: 11px; height: 11px; border-radius: 50%; }
        .editor-file {
          font-family: 'DM Mono', monospace;
          font-size: 12px; color: var(--text3); margin-left: 7px;
        }
        .editor-body { padding: 1.5rem 1.75rem; overflow-x: auto; }
        .editor-body pre {
          font-family: 'DM Mono', monospace;
          font-size: 13px; line-height: 2; color: var(--text2);
          margin: 0; white-space: pre;
        }
        .tok-kw  { color: #c084fc; }
        .tok-fn  { color: #86efac; }
        .tok-str { color: #fde68a; }
        .tok-cm  { color: var(--text3); }
        .tok-cls { color: var(--accent); }
        .cursor {
          display: inline-block; width: 2px; height: 14px;
          background: var(--accent); vertical-align: middle;
          animation: blink 1s step-end infinite;
        }

        /* ── sections ── */
        .section { padding: 100px 0; border-top: 1px solid var(--border); }
        .section-eyebrow {
          font-family: 'DM Mono', monospace; font-size: 11px;
          color: var(--accent); letter-spacing: .13em;
          text-transform: uppercase; margin-bottom: 1rem;
        }
        .section-h2 {
          font-family: 'DM Mono', monospace; font-size: 32px;
          font-weight: 500; letter-spacing: -.5px;
          color: var(--heading); margin-bottom: .75rem;
        }
        .section-lead {
          font-size: 17px; color: var(--text2); line-height: 1.75;
          max-width: 520px; margin-bottom: 2.75rem;
        }

        /* ── yaml tabs ── */
        .yaml-tabs { display: flex; gap: 4px; }
        .yaml-tab {
          font-family: 'DM Mono', monospace; font-size: 13px;
          padding: 7px 16px; border-radius: 7px 7px 0 0;
          cursor: pointer; transition: all .18s;
          border: 1px solid transparent; border-bottom: none;
          color: var(--text3); background: transparent;
        }
        .yaml-tab.active {
          color: var(--accent); background: var(--bg2); border-color: var(--border2);
        }
        .yaml-tab:not(.active):hover { color: var(--text2); }
        .yaml-block {
          background: var(--bg2); border: 1px solid var(--border2);
          border-radius: 0 10px 10px 10px;
          padding: 1.75rem; overflow-x: auto;
        }
        .yaml-block pre {
          font-family: 'DM Mono', monospace; font-size: 13px;
          color: var(--text2); line-height: 1.85; margin: 0; white-space: pre;
        }

        /* ── config builder ── */
        .builder-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 3.5rem;
          align-items: start;
        }
        @media (max-width: 820px) {
          .builder-grid { grid-template-columns: 1fr; gap: 2.5rem; }
        }
        .builder-category { margin-bottom: 1.6rem; }
        .builder-cat-label {
          font-family: 'DM Mono', monospace; font-size: 11px;
          font-weight: 500; text-transform: uppercase; letter-spacing: .07em;
          color: var(--text3); margin-bottom: .65rem;
        }
        .builder-chips { display: flex; flex-wrap: wrap; gap: 7px; }
        .builder-chip {
          font-family: 'DM Mono', monospace; font-size: 12px;
          color: var(--text2); padding: 5px 14px;
          border: 1px solid var(--border); border-radius: 6px;
          cursor: pointer; background: var(--bg2);
          transition: color .18s, border-color .18s, background .18s;
          user-select: none;
        }
        .builder-chip:hover { color: var(--text); border-color: var(--border2); }
        .builder-chip.selected {
          color: var(--accent); border-color: var(--accent);
          background: var(--accent-dim);
          animation: chipPop .2s ease;
        }
        .builder-out-label {
          font-family: 'DM Mono', monospace; font-size: 11px;
          font-weight: 500; text-transform: uppercase; letter-spacing: .07em;
          color: var(--text3); margin-bottom: .6rem;
        }
        .builder-badges {
          display: flex; flex-wrap: wrap; gap: 5px;
          margin-bottom: .85rem; min-height: 26px;
        }
        .builder-badge {
          font-family: 'DM Mono', monospace; font-size: 10.5px;
          padding: 2px 9px; border-radius: 4px;
          background: var(--accent-dim); color: var(--accent);
          border: 1px solid var(--accent-border);
        }
        .builder-badge strong { font-weight: 500; }
        .builder-yaml {
          background: var(--bg2); border: 1px solid var(--border2);
          border-radius: 10px; padding: 1.35rem 1.5rem;
          overflow-x: auto; margin-bottom: .85rem;
          transition: border-color .2s;
        }
        .builder-yaml:hover { border-color: var(--accent-border); }
        .builder-yaml pre {
          font-family: 'DM Mono', monospace; font-size: 12.5px;
          line-height: 1.85; color: var(--text2);
          margin: 0; white-space: pre;
        }
        .builder-download {
          width: 100%;
          font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500;
          background: var(--accent); color: #fff;
          border: none; border-radius: 8px; padding: 12px 20px;
          cursor: pointer; display: flex; align-items: center;
          justify-content: center; gap: 8px;
          transition: background .18s, transform .15s;
          position: relative; overflow: hidden;
        }
        .builder-download::after {
          content: ''; position: absolute; top: 0; left: -130%;
          width: 55%; height: 100%;
          background: rgba(255,255,255,0.18);
          transform: skewX(-22deg);
          animation: shimmer 3.5s ease-in-out infinite;
        }
        .builder-download:hover { background: #2aa6ef; transform: translateY(-2px); }
        .builder-download:active { transform: translateY(0); }

        /* ── faq ── */
        .faq-item { border-bottom: 1px solid var(--border); }
        .faq-q {
          display: flex; justify-content: space-between; align-items: center;
          padding: 1.35rem 0; cursor: pointer;
          font-size: 16px; color: var(--text2); gap: 1rem; transition: color .15s;
        }
        .faq-q:hover { color: var(--text); }
        .faq-icon {
          font-size: 22px; line-height: 1; color: var(--text3);
          transition: transform .22s, color .15s; flex-shrink: 0;
        }
        .faq-icon.open { transform: rotate(45deg); color: var(--accent); }
        .faq-a {
          font-size: 15px; color: var(--text2); line-height: 1.85;
          padding-bottom: 1.35rem; max-width: 680px;
        }

        /* ── footer ── */
        .footer {
          border-top: 1px solid var(--border);
          padding: 3rem 2.5rem; text-align: center;
        }
        .footer-name {
          font-family: 'DM Mono', monospace;
          font-size: 16px; color: var(--text3); margin-bottom: 1.1rem;
        }
        .footer-links { display: flex; justify-content: center; gap: 2rem; }
        .footer-links a {
          font-family: 'DM Mono', monospace;
          font-size: 13px; color: var(--text3); transition: color .15s;
        }
        .footer-links a:hover { color: var(--accent); }
      `}</style>

      <div className={`page-shell ${themeClass}`}>
        {/* ── HEADER ── */}
        <header className="header">
          <div className="header-logo">ragway</div>
          <nav className="header-nav">
            <Link href="/docs">docs</Link>
            <Link href="/docs/quickstart">quickstart</Link>
            <a href="https://pypi.org/project/ragway/" target="_blank" rel="noreferrer">pypi</a>
            <a href="https://github.com/swapanth/ragway" target="_blank" rel="noreferrer" className="gh-btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </a>
            {/* ── LIGHT / DARK TOGGLE ── */}
            <button className="theme-btn" onClick={toggleTheme} aria-label="Toggle theme">
              {isDark
                ? <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>
                : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
              }
            </button>
          </nav>
        </header>

        {/* ── MAIN ── */}

        {/* ── HERO ── */}
        <div className="hero-wrap">
          <div className="container">
            <div className="hero-grid">

              {/* left */}
              <div>
                <div className="hero-pill">
                  <div className="pill-dot" />
                  v0.1.0 — now on PyPI
                </div>
                <h1 className="hero-h1">
                  The way to<br />
                  <span>build RAG</span>
                </h1>
                <p className="hero-p">
                  Modular RAG library for Python. Swap any component — LLM,
                  vectorstore, reranker — with one line in a YAML file.
                  No code changes. Just config.
                </p>
                <div className="hero-actions">
                  <Link href="/docs/quickstart" className="btn-primary">Get started →</Link>
                  <Link href="/docs" className="btn-secondary">Documentation</Link>
                </div>
                <div className="hero-install">
                  <span>install</span>
                  <code>pip install ragway</code>
                </div>
              </div>

              {/* right — code editor */}
              <div className="code-editor">
                <div className="editor-header">
                  <div className="editor-dot" style={{ background: '#ff5f57' }} />
                  <div className="editor-dot" style={{ background: '#febc2e' }} />
                  <div className="editor-dot" style={{ background: '#28c840' }} />
                  <span className="editor-file">quickstart.py</span>
                </div>
                <div className="editor-body">
                  <pre dangerouslySetInnerHTML={{ __html: CODE_HTML }} />
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* ── HOW TO USE ── */}
        <div className="container">
          <div className="section">
            <div className="section-eyebrow">how it works</div>
            <h2 className="section-h2">One YAML file.<br />Everything configured.</h2>
            <p className="section-lead">
              Pick a provider for each component. Add your API key.
              ragway wires it all together — no Python changes needed.
            </p>
            <div className="yaml-tabs">
              {(Object.keys(YAML_EXAMPLES) as Array<keyof typeof YAML_EXAMPLES>).map(k => (
                <button
                  key={k}
                  className={`yaml-tab${activeYaml === k ? ' active' : ''}`}
                  onClick={() => setActiveYaml(k)}
                >
                  {k === 'groq' ? 'groq + qdrant' : k === 'anthropic' ? 'anthropic + pinecone' : 'fully local'}
                </button>
              ))}
            </div>
            <div className="yaml-block">
              <pre>{YAML_EXAMPLES[activeYaml]}</pre>
            </div>
          </div>
        </div>

        {/* ── CONFIG BUILDER ── */}
        <div className="container">
          <div className="section">
            <div className="section-eyebrow">config builder</div>
            <h2 className="section-h2">Build your rag.yaml</h2>
            <p className="section-lead">
              Select one from each category. Your config generates live.
              Download and drop it straight into your project.
            </p>
            <div className="builder-grid">

              {/* left — chips */}
              <div>
                {REGISTRY_CATEGORIES.map(({ cat, label, items }) => (
                  <div className="builder-category" key={cat}>
                    <div className="builder-cat-label">{label}</div>
                    <div className="builder-chips">
                      {items.map(item => (
                        <button
                          key={item}
                          className={`builder-chip${selections[cat] === item ? ' selected' : ''}`}
                          onClick={() => select(cat as CatKey, item)}
                        >
                          {item}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* right — live yaml */}
              <div>
                <div className="builder-out-label">generated config</div>
                <div className="builder-badges">
                  {Object.entries(selections).map(([k, v]) => (
                    <span className="builder-badge" key={k}>
                      {k}: <strong>{v}</strong>
                    </span>
                  ))}
                </div>
                <div className="builder-yaml">
                  <pre>{yaml}</pre>
                </div>
                <button className="builder-download" onClick={downloadYaml}>
                  <svg width="15" height="15" viewBox="0 0 16 16" fill="none"
                    stroke="currentColor" strokeWidth="1.6"
                    strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 2v8m0 0L5 7m3 3 3-3M2 13h12" />
                  </svg>
                  download rag.yaml
                </button>
              </div>

            </div>
          </div>
        </div>

        {/* ── FAQ ── */}
        <div className="container">
          <div className="section">
            <div className="section-eyebrow">faq</div>
            <h2 className="section-h2">Common questions</h2>
            {FAQS.map((item, i) => (
              <div className="faq-item" key={i}>
                <div className="faq-q" onClick={() => setOpenFaq(openFaq === i ? null : i)}>
                  <span>{item.q}</span>
                  <span className={`faq-icon${openFaq === i ? ' open' : ''}`}>+</span>
                </div>
                {openFaq === i && <div className="faq-a">{item.a}</div>}
              </div>
            ))}
          </div>
        </div>

        {/* ── FOOTER ── */}
        <footer className="footer">
          <div className="footer-name">ragway</div>
          <div className="footer-links">
            <a href="https://pypi.org/project/ragway/" target="_blank" rel="noreferrer">pypi</a>
            <a href="https://github.com/swapanth/ragway" target="_blank" rel="noreferrer">github</a>
            <Link href="/docs">docs</Link>
            <Link href="/docs/quickstart">quickstart</Link>
            <a href="mailto:swapanthvakapalli@gmail.com">swapanthvakapalli@gmail.com</a>
          </div>
        </footer>
      </div>

    </>
  )
}
