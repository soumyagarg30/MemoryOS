import { useCallback, useEffect, useRef, useState } from 'react'
import { Archive, ArrowDownToLine, ArrowRight, Check, Clock3, FileText, Folder, Layers, MessageSquare, RefreshCw, Search, Send, X } from 'lucide-react'
import { api } from './api'
import type { Activity, ChatResult, Consolidation, Memory, MemoryType, Provenance, Retrieved, Status } from './types'

const types: MemoryType[] = ['PREFERENCE', 'EPISODIC', 'SEMANTIC', 'TASK', 'WORKING']
const statuses: Status[] = ['ACTIVE', 'STALE', 'ARCHIVED', 'SUPERSEDED']
const short = (id: string) => id.slice(0, 8).toUpperCase()
const date = (value: string | null) => value ? new Date(value).toLocaleString() : 'Not recorded'
interface Turn {role: 'user' | 'assistant'; content: string; recalled?: Retrieved[]}
function Stamp({ value }: { value: string }) { return <span className={`stamp ${value.toLowerCase()}`}>{value}</span> }

export default function App() {
  const [user, setUser] = useState(() => localStorage.getItem('memoryos-user') || crypto.randomUUID())
  const [draftUser, setDraftUser] = useState(user)
  const [memories, setMemories] = useState<Memory[]>([])
  const [selected, setSelected] = useState<Memory | null>(null)
  const [provenance, setProvenance] = useState<Provenance | null>(null)
  const [provenanceError, setProvenanceError] = useState('')
  const [view, setView] = useState<'archive' | 'session'>('archive')
  const [type, setType] = useState('ALL')
  const [status, setStatus] = useState('ALL')
  const [search, setSearch] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [message, setMessage] = useState('')
  const [events, setEvents] = useState<Activity[]>([])
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(false)
  const [connection, setConnection] = useState('Checking')
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [consolidated, setConsolidated] = useState<Consolidation | null>(null)
  const [cursor, setCursor] = useState<string | null>(null)
  const generation = useRef(0)
  const operation = useRef(false)
  const query = useCallback((page: number) => {
    const params = new URLSearchParams({ user_id: user, limit: '50', offset: String(page) })
    if (type !== 'ALL') params.set('memory_type', type)
    if (status !== 'ALL') params.set('status', status)
    return `/memories?${params}`
  }, [user, type, status])
  const refresh = useCallback(async (page = 0) => {
    const version = ++generation.current
    setLoading(true)
    try {
      const records = await api<Memory[]>(query(page), undefined, 15000)
      if (version !== generation.current) return
      setMemories(records); setOffset(page); setHasMore(records.length === 50); setConnection('Connected')
    } catch (e) { if (version === generation.current) { setConnection('Unavailable'); setError((e as Error).message) } }
    finally { if (version === generation.current) setLoading(false) }
  }, [query])
  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => { localStorage.setItem('memoryos-user', user) }, [user])
  useEffect(() => {
    let active = true
    setProvenance(null); setProvenanceError('')
    if (selected) api<Provenance>(`/memories/${selected.id}/provenance`, undefined, 15000).then(p => { if (active) setProvenance(p) }).catch(e => { if (active) setProvenanceError(e.message) })
    return () => { active = false }
  }, [selected])

  async function perform(label: string, action: () => Promise<void>) {
    if (operation.current) return
    operation.current = true; setBusy(label); setError(''); setNotice('')
    try { await action(); await refresh(); setSelected(null) }
    catch (e) { setError((e as Error).message); await refresh() }
    finally { operation.current = false; setBusy('') }
  }
  const addEvents = (items: Activity[]) => setEvents(old => [...items.reverse(), ...old].slice(0, 100))
  const event = (kind: string, text: string, ids: string[] = []) => addEvents([{kind, message: text, memory_ids: ids, at: new Date().toISOString()}])
  async function send() {
    const content = message.trim()
    if (!content || busy) return
    await perform('Processing conversation', async () => {
      const result = await api<ChatResult>('/chat', {user_id: user, message: content})
      setTurns(old => [...old, {role: 'user', content}, {role: 'assistant', content: result.assistant_response, recalled: result.retrieved_memories}])
      setMessage(''); addEvents(result.memory_activity_events); setNotice(result.warnings.join(' '))
    })
  }
  async function consolidate() {
    await perform('Consolidating archive', async () => {
      const result = await api<Consolidation>('/memory/consolidate', {user_id: user, after_id: cursor})
      setConsolidated(result); setCursor(result.next_cursor)
      for (const row of result.memories) event('CONSOLIDATION_CREATED', row.memory.content, [...row.source_memory_ids, row.memory.id])
      setNotice(result.memories.length ? `${result.memories.length} semantic records created.` : `No eligible clusters on this page. ${result.candidates_examined} candidates examined.`)
    })
  }
  async function inspect(id: string) {
    try { setSelected(await api<Memory>(`/memories/${id}`, undefined, 15000)) }
    catch (e) { setError((e as Error).message) }
  }
  const filtered = memories.filter(m => m.content.toLowerCase().includes(search.toLowerCase()) || m.id.includes(search.toLowerCase()))
  return <div className="terminal">
    <header className="masthead"><div className="brand"><Archive size={27}/><div><h1>MEMORYOS</h1><span>MEMORY CORE ONLINE / ARCHIVE TERMINAL 01</span></div></div><div className="header-right"><span className={`connection ${connection === 'Connected' ? 'online' : ''}`}>DB: {connection}</span><span className="edition">LOCAL EDITION</span></div></header>
    <div className="session-bar"><span><span className="tiny-square"/> PERSONAL MEMORY ARCHIVE</span><form onSubmit={e => { e.preventDefault(); if (busy) return; if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(draftUser)) {setError('Enter a valid user UUID.'); return} setUser(draftUser); setTurns([]); setEvents([]); setSelected(null); setCursor(null); setConsolidated(null) }}><label htmlFor="user">SESSION</label><input id="user" aria-label="User ID" value={draftUser} onChange={e => setDraftUser(e.target.value)} disabled={!!busy}/><button title="Apply session" aria-label="Apply session" disabled={!!busy}><Check size={15}/></button></form></div>
    <div className="workspace"><aside className="sidebar"><div className="section-label">01 / COLLECTIONS</div><button className={`nav ${type === 'ALL' && status === 'ALL' ? 'chosen' : ''}`} onClick={() => {setType('ALL');setStatus('ALL');setView('archive')}}><Archive size={16}/>All memories</button><div className="nav-divider"/>{types.map(t => <button key={t} className={`nav ${type === t ? 'chosen' : ''}`} onClick={() => {setType(t);setStatus('ALL');setView('archive')}}><Folder size={16}/>{t.charAt(0) + t.slice(1).toLowerCase()}</button>)}<div className="nav-divider"/>{['ARCHIVED','SUPERSEDED'].map(s => <button key={s} className={`nav ${status === s ? 'chosen' : ''}`} onClick={() => {setStatus(s);setType('ALL');setView('archive')}}><Clock3 size={16}/>{s.charAt(0) + s.slice(1).toLowerCase()}</button>)}
    <div className="archive-photo"><img src="https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=400&q=80" alt="Shelves in a library archive"/><span>THE THINGS WE KEEP.</span></div><button className="demo-button" disabled={!!busy} onClick={() => perform('Loading demo archive', async () => { const r = await api<{already_loaded: boolean; memory_activity_events: Activity[]}>('/demo', {user_id:user}); addEvents(r.memory_activity_events); setNotice(r.already_loaded ? 'Demo records are already in this archive.' : 'Demo archive loaded.'); setView('session') })}><ArrowDownToLine size={16}/>LOAD ARCHIVE TAPE</button></aside>
    <main><div className="main-top"><div><div className="section-label">02 / {view === 'archive' ? 'THE CATALOG' : 'LIVE SESSION'}</div><h2>{view === 'archive' ? 'Memory archive' : 'At the writing desk'}</h2></div><div className="tabs" role="tablist"><button role="tab" aria-selected={view === 'archive'} onClick={() => setView('archive')}><Archive size={16}/>Archive</button><button role="tab" aria-selected={view === 'session'} onClick={() => setView('session')}><MessageSquare size={16}/>Session</button></div></div>
    {error && <div role="alert" className="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={16}/></button></div>}{notice && <div role="status" className="notice">{notice}</div>}{busy && <div role="status" className="busy"><span className="pulse"/>{busy}...</div>}
    {view === 'archive' ? <><div className="filters"><label className="search"><Search size={16}/><input aria-label="Search current page" placeholder="Search this page..." value={search} onChange={e => setSearch(e.target.value)}/></label><select aria-label="Memory type" value={type} onChange={e => setType(e.target.value)}><option value="ALL">All types</option>{types.map(t => <option key={t}>{t}</option>)}</select><select aria-label="Memory status" value={status} onChange={e => setStatus(e.target.value)}><option value="ALL">All states</option>{statuses.map(s => <option key={s}>{s}</option>)}</select><button title="Refresh archive" aria-label="Refresh archive" onClick={() => {setError('');void refresh(offset)}} disabled={loading}><RefreshCw size={17}/></button></div><div className="catalog-meta"><span>{filtered.length.toString().padStart(2,'0')} RECORDS / PAGE {Math.floor(offset/50)+1}</span><span>FILED BY MOST RECENT</span></div>
    <div className="records" aria-busy={loading}>{loading ? <div className="empty">Opening archive...</div> : filtered.length ? filtered.map(m => <button key={m.id} className={`record ${selected?.id === m.id ? 'selected' : ''}`} onClick={() => setSelected(m)}><div className="record-top"><span className="record-id">NO. {short(m.id)}</span><Stamp value={m.status}/></div><p>{m.content}</p><div className="record-type"><FileText size={13}/>{m.memory_type}{m.consolidated_at && <Stamp value="CONSOLIDATED"/>}</div><div className="record-metrics"><span>IMP <b>{m.importance.toFixed(2)}</b></span><span>CONF <b>{m.confidence.toFixed(2)}</b></span><span>READS <b>{m.access_count}</b></span></div><div className="record-dates">FILED {date(m.created_at)}<br/>LAST READ {date(m.last_accessed_at)}</div></button>) : <div className="empty"><Archive size={36}/><h3>{search || type !== 'ALL' || status !== 'ALL' ? 'No matching records' : 'An archive yet to be written.'}</h3><span>{search || type !== 'ALL' || status !== 'ALL' ? '0 records in this view' : 'NO RECORDS FILED'}</span></div>}</div><div className="pagination"><button disabled={offset === 0 || loading} onClick={() => refresh(Math.max(0,offset-50))}>Previous</button><button disabled={!hasMore || loading} onClick={() => refresh(offset+50)}>Next</button></div></> : <div className="session"><div className="conversation">{turns.length === 0 && <div className="session-empty"><span className="section-label">A NEW ENTRY</span><h3>What is worth remembering?</h3><div className="prompts">{['I switched to PostgreSQL for new projects.', 'What database do I use for new projects?', 'I prefer Python at work, but C++ for contests.'].map(p => <button key={p} onClick={() => setMessage(p)}>{p}<ArrowRight size={15}/></button>)}</div></div>}{turns.map((t,i) => <article className={`turn ${t.role}`} key={i}><span className="section-label">{t.role === 'user' ? 'YOU / NEW ENTRY' : 'MEMORYOS / RESPONSE'}</span><p>{t.content}</p>{t.recalled && t.recalled.length > 0 && <details><summary>{t.recalled.length} recalled records</summary>{t.recalled.map(r => <div className="recall" key={r.memory.id}><button onClick={() => setSelected(r.memory)}>{short(r.memory.id)} / {r.memory.content}</button><span>RANK {r.score.toFixed(3)}</span>{r.components && <div className="scores">{Object.entries(r.components).map(([k,v]) => <span key={k}>{k.replaceAll('_',' ')}: {v.toFixed(3)}</span>)}</div>}</div>)}</details>}</article>)}</div><form className="composer" onSubmit={e => {e.preventDefault();void send()}}><textarea aria-label="Message" placeholder="Add a thought to the record..." maxLength={8000} value={message} onChange={e => setMessage(e.target.value)} disabled={!!busy}/><button className="primary" disabled={!!busy || !message.trim()} title="Send message" aria-label="Send message"><Send size={18}/></button></form></div>}
    <div className="archive-actions"><button disabled={!!busy} onClick={consolidate}><Layers size={16}/>{cursor ? 'Consolidate next page' : 'Consolidate archive'}</button><button disabled={!!busy} onClick={() => perform('Applying memory decay', async () => {const r = await api<{stale:number;archived:number;processed:number}>('/memory/decay', {user_id:user}); event('DECAY_APPLIED', `${r.processed} evaluated / ${r.stale} stale / ${r.archived} archived`); setNotice(`${r.stale} memories became stale; ${r.archived} archived.`)})}><Clock3 size={16}/>Apply decay</button></div>
    {consolidated && consolidated.memories.length > 0 && <section className="consolidation"><h3>Consolidation ledger</h3>{consolidated.memories.map(c => <div key={c.memory.id}><div className="source-links">{c.source_memory_ids.map(id => <button key={id} onClick={() => inspect(id)}>{short(id)}</button>)}</div><ArrowRight size={18}/><button onClick={() => setSelected(c.memory)}>{c.memory.content}</button></div>)}</section>}
    </main>
    <aside className="inspector"><div className="section-label">03 / RECORD INSPECTOR {selected && <button aria-label="Close inspector" onClick={() => setSelected(null)}><X size={15}/></button>}</div>{selected ? <><div className="inspector-id">FILE / {short(selected.id)}</div><Stamp value={selected.status}/>{selected.consolidated_at && <Stamp value="CONSOLIDATED"/>}<h3>{selected.content}</h3><div className="meter-label">IMPORTANCE <b>{selected.importance.toFixed(2)}</b></div><meter min="0" max="1" value={selected.importance}/><div className="meter-label">CONFIDENCE <b>{selected.confidence.toFixed(2)}</b></div><meter min="0" max="1" value={selected.confidence}/><dl>{Object.entries({'Memory ID':selected.id,'Type':selected.memory_type,'Source':selected.source || 'Not recorded','Created':date(selected.created_at),'Updated':date(selected.updated_at),'Last accessed':date(selected.last_accessed_at),'Access count':selected.access_count,'Consolidated':date(selected.consolidated_at),'Expires':date(selected.expires_at)}).map(([k,v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl><h4>PROVENANCE</h4>{provenanceError ? <p role="alert">{provenanceError}</p> : !provenance ? <p>Loading links...</p> : <>{(['derived_from','consolidated_into','supersedes'] as const).map(key => <div className="provenance" key={key}><span>{key.replaceAll('_',' ')}</span>{provenance[key].length ? provenance[key].map(m => <button key={m.id} onClick={() => setSelected(m)}>{short(m.id)} / {m.content}<ArrowRight size={12}/></button>) : <p>None</p>}</div>)}{provenance.superseded_by && <div className="provenance"><span>Superseded by</span><button onClick={() => setSelected(provenance.superseded_by)}>{provenance.superseded_by.content}<ArrowRight size={12}/></button></div>}</>}</> : <div className="inspector-empty"><FileText size={48} strokeWidth={1}/><span>NO FILE OPEN</span><div className="file-lines"/><span className="stamp">AWAITING SELECTION</span></div>}</aside></div>
    <section className="activity"><div className="activity-heading"><span><span className="tiny-square"/> MEMORY ACTIVITY LOG</span><span>{events.length.toString().padStart(2,'0')} EVENTS / THIS SESSION</span></div><div className="activity-rows" aria-live="polite">{events.length ? events.map((e,i) => <div className="activity-row" key={`${e.at}-${i}`}><time>{new Date(e.at).toLocaleTimeString()}</time><Stamp value={e.kind}/><span>{e.message}</span><div>{e.memory_ids.map(id => <button key={id} onClick={() => inspect(id)}>{short(id)}</button>)}</div></div>) : <div className="activity-idle"><span>--:--:--</span>Archive idle. No activity recorded.</div>}</div></section><footer><span>MEMORYOS / ADAPTIVE LONG-TERM MEMORY</span><span>REMEMBER. RECONSIDER. RECALL.</span></footer>
  </div>
}
