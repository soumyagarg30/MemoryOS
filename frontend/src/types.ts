export type MemoryType = 'WORKING' | 'EPISODIC' | 'SEMANTIC' | 'PREFERENCE' | 'TASK'
export type Status = 'ACTIVE' | 'STALE' | 'ARCHIVED' | 'SUPERSEDED'
export interface Memory {
  id: string; user_id: string; content: string; memory_type: MemoryType; status: Status;
  importance: number; confidence: number; created_at: string; updated_at: string;
  last_accessed_at: string | null; access_count: number; expires_at: string | null;
  source: string | null; superseded_by: string | null; consolidated_at: string | null;
  metadata: Record<string, unknown>
}
export interface Activity {kind: string; message: string; memory_ids: string[]; at: string}
export interface Retrieved { memory: Memory; score: number; components: Record<string, number> | null }
export interface ChatResult {assistant_response: string; retrieved_memories: Retrieved[]; created_memories: Memory[]; reinforced_memories: Memory[]; superseded_memories: Memory[]; memory_activity_events: Activity[]; warnings: string[]}
export interface Provenance {derived_from: Memory[]; consolidated_into: Memory[]; supersedes: Memory[]; superseded_by: Memory | null}
export interface Consolidation {memories: {memory: Memory; source_memory_ids: string[]}[]; candidates_examined: number; clusters_skipped: number; next_cursor: string | null}
