const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')
export async function api<T>(path: string, body?: unknown, timeout = Number(import.meta.env.VITE_API_TIMEOUT_MS) || 600000): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  try {
    const response = await fetch(`${base}${path}`, { method: body === undefined ? 'GET' : 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal })
    const data = await response.json().catch(() => null)
    if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : `Request failed (${response.status}). Check the archive connection.`)
    return data as T
  } catch (error) {
    if (controller.signal.aborted) throw new Error('Request timed out. The server may still be processing changes; refresh before retrying.')
    if (error instanceof TypeError) throw new Error('Archive connection unavailable. Check that the backend is running.')
    throw error
  } finally { clearTimeout(timer) }
}
