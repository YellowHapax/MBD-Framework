/* MBD Lab — API client */

const BASE = '/api';

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

/* Paper Labs */
export function listLabs(): Promise<Record<string, unknown>[]> {
  return json('/labs');
}

export function describeLab(paper: string, lab: string): Promise<Record<string, unknown>> {
  return json(`/labs/${paper}/${lab}/describe`);
}

export function runLab(paper: string, lab: string, params?: Record<string, unknown>): Promise<Record<string, unknown>> {
  return json(`/labs/${paper}/${lab}/run`, {
    method: 'POST',
    body: JSON.stringify(params ?? {}),
  });
}
