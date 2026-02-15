/* MBD Lab — API client */

import type {
  CubeGeometry,
  TrajectoryRequest,
  TrajectoryResult,
  SimulationRequest,
  SimulationResult,
} from './types';

const BASE = '/api';

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

export function getCubeGeometry(): Promise<CubeGeometry> {
  return json('/cube');
}

export function runTrajectory(req: TrajectoryRequest): Promise<TrajectoryResult> {
  return json('/cube/trajectory', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function runTraumaSimulation(req: SimulationRequest): Promise<SimulationResult> {
  return json('/trauma/simulate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}
