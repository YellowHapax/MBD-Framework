/* MBD Lab — Shared TypeScript types */

export interface Vertex {
  name: string;
  symbol: string;
  coords: [number, number, number];
  constructive: boolean;
  description: string;
  dual: string;
}

export interface CubeGeometry {
  vertices: Vertex[];
  constructive_regular: boolean;
  destructive_regular: boolean;
  is_stella_octangula: boolean;
  nature_capture_diagonal: number;
}

export interface Influences {
  nature: number;
  nurture: number;
  heaven: number;
  home: number;
  displacement: number;
  fixation: number;
  degeneration: number;
  capture: number;
}

export interface Lambdas {
  values: number[];
  river: number;
}

export interface BaselineStepResult {
  B_prev: number[];
  B_next: number[];
  delta: number[];
  constructive_sum: number;
  destructive_sum: number;
  balance: number;
  centroid: number[];
}

export interface TrajectoryRequest {
  initial_B: number[];
  steps: number;
  influences: Influences;
  lambdas: Lambdas;
  noise_scale: number;
  seed: number;
}

export interface TrajectoryResult {
  history: number[][];
  centroids: number[][];
  balances: number[];
}

export interface TraumaEvent {
  input_signal: number[];
  lambda_rate: number;
  label: string;
}

export interface InteractionEvent {
  novelty: number;
  duration: number;
  label: string;
}

export interface SimulationRequest {
  initial_baseline: number[];
  initial_kappa: number;
  alpha: number;
  beta: number;
  traumas: TraumaEvent[];
  interactions: InteractionEvent[];
}

export interface SimulationResult {
  baseline_history: number[][];
  kappa_history: number[];
  trauma_labels: string[];
  interaction_labels: string[];
}

/* Baseline component labels */
export const BASELINE_LABELS = ['Trust', 'Aggression', 'Curiosity', 'Status'] as const;
export const BASELINE_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'] as const;

export const VERTEX_NAMES = [
  'nature', 'nurture', 'heaven', 'home',
  'displacement', 'fixation', 'degeneration', 'capture',
] as const;
