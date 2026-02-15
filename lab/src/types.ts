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
  haven: number;
  home: number;
  displacement: number;
  fixation: number;
  erosion: number;
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
  'nature', 'nurture', 'haven', 'home',
  'displacement', 'fixation', 'erosion', 'capture',
] as const;

/* ---- Field Translation ---- */

export interface FieldPole {
  pole: string;
  field: string;
  value: number;
  magnitude: number;
  field_effect: string;
  somatic: string;
  agency: string;
  prompt: string;
}

export interface FieldTranslationResult {
  poles: FieldPole[];
  narrative_prompt: string;
  gravity: number | null;
}

/* ---- Coupling Explorer ---- */

export interface CouplingSeriesResult {
  kappa_history: number[];
  labels: string[];
  alpha: number;
  beta: number;
}

export interface CouplingGridResult {
  alpha_values: number[];
  beta_values: number[];
  final_kappa: number[][];
}

/* ---- Social Fabric ---- */

export interface SocialAgent {
  id: string;
  name: string;
  race: string;
  sex: string;
  age: number;
  trust: number;
  playful: number;
  aggression: number;
  reproductive_drive: number;
  frustration: number;
}

export interface SocialEdge {
  a: string;
  b: string;
  intimacy: number;
  love: number;
  conflict: number;
  pair_bonding: number;
}

export interface SocialSnapshot {
  agents: SocialAgent[];
  edges: SocialEdge[];
  tick: number;
}

export interface SocialSimulationResult {
  snapshots: SocialSnapshot[];
  events: { tick: number; a: string; b: string; probability: number }[];
}

/* ---- Agent Architecture ---- */

export interface AgentState {
  agent_id: string;
  beliefs: Record<string, { value: unknown; certainty: number }>;
  needs: Record<string, number>;
  action: { type: string; direction?: string; target?: unknown };
  nearby_agents: number;
}

export interface AgentStepResult {
  history: AgentState[];
}

/* ---- Resonance Tiers ---- */

export interface ResonanceTier {
  tier: number;
  name: string;
  model: string;
  ai_experience: string;
  description: string;
  example_output: Record<string, unknown>;
}
