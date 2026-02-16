/* MBD Lab — Shared TypeScript types */

/* Paper Labs */
export interface LabDescriptor {
  key: string;
  title: string;
  paper: string;
  description: string;
  parameters?: Record<string, unknown>;
}

export interface LabResult {
  timeseries?: Record<string, number[]>;
  summary?: Record<string, unknown>;
  params?: Record<string, unknown>;
  [key: string]: unknown;
}
