export type AgentStatus = 'active' | 'inactive' | 'idle';

export interface AgentMetrics {
  tasks_completed: number;
  tasks_failed: number;
  avg_response_time?: number;
  avg_response_time_ms?: number;
}

export interface Agent {
  status: AgentStatus;
  metrics: {
    tasks_completed: number;
    tasks_failed: number;
    avg_response_time?: number;
    avg_response_time_ms?: number;
  };
}

export type AgentsMap = Record<string, Agent>;

export interface KnowledgeBase {
  total_items: number;
}

export interface EventStream {
  total_events: number;
}

export interface DashboardData {
  agents: AgentsMap;
  knowledge_base: {
    total_items: number;
  };
  event_stream: {
    total_events: number;
  };
}

export interface PolicyRule {
  id: string;
  text: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface PolicyTestSpecification {
  id?: string;
  description?: string;
}


export interface Policy {
  id: string;
  name: string;
  description: string;
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  version: string;
  package_id: string;

  active: boolean;

  created_at: string;   // ISO datetime
  updated_at: string;   // ISO datetime

  regulatory_references: string[];

  rules: PolicyRule[];

  test_specifications: PolicyTestSpecification[];
}

export type AIModel = {
  id: string;
  name: string;
  model_name: string;
  model_type: "api" | "local";
  provider: string;

  description: string;

  api_key: string | null;
  endpoint_url: string | null;

  file_path: string | null;
  file_format: string | null;

  max_tokens: number;
  temperature: number;
  timeout_seconds: number;

  avg_latency_ms: number;
  total_requests: number;
  total_tokens: number;

  status: "active" | "inactive" | "disabled";

  tags: string[];

  created_at: string; // ISO date
  last_tested: string | null;

  test_result: string | null;
};


export interface Recommendation {
  id: string;
  priority: "immediate" | "high" | "medium" | "low";
  title: string;
  description: string;
  finding_id: string;
}

