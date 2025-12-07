// API client for FastAPI backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Types matching backend Pydantic models
export interface StartSessionResponse {
  session_id: string;
  first_question: any;
  progress: number;
}

export interface AnswerResponse {
  next_question: any | null;
  progress: number;
  is_complete: boolean;
  risk_band?: string;
}

export interface SummaryResponse {
  session_id: string;
  ticket_id: string;
  risk_band: string;
  risk_color: string;
  triggered_rules: Array<{ rule: string; description: string }>;
  summary: string;
  key_flags: string[];
  waiting_instruction: string;
  demographics: any;
  answers: any;
  disclaimer: string;
}

export interface CaseResponse {
  id: string;
  session_id: string;
  ticket_id: string;
  created_at: string;
  risk_band: string;
  demographics: any;
  answers: any;
  summary: string;
  key_flags: string[];
  triggered_rules: any[];
  status: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  description: string;
  filename: string;
  download_url: string;
  size_mb: number;
  size_category: string;
  quality: string;
  context_length: number;
  quantization: string;
  speed_rating: number;
  quality_rating: number;
  memory_mb: number;
  tags: string[];
  is_available: boolean;
  is_loaded: boolean;
}

export interface CurrentModel {
  model_id: string;
  name: string;
  family: string;
  loaded_at: string;
  inference_count: number;
  is_loaded: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  current_model: CurrentModel | null;
}

// API functions
export const api = {
  // Patient endpoints
  async startSession(language: string = "en"): Promise<StartSessionResponse> {
    const response = await fetch(`${API_BASE_URL}/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language }),
    });
    if (!response.ok) throw new Error("Failed to start session");
    return response.json();
  },

  async submitDemographics(
    sessionId: string,
    demographics: { age: number; sex: string; pregnant?: boolean }
  ): Promise<any> {
    const response = await fetch(
      `${API_BASE_URL}/session/${sessionId}/demographics`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(demographics),
      }
    );
    if (!response.ok) throw new Error("Failed to submit demographics");
    return response.json();
  },

  async submitComplaint(
    sessionId: string,
    complaint: { complaint_id: string; free_text?: string }
  ): Promise<any> {
    const response = await fetch(
      `${API_BASE_URL}/session/${sessionId}/complaint`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(complaint),
      }
    );
    if (!response.ok) throw new Error("Failed to submit complaint");
    return response.json();
  },

  async submitAnswer(
    sessionId: string,
    questionId: string,
    answer: any
  ): Promise<AnswerResponse> {
    const response = await fetch(
      `${API_BASE_URL}/session/${sessionId}/answer`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, answer }),
      }
    );
    if (!response.ok) throw new Error("Failed to submit answer");
    return response.json();
  },

  async getSummary(sessionId: string): Promise<SummaryResponse> {
    const response = await fetch(
      `${API_BASE_URL}/session/${sessionId}/summary`
    );
    if (!response.ok) throw new Error("Failed to get summary");
    return response.json();
  },

  // Staff endpoints
  async staffAuth(pin: string): Promise<{ authenticated: boolean }> {
    const response = await fetch(`${API_BASE_URL}/staff/auth`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pin }),
    });
    if (!response.ok) throw new Error("Invalid PIN");
    return response.json();
  },

  async getCases(staffPin: string): Promise<{ cases: CaseResponse[] }> {
    const response = await fetch(`${API_BASE_URL}/staff/cases`, {
      headers: { "X-Staff-Pin": staffPin },
    });
    if (!response.ok) throw new Error("Failed to fetch cases");
    return response.json();
  },

  async getCase(caseId: string, staffPin: string): Promise<CaseResponse> {
    const response = await fetch(`${API_BASE_URL}/staff/case/${caseId}`, {
      headers: { "X-Staff-Pin": staffPin },
    });
    if (!response.ok) throw new Error("Failed to fetch case");
    return response.json();
  },

  async askQuestion(
    caseId: string,
    question: string,
    staffPin: string
  ): Promise<{ answer: string; cited_data: string[]; disclaimer: string }> {
    const response = await fetch(
      `${API_BASE_URL}/staff/case/${caseId}/ask`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Staff-Pin": staffPin,
        },
        body: JSON.stringify({ question }),
      }
    );
    if (!response.ok) throw new Error("Failed to ask question");
    return response.json();
  },

  async updateCaseStatus(
    caseId: string,
    status: string,
    staffPin: string
  ): Promise<any> {
    const response = await fetch(
      `${API_BASE_URL}/staff/case/${caseId}/status`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Staff-Pin": staffPin,
        },
        body: JSON.stringify({ status }),
      }
    );
    if (!response.ok) throw new Error("Failed to update case status");
    return response.json();
  },

  // Model management endpoints
  async getModels(): Promise<ModelsResponse> {
    const response = await fetch(`${API_BASE_URL}/models`);
    if (!response.ok) throw new Error("Failed to fetch models");
    return response.json();
  },

  async getCurrentModel(): Promise<{ loaded: boolean; model?: CurrentModel }> {
    const response = await fetch(`${API_BASE_URL}/models/current`);
    if (!response.ok) throw new Error("Failed to fetch current model");
    return response.json();
  },

  async getModelInfo(modelId: string): Promise<ModelInfo> {
    const response = await fetch(`${API_BASE_URL}/models/${modelId}`);
    if (!response.ok) throw new Error("Failed to fetch model info");
    return response.json();
  },

  async switchModel(
    modelId: string,
    staffPin: string
  ): Promise<{ success: boolean; message: string; model: CurrentModel }> {
    const response = await fetch(`${API_BASE_URL}/models/switch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Staff-Pin": staffPin,
      },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to switch model");
    }
    return response.json();
  },

  async getModelStats(staffPin: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/models/stats`, {
      headers: { "X-Staff-Pin": staffPin },
    });
    if (!response.ok) throw new Error("Failed to fetch model stats");
    return response.json();
  },

  async askQuestionWithModel(
    caseId: string,
    question: string,
    staffPin: string,
    modelId?: string
  ): Promise<{
    answer: string;
    reasoning: string | null;
    has_reasoning: boolean;
    suggested_questions: string[];
    cited_data: string[];
    model_used: string;
    disclaimer: string;
  }> {
    const response = await fetch(
      `${API_BASE_URL}/staff/case/${caseId}/ask`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Staff-Pin": staffPin,
        },
        body: JSON.stringify({ question, model_id: modelId }),
      }
    );
    if (!response.ok) throw new Error("Failed to ask question");
    return response.json();
  },
};
