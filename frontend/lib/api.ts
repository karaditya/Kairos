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
  filename?: string;
  download_url?: string;
  size_mb: number;
  size_category?: string;
  quality?: string;
  context_length?: number;
  quantization?: string;
  speed_rating?: number;
  quality_rating?: number;
  memory_mb?: number;
  tags: string[];
  is_available?: boolean;
  is_loaded: boolean;
  is_downloaded: boolean;
  type: "gguf" | "drbert";
  hf_model_id?: string;
  training_data_gb?: number;
}

export interface GpuInfo {
  cuda_available: boolean;
  gpu_name: string | null;
  total_vram_mb: number;
  free_vram_mb: number;
  recommended_gpu_layers: number;
  can_use_gpu: boolean;
}

export interface AllModelsResponse {
  models: ModelInfo[];
  gpu_info: GpuInfo;
  current_gguf: CurrentModel | null;
  current_drbert: CurrentModel | null;
}

export interface DownloadProgress {
  type: "start" | "progress" | "complete" | "error" | "loading" | "loaded" | "gpu_detected";
  model_id?: string;
  progress?: number;
  downloaded_mb?: number;
  total_mb?: number;
  speed_mbps?: number;
  status?: string;
  error?: string;
  success?: boolean;
  gpu_info?: GpuInfo;
  result?: any;
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

// New model types for Ollama-based models
export interface OllamaModel {
  id: string;
  name: string;
  size_gb: number;
  description: string;
  status: 'ready' | 'available' | 'downloading';
  source: 'local' | 'ollama';
  is_active: boolean;
  is_ready: boolean;
  is_available: boolean;
}

export interface OllamaModelsResponse {
  models: OllamaModel[];
  current_model: string;
  ollama_available: boolean;
  ready_count: number;
  available_count: number;
}

export interface PullProgress {
  model_id: string;
  status: 'downloading' | 'verifying' | 'complete' | 'error';
  progress: number;
  downloaded_gb: number;
  total_gb: number;
  message: string;
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

  async generateSummaryPDF(sessionId: string): Promise<Blob> {
    const response = await fetch(
      `${API_BASE_URL}/session/${sessionId}/summary/pdf`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }
    );
    if (!response.ok) throw new Error("Failed to generate PDF");
    return response.blob();
  },

  downloadBlob(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
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
    // RAG-specific fields
    rag_used: boolean;
    protocol_applied: string | null;
    protocol_source: string | null;
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

  // =========================================================================
  // Unified Model Management (GGUF + DrBERT)
  // =========================================================================

  async getAllModels(): Promise<AllModelsResponse> {
    const response = await fetch(`${API_BASE_URL}/models/all`);
    if (!response.ok) throw new Error("Failed to fetch all models");
    return response.json();
  },

  async getGpuInfo(): Promise<GpuInfo> {
    const response = await fetch(`${API_BASE_URL}/gpu/info`);
    if (!response.ok) throw new Error("Failed to fetch GPU info");
    return response.json();
  },

  async downloadModel(
    modelId: string,
    autoLoad: boolean = true,
    onProgress?: (progress: DownloadProgress) => void
  ): Promise<boolean> {
    const response = await fetch(
      `${API_BASE_URL}/download/${modelId}?auto_load=${autoLoad}`,
      { method: "POST" }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to start download");
    }

    // Check if it's already downloaded (JSON response)
    const contentType = response.headers.get("content-type");
    if (contentType?.includes("application/json")) {
      const result = await response.json();
      if (onProgress) {
        onProgress({
          type: "complete",
          model_id: modelId,
          success: true,
          result: result
        });
      }
      return result.loaded || result.status === "already_downloaded";
    }

    // Handle SSE stream for download progress
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let success = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      const lines = text.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6)) as DownloadProgress;
            if (onProgress) onProgress(data);

            if (data.type === "complete" || data.type === "loaded") {
              success = data.success ?? true;
            } else if (data.type === "error") {
              throw new Error(data.error || "Download failed");
            }
          } catch (e) {
            // Ignore parse errors for incomplete chunks
          }
        }
      }
    }

    return success;
  },

  async getDownloadStatus(modelId?: string): Promise<any> {
    const url = modelId
      ? `${API_BASE_URL}/download/status/${modelId}`
      : `${API_BASE_URL}/download/status`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("Failed to fetch download status");
    return response.json();
  },

  // DrBERT specific endpoints
  async loadDrBertModel(modelId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/drbert/load/${modelId}`, {
      method: "POST",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to load DrBERT model");
    }
    return response.json();
  },

  async unloadDrBertModel(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/drbert/unload`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to unload DrBERT model");
    return response.json();
  },

  // =========================================================================
  // Ollama Model Management (New endpoints)
  // =========================================================================

  async getOllamaModels(): Promise<OllamaModelsResponse> {
    const response = await fetch(`${API_BASE_URL}/models`);
    if (!response.ok) throw new Error("Failed to fetch Ollama models");
    return response.json();
  },

  async getReadyModels(): Promise<{ models: OllamaModel[]; current_model: string }> {
    const response = await fetch(`${API_BASE_URL}/models/ready`);
    if (!response.ok) throw new Error("Failed to fetch ready models");
    return response.json();
  },

  async getAvailableModels(): Promise<{ models: OllamaModel[] }> {
    const response = await fetch(`${API_BASE_URL}/models/available`);
    if (!response.ok) throw new Error("Failed to fetch available models");
    return response.json();
  },

  async selectModel(modelId: string): Promise<{ success: boolean; model_id: string; message: string }> {
    const response = await fetch(`${API_BASE_URL}/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to select model");
    }
    return response.json();
  },

  async pullModel(
    modelId: string,
    onProgress?: (progress: PullProgress) => void
  ): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/models/pull/${modelId}`, {
      method: "POST",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to pull model");
    }

    // Check if already pulled (JSON response)
    const contentType = response.headers.get("content-type");
    if (contentType?.includes("application/json")) {
      const result = await response.json();
      if (onProgress) {
        onProgress({
          model_id: modelId,
          status: "complete",
          progress: 1,
          downloaded_gb: 0,
          total_gb: 0,
          message: result.message || "Model ready",
        });
      }
      return result.status === "already_ready";
    }

    // Handle SSE stream for pull progress
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let success = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      const lines = text.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6)) as PullProgress;
            if (onProgress) onProgress(data);

            if (data.status === "complete") {
              success = true;
            } else if (data.status === "error") {
              throw new Error(data.message || "Pull failed");
            }
          } catch (e) {
            // Ignore parse errors for incomplete chunks
          }
        }
      }
    }

    return success;
  },

  async getModelStatus(modelId: string): Promise<{
    model_id: string;
    name: string;
    description: string;
    size_gb: number;
    status: string;
    is_active: boolean;
    is_ready: boolean;
  }> {
    const response = await fetch(`${API_BASE_URL}/models/${modelId}/status`);
    if (!response.ok) throw new Error("Failed to fetch model status");
    return response.json();
  },
};
