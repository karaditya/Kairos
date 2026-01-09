// =============================================================================
// Clinical Admin Edge (CAE) System - API Client
// =============================================================================

import type {
  // Session
  SessionStartRequest,
  SessionStartResponse,
  SessionStateResponse,
  SessionListItem,
  // Audio
  TranscriptResponse,
  AudioUploadResponse,
  AudioStatusResponse,
  KeywordsResponse,
  // Vision
  VisionScrapeRequest,
  VisionScrapeResponse,
  VisionCalibrateRequest,
  VisionCalibrateResponse,
  CoordinateMap,
  // Agent
  AgentDraftResponse,
  SyncPreviewResponse,
  SyncApproveRequest,
  SyncResultResponse,
  PendingVerification,
  // Admin
  ProtocolInput,
  ProtocolSearchResult,
  ProtocolStats,
  ModelsResponse,
  PullProgress,
  SystemStatus,
  HealthResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: "Request failed" }));
    throw new Error(error.detail || error.message || `HTTP ${response.status}`);
  }
  return response.json();
}

function getHeaders(staffPin?: string): HeadersInit {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };
  if (staffPin) {
    headers["X-Staff-Pin"] = staffPin;
  }
  return headers;
}

// =============================================================================
// CAE API CLIENT
// =============================================================================

export const caeApi = {
  // ===========================================================================
  // SESSION ENDPOINTS
  // ===========================================================================
  session: {
    async start(request: SessionStartRequest): Promise<SessionStartResponse> {
      const response = await fetch(`${API_BASE_URL}/session/start`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(request),
      });
      return handleResponse(response);
    },

    async get(sessionId: string): Promise<SessionStateResponse> {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);
      return handleResponse(response);
    },

    async list(): Promise<{ sessions: SessionListItem[]; total: number }> {
      const response = await fetch(`${API_BASE_URL}/session/`);
      return handleResponse(response);
    },

    async pause(sessionId: string): Promise<{ status: string }> {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/pause`, {
        method: "POST",
      });
      return handleResponse(response);
    },

    async resume(sessionId: string): Promise<{ status: string }> {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/resume`, {
        method: "POST",
      });
      return handleResponse(response);
    },

    async end(sessionId: string): Promise<{ status: string; duration_seconds: number }> {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/end`, {
        method: "POST",
      });
      return handleResponse(response);
    },
  },

  // ===========================================================================
  // AUDIO ENDPOINTS
  // ===========================================================================
  audio: {
    getWebSocketUrl(sessionId: string): string {
      const wsBase = API_BASE_URL.replace("http", "ws").replace("https", "wss");
      return `${wsBase}/audio/stream/${sessionId}`;
    },

    async upload(
      sessionId: string,
      file: File,
      language: string = "fr"
    ): Promise<AudioUploadResponse> {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_BASE_URL}/audio/upload?session_id=${sessionId}&language=${language}`,
        {
          method: "POST",
          body: formData,
        }
      );
      return handleResponse(response);
    },

    async getTranscript(sessionId: string): Promise<TranscriptResponse> {
      const response = await fetch(`${API_BASE_URL}/audio/transcript/${sessionId}`);
      return handleResponse(response);
    },

    async getKeywords(sessionId: string): Promise<KeywordsResponse> {
      const response = await fetch(`${API_BASE_URL}/audio/keywords/${sessionId}`);
      return handleResponse(response);
    },

    async getStatus(): Promise<AudioStatusResponse> {
      const response = await fetch(`${API_BASE_URL}/audio/status`);
      return handleResponse(response);
    },
  },

  // ===========================================================================
  // VISION ENDPOINTS
  // ===========================================================================
  vision: {
    async scrape(request: VisionScrapeRequest): Promise<VisionScrapeResponse> {
      const response = await fetch(`${API_BASE_URL}/vision/scrape`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(request),
      });
      return handleResponse(response);
    },

    async calibrate(request: VisionCalibrateRequest): Promise<VisionCalibrateResponse> {
      const response = await fetch(`${API_BASE_URL}/vision/calibrate`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(request),
      });
      return handleResponse(response);
    },

    async detectLayout(windowTitle?: string): Promise<any> {
      const params = windowTitle ? `?window_title=${encodeURIComponent(windowTitle)}` : "";
      const response = await fetch(`${API_BASE_URL}/vision/detect-layout${params}`, {
        method: "POST",
      });
      return handleResponse(response);
    },

    async listWindows(): Promise<{ windows: string[] }> {
      const response = await fetch(`${API_BASE_URL}/vision/windows`);
      return handleResponse(response);
    },

    async listCoordinateMaps(ehrType?: string): Promise<{ maps: CoordinateMap[] }> {
      const params = ehrType ? `?ehr_type=${encodeURIComponent(ehrType)}` : "";
      const response = await fetch(`${API_BASE_URL}/vision/coordinate-maps${params}`);
      return handleResponse(response);
    },

    async getCoordinateMap(ehrType: string, name: string): Promise<CoordinateMap> {
      const response = await fetch(
        `${API_BASE_URL}/vision/coordinate-maps/${encodeURIComponent(ehrType)}/${encodeURIComponent(name)}`
      );
      return handleResponse(response);
    },

    async deleteCoordinateMap(ehrType: string, name: string): Promise<{ deleted: boolean }> {
      const response = await fetch(
        `${API_BASE_URL}/vision/coordinate-maps/${encodeURIComponent(ehrType)}/${encodeURIComponent(name)}`,
        { method: "DELETE" }
      );
      return handleResponse(response);
    },
  },

  // ===========================================================================
  // AGENT ENDPOINTS
  // ===========================================================================
  agent: {
    async generateDraft(
      sessionId: string,
      includeTranscript: boolean = true,
      includeEhr: boolean = true
    ): Promise<AgentDraftResponse> {
      const params = new URLSearchParams({
        session_id: sessionId,
        include_transcript: String(includeTranscript),
        include_ehr: String(includeEhr),
      });
      const response = await fetch(`${API_BASE_URL}/agent/draft?${params}`);
      return handleResponse(response);
    },

    async requestSync(
      sessionId: string,
      compteRenduId: string,
      staffPin: string,
      targetFields?: string[]
    ): Promise<SyncPreviewResponse> {
      const response = await fetch(`${API_BASE_URL}/agent/sync`, {
        method: "POST",
        headers: getHeaders(staffPin),
        body: JSON.stringify({
          session_id: sessionId,
          compte_rendu_id: compteRenduId,
          target_fields: targetFields,
        }),
      });
      return handleResponse(response);
    },

    async approveSync(
      verificationId: string,
      request: SyncApproveRequest,
      staffPin: string
    ): Promise<SyncResultResponse> {
      const response = await fetch(
        `${API_BASE_URL}/agent/sync/${verificationId}/approve`,
        {
          method: "POST",
          headers: getHeaders(staffPin),
          body: JSON.stringify(request),
        }
      );
      return handleResponse(response);
    },

    async rejectSync(
      verificationId: string,
      userId: string,
      staffPin: string
    ): Promise<{ status: string; verification_id: string }> {
      const response = await fetch(
        `${API_BASE_URL}/agent/sync/${verificationId}/reject?user_id=${encodeURIComponent(userId)}`,
        {
          method: "POST",
          headers: getHeaders(staffPin),
        }
      );
      return handleResponse(response);
    },

    async getPending(
      staffPin: string,
      sessionId?: string
    ): Promise<{ pending: PendingVerification[]; count: number }> {
      const params = sessionId ? `?session_id=${sessionId}` : "";
      const response = await fetch(`${API_BASE_URL}/agent/pending${params}`, {
        headers: getHeaders(staffPin),
      });
      return handleResponse(response);
    },

    async getCompteRendu(crId: string): Promise<any> {
      const response = await fetch(`${API_BASE_URL}/agent/compte-rendu/${crId}`);
      return handleResponse(response);
    },

    async verifyCompteRendu(
      crId: string,
      userId: string,
      staffPin: string
    ): Promise<{ verified: boolean; verified_by: string }> {
      const response = await fetch(
        `${API_BASE_URL}/agent/compte-rendu/${crId}/verify?user_id=${encodeURIComponent(userId)}`,
        {
          method: "POST",
          headers: getHeaders(staffPin),
        }
      );
      return handleResponse(response);
    },
  },

  // ===========================================================================
  // ADMIN ENDPOINTS
  // ===========================================================================
  admin: {
    // Protocol Management
    async ingestProtocols(
      protocols: ProtocolInput[],
      staffPin: string
    ): Promise<{ ingested: number; message: string }> {
      const response = await fetch(`${API_BASE_URL}/admin/protocols/ingest`, {
        method: "POST",
        headers: getHeaders(staffPin),
        body: JSON.stringify({ protocols }),
      });
      return handleResponse(response);
    },

    async searchProtocols(
      query: string,
      limit: number = 5,
      language?: string,
      category?: string
    ): Promise<{ results: ProtocolSearchResult[]; count: number }> {
      const response = await fetch(`${API_BASE_URL}/admin/protocols/search`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify({ query, limit, language, category }),
      });
      return handleResponse(response);
    },

    async getProtocolStats(): Promise<ProtocolStats> {
      const response = await fetch(`${API_BASE_URL}/admin/protocols/stats`);
      return handleResponse(response);
    },

    async clearProtocols(staffPin: string): Promise<{ cleared: boolean }> {
      const response = await fetch(`${API_BASE_URL}/admin/protocols/clear`, {
        method: "DELETE",
        headers: getHeaders(staffPin),
      });
      return handleResponse(response);
    },

    // Model Management
    async listModels(): Promise<ModelsResponse> {
      const response = await fetch(`${API_BASE_URL}/admin/models`);
      return handleResponse(response);
    },

    async selectModel(modelId: string, staffPin: string): Promise<{ selected: string }> {
      const response = await fetch(
        `${API_BASE_URL}/admin/models/select?model_id=${encodeURIComponent(modelId)}`,
        {
          method: "POST",
          headers: getHeaders(staffPin),
        }
      );
      return handleResponse(response);
    },

    async pullModel(
      modelId: string,
      staffPin: string,
      onProgress?: (progress: PullProgress) => void
    ): Promise<boolean> {
      const response = await fetch(`${API_BASE_URL}/admin/models/pull/${encodeURIComponent(modelId)}`, {
        headers: getHeaders(staffPin),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Pull failed" }));
        throw new Error(error.detail || "Failed to pull model");
      }

      // Handle NDJSON streaming
      const reader = response.body?.getReader();
      if (!reader) return false;

      const decoder = new TextDecoder();
      let success = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n").filter((l) => l.trim());

        for (const line of lines) {
          try {
            const progress = JSON.parse(line) as PullProgress;
            onProgress?.(progress);
            if (progress.status === "complete") {
              success = true;
            }
          } catch {
            // Ignore parse errors
          }
        }
      }

      return success;
    },

    // System Status
    async getSystemStatus(): Promise<SystemStatus> {
      const response = await fetch(`${API_BASE_URL}/admin/status`);
      return handleResponse(response);
    },

    async healthCheck(): Promise<HealthResponse> {
      const response = await fetch(`${API_BASE_URL}/admin/health`);
      return handleResponse(response);
    },
  },

  // ===========================================================================
  // GENERAL HEALTH CHECK
  // ===========================================================================
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return handleResponse(response);
  },
};

// Export default for convenience
export default caeApi;

// Re-export types
export type * from "./types";
