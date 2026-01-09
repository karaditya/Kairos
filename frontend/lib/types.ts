// =============================================================================
// Clinical Admin Edge (CAE) System - TypeScript Type Definitions
// =============================================================================

// =============================================================================
// SESSION TYPES
// =============================================================================

export interface SessionStartRequest {
  patient_id?: string;
  ehr_window_title?: string;
  language: "fr" | "en";
}

export interface SessionStartResponse {
  session_id: string;
  websocket_url: string;
  status: string;
}

export interface SessionStateResponse {
  session_id: string;
  status: "active" | "paused" | "completed" | "idle";
  language: string;
  patient_id?: string;
  ehr_window_title?: string;
  transcript?: TranscriptData;
  ehr_data?: EHRData;
  created_at: string;
  updated_at: string;
}

export interface SessionListItem {
  session_id: string;
  status: string;
  language: string;
  patient_id?: string;
  created_at: string;
  updated_at: string;
}

// =============================================================================
// AUDIO / TRANSCRIPT TYPES
// =============================================================================

export interface TranscriptSegment {
  text: string;
  start_time: number;
  end_time: number;
  confidence: number;
  speaker?: string;
  keywords: string[];
}

export interface TranscriptData {
  transcript_id: string;
  session_id: string;
  language: string;
  full_text: string;
  segments: TranscriptSegment[];
  keywords: string[];
  duration_seconds: number;
}

export interface TranscriptResponse extends TranscriptData {}

export interface AudioUploadResponse {
  transcript_id: string;
  status: string;
  message: string;
}

export interface AudioStatusResponse {
  model_loaded: boolean;
  model_name: string;
  device: string;
  language: string;
}

export interface KeywordsResponse {
  session_id: string;
  keywords: string[];
  keywords_by_category: Record<string, string[]>;
}

// WebSocket message types
export interface WSTranscriptMessage {
  type: "transcript";
  segment: TranscriptSegment;
}

export interface WSStatusMessage {
  type: "status";
  status: string;
  message?: string;
}

export interface WSErrorMessage {
  type: "error";
  error: string;
}

export type WSMessage = WSTranscriptMessage | WSStatusMessage | WSErrorMessage;

// =============================================================================
// VISION TYPES
// =============================================================================

export interface ExtractedField {
  value: string;
  confidence: number;
  source: string;
}

export interface UIElement {
  type: string;
  label: string;
  bbox: number[];
}

export interface EHRData {
  extracted_data: Record<string, ExtractedField>;
  ui_elements: UIElement[];
  screenshot_id?: string;
  raw_text?: string;
}

export interface VisionScrapeRequest {
  session_id: string;
  window_title?: string;
  region?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  extract_fields?: string[];
}

export interface VisionScrapeResponse {
  success: boolean;
  extracted_data: Record<string, ExtractedField>;
  ui_elements: UIElement[];
  screenshot_id: string;
  raw_text: string;
}

export interface FieldMapping {
  field_name: string;
  bbox: number[];
}

export interface VisionCalibrateRequest {
  ehr_type: string;
  name: string;
  field_mappings: FieldMapping[];
}

export interface VisionCalibrateResponse {
  success: boolean;
  coordinate_map_id: string;
  message: string;
}

export interface CoordinateMap {
  id: string;
  ehr_type: string;
  name: string;
  fields: Record<string, FieldCoordinate>;
  created_at?: string;
}

export interface FieldCoordinate {
  field_name: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

// =============================================================================
// AGENT / COMPTE RENDU TYPES
// =============================================================================

export interface MissingField {
  field_name: string;
  reason: string;
  required_for: string;
}

export interface FlaggedField {
  field_name: string;
  reason: string;
  requires_clinical_input: boolean;
}

export interface RAGCitation {
  protocol: string;
  source: string;
  relevance: number;
}

export interface CompteRenduContent {
  patient_name?: string;
  patient_dob?: string;
  patient_mrn?: string;
  motif_consultation?: string;
  anamnese?: string;
  antecedents?: string;
  allergies?: string;
  traitements_actuels?: string;
  examen_clinique?: string;
  examens_complementaires?: string;
  hypotheses_diagnostiques?: string;
  plan_therapeutique?: string;
  missing_fields: MissingField[];
  flagged_fields: FlaggedField[];
}

export interface AgentDraftResponse {
  session_id: string;
  compte_rendu_id: string;
  compte_rendu: CompteRenduContent;
  rag_citations: RAGCitation[];
  model_used: string;
  disclaimer: string;
}

export interface SyncRequest {
  session_id: string;
  compte_rendu_id: string;
  target_fields?: string[];
}

export interface SyncPreviewResponse {
  status: string;
  verification_id: string;
  preview: Record<string, any>;
}

export interface SyncApproveRequest {
  user_id: string;
  modifications?: Record<string, string>;
}

export interface SyncResultResponse {
  status: string;
  actions_executed: number;
  timestamp: string;
  errors: string[];
}

export interface PendingVerification {
  verification_id: string;
  session_id: string;
  compte_rendu_id: string;
  status: string;
  created_at: string;
  preview: Record<string, any>;
}

// =============================================================================
// ADMIN / PROTOCOL TYPES
// =============================================================================

export interface ProtocolInput {
  title: string;
  content: string;
  source: string;
  language: string;
  category?: string;
  keywords?: string[];
}

export interface ProtocolSearchResult {
  id: string;
  title: string;
  content: string;
  source: string;
  language: string;
  category?: string;
  relevance: number;
}

export interface ProtocolStats {
  total: number;
  by_language: Record<string, number>;
  by_category: Record<string, number>;
}

// =============================================================================
// MODEL TYPES
// =============================================================================

export interface OllamaModel {
  id: string;
  name: string;
  size_gb: number;
  description: string;
  status: "ready" | "available" | "downloading";
  is_active: boolean;
}

export interface ModelsResponse {
  available: boolean;
  models: OllamaModel[];
  active_model?: string;
  error?: string;
}

export interface PullProgress {
  model_id: string;
  status: "downloading" | "verifying" | "complete" | "error";
  progress: number;
  downloaded_gb: number;
  total_gb: number;
  message: string;
}

// =============================================================================
// SYSTEM STATUS TYPES
// =============================================================================

export interface SystemStatus {
  database: string;
  ollama: string;
  parlant: string;
  whisper: string;
  vision: string;
  qdrant: string;
}

export interface HealthResponse {
  status: string;
}

// =============================================================================
// ERROR TYPES
// =============================================================================

export interface APIError {
  error: string;
  message: string;
  details?: Record<string, any>;
}
